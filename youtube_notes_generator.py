import os
import sys
import re
import argparse
import ssl
import logging
import subprocess
import json
import requests
from datetime import datetime

# python3 youtube_notes_generator.py "https://www.youtube.com/watch?v=s9Qh9fWeOAk&t=444s" --api-key AIzaSyD5JceNWUWyozRCp-xcObzcbUzvuhkIR9s

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def fix_ssl_certificates():
    """Fix SSL certificate issues for macOS and other systems"""
    # For macOS, automatically run the certificate installation command
    if sys.platform == 'darwin':
        try:
            # Look for the certificate command
            cert_command = None
            for path in ['/Applications/Python*/Install Certificates.command',
                         f'{sys.prefix}/Install Certificates.command']:
                expanded_paths = subprocess.getoutput(f'ls {path} 2>/dev/null')
                if expanded_paths and 'No such file' not in expanded_paths:
                    cert_command = expanded_paths.split('\n')[0]
                    break

            if cert_command:
                logger.info(f"Found certificate installation command: {cert_command}")
                logger.info("Attempting to run certificate installation (may require password)...")
                result = subprocess.run(['sudo', cert_command], capture_output=True, text=True)
                if result.returncode == 0:
                    logger.info("Successfully installed certificates")
                    return True
                else:
                    logger.warning(f"Certificate installation failed: {result.stderr}")
            else:
                logger.warning("Could not find certificate installation command")
        except Exception as e:
            logger.warning(f"Error during certificate installation: {str(e)}")

    # Try to ensure certifi is installed
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--user", "certifi"])
        import certifi
        ssl_context = ssl.create_default_context(cafile=certifi.where())
        logger.info("Successfully set up SSL with certifi")
        return True
    except Exception as e:
        logger.warning(f"Could not set up SSL certificates: {str(e)}")
        logger.warning("Will continue with SSL verification disabled (not recommended)")
        return False

def install_requirements():
    """Install or upgrade required packages"""
    requirements = ["pytube", "yt-dlp", "requests"]

    for package in requirements:
        try:
            logger.info(f"Installing/upgrading {package}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "--user", package])
            logger.info(f"Successfully installed {package}")
        except Exception as e:
            logger.warning(f"Failed to install {package}: {str(e)}")
            if package == "pytube":
                # If pytube fails, specifically try to install from git for latest fixes
                try:
                    logger.info("Attempting to install pytube from git...")
                    subprocess.check_call([
                        sys.executable, "-m", "pip", "install", "--upgrade", "--user",
                        "git+https://github.com/pytube/pytube.git"
                    ])
                    logger.info("Successfully installed pytube from git")
                except Exception as e2:
                    logger.warning(f"Failed to install pytube from git: {str(e2)}")

def extract_transcript(youtube_url, disable_ssl_verify=False):
    """Extract transcript from a YouTube video using multiple methods"""
    # Disable SSL verification if requested
    original_context = None
    if disable_ssl_verify:
        logger.warning("SSL verification disabled for transcript extraction")
        original_context = ssl._create_default_https_context
        ssl._create_default_https_context = ssl._create_unverified_context

    video_info = {
        "title": "Unknown",
        "author": "Unknown",
        "length": 0,
        "views": 0,
        "publish_date": None
    }

    # Extract video ID
    video_id_match = re.search(r'(?:v=|\/)([0-9A-Za-z_-]{11}).*', youtube_url)
    video_id = video_id_match.group(1) if video_id_match else None

    if not video_id:
        logger.error(f"Could not extract video ID from URL: {youtube_url}")
        if original_context:
            ssl._create_default_https_context = original_context
        return {"error": "Invalid YouTube URL format"}

    # Try multiple methods to extract transcript
    transcript_text = None
    errors = []

    # Method 1: Try pytube
    try:
        logger.info("Attempting to extract transcript using pytube...")
        from pytube import YouTube

        # Add user-agent to look more like a browser
        yt = YouTube(
            youtube_url,
            use_oauth=False,
            allow_oauth_cache=False,
            use_oauth_cache=False
        )

        # Get video information
        video_info = {
            "title": yt.title,
            "author": yt.author,
            "length": yt.length,
            "views": yt.views,
            "publish_date": str(yt.publish_date) if yt.publish_date else None
        }

        # Get captions
        caption_tracks = yt.captions

        # Try to get English captions, fallback to auto-generated if needed
        captions = None
        for track in caption_tracks.all():
            track_code = track.code.lower()
            if 'en' in track_code:
                captions = track
                break

        if captions is None and caption_tracks.all():
            # Fallback to the first available caption
            captions = caption_tracks.all()[0]

        if captions:
            transcript_text = captions.generate_srt_captions()
            # Clean up the SRT format to plain text
            transcript_text = re.sub(r'\d+\n\d{2}:\d{2}:\d{2},\d{3} --> \d{2}:\d{2}:\d{2},\d{3}\n', '', transcript_text)
            transcript_text = re.sub(r'<.*?>', '', transcript_text)
            transcript_text = transcript_text.replace('\n\n', ' ').strip()
            logger.info("Successfully extracted transcript using pytube")
        else:
            errors.append("No captions found using pytube")

    except Exception as e:
        errors.append(f"pytube extraction failed: {str(e)}")
        logger.warning(f"pytube extraction failed: {str(e)}")

    # Method 2: Try yt-dlp if pytube failed
    if not transcript_text:
        try:
            logger.info("Attempting to extract transcript using yt-dlp...")
            import yt_dlp

            ydl_opts = {
                'skip_download': True,
                'writeautomaticsub': True,
                'subtitlesformat': 'srt',
                'quiet': True,
                'no_warnings': True
            }

            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(youtube_url, download=False)

                # Update video info if we didn't get it from pytube
                if video_info["title"] == "Unknown":
                    video_info = {
                        "title": info.get('title', 'Unknown'),
                        "author": info.get('uploader', 'Unknown'),
                        "length": info.get('duration', 0),
                        "views": info.get('view_count', 0),
                        "publish_date": info.get('upload_date', None)
                    }

                # Look for subtitles
                if 'subtitles' in info and info['subtitles']:
                    # Try English first
                    if 'en' in info['subtitles']:
                        subs = info['subtitles']['en']
                    # Then auto-generated English
                    elif 'en-GB' in info['subtitles']:
                        subs = info['subtitles']['en-GB']
                    # Then any other language
                    elif len(info['subtitles']) > 0:
                        lang = list(info['subtitles'].keys())[0]
                        subs = info['subtitles'][lang]
                    else:
                        subs = None

                    if subs and len(subs) > 0:
                        # Get the URL of the first subtitle format
                        sub_url = subs[0]['url']

                        # Download the subtitle file
                        import urllib.request
                        with urllib.request.urlopen(sub_url) as response:
                            srt_content = response.read().decode('utf-8')

                        # Convert SRT to plain text
                        transcript_text = re.sub(r'\d+\n\d{2}:\d{2}:\d{2},\d{3} --> \d{2}:\d{2}:\d{2},\d{3}\n', '', srt_content)
                        transcript_text = re.sub(r'<.*?>', '', transcript_text)
                        transcript_text = transcript_text.replace('\n\n', ' ').strip()
                        logger.info("Successfully extracted transcript using yt-dlp")
                elif 'automatic_captions' in info and info['automatic_captions']:
                    # Try auto-generated captions
                    if 'en' in info['automatic_captions']:
                        subs = info['automatic_captions']['en']
                    elif len(info['automatic_captions']) > 0:
                        lang = list(info['automatic_captions'].keys())[0]
                        subs = info['automatic_captions'][lang]
                    else:
                        subs = None

                    if subs and len(subs) > 0:
                        # Get the URL of the first subtitle format
                        sub_url = subs[0]['url']

                        # Download the subtitle file
                        import urllib.request
                        with urllib.request.urlopen(sub_url) as response:
                            srt_content = response.read().decode('utf-8')

                        # Convert SRT to plain text
                        transcript_text = re.sub(r'\d+\n\d{2}:\d{2}:\d{2},\d{3} --> \d{2}:\d{2}:\d{2},\d{3}\n', '', srt_content)
                        transcript_text = re.sub(r'<.*?>', '', transcript_text)
                        transcript_text = transcript_text.replace('\n\n', ' ').strip()
                        logger.info("Successfully extracted auto-generated transcript using yt-dlp")
                else:
                    errors.append("No subtitles or automatic captions found using yt-dlp")

        except Exception as e:
            errors.append(f"yt-dlp extraction failed: {str(e)}")
            logger.warning(f"yt-dlp extraction failed: {str(e)}")

    # Restore original SSL context if needed
    if original_context:
        ssl._create_default_https_context = original_context

    if transcript_text:
        return {
            "video_info": video_info,
            "transcript": transcript_text
        }
    else:
        error_msg = "; ".join(errors)
        logger.error(f"All transcript extraction methods failed: {error_msg}")
        return {"error": f"Failed to extract transcript: {error_msg}"}

def split_transcript_into_batches(transcript, batch_size=15000, overlap=1000):
    """Split transcript into overlapping batches for processing"""
    batches = []
    start = 0

    while start < len(transcript):
        end = min(start + batch_size, len(transcript))

        # If we're not at the end, try to break at a sentence
        if end < len(transcript):
            # Try to find a sentence end within the last 20% of the batch
            search_start = max(start + int(batch_size * 0.8), start)
            sentence_end = transcript.rfind('. ', search_start, end)
            if sentence_end > search_start:
                end = sentence_end + 1  # Include the period

        batch = transcript[start:end]
        batches.append(batch)

        # Move start position for next batch, with overlap
        if end == len(transcript):
            break
        start = max(0, end - overlap)

    logger.info(f"Split transcript into {len(batches)} batches")
    return batches

def generate_batch_notes(batch, api_key, batch_number, total_batches):
    """Generate notes for a single batch of transcript using Google AI Studio API"""
    try:
        # Create prompt for Google Gemini API
        prompt = f"""
        Create structured notes from this transcript segment from a video (batch {batch_number} of {total_batches}). 
        Include:
        - Key points and concepts
        - Extra insights and explanations 
        - Real-world applications
        - Research-backed insights
        - Common misconceptions
        - Practical implementation tips
        
        Format with clear headings and bullet points. Be detailed yet concise.
        
        Transcript:
        {batch}
        """

        # Generate notes with Google Gemini API
        logger.info(f"Generating notes for batch {batch_number}/{total_batches}...")
        
        url = "https://generativelanguage.googleapis.com/v1/models/gemini-1.5-pro:generateContent"
        headers = {
            "Content-Type": "application/json",
            "x-goog-api-key": api_key
        }
        
        data = {
            "contents": [
                {
                    "parts": [
                        {
                            "text": prompt
                        }
                    ]
                }
            ],
            "generationConfig": {
                "temperature": 0.7,
                "maxOutputTokens": 2000
            }
        }
        
        response = requests.post(url, headers=headers, data=json.dumps(data))
        
        if response.status_code != 200:
            raise Exception(f"API request failed with status {response.status_code}: {response.text}")
            
        result = response.json()
        
        # Extract the generated text from the response
        if "candidates" in result and len(result["candidates"]) > 0:
            if "content" in result["candidates"][0] and "parts" in result["candidates"][0]["content"]:
                generated_text = result["candidates"][0]["content"]["parts"][0]["text"]
            else:
                generated_text = "Error: Unexpected response format"
        else:
            generated_text = "Error: No candidates in response"

        logger.info(f"Successfully generated notes for batch {batch_number}")
        return generated_text

    except Exception as e:
        logger.error(f"Error generating notes for batch {batch_number}: {str(e)}")
        return f"Failed to generate notes for segment {batch_number}: {str(e)}"

def generate_summary(all_batch_notes, api_key, video_title):
    """Generate a final summary based on all the batch notes"""
    try:
        # Combine all batch notes, but limit length to avoid context issues
        combined_notes = "\n\n".join(all_batch_notes)
        if len(combined_notes) > 20000:
            combined_notes = combined_notes[:20000] + "... (notes truncated)"

        # Create prompt for Google Gemini API
        prompt = f"""
        Create a comprehensive summary for a video titled "{video_title}" based on these notes from different segments.
        Include:
        1. A brief overview (1-2 paragraphs)
        2. Key points and main topics
        3. Practical applications
        4. Common misconceptions
        
        Organize with clear headings. Be comprehensive yet concise.
        
        Notes:
        {combined_notes}
        """

        # Generate summary with Google Gemini API
        logger.info("Generating final summary from all batch notes...")
        
        url = "https://generativelanguage.googleapis.com/v1/models/gemini-1.5-pro:generateContent"
        headers = {
            "Content-Type": "application/json",
            "x-goog-api-key": api_key
        }
        
        data = {
            "contents": [
                {
                    "parts": [
                        {
                            "text": prompt
                        }
                    ]
                }
            ],
            "generationConfig": {
                "temperature": 0.7,
                "maxOutputTokens": 2500
            }
        }
        
        response = requests.post(url, headers=headers, data=json.dumps(data))
        
        if response.status_code != 200:
            raise Exception(f"API request failed with status {response.status_code}: {response.text}")
            
        result = response.json()
        
        # Extract the generated text from the response
        if "candidates" in result and len(result["candidates"]) > 0:
            if "content" in result["candidates"][0] and "parts" in result["candidates"][0]["content"]:
                generated_text = result["candidates"][0]["content"]["parts"][0]["text"]
            else:
                generated_text = "Error: Unexpected response format"
        else:
            generated_text = "Error: No candidates in response"

        logger.info("Successfully generated final summary")
        return generated_text

    except Exception as e:
        logger.error(f"Error generating final summary: {str(e)}")
        return f"Failed to generate final summary: {str(e)}"

def clean_llm_output(text):
    """Clean up the LLM output to remove typical artifacts"""
    # Remove any variations of system/user/assistant prefixes
    text = re.sub(r'^\s*(System|User|Assistant):\s*', '', text, flags=re.MULTILINE)

    # Remove any leftover prompt format instructions
    text = re.sub(r'^\s*\{.*?\}\s*$', '', text, flags=re.MULTILINE)

    # Remove transcript segment mentions
    text = re.sub(r'Transcript segment:', '', text)

    # Remove batch references
    text = re.sub(r'This is batch \d+ of \d+ from a video transcript\.', '', text)

    # Remove any markdown code blocks that might contain the original prompt
    text = re.sub(r'```.*?```', '', text, flags=re.DOTALL)

    # Normalize whitespace
    text = re.sub(r'\n{3,}', '\n\n', text)

    return text.strip()

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Generate clean notes from YouTube videos using Google AI Studio API')
    parser.add_argument('url', help='YouTube video URL')
    parser.add_argument('--api-key', required=True, help='Google AI Studio API key')
    parser.add_argument('--output', help='Output file path (default: notes_[video_id].md)')
    parser.add_argument('--disable-ssl-verify', action='store_true',
                        help='Disable SSL verification (use only if necessary)')
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug logging')
    parser.add_argument('--batch-size', type=int, default=15000,
                        help='Size of transcript batches in characters (default: 15000)')
    parser.add_argument('--overlap', type=int, default=1000,
                        help='Overlap between batches in characters (default: 1000)')
    args = parser.parse_args()

    # Set debug logging if requested
    if args.debug:
        logger.setLevel(logging.DEBUG)
        logger.debug("Debug logging enabled")

    # Disable SSL verification if requested (or on macOS by default)
    if args.disable_ssl_verify or sys.platform == 'darwin':
        if sys.platform == 'darwin' and not args.disable_ssl_verify:
            logger.info("macOS detected, disabling SSL verification by default")
        else:
            logger.warning("SSL verification disabled. This is not recommended for security reasons.")
        ssl._create_default_https_context = ssl._create_unverified_context

    # Fix SSL certificates (will be a no-op if SSL verification is disabled)
    fix_ssl_certificates()

    # Install or upgrade required packages
    install_requirements()

    # Extract video ID for filename
    video_id_match = re.search(r'(?:v=|\/)([0-9A-Za-z_-]{11}).*', args.url)
    video_id = video_id_match.group(1) if video_id_match else "unknown"
    output_file = args.output if args.output else f"notes_{video_id}.md"

    # Verify API key is provided
    if not args.api_key:
        logger.error("No Google AI Studio API key provided. Use --api-key to specify your API key.")
        sys.exit(1)

    logger.info(f"Step 1: Extracting transcript from {args.url}")
    result = extract_transcript(args.url, args.disable_ssl_verify)

    if "error" in result:
        logger.error(f"Error: {result['error']}")
        sys.exit(1)

    transcript = result["transcript"]
    video_info = result["video_info"]

    if transcript == "No transcript available for this video.":
        logger.error("Error: No transcript available for this video.")
        sys.exit(1)

    logger.info(f"Successfully extracted transcript ({len(transcript)} characters)")
    logger.info(f"Video: {video_info['title']} by {video_info['author']}")

    # Split transcript into batches
    batches = split_transcript_into_batches(transcript, args.batch_size, args.overlap)

    # Process each batch
    logger.info(f"\nStep 2: Generating notes for {len(batches)} transcript batches")
    batch_notes = []
    for i, batch in enumerate(batches):
        notes = generate_batch_notes(batch, args.api_key, i+1, len(batches))
        # Clean the notes to remove artifacts
        cleaned_notes = clean_llm_output(notes)
        batch_notes.append(cleaned_notes)

    # Generate final summary
    logger.info("\nStep 3: Generating final summary")
    final_summary = generate_summary(batch_notes, args.api_key, video_info['title'])
    # Clean the summary to remove artifacts
    final_summary = clean_llm_output(final_summary)

    # Create markdown file
    logger.info(f"\nStep 4: Saving notes to {output_file}")
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(f"# Notes: {video_info['title']}\n\n")
        f.write("## Video Information\n\n")
        f.write(f"- **Title:** {video_info['title']}\n")
        f.write(f"- **Author:** {video_info['author']}\n")
        f.write(f"- **Length:** {video_info['length']} seconds\n")
        f.write(f"- **Views:** {video_info['views']}\n")
        if video_info['publish_date']:
            f.write(f"- **Published:** {video_info['publish_date']}\n")
        f.write(f"- **Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("\n## Summary\n\n")
        f.write(final_summary)

        f.write("\n\n## Detailed Notes\n\n")
        for i, notes in enumerate(batch_notes):
            f.write(f"### Part {i+1}\n\n")
            f.write(notes)
            f.write("\n\n")

    logger.info(f"✅ Notes successfully generated and saved to {output_file}")

if __name__ == "__main__":
    main()