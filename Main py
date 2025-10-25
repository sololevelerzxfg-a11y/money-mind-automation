# main.py
# MoneyMind - automated script -> TTS -> visuals -> music -> assemble -> (optional) upload
# NOTE: requires environment secrets:
# - CHATGPT_KEY (OpenAI)
# - PEXELS_API_KEY
# - PIXABAY_API_KEY
# - YOUTUBE_API_KEY (for some lookups). For uploads provide optional YT_CLIENT_ID, YT_CLIENT_SECRET, YT_REFRESH_TOKEN

import os
import json
import tempfile
import random
import textwrap
from pathlib import Path
from datetime import datetime
import requests
from PIL import Image, ImageDraw, ImageFont
from moviepy.editor import (
    AudioFileClip, ImageClip, concatenate_videoclips, CompositeVideoClip, TextClip, VideoFileClip
)

# ---------- Config ----------
OUTDIR = Path("outputs")
CLIPS_DIR = Path("clips")
MUSIC_DIR = Path("music")
THUMBS_DIR = Path("thumbnails")
OUTDIR.mkdir(exist_ok=True)
CLIPS_DIR.mkdir(exist_ok=True)
MUSIC_DIR.mkdir(exist_ok=True)
THUMBS_DIR.mkdir(exist_ok=True)

OPENAI_KEY = os.getenv("CHATGPT_KEY")
PEXELS_KEY = os.getenv("PEXELS_API_KEY")
PIXABAY_KEY = os.getenv("PIXABAY_API_KEY")
YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")

# Optional YouTube OAuth (for real upload; not required to run generation)
YT_CLIENT_ID = os.getenv("YT_CLIENT_ID")
YT_CLIENT_SECRET = os.getenv("YT_CLIENT_SECRET")
YT_REFRESH_TOKEN = os.getenv("YT_REFRESH_TOKEN")

# Basic headers
OPENAI_HEADERS = {"Authorization": f"Bearer {OPENAI_KEY}"}
PEXELS_HEADERS = {"Authorization": PEXELS_KEY}

# ---------- Helpers ----------
def gpt_generate(prompt, max_tokens=800):
    """Generate script or metadata using OpenAI ChatCompletion (gpt-3.5-turbo)."""
    url = "https://api.openai.com/v1/chat/completions"
    body = {
        "model": "gpt-3.5-turbo",
        "messages": [{"role":"user","content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.7
    }
    r = requests.post(url, headers=OPENAI_HEADERS, json=body, timeout=60)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"].strip()

def generate_script_and_meta(topic, long_minutes=10):
    """Return dict with script_long, script_short, title, description, tags."""
    # Long video prompt
    p_long = f"""You are an expert motivational financial coach. Produce an {long_minutes}-minute YouTube video script for the topic:
{topic}
Structure: Hook (10-20s), 6-8 points with short examples, transitions, and a strong call-to-action. Return only the script (no headers)."""
    script_long = gpt_generate(p_long, max_tokens=1400)

    p_short = f"""Write a 50-60 second high-energy hook script for the same topic: {topic}. Make it punchy and perfect for a Short."""
    script_short = gpt_generate(p_short, max_tokens=300)

    p_meta = f"""From this topic: {topic}, create:
1) A clickable YouTube title under 80 chars.
2) A 2-sentence SEO-friendly description with a call to action and placeholder for affiliate links: [AFF_LINKS]
3) 8 tags (comma-separated).
Return as JSON: {{ "title": "...", "description": "...", "tags": ["a","b"] }} only."""
    meta_json = gpt_generate(p_meta, max_tokens=250)
    try:
        meta = json.loads(meta_json)
    except Exception:
        # fallback parse
        meta = {"title": " ".join(topic.split()[:6]) + " - Money Mind", "description": f"{topic} - watch to learn.", "tags": ["finance","money","mindset"]}
    return {"script_long": script_long, "script_short": script_short, **meta}

# ---------- TTS (OpenAI) ----------
def text_to_speech_openai(text, out_path: Path, voice="alloy"):
    """Uses OpenAI TTS endpoint (if available) else fallback to gTTS."""
    try:
        # Try OpenAI TTS (newer endpoints may differ per SDK; using REST call)
        url = "https://api.openai.com/v1/audio/speech"
        payload = {
            "model": "gpt-4o-mini-tts",
            "voice": voice
        }
        headers = {"Authorization": f"Bearer {OPENAI_KEY}", "Content-Type": "application/json"}
        # The API expects binary streaming of audio; many accounts may not have access.
        # For robustness we fallback to gTTS if OpenAI TTS fails.
        r = requests.post(url, headers=headers, json={"input": text, "voice": voice})
        if r.status_code == 200 and r.headers.get("content-type","").startswith("audio"):
            out_path.write_bytes(r.content)
            return str(out_path)
    except Exception:
        pass

    # Fallback: gTTS (very reliable, needs `gTTS` installed)
    try:
        from gtts import gTTS
        tts = gTTS(text=text, lang="en", slow=False)
        tts.save(str(out_path))
        return str(out_path)
    except Exception as e:
        print("TTS failed:", e)
        raise

# ---------- Stock fetch (Pexels + Pixabay) ----------
def fetch_pexels_videos(query, count=3):
    """Download top Pexels videos for a query."""
    url = "https://api.pexels.com/videos/search"
    params = {"query": query, "per_page": count}
    r = requests.get(url, headers=PEXELS_HEADERS, params=params, timeout=30)
    r.raise_for_status()
    data = r.json()
    files = []
    for v in data.get("videos", [])[:count]:
        # choose best video file
        file_url = v["video_files"][-1]["link"]
        dest = CLIPS_DIR / f"pexels_{v['id']}.mp4"
        if not dest.exists():
            dl = requests.get(file_url, stream=True, timeout=60)
            with open(dest, "wb") as f:
                for chunk in dl.iter_content(1024*1024):
                    f.write(chunk)
        files.append(str(dest))
    return files

def fetch_pixabay_media(query, count=3, media_type="video"):
    """Download videos or images from Pixabay."""
    if media_type == "video":
        url = "https://pixabay.com/api/videos/"
    else:
        url = "https://pixabay.com/api/"
    params = {"key": PIXABAY_KEY, "q": query, "per_page": count}
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    hits = r.json().get("hits", [])[:count]
    out = []
    for h in hits:
        if media_type == "video":
            video_url = h.get("videos", {}).get("medium", {}).get("url")
            if video_url:
                dest = CLIPS_DIR / f"pix_{h['id']}.mp4"
                if not dest.exists():
                    dl = requests.get(video_url, stream=True, timeout=60)
                    with open(dest, "wb") as f:
                        for chunk in dl.iter_content(1024*1024):
                            f.write(chunk)
                out.append(str(dest))
        else:
            img_url = h.get("largeImageURL")
            if img_url:
                dest = CLIPS_DIR / f"pix_img_{h['id']}.jpg"
                if not dest.exists():
                    dl = requests.get(img_url, stream=True, timeout=60)
                    with open(dest, "wb") as f:
                        for chunk in dl.iter_content(1024*1024):
                            f.write(chunk)
                out.append(str(dest))
    return out

# ---------- Music (Pixabay) ----------
def fetch_pixabay_music(query="motivational", count=5):
    """Pixabay music endpoint - fetch a random music track and download it."""
    url = "https://pixabay.com/api/music/"
    params = {"key": PIXABAY_KEY, "q": query, "per_page": count}
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    hits = r.json().get("hits", [])[:count]
    if not hits:
        return None
    track = random.choice(hits)
    track_url = track.get("url")
    # Pixabay music sometimes returns mp3 URL in assets
    if track_url:
        dest = MUSIC_DIR / f"pix_music_{track['id']}.mp3"
        if not dest.exists():
            dl = requests.get(track_url, stream=True, timeout=60)
            with open(dest, "wb") as f:
                for chunk in dl.iter_content(1024*1024):
                    f.write(chunk)
        return str(dest)
    return None

# ---------- Thumbnail generator ----------
def create_thumbnail(title_text, out_path: Path):
    """Simple thumbnail generator using PIL."""
    W, H = 1280, 720
    bg_color = (20, 20, 30)
    img = Image.new("RGB", (W, H), color=bg_color)
    draw = ImageDraw.Draw(img)
    # Try to load a bold font
    try:
        fnt = ImageFont.truetype("arialbd.ttf", 72)
    except:
        fnt = ImageFont.load_default()
    # Big title
    lines = textwrap.wrap(title_text, width=18)
    y = 120
    for line in lines[:3]:
        draw.text((60, y), line, font=fnt, fill=(255, 230, 0))
        y += 90
    # small footer
    draw.text((60, H - 80), "Money Mind • Subscribe", font=fnt, fill=(255,255,255))
    img.save(out_path)
    return str(out_path)

# ---------- Assemble video ----------
def assemble_video(voice_file, clip_files, music_file, out_file, duration_target=None):
    """
    Create a video: sequence of clips timed to narration length.
    Simple approach: stretch/cut clips to match audio length.
    """
    audio = AudioFileClip(str(voice_file))
    audio_dur = audio.duration
    clips = []
    if not clip_files:
        # if no clips, create a single image slide
        img_clip = ImageClip(str(THUMBS_DIR / "fallback.jpg")).set_duration(audio_dur).resize((1280,720))
        clips = [img_clip]
    else:
        # distribute clips to cover audio duration
        per_clip = max(2, audio_dur / max(1, len(clip_files)))
        for cf in clip_files:
            try:
                v = VideoFileClip(cf).subclip(0, min(per_clip, VideoFileClip(cf).duration)).resize((1280,720))
            except Exception:
                # fallback to image clip
                v = ImageClip(cf).set_duration(per_clip).resize((1280,720))
            clips.append(v)
    final = concatenate_videoclips(clips, method="compose")
    final = final.set_audio(audio)
    if music_file:
        bg = AudioFileClip(str(music_file)).volumex(0.12).set_duration(audio_dur)
        # composite audio: voice on top of bg
        final_audio = CompositeAudioClip([bg, audio.set_start(0)])
        final = final.set_audio(final_audio)
    final.write_videofile(str(out_file), fps=24, codec="libx264", audio_codec="aac", threads=2, verbose=False, logger=None)
    return str(out_file)

# ---------- Uploader (optional) ----------
def upload_to_youtube(video_path, title, description, tags, thumb_path=None, privacy="public"):
    """
    Upload only if OAuth credentials provided. If no OAuth, we skip.
    For a true upload you must provide YT_CLIENT_ID, YT_CLIENT_SECRET and YT_REFRESH_TOKEN as repo secrets.
    """
    if not (YT_CLIENT_ID and YT_CLIENT_SECRET and YT_REFRESH_TOKEN):
        print("YouTube OAuth credentials not provided. Skipping upload.")
        return None

    # Use googleapiclient to build an authenticated service by exchanging refresh token (not included here)
    # This part requires google-auth and oauth2client setup and is environment-specific.
    print("Upload logic here - OAuth found. Implement upload with google-auth client and youtube API.")
    return None

# ---------- Main workflow ----------
def run_cycle(topic):
    stamp = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    print("Generating scripts for:", topic)
    meta = generate_script_and_meta(topic, long_minutes=10)

    # create voice for long script and short
    long_voice_path = OUTDIR / f"voice_long_{stamp}.mp3"
    short_voice_path = OUTDIR / f"voice_short_{stamp}.mp3"
    print("TTS long...")
    text_to_speech_openai(meta["script_long"], long_voice_path)
    print("TTS short...")
    text_to_speech_openai(meta["script_short"], short_voice_path)

    # fetch clips
    keywords = topic
    clips = []
    try:
        clips += fetch_pexels_videos(keywords, count=3)
    except Exception as e:
        print("Pexels fail:", e)
    try:
        clips += fetch_pixabay_media(keywords, count=3, media_type="video")
    except Exception as e:
        print("Pixabay fail:", e)

    # fetch music
    music = fetch_pixabay_music(query="motivational")
    print("Music:", music)

    # thumbnail
    thumb_file = THUMBS_DIR / f"thumb_{stamp}.jpg"
    create_thumbnail(meta.get("title","Money Mind"), thumb_file)

    # assemble long video
    long_out = OUTDIR / f"money_mind_long_{stamp}.mp4"
    print("Assembling long video...")
    assemble_video(long_voice_path, clips, music, long_out)

    # assemble short video (use same audio trimmed to 55s)
    short_out = OUTDIR / f"money_mind_short_{stamp}.mp4"
    print("Assembling short video...")
    # For short we create a clipped audio file
    # Here we rely on the TTS output being long enough; we trim movie accordingly
    assemble_video(short_voice_path, clips[:2], music, short_out)

    # upload (optional)
    print("Attempting upload...")
    upload_to_youtube(long_out, meta.get("title","Money Mind"), meta.get("description",""), meta.get("tags",[]), thumb_file)
    upload_to_youtube(short_out, meta.get("title","Money Mind - Short"), meta.get("description",""), meta.get("tags",[]), thumb_file)

    print("Cycle complete. Files saved to", OUTDIR)

if __name__ == "__main__":
    # Example topics rotation (you can replace or connect to a trends API)
    topics = [
        "5 passive income ideas for teens",
        "How to save and invest your first $100",
        "The brutal truth about working a 9-5",
        "How the rich think about money"
    ]
    topic = random.choice(topics)
    run_cycle(topic)
