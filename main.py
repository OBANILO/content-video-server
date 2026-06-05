import os
import time
import uuid
import shutil
import subprocess
import re
from pathlib import Path
from typing import Dict, Any, Optional, List

import requests
from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

APP_NAME = "content-video-server"
BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"
TEMP_DIR = BASE_DIR / "tmp"
OUTPUT_DIR.mkdir(exist_ok=True)
TEMP_DIR.mkdir(exist_ok=True)

app = FastAPI(title=APP_NAME)
JOBS: Dict[str, Dict[str, Any]] = {}

class GenerateRequest(BaseModel):
    api_key: str = Field(..., description="Same acg_api_key from WordPress")
    script: str
    title: str = "Generated Video"
    language: str = "English"
    website: str = ""
    video_type: str = "marketing"
    search_query: str = "business office technology"
    elevenlabs_key: str
    elevenlabs_voice: str
    pexels_key: str
    openai_key: Optional[str] = None

@app.get("/")
def home():
    return {
        "ok": True,
        "service": APP_NAME,
        "endpoints": {
            "generate": "POST /generate",
            "status": "GET /status/{api_key}",
            "outputs": "GET /outputs/{file}.mp4"
        }
    }

@app.post("/generate")
def generate_video(req: GenerateRequest, background_tasks: BackgroundTasks):
    if len(req.script.strip()) < 50:
        raise HTTPException(status_code=400, detail="script too short")
    if not req.elevenlabs_key or not req.elevenlabs_voice:
        raise HTTPException(status_code=400, detail="ElevenLabs key/voice missing")
    if not req.pexels_key:
        raise HTTPException(status_code=400, detail="Pexels key missing")

    job_id = str(uuid.uuid4())
    JOBS[req.api_key] = {
        "job_id": job_id,
        "status": "processing",
        "step": "queued",
        "created_at": time.time(),
        "video_url": "",
        "error": "",
        "title": req.title,
        "video_type": req.video_type,
    }

    background_tasks.add_task(run_generation, req.model_dump(), job_id)
    return {"ok": True, "status": "processing", "job_id": job_id}

@app.get("/status/{api_key}")
def status(api_key: str):
    job = JOBS.get(api_key)
    if not job:
        return {"status": "idle", "step": "no job found"}
    return job

app.mount("/outputs", StaticFiles(directory=str(OUTPUT_DIR)), name="outputs")

def update_job(api_key: str, **kwargs):
    if api_key in JOBS:
        JOBS[api_key].update(kwargs)

def run_generation(data: Dict[str, Any], job_id: str):
    api_key = data["api_key"]
    work = TEMP_DIR / job_id
    work.mkdir(exist_ok=True)

    try:
        update_job(api_key, step="creating voiceover")
        audio_path = work / "voice.mp3"
        make_voiceover(
            text=data["script"],
            elevenlabs_key=data["elevenlabs_key"],
            voice_id=data["elevenlabs_voice"],
            output_path=audio_path
        )

        update_job(api_key, step="getting Pexels clips")
        clips = download_pexels_clips(
            query=data.get("search_query") or "business office technology",
            title=data.get("title", ""),
            script=data.get("script", ""),
            pexels_key=data["pexels_key"],
            work_dir=work,
            max_clips=10
        )

        if not clips:
            update_job(api_key, step="no clips found, creating fallback background")
            clips = [create_color_video(work / "fallback.mp4", duration=30)]

        update_job(api_key, step="creating captions")
        subtitles_path = work / "captions.srt"
        make_simple_srt(data["script"], subtitles_path)

        update_job(api_key, step="rendering final video")
        final_path = OUTPUT_DIR / f"{job_id}.mp4"
        render_video_ffmpeg(
            clips=clips,
            audio_path=audio_path,
            subtitles_path=subtitles_path,
            output_path=final_path
        )

        public_base = os.environ.get("PUBLIC_BASE_URL") or os.environ.get("RENDER_EXTERNAL_URL", "")
        if public_base:
            video_url = public_base.rstrip("/") + f"/outputs/{job_id}.mp4"
        else:
            video_url = f"/outputs/{job_id}.mp4"

        update_job(api_key, status="completed", step="done", video_url=video_url)

    except Exception as e:
        update_job(api_key, status="error", step="failed", error=str(e))
    finally:
        shutil.rmtree(work, ignore_errors=True)

def make_voiceover(text: str, elevenlabs_key: str, voice_id: str, output_path: Path):
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
    payload = {
        "text": text[:5000],
        "model_id": "eleven_multilingual_v2",
        "voice_settings": {
            "stability": 0.45,
            "similarity_boost": 0.75,
            "style": 0.25,
            "use_speaker_boost": True
        }
    }
    headers = {
        "xi-api-key": elevenlabs_key,
        "Content-Type": "application/json",
        "Accept": "audio/mpeg"
    }
    r = requests.post(url, headers=headers, json=payload, timeout=180)
    if r.status_code >= 400:
        raise RuntimeError(f"ElevenLabs error {r.status_code}: {r.text[:300]}")
    output_path.write_bytes(r.content)
    if output_path.stat().st_size < 1000:
        raise RuntimeError("Voiceover file is empty")

def build_pexels_queries(query: str, title: str = "", script: str = "") -> List[str]:
    """
    Build multiple topic-specific Pexels searches.
    This prevents every video from using the same generic b-roll.
    """
    text = f"{query} {title} {script[:1200]}".lower()
    queries = []

    # Use comma/pipe-separated queries from WordPress first.
    for part in str(query or "").replace("|", ",").split(","):
        part = part.strip()
        if part and len(part) >= 3:
            queries.append(part)

    # Topic-specific query packs.
    if any(w in text for w in ["iptv", "live tv", "streaming", "4k", "uhd", "sports channel", "football", "soccer", "nba", "nfl", "world cup"]):
        queries += [
            "watching football on tv",
            "watching sports on television",
            "smart tv remote control",
            "home theater tv",
            "live sports tv screen",
            "soccer match television",
            "family watching tv",
            "man watching tv remote",
            "4k television living room",
            "streaming television remote"
        ]
    elif any(w in text for w in ["instagram", "tiktok", "followers", "social media", "likes", "creator"]):
        queries += [
            "social media phone",
            "creator using phone",
            "instagram phone app",
            "woman using smartphone",
            "content creator laptop",
            "phone scrolling social media",
            "influencer recording video",
            "online marketing phone"
        ]
    elif any(w in text for w in ["chatbot", "ai bot", "ai assistant", "customer support", "live chat"]):
        queries += [
            "customer support chatbot",
            "business ai technology",
            "support agent computer",
            "chat app computer",
            "business dashboard screen",
            "call center support",
            "website chat support"
        ]
    elif any(w in text for w in ["shopify", "payment", "checkout", "ecommerce", "online store"]):
        queries += [
            "online shopping checkout",
            "ecommerce payment laptop",
            "credit card online payment",
            "small business online store",
            "packing ecommerce order",
            "shopping cart checkout"
        ]
    elif any(w in text for w in ["youtube", "subscribers", "channel", "monetization"]):
        queries += [
            "youtube creator camera",
            "content creator desk",
            "video editing laptop",
            "creator recording video",
            "analytics dashboard computer",
            "vlogger filming camera"
        ]
    else:
        queries += [
            "business website laptop",
            "person using laptop",
            "online service computer",
            "digital marketing office",
            "website dashboard screen"
        ]

    # Clean + de-duplicate while preserving order.
    cleaned = []
    seen = set()
    for q in queries:
        q = re.sub(r"[^a-zA-Z0-9\\s-]", " ", q).strip().lower()
        q = re.sub(r"\\s+", " ", q)
        if q and q not in seen:
            cleaned.append(q)
            seen.add(q)

    return cleaned[:12]


def download_pexels_clips(query: str, title: str, script: str, pexels_key: str, work_dir: Path, max_clips: int = 10) -> List[Path]:
    import random

    headers = {"Authorization": pexels_key}
    queries = build_pexels_queries(query, title, script)

    paths = []
    used_video_ids = set()

    # Try several topic-specific searches, random pages, and avoid duplicate videos.
    for search_i, q in enumerate(queries):
        if len(paths) >= max_clips:
            break

        params = {
            "query": q,
            "per_page": 6,
            "orientation": "landscape",
            "size": "medium",
            "page": random.randint(1, 4)
        }

        try:
            r = requests.get("https://api.pexels.com/videos/search", headers=headers, params=params, timeout=45)
            if r.status_code >= 400:
                continue
            videos = r.json().get("videos", [])
        except Exception:
            continue

        random.shuffle(videos)

        for video in videos:
            if len(paths) >= max_clips:
                break

            vid = str(video.get("id", ""))
            if vid and vid in used_video_ids:
                continue

            files = video.get("video_files", [])
            candidates = sorted(
                [f for f in files if f.get("file_type") == "video/mp4" and f.get("link")],
                key=lambda f: abs((f.get("height") or 720) - 720)
            )
            if not candidates:
                continue

            link = candidates[0]["link"]
            out = work_dir / f"clip_{len(paths)}_{search_i}.mp4"

            try:
                with requests.get(link, stream=True, timeout=120) as resp:
                    resp.raise_for_status()
                    with open(out, "wb") as f:
                        for chunk in resp.iter_content(chunk_size=1024 * 1024):
                            if chunk:
                                f.write(chunk)
                if out.exists() and out.stat().st_size > 50000:
                    paths.append(out)
                    if vid:
                        used_video_ids.add(vid)
            except Exception:
                continue

    return paths

def create_color_video(output_path: Path, duration: int = 30) -> Path:
    cmd = [
        "ffmpeg", "-y",
        "-f", "lavfi", "-i", f"color=c=black:s=1280x720:d={duration}",
        "-pix_fmt", "yuv420p", str(output_path)
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return output_path

def get_duration(path: Path) -> float:
    cmd = [
        "ffprobe", "-v", "error", "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1", str(path)
    ]
    out = subprocess.check_output(cmd).decode().strip()
    return float(out)

def make_simple_srt(script: str, output_path: Path):
    words = script.replace("\n", " ").split()
    chunks = []
    for i in range(0, len(words), 9):
        chunks.append(" ".join(words[i:i+9]))

    def fmt(sec: float) -> str:
        h = int(sec // 3600)
        m = int((sec % 3600) // 60)
        s = int(sec % 60)
        ms = int((sec - int(sec)) * 1000)
        return f"{h:02}:{m:02}:{s:02},{ms:03}"

    lines = []
    t = 0.0
    for idx, chunk in enumerate(chunks[:400], start=1):
        start = t
        end = t + 3.5
        lines.append(str(idx))
        lines.append(f"{fmt(start)} --> {fmt(end)}")
        lines.append(chunk)
        lines.append("")
        t = end

    output_path.write_text("\n".join(lines), encoding="utf-8")

def render_video_ffmpeg(clips: List[Path], audio_path: Path, subtitles_path: Path, output_path: Path):
    audio_duration = max(10, get_duration(audio_path))
    concat_file = TEMP_DIR / f"concat_{output_path.stem}.txt"

    entries = []
    total = 0.0
    index = 0
    while total < audio_duration + 3:
        clip = clips[index % len(clips)]
        entries.append(f"file '{clip.as_posix()}'")
        try:
            total += max(2, get_duration(clip))
        except Exception:
            total += 5
        index += 1

    concat_file.write_text("\n".join(entries), encoding="utf-8")

    sub_path = subtitles_path.as_posix().replace(":", "\\:")

    # ✅ NO TOP TITLE AND NO WEBSITE WATERMARK.
    # Only subtitles/captions remain at the bottom.
    vf = (
        "scale=1280:720:force_original_aspect_ratio=increase,"
        "crop=1280:720,"
        "format=yuv420p,"
        f"subtitles='{sub_path}':force_style='Fontsize=24,Outline=2,Shadow=1,Alignment=2'"
    )

    cmd = [
        "ffmpeg", "-y",
        "-f", "concat", "-safe", "0", "-i", str(concat_file),
        "-i", str(audio_path),
        "-t", str(audio_duration),
        "-vf", vf,
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "24",
        "-c:a", "aac", "-b:a", "160k",
        "-shortest",
        str(output_path)
    ]

    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        concat_file.unlink(missing_ok=True)
    except Exception:
        pass

    if result.returncode != 0:
        raise RuntimeError(result.stderr.decode(errors="ignore")[-1200:])
    if not output_path.exists() or output_path.stat().st_size < 100000:
        raise RuntimeError("Final video render failed or file too small")