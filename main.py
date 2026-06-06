import os
import time
import uuid
import shutil
import subprocess
import random
import re
from pathlib import Path
from typing import Dict, Any, Optional, List

import requests
from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

APP_NAME = "content-video-server-conversion-engine"
BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"
TEMP_DIR = BASE_DIR / "tmp"

OUTPUT_DIR.mkdir(exist_ok=True)
TEMP_DIR.mkdir(exist_ok=True)

app = FastAPI(title=APP_NAME)
JOBS: Dict[str, Dict[str, Any]] = {}

class GenerateRequest(BaseModel):
    api_key: str
    script: str
    title: str = "Generated Video"
    language: str = "English"
    website: str = ""
    video_type: str = "marketing"
    search_query: str = "business website laptop"
    elevenlabs_key: str
    elevenlabs_voice: str
    pexels_key: str
    openai_key: Optional[str] = None

    niche: Optional[str] = ""
    service_name: Optional[str] = ""
    main_offer: Optional[str] = ""
    benefits: Optional[List[str]] = []
    cta: Optional[str] = ""
    screenshot_urls: Optional[List[str]] = []
    conversion_goal: Optional[str] = ""

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
    if not req.api_key:
        raise HTTPException(status_code=400, detail="api_key missing")
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
        "updated_at": time.time(),
        "video_url": "",
        "error": "",
        "title": req.title,
        "video_type": req.video_type,
        "niche": req.niche or ""
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
        JOBS[api_key]["updated_at"] = time.time()

def run_generation(data: Dict[str, Any], job_id: str):
    api_key = data["api_key"]
    work = TEMP_DIR / job_id
    work.mkdir(exist_ok=True)

    try:
        niche = data.get("niche") or detect_niche(data)

        update_job(api_key, step="creating voiceover", niche=niche)
        audio_path = work / "voice.mp3"
        make_voiceover(
            text=data["script"],
            elevenlabs_key=data["elevenlabs_key"],
            voice_id=data["elevenlabs_voice"],
            output_path=audio_path
        )

        update_job(api_key, step="creating conversion images")
        conversion_images = generate_conversion_images(data, niche, work)

        update_job(api_key, step="capturing website screenshots")
        screenshots = capture_website_screenshots(data, work)

        update_job(api_key, step="getting Pexels clips")
        clips = download_pexels_clips(
            query=data.get("search_query") or "business website laptop",
            title=data.get("title", ""),
            script=data.get("script", ""),
            niche=niche,
            pexels_key=data["pexels_key"],
            work_dir=work,
            max_clips=32
        )

        update_job(api_key, step="creating captions")
        subtitles_path = work / "captions.srt"
        make_simple_srt(data["script"], subtitles_path)

        update_job(api_key, step="rendering final video")
        final_path = OUTPUT_DIR / f"{job_id}.mp4"
        render_mixed_video(
            clips=clips,
            images=conversion_images + screenshots,
            audio_path=audio_path,
            subtitles_path=subtitles_path,
            output_path=final_path
        )

        public_base = os.environ.get("PUBLIC_BASE_URL") or os.environ.get("RENDER_EXTERNAL_URL", "")
        video_url = (public_base.rstrip("/") + f"/outputs/{job_id}.mp4") if public_base else f"/outputs/{job_id}.mp4"
        update_job(api_key, status="completed", step="done", video_url=video_url)

    except Exception as e:
        update_job(api_key, status="error", step="failed", error=str(e))
    finally:
        shutil.rmtree(work, ignore_errors=True)

def detect_niche(data: Dict[str, Any]) -> str:
    text = " ".join([
        str(data.get("website", "")),
        str(data.get("title", "")),
        str(data.get("script", ""))[:1500],
        str(data.get("search_query", "")),
        str(data.get("service_name", "")),
        str(data.get("main_offer", "")),
    ]).lower()

    if any(x in text for x in ["iptv", "live tv", "4k tv", "uhd", "streaming", "world cup", "sports channels"]):
        return "iptv"
    if "facebook" in text and any(x in text for x in ["followers", "likes", "page growth", "social proof"]):
        return "facebook_followers"
    if "instagram" in text and any(x in text for x in ["followers", "likes", "growth"]):
        return "instagram_growth"
    if "tiktok" in text and any(x in text for x in ["followers", "likes", "views"]):
        return "tiktok_growth"
    if "youtube" in text and any(x in text for x in ["subscribers", "watch hours", "monetization", "channel"]):
        return "youtube_growth"
    if any(x in text for x in ["chatbot", "ai bot", "ai assistant", "customer support", "lead capture"]):
        return "ai_chatbot"
    if any(x in text for x in ["shopify", "payment", "checkout", "ecommerce", "stripe"]):
        return "ecommerce_payment"
    return "general_business"

def make_voiceover(text: str, elevenlabs_key: str, voice_id: str, output_path: Path):
    text = text.strip()
    if len(text) > 5000:
        text = text[:5000]

    url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
    payload = {
        "text": text,
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
    r = requests.post(url, headers=headers, json=payload, timeout=240)
    if r.status_code >= 400:
        raise RuntimeError(f"ElevenLabs error {r.status_code}: {r.text[:300]}")
    output_path.write_bytes(r.content)
    if output_path.stat().st_size < 1000:
        raise RuntimeError("Voiceover file is empty")

def generate_conversion_images(data: Dict[str, Any], niche: str, work_dir: Path) -> List[Path]:
    openai_key = (data.get("openai_key") or "").strip()
    if not openai_key:
        return []

    prompts = build_conversion_image_prompts(data, niche)
    image_paths: List[Path] = []

    for i, prompt in enumerate(prompts[:5]):
        out_raw = work_dir / f"conversion_{i}.png"
        out_jpg = work_dir / f"conversion_{i}.jpg"

        try:
            response = requests.post(
                "https://api.openai.com/v1/images/generations",
                headers={
                    "Authorization": f"Bearer {openai_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": "gpt-image-1",
                    "size": "1536x1024",
                    "prompt": prompt,
                },
                timeout=240,
            )
            if response.status_code >= 400:
                continue

            data_json = response.json()
            b64 = data_json.get("data", [{}])[0].get("b64_json")
            if not b64:
                continue

            import base64
            out_raw.write_bytes(base64.b64decode(b64))
            convert_image_to_video_frame(out_raw, out_jpg)
            if out_jpg.exists() and out_jpg.stat().st_size > 50000:
                image_paths.append(out_jpg)
        except Exception:
            continue

    return image_paths

def build_conversion_image_prompts(data: Dict[str, Any], niche: str) -> List[str]:
    title = data.get("title", "Generated Video")
    website = data.get("website", "")
    service = data.get("service_name", "") or title
    cta = data.get("cta", "")

    base = (
        "Create a realistic high-conversion marketing image for a YouTube video. "
        "16:9 horizontal. Modern premium style. Clear visual message. "
        "No tiny text, no paragraphs, no watermark, no fake brand logos. "
        "The image should look like a conversion scene inside a video, not a thumbnail. "
    )

    if niche == "facebook_followers":
        return [
            base + f"Show a realistic Facebook business page/profile mockup on a phone or laptop with strong social proof, visible follower growth, active posts, likes, and engagement. Topic: {service}. Website: {website}.",
            base + "Create a before-and-after social proof scene: left side small empty Facebook page with low followers, right side trusted active Facebook page with more followers and better engagement. Use clear growth arrow, professional style.",
            base + "Create a safe order process visual for buying Facebook followers: show laptop checkout style, no password needed, safe growth, fast delivery, real followers. Do not show real private data.",
            base + "Create a business owner looking happy while checking a growing Facebook page on smartphone, with notifications and follower growth visuals. Realistic, high trust, conversion focused.",
            base + f"Create a final CTA image for {website}: professional social media growth look, strong trust feeling, Facebook page growth, clear call-to-action mood. {cta}",
        ]

    if niche == "iptv":
        return [
            base + "Show premium 4K IPTV service visual: large smart TV with live sports channels, remote control, TV box, dark cinematic living room, 4K/UHD feeling.",
            base + "Show no buffering IPTV benefit: smooth live sports on TV, strong WiFi/streaming symbol, happy viewer, premium sports entertainment vibe.",
            base + "Show IPTV setup process: smart TV, app login screen, remote, simple steps, high quality streaming look. Avoid fake brand logos.",
            base + "Show IPTV packages/checkout style on laptop or phone, safe subscription purchase, live TV and sports background, premium conversion image.",
            base + "Final CTA IPTV visual: sports, movies, live TV, 4K streaming, remote control, dark premium background, high trust feel.",
        ]

    if niche == "instagram_growth":
        return [
            base + "Show Instagram profile growth mockup on phone, more followers, likes, engagement, clean professional creator profile. High-conversion social proof style.",
            base + "Before and after Instagram profile growth: low followers vs trusted profile with strong engagement, arrow, clean bright modern layout.",
            base + "Safe order visual for Instagram growth: no password needed, real followers, fast delivery, simple checkout style on laptop.",
            base + "Creator smiling while checking Instagram growth analytics on phone, notifications, social proof, premium digital marketing style.",
        ]

    if niche == "youtube_growth":
        return [
            base + "Show YouTube channel growth dashboard mockup with subscribers increasing, watch hours, monetization progress, professional creator studio vibe.",
            base + "Before/after YouTube channel growth: zero traction versus strong subscriber count and views, high conversion style.",
            base + "Show 1000 subscribers and 4000 watch hours concept with progress bars and creator looking motivated, premium YouTube growth style.",
            base + "Safe order process visual for YouTube subscribers/watch hours, no password needed, fast delivery, clean checkout style.",
        ]

    if niche == "ai_chatbot":
        return [
            base + "Show AI chatbot on a business website helping customers, chat window, lead capture, modern SaaS dashboard, premium tech style.",
            base + "Show before/after customer support: busy manual support vs automated AI chatbot handling messages quickly, clear business value.",
            base + "Show lead generation with AI chatbot: website visitor chatting, captured email/phone lead, business owner happy.",
            base + "Show chatbot dashboard analytics with conversations, leads, automation, clean modern UI, high conversion SaaS image.",
        ]

    return [
        base + f"Show a website/service growth visual for {service}. Website: {website}. Use professional laptop/phone mockups and high trust conversion style.",
        base + "Show before and after business growth online, more users, more trust, more sales, clean premium design.",
        base + "Show a safe checkout/order process visual, no private data, clean professional high-conversion layout.",
        base + "Show final call-to-action visual for an online service, website on screen, happy user, trust and growth mood.",
    ]

def capture_website_screenshots(data: Dict[str, Any], work_dir: Path) -> List[Path]:
    template = os.environ.get("WEBSITE_SCREENSHOT_API", "").strip()
    if not template:
        return []

    urls = data.get("screenshot_urls") or []
    website = data.get("website", "")
    if website:
        urls.insert(0, website)

    clean_urls, seen = [], set()
    for u in urls:
        u = str(u).strip()
        if not u:
            continue
        if not u.startswith("http"):
            u = "https://" + u
        if u not in seen:
            clean_urls.append(u)
            seen.add(u)

    paths = []
    for i, url in enumerate(clean_urls[:4]):
        try:
            shot_url = template.replace("{url}", requests.utils.quote(url, safe=""))
            resp = requests.get(shot_url, timeout=90)
            if resp.status_code >= 400 or len(resp.content) < 10000:
                continue

            raw = work_dir / f"screenshot_{i}.png"
            jpg = work_dir / f"screenshot_{i}.jpg"
            raw.write_bytes(resp.content)
            convert_image_to_video_frame(raw, jpg)
            if jpg.exists() and jpg.stat().st_size > 50000:
                paths.append(jpg)
        except Exception:
            continue

    return paths

def convert_image_to_video_frame(input_path: Path, output_path: Path):
    cmd = [
        "ffmpeg", "-y",
        "-i", str(input_path),
        "-vf", "scale=1280:720:force_original_aspect_ratio=increase,crop=1280:720,format=yuv420p",
        "-frames:v", "1",
        str(output_path)
    ]
    subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)

def build_pexels_queries(query: str, title: str = "", script: str = "", niche: str = "") -> List[str]:
    text = f"{niche} {query} {title} {script[:1500]}".lower()
    queries = []

    for part in str(query or "").replace("|", ",").replace(";", ",").split(","):
        part = part.strip()
        if part and len(part) >= 3:
            queries.append(part)

    if niche == "iptv" or any(w in text for w in ["iptv", "live tv", "streaming", "4k", "uhd", "sports", "football", "soccer", "watch tv", "watching tv"]):
        queries += ["watching tv", "watching television", "people watching tv", "watching football on tv", "watching sports on tv", "family watching tv", "friends watching tv", "living room tv", "smart tv remote", "remote control tv", "home theater tv", "football match television", "soccer match tv", "sports bar tv", "streaming tv"]
    elif niche == "facebook_followers":
        queries += ["social media phone", "person using phone", "business owner laptop", "digital marketing office", "creator using smartphone", "online business growth", "marketing phone", "social media notification"]
    elif niche in ["instagram_growth", "tiktok_growth"]:
        queries += ["social media phone", "creator using phone", "content creator phone", "influencer recording video", "woman using smartphone", "man using smartphone", "laptop social media", "marketing phone", "scrolling phone"]
    elif niche == "youtube_growth":
        queries += ["youtube creator", "content creator camera", "video editing laptop", "creator recording video", "vlogger camera", "analytics dashboard", "studio recording video", "person filming video"]
    elif niche == "ai_chatbot":
        queries += ["customer support computer", "business technology office", "chat support computer", "call center support", "business dashboard screen", "website chat support", "ai technology office"]
    else:
        queries += ["person using laptop", "business website laptop", "online service computer", "digital marketing office", "people watching screen", "website dashboard screen"]

    cleaned, seen = [], set()
    for q in queries:
        q = re.sub(r"[^a-zA-Z0-9\s-]", " ", q).strip().lower()
        q = re.sub(r"\s+", " ", q)
        if q and q not in seen:
            cleaned.append(q)
            seen.add(q)
    return cleaned[:18]

def pick_best_video_file(files: list) -> Optional[str]:
    candidates = []
    for f in files:
        if f.get("file_type") != "video/mp4" or not f.get("link"):
            continue
        w = int(f.get("width") or 0)
        h = int(f.get("height") or 0)
        if w <= 0 or h <= 0 or w < h:
            continue
        score = abs(h - 720) + abs((w / max(h, 1)) - (16/9)) * 400
        candidates.append((score, f["link"]))

    if candidates:
        candidates.sort(key=lambda x: x[0])
        return candidates[0][1]

    for f in files:
        if f.get("file_type") == "video/mp4" and f.get("link"):
            return f["link"]
    return None

def download_pexels_clips(query: str, title: str, script: str, niche: str, pexels_key: str, work_dir: Path, max_clips: int = 32) -> List[Path]:
    headers = {"Authorization": pexels_key}
    queries = build_pexels_queries(query, title, script, niche)

    paths: List[Path] = []
    used_video_ids = set()
    used_links = set()
    pages = [1, 2, 3, 4, 5]

    for q in queries:
        if len(paths) >= max_clips:
            break
        for page in pages:
            if len(paths) >= max_clips:
                break
            params = {"query": q, "per_page": 12, "orientation": "landscape", "size": "medium", "page": page}
            try:
                r = requests.get("https://api.pexels.com/videos/search", headers=headers, params=params, timeout=45)
                if r.status_code >= 400:
                    continue
                videos = r.json().get("videos", [])
            except Exception:
                continue

            top = videos[:4]
            rest = videos[4:]
            random.shuffle(rest)
            videos = top + rest

            for video in videos:
                if len(paths) >= max_clips:
                    break
                vid = str(video.get("id", ""))
                if vid and vid in used_video_ids:
                    continue
                link = pick_best_video_file(video.get("video_files", []))
                if not link or link in used_links:
                    continue
                out = work_dir / f"pexels_{len(paths):02d}_{vid or random.randint(1000,9999)}.mp4"
                try:
                    with requests.get(link, stream=True, timeout=120) as resp:
                        resp.raise_for_status()
                        with open(out, "wb") as f:
                            for chunk in resp.iter_content(chunk_size=1024 * 1024):
                                if chunk:
                                    f.write(chunk)
                    if out.exists() and out.stat().st_size > 100000:
                        paths.append(out)
                        if vid:
                            used_video_ids.add(vid)
                        used_links.add(link)
                except Exception:
                    try:
                        out.unlink(missing_ok=True)
                    except Exception:
                        pass
                    continue

    return paths

def create_color_video(output_path: Path, duration: int = 30) -> Path:
    cmd = ["ffmpeg", "-y", "-f", "lavfi", "-i", f"color=c=black:s=1280x720:d={duration}", "-pix_fmt", "yuv420p", str(output_path)]
    subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return output_path

def get_duration(path: Path) -> float:
    cmd = ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", str(path)]
    out = subprocess.check_output(cmd).decode().strip()
    return float(out)

def make_simple_srt(script: str, output_path: Path):
    words = script.replace("\n", " ").split()
    chunks = [" ".join(words[i:i+9]) for i in range(0, len(words), 9)]
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
        lines.extend([str(idx), f"{fmt(start)} --> {fmt(end)}", chunk, ""])
        t = end
    output_path.write_text("\n".join(lines), encoding="utf-8")

def make_image_segment(image_path: Path, output_path: Path, duration: float = 6.0):
    vf = "scale=1280:720:force_original_aspect_ratio=increase,crop=1280:720,zoompan=z='min(zoom+0.0015,1.08)':d=150:s=1280x720,format=yuv420p"
    cmd = ["ffmpeg", "-y", "-loop", "1", "-i", str(image_path), "-t", str(duration), "-vf", vf, "-an", "-c:v", "libx264", "-preset", "veryfast", "-crf", "24", str(output_path)]
    subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)

def render_mixed_video(clips: List[Path], images: List[Path], audio_path: Path, subtitles_path: Path, output_path: Path):
    audio_duration = max(10, get_duration(audio_path))
    segment_dir = TEMP_DIR / f"segments_{output_path.stem}"
    segment_dir.mkdir(exist_ok=True)

    segment_duration = 6.0
    needed_segments = int(audio_duration / segment_duration) + 2
    segment_paths: List[Path] = []

    image_queue = images[:]
    random.shuffle(image_queue)
    pexels_queue = clips[:]
    random.shuffle(pexels_queue)
    img_i = 0
    vid_i = 0

    for i in range(needed_segments):
        use_image = bool(image_queue and (i in [0, 2, 5, needed_segments - 2] or i % 4 == 0))
        if use_image and img_i < len(image_queue):
            seg = segment_dir / f"img_{i:03d}.mp4"
            make_image_segment(image_queue[img_i], seg, duration=segment_duration)
            img_i += 1
            if seg.exists() and seg.stat().st_size > 50000:
                segment_paths.append(seg)
                continue

        if vid_i < len(pexels_queue):
            clip = pexels_queue[vid_i]
            vid_i += 1
            try:
                dur = get_duration(clip)
            except Exception:
                dur = segment_duration
            start_at = random.uniform(0, max(0, dur - segment_duration - 0.5)) if dur > segment_duration + 1 else 0
            seg = segment_dir / f"vid_{i:03d}.mp4"
            cmd = ["ffmpeg", "-y", "-ss", str(start_at), "-i", str(clip), "-t", str(segment_duration), "-an", "-vf", "scale=1280:720:force_original_aspect_ratio=increase,crop=1280:720,format=yuv420p", "-c:v", "libx264", "-preset", "veryfast", "-crf", "24", str(seg)]
            res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            if res.returncode == 0 and seg.exists() and seg.stat().st_size > 50000:
                segment_paths.append(seg)
                continue

        fallback = segment_dir / f"fallback_{i:03d}.mp4"
        create_color_video(fallback, duration=int(segment_duration))
        segment_paths.append(fallback)

    concat_file = TEMP_DIR / f"concat_{output_path.stem}.txt"
    concat_file.write_text("\n".join([f"file '{p.as_posix()}'" for p in segment_paths]), encoding="utf-8")
    sub_path = subtitles_path.as_posix().replace(":", "\\:")
    vf = "scale=1280:720:force_original_aspect_ratio=increase,crop=1280:720,format=yuv420p," + f"subtitles='{sub_path}':force_style='Fontsize=24,Outline=2,Shadow=1,Alignment=2'"

    cmd = ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(concat_file), "-i", str(audio_path), "-t", str(audio_duration), "-vf", vf, "-map", "0:v:0", "-map", "1:a:0", "-c:v", "libx264", "-preset", "veryfast", "-crf", "24", "-c:a", "aac", "-b:a", "192k", "-ar", "44100", "-ac", "2", "-shortest", str(output_path)]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        concat_file.unlink(missing_ok=True)
    except Exception:
        pass

    if result.returncode != 0:
        raise RuntimeError(result.stderr.decode(errors="ignore")[-1200:])
    if not output_path.exists() or output_path.stat().st_size < 100000:
        raise RuntimeError("Final video render failed or file too small")

    probe_cmd = ["ffprobe", "-v", "error", "-select_streams", "a:0", "-show_entries", "stream=codec_type", "-of", "default=noprint_wrappers=1:nokey=1", str(output_path)]
    audio_probe = subprocess.run(probe_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if b"audio" not in audio_probe.stdout:
        raise RuntimeError("Final video has no audio stream")