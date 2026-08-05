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

SEGMENT_MIN = 2.5      # shortest a single scene may be on screen
SEGMENT_MAX = 9.0      # longest a single scene may be on screen

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

    # NEW: ordered scene plan from GPT.
    # [{"text": "spoken line", "query": "pexels phrase", "visual": "broll"|"screenshot"}]
    scenes: Optional[List[Dict[str, Any]]] = []
    # NEW: AI conversion images are off by default (they are the biggest memory spike)
    use_ai_images: bool = False
    # NEW: user-recorded screen captures, in order.
    # [{"url": "...", "label": "sign up", "position": "middle"|"end"|"auto"}]
    # A scene whose "visual" is "clip1".."clip9" gets that recording.
    custom_clips: Optional[List[Dict[str, Any]]] = []

    # NEW: keeps stock footage inside the niche.
    scene_anchor: str = ""                     # e.g. "television"
    ban_terms: Optional[List[str]] = []        # e.g. ["instagram", "office meeting"]

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
        audio_duration = max(10.0, get_duration(audio_path))

        conversion_images: List[Path] = []
        if data.get("use_ai_images"):
            update_job(api_key, step="creating conversion images")
            conversion_images = generate_conversion_images(data, niche, work)

        update_job(api_key, step="capturing website screenshots")
        screenshots = capture_website_screenshots(data, work)

        update_job(api_key, step="planning scenes")
        scenes = build_scene_plan(data, niche, audio_duration)
        update_job(api_key, scene_count=len(scenes))

        # ✅ measure when each word is ACTUALLY spoken, instead of guessing from
        # word counts. Without this the captions and the clips drift apart.
        spoken_words: List[Dict[str, Any]] = []
        if data.get("openai_key"):
            update_job(api_key, step="listening back to the voiceover")
            spoken_words = transcribe_words(audio_path, data["openai_key"])

        if spoken_words:
            scenes = align_scenes_to_audio(scenes, spoken_words, audio_duration)
            update_job(api_key, aligned="whisper", scene_count=len(scenes))
        else:
            update_job(api_key, aligned="estimated")

        update_job(api_key, step="matching a clip to each scene")
        scenes = attach_visuals(
            scenes=scenes,
            pexels_key=data["pexels_key"],
            work_dir=work,
            screenshots=screenshots,
            extra_images=conversion_images,
            anchor=str(data.get("scene_anchor") or ""),
            ban_terms=[str(b).strip().lower() for b in (data.get("ban_terms") or []) if str(b).strip()]
        )

        if data.get("custom_clips"):
            update_job(api_key, step="adding your recorded clips")
            customs = download_custom_clips(data["custom_clips"], work)
            place_custom_clips(scenes, customs)
            update_job(api_key, custom_clips_used=len(customs))

        update_job(api_key, step="creating captions")
        subtitles_path = work / "captions.srt"
        if spoken_words:
            build_srt_from_words(spoken_words, subtitles_path)
        else:
            build_srt_from_scenes(scenes, subtitles_path)

        update_job(api_key, step="rendering final video")
        final_path = OUTPUT_DIR / f"{job_id}.mp4"
        render_scene_video(
            scenes=scenes,
            audio_path=audio_path,
            audio_duration=audio_duration,
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

# ══════════════════════════════════════════════════════════════════
# SCENE PLANNING
# ══════════════════════════════════════════════════════════════════

SENTENCE_SPLIT = re.compile(r'(?<=[.!?])\s+')

def split_script_into_chunks(script: str, target_words: int = 20) -> List[str]:
    """Group sentences into chunks of roughly target_words each."""
    sentences = [s.strip() for s in SENTENCE_SPLIT.split(script.replace("\n", " ")) if s.strip()]
    chunks: List[str] = []
    current: List[str] = []
    count = 0
    for s in sentences:
        words = len(s.split())
        current.append(s)
        count += words
        if count >= target_words:
            chunks.append(" ".join(current))
            current, count = [], 0
    if current:
        if chunks and count < 6:
            chunks[-1] = chunks[-1] + " " + " ".join(current)
        else:
            chunks.append(" ".join(current))
    return chunks or [script]

def build_scene_plan(data: Dict[str, Any], niche: str, audio_duration: float) -> List[Dict[str, Any]]:
    """
    Returns an ordered list of scenes:
      {"text": str, "query": str, "visual": "broll"|"screenshot", "duration": float}
    Uses GPT's scene plan when present, otherwise derives one from the script.
    """
    raw_scenes = data.get("scenes") or []
    scenes: List[Dict[str, Any]] = []

    if raw_scenes:
        for s in raw_scenes:
            text = str(s.get("text") or "").strip()
            query = str(s.get("query") or "").strip()
            visual = str(s.get("visual") or "broll").strip().lower()
            if not text:
                continue
            if visual not in ("broll", "screenshot"):
                visual = "broll"
            scenes.append({"text": text, "query": query, "visual": visual})

    if not scenes:
        # Fallback: chunk the script and rotate through the generic query list
        fallback_queries = build_pexels_queries(
            data.get("search_query") or "",
            data.get("title", ""),
            data.get("script", ""),
            niche
        )
        chunks = split_script_into_chunks(data.get("script", ""))
        for i, chunk in enumerate(chunks):
            q = fallback_queries[i % len(fallback_queries)] if fallback_queries else "business office"
            scenes.append({"text": chunk, "query": q, "visual": "broll"})

    # Any scene missing a query falls back to keywords pulled from its own text
    for s in scenes:
        if not s["query"]:
            s["query"] = keywords_from_text(s["text"], niche)

    allocate_durations(scenes, audio_duration)
    return scenes

STOPWORDS = set("""
a an the and or but if then than that this these those is are was were be been being am
i you he she it we they me him her us them my your his its our their
of in on at to for with from by about into over after before under as
so very just really more most much many can will would should could do does did
not no yes what when where who how why which
""".split())

def keywords_from_text(text: str, niche: str = "") -> str:
    words = re.findall(r"[a-zA-Z]{4,}", text.lower())
    keep = [w for w in words if w not in STOPWORDS]
    if not keep:
        return f"{niche} business" if niche else "business office"
    return " ".join(keep[:3])

def allocate_durations(scenes: List[Dict[str, Any]], audio_duration: float):
    """Give each scene screen time proportional to how long its line takes to say."""
    weights = [max(1, len(str(s.get("text", "")).split())) for s in scenes]
    total = float(sum(weights)) or 1.0
    for s, w in zip(scenes, weights):
        d = audio_duration * (w / total)
        s["duration"] = round(min(SEGMENT_MAX, max(SEGMENT_MIN, d)), 2)

# ══════════════════════════════════════════════════════════════════
# REAL TIMING — transcribe the voiceover and use its word timestamps
# ══════════════════════════════════════════════════════════════════

def transcribe_words(audio_path: Path, openai_key: str) -> List[Dict[str, Any]]:
    """Whisper with word-level timestamps. Returns [] on any failure."""
    try:
        with open(audio_path, "rb") as fh:
            r = requests.post(
                "https://api.openai.com/v1/audio/transcriptions",
                headers={"Authorization": f"Bearer {openai_key}"},
                files={"file": ("voice.mp3", fh, "audio/mpeg")},
                data={
                    "model": "whisper-1",
                    "response_format": "verbose_json",
                    "timestamp_granularities[]": "word",
                },
                timeout=300,
            )
        if r.status_code >= 400:
            return []
        payload = r.json()
    except Exception:
        return []

    words: List[Dict[str, Any]] = []
    for w in payload.get("words", []) or []:
        text = str(w.get("word") or "").strip()
        start, end = w.get("start"), w.get("end")
        if not text or start is None or end is None:
            continue
        start, end = float(start), float(end)
        if end <= start:
            continue
        words.append({"word": text, "start": start, "end": end})

    if words:
        return words

    # some responses only carry segments — better than nothing
    for seg in payload.get("segments", []) or []:
        text = str(seg.get("text") or "").strip()
        start, end = seg.get("start"), seg.get("end")
        if not text or start is None or end is None:
            continue
        words.append({"word": text, "start": float(start), "end": float(end)})

    return words

def align_scenes_to_audio(scenes: List[Dict[str, Any]], words: List[Dict[str, Any]],
                          audio_duration: float) -> List[Dict[str, Any]]:
    """
    Give every scene the time its own line is actually spoken.
    The transcript follows the script in order, so cumulative word position maps
    across cleanly even when Whisper drops or merges the odd word.
    """
    if not scenes or not words:
        return scenes

    total_spoken = len(words)
    counts = [max(1, len(str(s.get("text", "")).split())) for s in scenes]
    total_script = float(sum(counts)) or 1.0

    cursor = 0
    for s, c in zip(scenes, counts):
        start_i = min(total_spoken - 1, int(round(cursor * total_spoken / total_script)))
        cursor += c
        end_i = min(total_spoken - 1, int(round(cursor * total_spoken / total_script)) - 1)
        if end_i < start_i:
            end_i = start_i

        s["start"] = round(words[start_i]["start"], 2)
        s["end"] = round(words[end_i]["end"], 2)

    # close gaps so one scene runs straight into the next
    for i in range(len(scenes) - 1):
        scenes[i]["end"] = scenes[i + 1]["start"]
    scenes[0]["start"] = 0.0
    scenes[-1]["end"] = max(scenes[-1]["end"], audio_duration)

    # fold anything too short to register into its neighbour
    merged: List[Dict[str, Any]] = []
    for s in scenes:
        dur = float(s["end"]) - float(s["start"])
        if merged and dur < 1.6:
            merged[-1]["end"] = s["end"]
            merged[-1]["text"] = (merged[-1].get("text", "") + " " + s.get("text", "")).strip()
            continue
        merged.append(s)

    for s in merged:
        s["duration"] = round(max(1.0, float(s["end"]) - float(s["start"])), 2)

    return merged

def _srt_time(sec: float) -> str:
    h = int(sec // 3600)
    m = int((sec % 3600) // 60)
    s = int(sec % 60)
    ms = int(round((sec - int(sec)) * 1000))
    return f"{h:02}:{m:02}:{s:02},{ms:03}"

def build_srt_from_words(words: List[Dict[str, Any]], output_path: Path, per_line: int = 7):
    """Captions straight off the measured timings — they cannot drift."""
    lines: List[str] = []
    idx = 1

    for i in range(0, len(words), per_line):
        chunk = words[i:i + per_line]
        if not chunk:
            continue
        text = " ".join(w["word"] for w in chunk).strip()
        if not text:
            continue

        start = float(chunk[0]["start"])
        end = float(chunk[-1]["end"])
        if end <= start:
            end = start + 0.6

        lines.extend([str(idx), f"{_srt_time(start)} --> {_srt_time(end)}", text, ""])
        idx += 1

    output_path.write_text("\n".join(lines), encoding="utf-8")

def build_srt_from_scenes(scenes: List[Dict[str, Any]], output_path: Path):
    """Captions built from the same timeline as the visuals, so they stay in sync."""
    def fmt(sec: float) -> str:
        h = int(sec // 3600)
        m = int((sec % 3600) // 60)
        s = int(sec % 60)
        ms = int(round((sec - int(sec)) * 1000))
        return f"{h:02}:{m:02}:{s:02},{ms:03}"

    lines: List[str] = []
    idx = 1
    t = 0.0
    for scene in scenes:
        dur = float(scene.get("duration", SEGMENT_MIN))
        words = str(scene.get("text", "")).split()
        if not words:
            t += dur
            continue
        groups = [" ".join(words[i:i + 8]) for i in range(0, len(words), 8)]
        per = dur / len(groups)
        for g in groups:
            start, end = t, t + per
            lines.extend([str(idx), f"{fmt(start)} --> {fmt(end)}", g, ""])
            idx += 1
            t = end
    output_path.write_text("\n".join(lines), encoding="utf-8")

# ══════════════════════════════════════════════════════════════════
# VISUALS: one clip per scene, in order
# ══════════════════════════════════════════════════════════════════

def attach_visuals(scenes: List[Dict[str, Any]], pexels_key: str, work_dir: Path,
                   screenshots: List[Path], extra_images: List[Path],
                   anchor: str = "", ban_terms: Optional[List[str]] = None) -> List[Dict[str, Any]]:
    ban_terms = ban_terms or []
    used_video_ids: set = set()
    used_links: set = set()
    shot_queue = list(screenshots)
    image_queue = list(extra_images)
    shot_i = 0
    img_i = 0

    for i, scene in enumerate(scenes):
        scene["clip"] = None
        scene["image"] = None

        if scene["visual"] == "screenshot":
            if shot_i < len(shot_queue):
                scene["image"] = shot_queue[shot_i]
                shot_i += 1
                continue
            if img_i < len(image_queue):
                scene["image"] = image_queue[img_i]
                img_i += 1
                continue
            scene["visual"] = "broll"  # nothing to show, fall back to b-roll

        clip = download_clip_for_query(
            query=scene["query"],
            pexels_key=pexels_key,
            work_dir=work_dir,
            idx=i,
            used_video_ids=used_video_ids,
            used_links=used_links,
            anchor=anchor,
            ban_terms=ban_terms
        )
        scene["clip"] = clip

    # Any screenshots GPT never asked for get dropped in at the end of the video
    leftovers = shot_queue[shot_i:] + image_queue[img_i:]
    if leftovers and scenes:
        for j, extra in enumerate(reversed(leftovers)):
            pos = len(scenes) - 2 - j
            if pos < 1:
                break
            if scenes[pos]["visual"] == "broll":
                scenes[pos]["visual"] = "screenshot"
                scenes[pos]["image"] = extra
                scenes[pos]["clip"] = None

    return scenes

def _clip_is_off_topic(video: Dict[str, Any], ban_terms: List[str]) -> bool:
    """Pexels puts the description in the page URL slug — use it to reject junk."""
    if not ban_terms:
        return False
    slug = str(video.get("url", "")).lower().replace("-", " ")
    return any(b and b in slug for b in ban_terms)

def download_clip_for_query(query: str, pexels_key: str, work_dir: Path, idx: int,
                            used_video_ids: set, used_links: set,
                            anchor: str = "", ban_terms: Optional[List[str]] = None) -> Optional[Path]:
    """Download ONE landscape clip that matches this scene's query, inside the niche."""
    headers = {"Authorization": pexels_key}
    ban_terms = ban_terms or []
    anchor = (anchor or "").strip()

    words = query.split()
    attempts: List[str] = []

    # anchored first: "buffering wheel" alone drifts, "buffering wheel television" does not
    if anchor and anchor.lower() not in query.lower():
        attempts.append(f"{query} {anchor}")
    attempts.append(query)
    if len(words) > 2:
        attempts.append(" ".join(words[:2]) + (f" {anchor}" if anchor else ""))
    # last resort stays inside the niche instead of "business office technology"
    attempts.append(anchor if anchor else "business office technology")

    seen_q = set()
    attempts = [q for q in attempts if q.strip() and not (q in seen_q or seen_q.add(q))]

    for q in attempts:
        for page in (1, 2):
            params = {"query": q, "per_page": 10, "orientation": "landscape", "size": "medium", "page": page}
            try:
                r = requests.get("https://api.pexels.com/videos/search", headers=headers, params=params, timeout=40)
                if r.status_code >= 400:
                    continue
                videos = r.json().get("videos", [])
            except Exception:
                continue

            for video in videos:
                vid = str(video.get("id", ""))
                if vid and vid in used_video_ids:
                    continue
                if _clip_is_off_topic(video, ban_terms):
                    continue
                link = pick_best_video_file(video.get("video_files", []))
                if not link or link in used_links:
                    continue

                out = work_dir / f"scene_{idx:03d}_{vid or random.randint(1000, 9999)}.mp4"
                try:
                    with requests.get(link, stream=True, timeout=120) as resp:
                        resp.raise_for_status()
                        with open(out, "wb") as f:
                            for chunk in resp.iter_content(chunk_size=1024 * 1024):
                                if chunk:
                                    f.write(chunk)
                    if out.exists() and out.stat().st_size > 100000:
                        if vid:
                            used_video_ids.add(vid)
                        used_links.add(link)
                        return out
                    out.unlink(missing_ok=True)
                except Exception:
                    try:
                        out.unlink(missing_ok=True)
                    except Exception:
                        pass
                    continue
    return None

# ══════════════════════════════════════════════════════════════════
# USER-RECORDED CLIPS (website walkthrough, how-to-subscribe, ...)
# ══════════════════════════════════════════════════════════════════

def download_custom_clips(clips: List[Dict[str, Any]], work_dir: Path) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for i, c in enumerate(clips or []):
        url = str((c or {}).get("url") or "").strip()
        position = str((c or {}).get("position") or "auto").strip().lower()
        label = str((c or {}).get("label") or "").strip()
        if not url:
            continue
        if position not in ("middle", "end", "start", "auto"):
            position = "auto"

        dest = work_dir / f"custom_{i}.mp4"
        try:
            with requests.get(url, stream=True, timeout=180) as resp:
                resp.raise_for_status()
                with open(dest, "wb") as f:
                    for chunk in resp.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            f.write(chunk)
        except Exception:
            continue

        if not dest.exists() or dest.stat().st_size < 50000:
            continue
        try:
            dur = get_duration(dest)
        except Exception:
            dur = 0.0
        if dur <= 0.5:
            continue

        out.append({
            "path": dest,
            "position": position,
            "label": label,
            "duration": dur,
            "slot": i + 1,          # matches a scene marked "clip1", "clip2", ...
        })
    return out

def _assign_clip(scenes: List[Dict[str, Any]], idxs: List[int], c: Dict[str, Any], taken: set):
    """Play one recording continuously across a run of scenes."""
    offset = 0.0
    for i in idxs:
        scenes[i]["custom_clip"] = c["path"]
        scenes[i]["custom_offset"] = round(offset, 2)
        scenes[i]["custom_total"] = c["duration"]
        scenes[i]["image"] = None
        scenes[i]["clip"] = None
        offset += float(scenes[i].get("duration", SEGMENT_MIN))
        taken.add(i)

def place_custom_clips(scenes: List[Dict[str, Any]], customs: List[Dict[str, Any]]):
    """
    Lay each recording over a run of consecutive scenes so the voiceover keeps
    talking while the recording plays through from start to finish.

    Preferred: the script marks scenes "clip1", "clip2", ... and each recording
    lands exactly where it is being talked about. Anything unmarked falls back
    to start / middle / end, then to an even spread.
    """
    if not scenes or not customs:
        return

    taken: set = set()
    placed: set = set()

    # ── 1. explicit marks from the script ─────────────────────────────
    for c in customs:
        mark = "clip%d" % int(c.get("slot", 0))
        idxs = [i for i, s in enumerate(scenes)
                if str(s.get("visual", "")).strip().lower() == mark and i not in taken]
        if not idxs:
            continue

        # extend forward until the whole recording has played
        acc = sum(float(scenes[i].get("duration", SEGMENT_MIN)) for i in idxs)
        j = idxs[-1] + 1
        while acc < c["duration"] and j < len(scenes) and j not in taken \
                and not str(scenes[j].get("visual", "")).lower().startswith("clip"):
            idxs.append(j)
            acc += float(scenes[j].get("duration", SEGMENT_MIN))
            j += 1

        _assign_clip(scenes, idxs, c, taken)
        placed.add(c["slot"])

    # ── 2. whatever the script did not mark ───────────────────────────
    leftovers = [c for c in customs if c["slot"] not in placed]
    if not leftovers:
        return

    order = {"start": 0, "middle": 1, "auto": 1, "end": 2}
    n = len(scenes)

    for k, c in enumerate(sorted(leftovers, key=lambda x: order.get(x["position"], 1))):
        idxs: List[int] = []
        acc = 0.0

        if c["position"] == "end":
            for i in range(n - 1, -1, -1):
                if i in taken:
                    break
                idxs.insert(0, i)
                acc += float(scenes[i].get("duration", SEGMENT_MIN))
                if acc >= c["duration"]:
                    break
        else:
            if c["position"] == "start":
                start = 0
            elif c["position"] == "middle":
                start = next((i for i, s in enumerate(scenes) if s.get("visual") == "screenshot"), None)
                if start is None:
                    start = max(0, n // 2)
            else:
                # spread the unmarked ones evenly through the back half
                start = min(n - 1, int(n * 0.45) + int(k * n * 0.5 / max(1, len(leftovers))))

            while start < n and start in taken:
                start += 1

            for i in range(start, n):
                if i in taken:
                    break
                idxs.append(i)
                acc += float(scenes[i].get("duration", SEGMENT_MIN))
                if acc >= c["duration"]:
                    break

        if idxs:
            _assign_clip(scenes, idxs, c, taken)

def make_custom_segment(clip_path: Path, output_path: Path, duration: float,
                        offset: float, clip_total: float) -> bool:
    """Cut `duration` seconds starting at `offset` so the recording plays continuously."""
    vf = "scale=1280:720:force_original_aspect_ratio=decrease,pad=1280:720:(ow-iw)/2:(oh-ih)/2:color=black,fps=30,setpts=PTS-STARTPTS,format=yuv420p"

    if clip_total > 0 and offset + duration <= clip_total:
        pre = ["-ss", str(round(offset, 2)), "-i", str(clip_path)]
        post_ss: List[str] = []
    else:
        # recording is shorter than the narration it covers: loop it
        pre = ["-stream_loop", "-1", "-i", str(clip_path)]
        post_ss = ["-ss", str(round(offset, 2))]

    cmd = ["ffmpeg", "-y"] + pre + post_ss + [
        "-t", str(duration), "-an", "-vf", vf,
        "-r", "30", "-vsync", "cfr", "-c:v", "libx264",
        "-preset", "veryfast", "-crf", "23", "-pix_fmt", "yuv420p",
        "-movflags", "+faststart", str(output_path)
    ]
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return res.returncode == 0 and output_path.exists() and output_path.stat().st_size > 50000

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

    for i, prompt in enumerate(prompts[:4]):
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
            del b64, data_json
            convert_image_to_video_frame(out_raw, out_jpg)
            out_raw.unlink(missing_ok=True)
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
            base + f"Create a final CTA image for {website}: professional social media growth look, strong trust feeling, Facebook page growth, clear call-to-action mood. {cta}",
        ]

    if niche == "iptv":
        return [
            base + "Show premium 4K IPTV service visual: large smart TV with live sports channels, remote control, TV box, dark cinematic living room, 4K/UHD feeling.",
            base + "Show no buffering IPTV benefit: smooth live sports on TV, strong WiFi/streaming symbol, happy viewer, premium sports entertainment vibe.",
            base + "Show IPTV setup process: smart TV, app login screen, remote, simple steps, high quality streaming look. Avoid fake brand logos.",
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
    """
    Needs the WEBSITE_SCREENSHOT_API env var, e.g.
      https://image.thum.io/get/width/1280/crop/720/{url_raw}
    {url_raw} = plain URL,  {url} = percent-encoded URL.
    """
    template = os.environ.get("WEBSITE_SCREENSHOT_API", "").strip()
    if not template:
        return []

    urls = list(data.get("screenshot_urls") or [])
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
    for i, url in enumerate(clean_urls[:5]):
        try:
            shot_url = template.replace("{url_raw}", url).replace(
                "{url}", requests.utils.quote(url, safe="")
            )
            resp = requests.get(shot_url, timeout=90)
            if resp.status_code >= 400 or len(resp.content) < 10000:
                continue

            raw = work_dir / f"screenshot_{i}.png"
            jpg = work_dir / f"screenshot_{i}.jpg"
            raw.write_bytes(resp.content)
            convert_image_to_video_frame(raw, jpg)
            raw.unlink(missing_ok=True)
            if jpg.exists() and jpg.stat().st_size > 30000:
                paths.append(jpg)
        except Exception:
            continue

    return paths

def convert_image_to_video_frame(input_path: Path, output_path: Path):
    cmd = [
        "ffmpeg", "-y",
        "-i", str(input_path),
        "-vf", "scale=1280:720:force_original_aspect_ratio=increase,crop=1280:720,fps=30,setpts=PTS-STARTPTS,format=yuv420p",
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

def create_color_video(output_path: Path, duration: float = 6.0) -> Path:
    cmd = ["ffmpeg", "-y", "-f", "lavfi", "-i", f"color=c=black:s=1280x720:d={duration}",
           "-r", "30", "-pix_fmt", "yuv420p", str(output_path)]
    subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return output_path

def get_duration(path: Path) -> float:
    cmd = ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", str(path)]
    out = subprocess.check_output(cmd).decode().strip()
    return float(out)

def make_image_segment(image_path: Path, output_path: Path, duration: float = 6.0):
    frames = int(duration * 30) + 30
    vf = ("scale=1280:720:force_original_aspect_ratio=increase,crop=1280:720,"
          f"zoompan=z='min(zoom+0.0010,1.06)':d={frames}:s=1280x720:fps=30,format=yuv420p")
    cmd = ["ffmpeg", "-y", "-loop", "1", "-i", str(image_path), "-t", str(duration),
           "-vf", vf, "-an", "-r", "30", "-vsync", "cfr", "-c:v", "libx264",
           "-preset", "veryfast", "-crf", "23", "-pix_fmt", "yuv420p",
           "-movflags", "+faststart", str(output_path)]
    subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)

def make_broll_segment(clip_path: Path, output_path: Path, duration: float) -> bool:
    try:
        clip_dur = get_duration(clip_path)
    except Exception:
        clip_dur = 0.0

    vf = "scale=1280:720:force_original_aspect_ratio=increase,crop=1280:720,fps=30,setpts=PTS-STARTPTS,format=yuv420p"

    if clip_dur > duration + 1.0:
        start_at = random.uniform(0, max(0.0, clip_dur - duration - 0.5))
        pre = ["-ss", str(round(start_at, 2)), "-i", str(clip_path)]
    else:
        # clip is shorter than the line being spoken: loop it instead of cutting away
        pre = ["-stream_loop", "-1", "-i", str(clip_path)]

    cmd = ["ffmpeg", "-y"] + pre + [
        "-t", str(duration), "-an", "-vf", vf,
        "-r", "30", "-vsync", "cfr", "-c:v", "libx264",
        "-preset", "veryfast", "-crf", "23", "-pix_fmt", "yuv420p",
        "-movflags", "+faststart", str(output_path)
    ]
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return res.returncode == 0 and output_path.exists() and output_path.stat().st_size > 50000

def render_scene_video(scenes: List[Dict[str, Any]], audio_path: Path, audio_duration: float,
                       subtitles_path: Path, output_path: Path):
    segment_dir = TEMP_DIR / f"segments_{output_path.stem}"
    segment_dir.mkdir(exist_ok=True)

    segment_paths: List[Path] = []
    last_good: Optional[Path] = None

    for i, scene in enumerate(scenes):
        duration = float(scene.get("duration", SEGMENT_MIN))
        seg = segment_dir / f"seg_{i:03d}.mp4"
        made = False

        if scene.get("custom_clip"):
            made = make_custom_segment(
                scene["custom_clip"], seg, duration,
                float(scene.get("custom_offset", 0.0)),
                float(scene.get("custom_total", 0.0))
            )
        elif scene.get("image"):
            make_image_segment(scene["image"], seg, duration=duration)
            made = seg.exists() and seg.stat().st_size > 50000
        elif scene.get("clip"):
            made = make_broll_segment(scene["clip"], seg, duration)

        if not made and last_good is not None:
            # reuse the previous scene's footage rather than showing a black gap
            made = make_broll_segment(last_good, seg, duration)

        if not made:
            create_color_video(seg, duration=duration)

        segment_paths.append(seg)
        if scene.get("clip"):
            last_good = scene["clip"]

    concat_file = TEMP_DIR / f"concat_{output_path.stem}.txt"
    concat_file.write_text("\n".join([f"file '{p.as_posix()}'" for p in segment_paths]), encoding="utf-8")
    sub_path = subtitles_path.as_posix().replace(":", "\\:")
    vf = ("scale=1280:720:force_original_aspect_ratio=increase,crop=1280:720,fps=30,setpts=PTS-STARTPTS,format=yuv420p,"
          + f"subtitles='{sub_path}':force_style='Fontsize=24,Outline=2,Shadow=1,Alignment=2'")

    cmd = [
        "ffmpeg", "-y",
        "-fflags", "+genpts",
        "-f", "concat", "-safe", "0", "-i", str(concat_file),
        "-i", str(audio_path),
        "-t", str(audio_duration),
        "-vf", vf + ",fps=30,setpts=PTS-STARTPTS",
        "-map", "0:v:0",
        "-map", "1:a:0",
        "-r", "30",
        "-vsync", "cfr",
        "-c:v", "libx264",
        "-preset", "veryfast",
        "-crf", "23",
        "-pix_fmt", "yuv420p",
        "-c:a", "aac",
        "-b:a", "192k",
        "-ar", "44100",
        "-ac", "2",
        "-af", "aresample=async=1:first_pts=0",
        "-movflags", "+faststart",
        "-shortest",
        str(output_path)
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        concat_file.unlink(missing_ok=True)
    except Exception:
        pass
    shutil.rmtree(segment_dir, ignore_errors=True)

    if result.returncode != 0:
        raise RuntimeError(result.stderr.decode(errors="ignore")[-1200:])
    if not output_path.exists() or output_path.stat().st_size < 100000:
        raise RuntimeError("Final video render failed or file too small")

    probe_cmd = ["ffprobe", "-v", "error", "-select_streams", "a:0", "-show_entries", "stream=codec_type", "-of", "default=noprint_wrappers=1:nokey=1", str(output_path)]
    audio_probe = subprocess.run(probe_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if b"audio" not in audio_probe.stdout:
        raise RuntimeError("Final video has no audio stream")
