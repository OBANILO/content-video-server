"""
Short video engine — the old my-short-server (Flask app.py), moved onto content-video-server
under /short/... so My Short Bot works again (the free Render service was shut down).

Same behaviour as before, plus:
  - 1080x1920 output (was 720x1280), all text/bars scaled to match
  - Whisper forced to English with the lyrics as a hint; non-Latin captions are dropped
  - the slow zoom on the image actually moves now (zoompan d=1 reset it every frame)
"""
import subprocess
import os
import requests
import threading
import math
import uuid
import json
import re

from fastapi import APIRouter, Request
from fastapi.responses import FileResponse, JSONResponse

router = APIRouter(prefix="/short")

UPLOAD_FOLDER = '/tmp/msb_short_jobs'
AUDIO_SEGMENTS_FOLDER = '/tmp/msb_audio_segments'
JOBS_STATE_FILE = '/tmp/msb_jobs_state.json'

OUT_W, OUT_H = 1080, 1920      # was 720x1280
UI = OUT_W / 720.0             # text/bar sizes below were designed at 720 wide
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(AUDIO_SEGMENTS_FOLDER, exist_ok=True)

EQ_CENTER_Y = 0.92
DARK_START  = 0.68
LYRICS_Y    = 0.80

# ─── Job Persistence ──────────────────────────────────────────────────────────

def load_jobs():
    try:
        if os.path.exists(JOBS_STATE_FILE):
            with open(JOBS_STATE_FILE, 'r') as f:
                return json.load(f)
    except:
        pass
    return {}

def save_job(job_id, data):
    jobs = load_jobs()
    jobs[job_id] = data
    try:
        with open(JOBS_STATE_FILE, 'w') as f:
            json.dump(jobs, f)
    except:
        pass

def get_job(job_id):
    return load_jobs().get(job_id)

def delete_job(job_id):
    jobs = load_jobs()
    jobs.pop(job_id, None)
    try:
        with open(JOBS_STATE_FILE, 'w') as f:
            json.dump(jobs, f)
    except:
        pass

# ─── Download ─────────────────────────────────────────────────────────────────

def download_file(url, dest_path):
    headers = {
        'Cache-Control': 'no-cache',
        'Pragma': 'no-cache',
        'User-Agent': 'Mozilla/5.0 (compatible; VideoServer/1.0)'
    }
    r = requests.get(url, timeout=180, stream=True, headers=headers)
    if r.status_code != 200:
        raise ValueError(f"Download failed: HTTP {r.status_code} for {url}")
    content_type = r.headers.get('content-type', '')
    if 'text/html' in content_type:
        raise ValueError(f"Got HTML instead of file from {url}")
    r.raise_for_status()
    with open(dest_path, 'wb') as f:
        for chunk in r.iter_content(chunk_size=8192):
            f.write(chunk)
    return dest_path

def download_pexels_video(pexels_url, dest_path, pexels_api_key=""):
    if '.mp4' in pexels_url.lower() or 'videos/download' in pexels_url:
        return download_file(pexels_url, dest_path)
    if 'pexels.com' not in pexels_url:
        return download_file(pexels_url, dest_path)
    match = re.search(r'/video/[^/]+-(\d+)/?', pexels_url)
    if not match:
        match = re.search(r'(\d{5,})/?$', pexels_url)
    if not match:
        return download_file(pexels_url, dest_path)
    video_id = match.group(1)
    api_key_to_use = pexels_api_key or 'xC87vhy3Cf152ByhxRtakfR4mM2rRHN2NxGIlVqzUHQQ5VlB5ebYoCva'
    try:
        api_resp = requests.get(
            f"https://api.pexels.com/videos/videos/{video_id}",
            headers={"Authorization": api_key_to_use},
            timeout=30
        )
        if api_resp.status_code == 200:
            data = api_resp.json()
            files = data.get('video_files', [])
            selected = None; max_h = 0
            for f in files:
                h = f.get('height', 0)
                if h <= 720 and h > max_h:
                    max_h = h; selected = f['link']
            if not selected:
                for f in files:
                    if f.get('quality') == 'sd':
                        selected = f['link']; break
            if not selected and files:
                selected = files[0]['link']
            if selected:
                print(f"[Pexels] Downloading: {selected[:80]}")
                return download_file(selected, dest_path)
    except Exception as e:
        print(f"[Pexels API] Error: {e}")
    return download_file(f"https://www.pexels.com/video/{video_id}/download/", dest_path)

# ─── Audio Helpers ────────────────────────────────────────────────────────────

def get_audio_duration(audio_path):
    result = subprocess.run(
        ['ffprobe', '-v', 'error', '-show_entries', 'format=duration',
         '-of', 'default=noprint_wrappers=1:nokey=1', audio_path],
        capture_output=True, text=True
    )
    v = result.stdout.strip()
    if v and v != 'N/A':
        try: return float(v)
        except: pass
    return 45.0

def find_best_segment(audio_path, segment_duration=45):
    total_duration = get_audio_duration(audio_path)
    if total_duration <= segment_duration:
        return 0.0
    step = 2.0
    volumes = []
    num_chunks = int(total_duration / step)
    for i in range(num_chunks):
        t = i * step
        result = subprocess.run([
            'ffmpeg', '-y', '-ss', str(t), '-t', str(step),
            '-i', audio_path, '-af', 'volumedetect',
            '-f', 'null', '/dev/null'
        ], capture_output=True, text=True, timeout=10)
        match = re.search(r'mean_volume:\s*([-\d.]+)\s*dB', result.stderr)
        volumes.append(float(match.group(1)) if match else -60.0)
    if not volumes:
        return 0.0
    window_chunks = int(segment_duration / step)
    best_start = 0.0
    best_score = -999.0
    for i in range(len(volumes) - window_chunks + 1):
        score = sum(volumes[i:i + window_chunks]) / window_chunks
        if score > best_score:
            best_score = score
            best_start = i * step
    print(f"[BestSegment] start={best_start:.1f}s score={best_score:.1f}dB total={total_duration:.1f}s")
    return best_start

# ─── Font Helpers ─────────────────────────────────────────────────────────────

def get_best_font():
    for path in [
        '/usr/share/fonts/truetype/ubuntu/Ubuntu-B.ttf',
        '/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf',
        '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf',
    ]:
        if os.path.exists(path): return path
    return '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'

def get_italic_font():
    for path in [
        '/usr/share/fonts/truetype/freefont/FreeSerifBoldItalic.ttf',
        '/usr/share/fonts/truetype/dejavu/DejaVuSerif-BoldItalic.ttf',
        '/usr/share/fonts/truetype/liberation/LiberationSerif-BoldItalic.ttf',
        '/usr/share/fonts/truetype/ubuntu/Ubuntu-BI.ttf',
        '/usr/share/fonts/truetype/dejavu/DejaVuSans-BoldOblique.ttf',
    ]:
        if os.path.exists(path): return path
    return get_best_font()

def get_lyrics_font():
    for path in [
        '/usr/share/fonts/truetype/freefont/FreeSerifBold.ttf',
        '/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf',
        '/usr/share/fonts/truetype/liberation/LiberationSerif-Bold.ttf',
    ]:
        if os.path.exists(path): return path
    return get_best_font()

# ─── FFmpeg Escape ────────────────────────────────────────────────────────────

def ffmpeg_escape(text):
    text = text.replace('\\', '\\\\')
    text = text.replace("'", "\u2019")
    text = text.replace(':', '\\:')
    text = text.replace('%', '\\%')
    text = text.replace('[', '\\[')
    text = text.replace(']', '\\]')
    text = text.replace(',', '\\,')
    return text

# ─── Watermark ────────────────────────────────────────────────────────────────

def build_artist_watermark(font_italic, artist_name="SORLUNE"):
    name       = ffmpeg_escape(artist_name.upper())
    padding    = int(28*UI)
    alpha_expr = "0.875+0.125*sin(6.2832/4.0*t)"
    watermark  = (
        f"drawtext=fontfile={font_italic}:text='{name}':"
        f"fontsize={int(34*UI)}:fontcolor=0xD4AF37@1.0:"
        f"borderw=2:bordercolor=black@0.80:"
        f"shadowcolor=black@0.70:shadowx=2:shadowy=2:"
        f"x=w-text_w-{padding}:y={padding}:alpha='{alpha_expr}'"
    )
    underline = (
        f"drawtext=fontfile={font_italic}:text='\u2014\u2014\u2014\u2014\u2014\u2014\u2014':"
        f"fontsize={int(14*UI)}:fontcolor=0xD4AF37@1.0:"
        f"x=w-text_w-{padding}:y={padding+int(42*UI)}:alpha='{alpha_expr}'"
    )
    return ",".join([watermark, underline])

# ─── Song Title ───────────────────────────────────────────────────────────────

def build_song_title(font, title=""):
    if not title: return ""
    safe_title = ffmpeg_escape(title[:40])
    alpha = "if(lt(t,1),0,if(lt(t,2.5),(t-1)/1.5,0.95))"
    return (
        f"drawtext=fontfile={font}:text='\u266b  {safe_title}  \u266b':"
        f"fontsize={int(26*UI)}:fontcolor=white@1.0:"
        f"borderw=2:bordercolor=black@0.90:"
        f"shadowcolor=black@0.80:shadowx=2:shadowy=2:"
        f"x=(w-text_w)/2:y=h*0.06:"
        f"alpha='{alpha}'"
    )

# ─── Subscribe CTA ────────────────────────────────────────────────────────────

def build_subscribe_cta(font, duration=45.0):
    """
    Follow / SUBSCRIBE / arrows — moved OFF the face (it used to sit on his sunglasses at h*0.20-0.34)
    down to the chest area just above the lyrics band, and only shown twice:
    seconds 3-10, then the last 7 seconds. The music emoji was dropped: the font has no emoji glyph.
    """
    end_on = max(10.5, duration - 7.0)
    # visible window: fade in at 3s, out at 10s; again from end_on to the end
    show = (f"if(between(t,3,4),(t-3),if(between(t,4,9.5),1,if(between(t,9.5,10.5),(10.5-t),"
            f"if(between(t,{end_on:.2f},{end_on+1:.2f}),(t-{end_on:.2f}),if(gte(t,{end_on+1:.2f}),1,0)))))")

    follow = (
        f"drawtext=fontfile={font}:text='Follow for more':"
        f"fontsize={int(22*UI)}:fontcolor=white@1.0:"
        f"borderw={int(2*UI)}:bordercolor=black@0.85:"
        f"shadowcolor=black@0.70:shadowx=1:shadowy=1:"
        f"x=(w-text_w)/2:y=h*0.565:"
        f"alpha='{show}'"
    )
    btn_box = (
        f"drawtext=fontfile={font}:text='  SUBSCRIBE  ':"
        f"fontsize={int(34*UI)}:fontcolor=white@1.0:"
        f"borderw=0:"
        f"box=1:boxcolor=0xCC0000@0.92:boxborderw={int(12*UI)}:"
        f"x=(w-text_w)/2:y=h*0.600:"
        f"alpha='({show})*(0.88+0.12*abs(sin(2.5*t)))'"
    )
    arr_y  = f"trunc(h*0.655)+trunc({int(8*UI)}*abs(sin(2.8*t)))"
    arrows = (
        f"drawtext=fontfile={font}:text='\u25BC   \u25BC   \u25BC':"
        f"fontsize={int(20*UI)}:fontcolor=0xFF3333@1.0:"
        f"borderw=1:bordercolor=black@0.80:"
        f"x=(w-text_w)/2:y={arr_y}:"
        f"alpha='{show}'"
    )
    return ",".join([follow, btn_box, arrows])

# ─── EQ Bar ───────────────────────────────────────────────────────────────────

def build_eq_bar(font):
    parts     = []
    bar_count = 24
    bar_gap   = int(12*UI)
    half      = bar_count // 2
    center_y  = f"h*{EQ_CENTER_Y}"
    freqs  = [1.3,2.1,2.7,1.9,3.1,2.4,1.7,2.9,2.2,3.5,2.0,2.8,
              2.8,2.0,3.5,2.2,2.9,1.7,2.4,3.1,1.9,2.7,2.1,1.3]
    phases = [0.0,0.5,1.1,1.7,0.3,0.9,1.5,0.2,0.8,1.4,0.6,1.2,
              1.2,0.6,1.4,0.8,0.2,1.5,0.9,0.3,1.7,1.1,0.5,0.0]
    for i in range(bar_count):
        dist      = abs(i - half) / half
        amplitude = int((4 + 28 * math.exp(-2.5 * dist * dist)) * UI)
        alpha_up  = 0.88 - 0.22 * dist
        alpha_dwn = 0.38 - 0.12 * dist
        offset    = (i - half) * bar_gap
        bar_x     = f"(w/2+({offset})-tw/2)"
        fs_expr   = f"{int(3*UI)}+{amplitude}*abs(sin(t*{freqs[i]}+{phases[i]}))"
        parts.append(
            f"drawtext=fontfile={font}:text='|':fontsize={fs_expr}:"
            f"fontcolor=0xD4AF37@{alpha_up:.2f}:x={bar_x}:y=({center_y})-text_h"
        )
        parts.append(
            f"drawtext=fontfile={font}:text='|':fontsize={fs_expr}:"
            f"fontcolor=0xB8860B@{alpha_dwn:.2f}:x={bar_x}:y={center_y}"
        )
    return ",".join(parts)

# ─── Lyrics ───────────────────────────────────────────────────────────────────

_SECTION_WORDS = r'verse|chorus|bridge|hook|outro|intro|pre[\-\s]?chorus|post[\-\s]?chorus|refrain|interlude|instrumental|spoken|rap|breakdown|solo|ad[\-\s]?lib|vamp|coda|tag|skit|fade'
SECTION_REGEX  = [re.compile(p, re.IGNORECASE) for p in [
    r'^\[.*\]$', r'^\(.*\)$',
    rf'^({_SECTION_WORDS})\s*[\d:.\-]*\s*$',
    rf'^({_SECTION_WORDS})\s*\d*\s*:$',
    r'^[\d\s\.\)\(\:\-]+$'
]]

def is_section_label(line):
    s = line.strip()
    return any(p.match(s) or p.match(s.rstrip(':').strip()) for p in SECTION_REGEX)

def split_lyrics_lines(text):
    if not text: return []
    return [l.strip() for l in text.replace('\r\n','\n').replace('\r','\n').split('\n')
            if l.strip() and not is_section_label(l.strip())]

def normalize_word(w):
    return re.sub(r"[^\w']", "", (w or "").lower()).strip()

def transcribe_audio_words_with_whisper(audio_path, openai_api_key, lyrics_text=""):
    if not openai_api_key or not os.path.exists(audio_path): return []
    # forced English: with music under the voice Whisper once guessed Sinhala
    form = {"model": "whisper-1", "response_format": "verbose_json",
            "timestamp_granularities[]": "word", "language": "en"}
    hint = " ".join(split_lyrics_lines(lyrics_text))[:600]
    if hint: form["prompt"] = hint
    try:
        with open(audio_path, "rb") as audio_file:
            response = requests.post(
                "https://api.openai.com/v1/audio/transcriptions",
                headers={"Authorization": f"Bearer {openai_api_key}"},
                files={"file": audio_file},
                data=form,
                timeout=300
            )
        if response.status_code != 200: return []
        data    = response.json()
        cleaned = []
        for w in data.get("words", []):
            word_text = (w.get("word") or "").strip()
            start     = w.get("start"); end = w.get("end")
            if not word_text or start is None or end is None: continue
            start, end = float(start), float(end)
            if end <= start: continue
            cleaned.append({"word": word_text, "norm": normalize_word(word_text), "start": start, "end": end})
        if cleaned: return cleaned
        seg_words = []
        for seg in data.get("segments", []):
            text  = (seg.get("text") or "").strip()
            start = seg.get("start"); end = seg.get("end")
            if not text or start is None or end is None: continue
            seg_words.append({"word": text, "norm": normalize_word(text), "start": float(start), "end": float(end)})
        return seg_words
    except Exception as e:
        print(f"[Whisper] Error: {e}"); return []

def build_lines_from_words(words, max_gap=0.45, max_words=6, max_duration=3.0):
    if not words: return []
    lines = []; current = [words[0]]
    def flush(lw):
        if not lw: return None
        text = " ".join(w["word"] for w in lw).strip()
        return {"start": round(lw[0]["start"], 2), "end": round(lw[-1]["end"], 2), "text": text} if text else None
    for w in words[1:]:
        prev = current[-1]
        if (w["start"] - prev["end"] > max_gap or
                len(current) >= max_words or
                w["end"] - current[0]["start"] > max_duration):
            item = flush(current)
            if item: lines.append(item)
            current = [w]
        else:
            current.append(w)
    item = flush(current)
    if item: lines.append(item)
    cleaned = []
    for seg in lines:
        start = float(seg["start"]); end = float(seg["end"]); text = seg["text"].strip()
        if not text: continue
        min_dur = max(0.60, min(1.40, len(text.split()) * 0.22))
        if end - start < min_dur: end = start + min_dur
        if cleaned and start < cleaned[-1]["end"]:
            start = round(cleaned[-1]["end"] + 0.03, 2)
            end   = max(end, start + min_dur)
        cleaned.append({"start": round(start, 2), "end": round(end, 2), "text": text})
    return cleaned

def _is_latin(text):
    return not re.search(r'[^\x00-\u024F\u1E00-\u1EFF\u2000-\u206F\u2190-\u21FF\s]', text or '')

def transcribe_lyrics_with_whisper(audio_path, openai_api_key, lyrics_text=""):
    lines = build_lines_from_words(transcribe_audio_words_with_whisper(audio_path, openai_api_key, lyrics_text))
    # never show non-English script — the caller falls back to the written lyrics
    return lines if all(_is_latin(l['text']) for l in lines) else []

def wrap_lyric_line(text, max_chars=32):
    if len(text) <= max_chars: return [text]
    words = text.split(); best_split = len(words) // 2; best_diff = float('inf')
    for i in range(1, len(words)):
        p1, p2 = " ".join(words[:i]), " ".join(words[i:])
        diff = abs(len(p1) - len(p2))
        if diff < best_diff and len(p1) <= max_chars and len(p2) <= max_chars:
            best_diff, best_split = diff, i
    return [" ".join(words[:best_split]), " ".join(words[best_split:])]

def build_karaoke_filter(segments, font, lyrics_font=None):
    if lyrics_font is None: lyrics_font = font
    if not segments: return ""
    parts = []; FONT_SIZE = int(36*UI); LINE_HEIGHT = int(44*UI); MAX_CHARS = 32
    for seg in segments:
        start, end, raw_text = seg["start"], seg["end"], seg["text"]
        dur = max(end - start, 0.5); fade_dur = min(0.18, dur / 5)
        alpha_expr = (
            f"if(between(t,{start},{start+fade_dur}),(t-{start})/{fade_dur},"
            f"if(between(t,{start+fade_dur},{end-fade_dur}),1,"
            f"if(between(t,{end-fade_dur},{end}),({end}-t)/{fade_dur},0)))"
        )
        lines = wrap_lyric_line(raw_text, max_chars=MAX_CHARS)
        if len(lines) == 1:
            parts.append(
                f"drawtext=fontfile={lyrics_font}:text='{ffmpeg_escape(lines[0])}':"
                f"fontsize={FONT_SIZE}:fontcolor=white@1.0:"
                f"borderw={int(3*UI)}:bordercolor=black@1.0:"
                f"shadowcolor=black@0.95:shadowx={int(2*UI)}:shadowy={int(2*UI)}:"
                f"x=(w-text_w)/2:y=h*{LYRICS_Y}:alpha='{alpha_expr}'"
            )
        else:
            base_y = LYRICS_Y - 0.04
            for li, line in enumerate(lines):
                parts.append(
                    f"drawtext=fontfile={lyrics_font}:text='{ffmpeg_escape(line)}':"
                    f"fontsize={FONT_SIZE}:fontcolor=white@1.0:"
                    f"borderw={int(3*UI)}:bordercolor=black@1.0:"
                    f"shadowcolor=black@0.95:shadowx={int(2*UI)}:shadowy={int(2*UI)}:"
                    f"x=(w-text_w)/2:y=h*{base_y}+{li*LINE_HEIGHT}:alpha='{alpha_expr}'"
                )
    return ",".join(parts)

# ─── Moving weather over the still image (rain / snow / embers) ──────────────
# One random particle frame is drawn once, stacked twice so it tiles seamlessly,
# then scrolled every frame — real falling motion at almost no CPU cost.
WEATHER_FX = ('rain', 'snow', 'embers')

def build_weather_fx(fx):
    """Returns a filtergraph piece ending in [fxl] (an RGBA particle layer), or '' for none.
    Particles are drawn at a lower resolution and scaled up, so they read as real
    rain streaks / snowflakes / embers on a phone instead of pixel dust."""
    w, h = OUT_W, OUT_H
    if fx == 'rain':
        k, density, blur, gain, speed, up = 2, 0.0040, "avgblur=sizeX=1:sizeY=16", 8, 1800, False
        color, alpha = '0xFFF4DC', 0.70
    elif fx == 'snow':
        k, density, blur, gain, speed, up = 4, 0.0022, "gblur=sigma=1.3", 14, 120, False
        color, alpha = '0xFFFFFF', 0.95
    elif fx == 'embers':
        k, density, blur, gain, speed, up = 3, 0.0016, "gblur=sigma=1.2", 14, 70, True
        color, alpha = '0xFFB347', 0.95
    else:
        return ''
    gw, gh = w // k, h // k
    y_expr = f"mod(t*{speed},{h})" if up else f"{h}-mod(t*{speed},{h})"
    return (
        f"nullsrc=s={gw}x{gh}:r=25:d=0.04,format=gray,"
        f"geq=lum='if(lt(random(1),{density}),255,0)',{blur},lutyuv=y='min(255,val*{gain})',"
        f"scale={w}:{h}:flags=bilinear,"
        f"split[pa][pb];[pa][pb]vstack,loop=loop=-1:size=1:start=0,setpts=N/25/TB,"
        f"crop={w}:{h}:0:'{y_expr}'[mask];"
        f"color=c={color}:s={w}x{h}:r=25[pcol];"
        f"[pcol][mask]alphamerge,colorchannelmixer=aa={alpha}[fxl]"
    )

# ─── Core FFmpeg — IMAGE MODE (loops image + audio) ───────────────────────────

def build_ffmpeg_command_image(image_path, audio_path, output_path, audio_duration,
                                font, font_italic, lyrics_font=None,
                                lyrics_segments=None, artist_name="SORLUNE",
                                song_title="", fx="rain"):
    """Build FFmpeg command using static image looped with audio — for shorts"""
    fade_out_st  = max(audio_duration - 3, audio_duration * 0.85)

    # Scale image to 720x1280 vertical 9:16
    scale_crop = (
        "scale=1080:1920:force_original_aspect_ratio=increase,"
        "crop=1080:1920"
    )
    # Subtle zoom effect on image
    # one still image -> d = every frame of the song, so the zoom really moves (d=1 reset it each frame)
    frames      = int(audio_duration * 25) + 25
    z_inc       = 0.06 / max(frames, 1)
    zoom_filter = (f"scale=2160:3840:flags=lanczos,"
                   f"zoompan=z='min(1.00+{z_inc:.8f}*on,1.06)':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)'"
                   f":d={frames}:s={OUT_W}x{OUT_H}:fps=25")

    grade_filter = (
        "eq=brightness=-0.02:contrast=1.05:saturation=0.95,"
        "curves=r='0/0 0.5/0.45 1/0.9':g='0/0 0.5/0.42 1/0.85':b='0/0 0.5/0.50 1/1.0'"
    )
    dark_overlay = (
        f"drawtext=fontfile={font}:text=' ':fontsize=1:fontcolor=black@0:"
        f"box=1:boxcolor=black@0.45:boxborderw=0:"
        f"x=0:y=h*{DARK_START}:fix_bounds=1"
    )
    # no fade-in: the very first frame is the Short's cover and its first-second hook (it was 2 s of black)
    fade_filter   = f"fade=t=out:st={fade_out_st:.2f}:d=3"
    artist_filter = build_artist_watermark(font_italic, artist_name)
    cta_filter    = build_subscribe_cta(font, audio_duration)
    eq_filter     = build_eq_bar(font)

    # background (image + zoom + grade), then the moving weather, then all text on top so it stays clean
    bg_chain = ",".join([zoom_filter, grade_filter, "format=yuv420p"])
    vf_parts = [dark_overlay, artist_filter]

    title_filter = build_song_title(font, song_title)
    if title_filter:
        vf_parts.append(title_filter)

    if lyrics_segments:
        karaoke = build_karaoke_filter(lyrics_segments, font, lyrics_font=lyrics_font)
        if karaoke:
            vf_parts.append(karaoke)

    vf_parts.append(cta_filter)
    vf_parts.append(eq_filter)
    vf_parts.append(fade_filter)

    fx_graph = build_weather_fx(fx)
    if fx_graph:
        graph = (f"[0:v]{bg_chain}[bg];{fx_graph};"
                 f"[bg][fxl]overlay=0:0:shortest=1,format=yuv420p," + ",".join(vf_parts) + "[v]")
    else:
        graph = f"[0:v]{bg_chain}," + ",".join(vf_parts) + "[v]"

    return [
        'ffmpeg', '-y',
        '-loop', '1',              # ✅ Loop image
        '-i', image_path,          # ✅ Input 0: image
        '-i', audio_path,          # ✅ Input 1: audio
        '-filter_complex', graph,
        '-map', '[v]',             # ✅ image + weather + text
        '-map', '1:a:0',           # ✅ audio from song
        '-c:v', 'libx264', '-preset', 'veryfast', '-crf', '20',
        '-threads', '2',
        '-c:a', 'aac', '-b:a', '192k',
        '-pix_fmt', 'yuv420p',
        '-t', str(audio_duration),
        '-shortest',
        output_path
    ]

# ─── Core FFmpeg — VIDEO MODE (loops video + audio) ───────────────────────────

def build_ffmpeg_command_short(video_path, audio_path, output_path, audio_duration,
                                font, font_italic, lyrics_font=None,
                                lyrics_segments=None, artist_name="SORLUNE",
                                song_title=""):
    """Build FFmpeg command using Pexels video — kept for manual short bot"""
    fade_out_st  = max(audio_duration - 3, audio_duration * 0.85)
    scale_crop   = (
        "scale=1080:1920:force_original_aspect_ratio=increase,"
        "crop=1080:1920"
    )
    grade_filter = (
        "eq=brightness=0.02:contrast=1.03:saturation=1.05,"
        "curves=r='0/0 0.5/0.53 1/1':g='0/0 0.5/0.48 1/0.95':b='0/0 0.5/0.43 1/0.86'"
    )
    dark_overlay = (
        f"drawtext=fontfile={font}:text=' ':fontsize=1:fontcolor=black@0:"
        f"box=1:boxcolor=black@0.55:boxborderw=0:"
        f"x=0:y=h*{DARK_START}:fix_bounds=1"
    )
    # no fade-in: the very first frame is the Short's cover and its first-second hook (it was 2 s of black)
    fade_filter   = f"fade=t=out:st={fade_out_st:.2f}:d=3"
    artist_filter = build_artist_watermark(font_italic, artist_name)
    cta_filter    = build_subscribe_cta(font, audio_duration)
    eq_filter     = build_eq_bar(font)

    vf_parts = [scale_crop, grade_filter, "format=yuv420p", dark_overlay, artist_filter]

    title_filter = build_song_title(font, song_title)
    if title_filter:
        vf_parts.append(title_filter)

    if lyrics_segments:
        karaoke = build_karaoke_filter(lyrics_segments, font, lyrics_font=lyrics_font)
        if karaoke:
            vf_parts.append(karaoke)

    vf_parts.append(cta_filter)
    vf_parts.append(eq_filter)
    vf_parts.append(fade_filter)

    return [
        'ffmpeg', '-y',
        '-stream_loop', '-1',
        '-i', video_path,
        '-i', audio_path,
        '-vf', ",".join(vf_parts),
        '-map', '0:v:0',
        '-map', '1:a:0',
        '-c:v', 'libx264', '-preset', 'veryfast', '-crf', '21',
        '-threads', '2',
        '-c:a', 'aac', '-b:a', '192k',
        '-pix_fmt', 'yuv420p',
        '-t', str(audio_duration),
        '-shortest',
        output_path
    ]

def generate_short_job(job_id, media_path, audio_path, output_path,
                       is_image=False, lyrics_segments=None,
                       artist_name="SORLUNE", song_title="", fx="rain"):
    try:
        save_job(job_id, {'status': 'processing'})
        audio_duration = get_audio_duration(audio_path)
        font           = get_best_font()
        font_italic    = get_italic_font()
        lyrics_font    = get_lyrics_font()

        if is_image:
            cmd = build_ffmpeg_command_image(
                media_path, audio_path, output_path,
                audio_duration, font, font_italic,
                lyrics_font=lyrics_font,
                lyrics_segments=lyrics_segments,
                artist_name=artist_name,
                song_title=song_title,
                fx=fx
            )
        else:
            cmd = build_ffmpeg_command_short(
                media_path, audio_path, output_path,
                audio_duration, font, font_italic,
                lyrics_font=lyrics_font,
                lyrics_segments=lyrics_segments,
                artist_name=artist_name,
                song_title=song_title
            )

        print(f"[FFmpeg] Starting job {job_id} (image={is_image})...")
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)

        if proc.returncode == 0 and os.path.exists(output_path):
            save_job(job_id, {
                'status': 'completed',
                'video_url': f"/videos/{job_id}/{job_id}.mp4",
                'duration': round(audio_duration, 1)
            })
            print(f"[Job {job_id}] ✅ Done!")
        else:
            error_msg = proc.stderr[-3000:] if proc.stderr else 'Unknown error'
            save_job(job_id, {'status': 'error', 'error': error_msg})
            print(f"[FFmpeg ERROR] {error_msg}")

    except Exception as e:
        save_job(job_id, {'status': 'error', 'error': str(e)})
        print(f"[Job {job_id}] ❌ {e}")

# ─── Routes ───────────────────────────────────────────────────────────────────


# ─── Routes (/short/...) — same requests and answers as the old Flask server ───

def _public_base(request: Request) -> str:
    base = os.environ.get("PUBLIC_BASE_URL") or os.environ.get("RENDER_EXTERNAL_URL", "")
    return (base or str(request.base_url)).rstrip('/')

async def _json(request: Request):
    try:
        return await request.json()
    except Exception:
        return None

@router.post('/generate-short')
async def generate_short(request: Request):
    data = await _json(request)
    if not data:
        return JSONResponse({'error': 'No JSON data'}, status_code=400)

    image_url      = (data.get('image_url') or '').strip()
    pexels_url     = (data.get('pexels_url') or '').strip()
    audio_url      = data.get('audio_url')
    api_key        = data.get('api_key') or str(uuid.uuid4())[:8]
    pexels_api_key = (data.get('pexels_api_key') or '').strip()
    artist_name    = (data.get('artist') or 'SORLUNE').strip()
    short_duration = int(data.get('duration', 45))
    lyrics_text    = (data.get('lyrics') or '').strip()
    openai_key     = (data.get('openai_key') or '').strip()
    song_title     = (data.get('title') or '').strip()
    fx             = (data.get('fx') or 'rain').strip().lower()
    if fx not in WEATHER_FX + ('none',): fx = 'rain'

    if not audio_url or (not image_url and not pexels_url):
        return JSONResponse({'error': 'Missing audio_url and image_url or pexels_url'}, status_code=400)

    use_image  = bool(image_url)
    job_id     = re.sub(r'[^A-Za-z0-9_-]', '', str(api_key)) or 'job'
    job_folder = os.path.join(UPLOAD_FOLDER, job_id)
    os.makedirs(job_folder, exist_ok=True)

    media_path  = os.path.join(job_folder, 'image.jpg' if use_image else 'pexels_video.mp4')
    audio_path  = os.path.join(job_folder, 'audio.mp3')
    output_path = os.path.join(job_folder, f'{job_id}.mp4')

    save_job(job_id, {'status': 'pending', 'video_url': None})

    def run():
        try:
            for f in [media_path, audio_path, output_path]:
                if os.path.exists(f): os.remove(f)

            if use_image:
                save_job(job_id, {'status': 'downloading_image'})
                download_file(image_url, media_path)
            else:
                save_job(job_id, {'status': 'downloading_video'})
                download_pexels_video(pexels_url, media_path, pexels_api_key)

            save_job(job_id, {'status': 'downloading_audio'})
            download_file(audio_url, audio_path)

            final_audio_path = audio_path
            lyrics_segments  = []
            try:
                save_job(job_id, {'status': 'finding_best_segment'})
                best_start = find_best_segment(audio_path, short_duration)
                total      = get_audio_duration(audio_path)

                # a few extra seconds after the loud part, so the cut can slide to where the VOICE starts
                pre_len = min(short_duration + 12, max(1.0, total - best_start))
                pre     = os.path.join(job_folder, 'audio_pre.mp3')
                subprocess.run(['ffmpeg', '-y', '-ss', str(best_start), '-i', audio_path, '-t', str(pre_len),
                                '-c:a', 'libmp3lame', '-b:a', '192k', pre], capture_output=True, timeout=120)
                src = pre if os.path.exists(pre) and os.path.getsize(pre) > 1000 else audio_path

                # ✅ VOICE FIRST: people swiped away in the first second because the Short opened on an instrumental bar.
                # Whisper finds the first sung word; the Short starts 0.25 s before it.
                offset = 0.0
                if openai_key:
                    try:
                        save_job(job_id, {'status': 'transcribing_lyrics'})
                        lines = transcribe_lyrics_with_whisper(src, openai_key, lyrics_text)
                        if lines:
                            first = float(lines[0]['start'])
                            if first > 0.6 and first < pre_len - short_duration + 0.5:
                                offset = max(0.0, first - 0.25)
                            # keep the captions that fall inside the final cut, moved to its timeline
                            for ln in lines:
                                st, en = float(ln['start']) - offset, float(ln['end']) - offset
                                if en <= 0.1 or st >= short_duration:
                                    continue
                                lyrics_segments.append({'start': round(max(0.0, st), 2),
                                                        'end': round(min(short_duration, en), 2),
                                                        'text': ln['text']})
                    except Exception as e:
                        print(f"[Lyrics] Whisper failed: {e}")

                trimmed_audio = os.path.join(job_folder, 'audio_best.mp3')
                proc_trim = subprocess.run([
                    'ffmpeg', '-y', '-ss', str(round(offset, 2)), '-i', src,
                    '-t', str(short_duration), '-c:a', 'libmp3lame', '-b:a', '192k', trimmed_audio
                ], capture_output=True, timeout=120)
                if proc_trim.returncode == 0 and os.path.exists(trimmed_audio) and os.path.getsize(trimmed_audio) > 1000:
                    final_audio_path = trimmed_audio
                save_job(job_id, {'status': 'cut_ready', 'voice_offset': round(offset, 2)})
            except Exception as trim_err:
                print(f"[Trim] Failed: {trim_err}")

            if not lyrics_segments and lyrics_text:
                duration = get_audio_duration(final_audio_path)
                lines    = split_lyrics_lines(lyrics_text)
                if lines:
                    step = max(duration / len(lines), 1.8)
                    current = 0.0
                    for line in lines:
                        lyrics_segments.append({"start": round(current, 2),
                                                "end": round(min(current + step, duration), 2), "text": line})
                        current += step

            generate_short_job(job_id, media_path, final_audio_path, output_path,
                               is_image=use_image, lyrics_segments=lyrics_segments,
                               artist_name=artist_name, song_title=song_title, fx=fx)
        except Exception as e:
            save_job(job_id, {'status': 'error', 'error': str(e)})
            print(f"[Job {job_id}] {e}")

    threading.Thread(target=run, daemon=True).start()
    return {'status': 'started', 'job_id': job_id}


@router.get('/status/{api_key}')
def short_status(api_key: str, request: Request):
    job = get_job(api_key)
    if not job:
        return {'status': 'not_found'}
    response = {'status': job['status']}
    if job['status'] == 'completed':
        url = job.get('video_url', '')
        # stored as /videos/... — served here under /short/videos/...
        response['video_url'] = url if url.startswith('http') else _public_base(request) + '/short' + url
        response['duration'] = job.get('duration')
    if job.get('error'):
        response['error'] = job['error']
    return response


@router.get('/videos/{job_id}/{filename}')
def short_video(job_id: str, filename: str):
    path = os.path.join(UPLOAD_FOLDER, os.path.basename(job_id), os.path.basename(filename))
    if not os.path.exists(path):
        return JSONResponse({'error': 'not found'}, status_code=404)
    return FileResponse(path, media_type='video/mp4')


@router.post('/clear-cache')
async def short_clear_cache(request: Request):
    data = await _json(request)
    api_key = data.get('api_key') if data else None
    if api_key:
        delete_job(api_key)
        import shutil
        job_folder = os.path.join(UPLOAD_FOLDER, os.path.basename(str(api_key)))
        if os.path.exists(job_folder):
            shutil.rmtree(job_folder, ignore_errors=True)
    return {'status': 'cleared'}


def _process_audio(audio_url, segment_duration):
    session_id = str(uuid.uuid4())[:8]
    audio_path = os.path.join(AUDIO_SEGMENTS_FOLDER, f'{session_id}_input.mp3')
    try:
        download_file(audio_url, audio_path)
    except Exception as e:
        return JSONResponse({'error': f'Download failed: {str(e)}'}, status_code=500)

    best_start = find_best_segment(audio_path, segment_duration)
    seg_fn     = f'{session_id}_seg000.mp3'
    seg_path   = os.path.join(AUDIO_SEGMENTS_FOLDER, seg_fn)
    proc = subprocess.run([
        'ffmpeg', '-y', '-ss', str(best_start), '-i', audio_path,
        '-t', str(segment_duration), '-c:a', 'libmp3lame', '-b:a', '192k', seg_path
    ], capture_output=True, timeout=120)
    os.remove(audio_path)
    if proc.returncode != 0 or not os.path.exists(seg_path):
        return JSONResponse({'error': 'Segment extraction failed'}, status_code=500)
    return {'segments': [seg_fn]}


@router.post('/process-audio')
async def short_process_audio(request: Request):
    data = await _json(request)
    if not data:
        return JSONResponse({'error': 'No JSON data'}, status_code=400)
    audio_url        = data.get('url')
    segment_duration = int(data.get('segment_duration', 45))
    if not audio_url:
        return JSONResponse({'error': 'Missing url'}, status_code=400)
    # scanning the song for its loudest part runs ffmpeg many times — keep it off the event loop
    import anyio
    return await anyio.to_thread.run_sync(_process_audio, audio_url, segment_duration)


@router.get('/audio_segments/{filename}')
def short_audio_segment(filename: str):
    path = os.path.join(AUDIO_SEGMENTS_FOLDER, os.path.basename(filename))
    if not os.path.exists(path):
        return JSONResponse({'error': 'not found'}, status_code=404)
    return FileResponse(path, media_type='audio/mpeg')


@router.get('/health')
def short_health():
    return {'status': 'ok', 'message': 'Short engine running — Image + Video modes, 1080x1920'}
