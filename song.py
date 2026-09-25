"""
Song video engine — the old video-generator-server (Flask app.py), moved onto
content-video-server so one Render service does both jobs.

Everything lives under /song/... and uses its own SONG_JOBS dict. The content
video side keys its JOBS by the same acg_api_key, so sharing one dict would let
a song job and a content video overwrite each other's status.
"""
import os
import re
import math
import time
import uuid
import shutil
import threading
import subprocess

import requests
from fastapi import APIRouter, Request
from fastapi.responses import FileResponse, JSONResponse

router = APIRouter(prefix="/song")

SONG_JOBS = {}
UPLOAD_FOLDER = '/tmp/video_jobs'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
AUDIO_SEGMENTS_FOLDER = '/tmp/audio_segments'
os.makedirs(AUDIO_SEGMENTS_FOLDER, exist_ok=True)

LYRICS_Y    = 0.80   # moved up — more space above EQ bar
EQ_CENTER_Y = 0.93
DARK_START  = 0.75   # dark band starts higher to cover lyrics area

def download_file(url, dest_path):
    headers = {'Cache-Control': 'no-cache', 'Pragma': 'no-cache'}
    r = requests.get(f"{url}?nocache={int(time.time())}", timeout=120, stream=True, headers=headers)
    if r.status_code != 200:
        r = requests.get(url, timeout=120, stream=True)
    r.raise_for_status()
    with open(dest_path, 'wb') as f:
        for chunk in r.iter_content(chunk_size=8192):
            f.write(chunk)
    return dest_path

def get_audio_duration(audio_path):
    result = subprocess.run(['ffprobe','-v','error','-show_entries','format=duration','-of','default=noprint_wrappers=1:nokey=1',audio_path],capture_output=True,text=True)
    return float(result.stdout.strip())

def get_best_font():
    for path in ['/usr/share/fonts/truetype/ubuntu/Ubuntu-B.ttf','/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf','/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf']:
        if os.path.exists(path): return path
    return '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'

def get_lyrics_font():
    # High-design serif font — cinematic, elegant, premium look
    for path in [
        '/usr/share/fonts/truetype/freefont/FreeSerifBold.ttf',
        '/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf',
        '/usr/share/fonts/truetype/liberation/LiberationSerif-Bold.ttf',
        '/usr/share/fonts/truetype/freefont/FreeSerifBoldItalic.ttf',
    ]:
        if os.path.exists(path):
            print(f"[Lyrics Font] {path}")
            return path
    return get_best_font()

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

_SECTION_WORDS = r'verse|chorus|bridge|hook|outro|intro|pre[\-\s]?chorus|post[\-\s]?chorus|refrain|interlude|instrumental|spoken|rap|breakdown|solo|ad[\-\s]?lib|vamp|coda|tag|skit|fade'
SECTION_REGEX = [re.compile(p, re.IGNORECASE) for p in [r'^\[.*\]$',r'^\(.*\)$',rf'^({_SECTION_WORDS})\s*[\d:.\-]*\s*$',rf'^({_SECTION_WORDS})\s*\d*\s*:$',r'^[\d\s\.\)\(\:\-]+$']]

def is_section_label(line):
    s = line.strip()
    return any(p.match(s) or p.match(s.rstrip(':').strip()) for p in SECTION_REGEX)

def split_lyrics_lines(text):
    if not text: return []
    return [l.strip() for l in text.replace('\r\n','\n').replace('\r','\n').split('\n') if l.strip() and not is_section_label(l.strip())]

def normalize_word(w):
    return re.sub(r"[^\w']","",(w or "").lower()).strip()

def transcribe_audio_words_with_whisper(audio_path, openai_api_key, lyrics_text=""):
    if not openai_api_key or not os.path.exists(audio_path): return []
    # language is forced: with music under the voice Whisper guessed Sinhala and the
    # captions came out in Sinhala script. The start of the lyrics is passed as a hint.
    hint = " ".join(split_lyrics_lines(lyrics_text))[:600]
    form = {"model":"whisper-1","response_format":"verbose_json","timestamp_granularities[]":"word","language":"en"}
    if hint: form["prompt"] = hint
    try:
        with open(audio_path,"rb") as audio_file:
            response = requests.post("https://api.openai.com/v1/audio/transcriptions",headers={"Authorization":f"Bearer {openai_api_key}"},files={"file":audio_file},data=form,timeout=300)
        if response.status_code != 200: return []
        data = response.json()
        cleaned = []
        for w in data.get("words",[]):
            word_text=(w.get("word") or "").strip()
            start=w.get("start"); end=w.get("end")
            if not word_text or start is None or end is None: continue
            start,end=float(start),float(end)
            if end<=start: continue
            cleaned.append({"word":word_text,"norm":normalize_word(word_text),"start":start,"end":end})
        if cleaned: return cleaned
        seg_words=[]
        for seg in data.get("segments",[]):
            text=(seg.get("text") or "").strip(); start=seg.get("start"); end=seg.get("end")
            if not text or start is None or end is None: continue
            seg_words.append({"word":text,"norm":normalize_word(text),"start":float(start),"end":float(end)})
        return seg_words
    except Exception as e:
        print(f"[Whisper] Error: {e}"); return []

def build_lines_from_words(words, max_gap=0.45, max_words=6, max_duration=3.0):
    if not words: return []
    lines=[]; current=[words[0]]
    def flush(lw):
        if not lw: return None
        text=" ".join(w["word"] for w in lw).strip()
        return {"start":round(lw[0]["start"],2),"end":round(lw[-1]["end"],2),"text":text} if text else None
    for w in words[1:]:
        prev=current[-1]
        if w["start"]-prev["end"]>max_gap or len(current)>=max_words or w["end"]-current[0]["start"]>max_duration:
            item=flush(current)
            if item: lines.append(item)
            current=[w]
        else: current.append(w)
    item=flush(current)
    if item: lines.append(item)
    cleaned=[]
    for seg in lines:
        start=float(seg["start"]); end=float(seg["end"]); text=seg["text"].strip()
        if not text: continue
        min_dur=max(0.60,min(1.40,len(text.split())*0.22))
        if end-start<min_dur: end=start+min_dur
        if cleaned and start<cleaned[-1]["end"]: start=round(cleaned[-1]["end"]+0.03,2); end=max(end,start+min_dur)
        cleaned.append({"start":round(start,2),"end":round(end,2),"text":text})
    return cleaned

def transcribe_lyrics_with_whisper(audio_path, openai_api_key, lyrics_text=""):
    return build_lines_from_words(transcribe_audio_words_with_whisper(audio_path, openai_api_key))

def ffmpeg_escape(text):
    text=text.replace('\\','\\\\'); text=text.replace("'","’"); text=text.replace(':','\\:')
    text=text.replace('%','\\%'); text=text.replace('[','\\['); text=text.replace(']','\\]'); text=text.replace(',','\\,')
    return text

def build_artist_watermark(font_italic, artist_name="SORLUNE"):
    name=ffmpeg_escape(artist_name.upper())
    padding=28
    alpha_expr="0.875+0.125*sin(6.2832/4.0*t)"
    # Gold italic name top-right
    watermark=(f"drawtext=fontfile={font_italic}:text='{name}':"
               f"fontsize=34:fontcolor=0xD4AF37@1.0:"
               f"borderw=2:bordercolor=black@0.80:"
               f"shadowcolor=black@0.70:shadowx=2:shadowy=2:"
               f"x=w-text_w-{padding}:y={padding}:alpha='{alpha_expr}'")
    # Gold underline decoration
    underline=(f"drawtext=fontfile={font_italic}:text='———————':"
               f"fontsize=14:fontcolor=0xD4AF37@1.0:"
               f"x=w-text_w-{padding}:y={padding+42}:alpha='{alpha_expr}'")
    return ",".join([watermark, underline])

def wrap_lyric_line(text, max_chars=44):
    if len(text)<=max_chars: return [text]
    words=text.split(); best_split,best_diff=len(words)//2,float('inf')
    for i in range(1,len(words)):
        p1,p2=" ".join(words[:i])," ".join(words[i:])
        diff=abs(len(p1)-len(p2))
        if diff<best_diff and len(p1)<=max_chars and len(p2)<=max_chars: best_diff,best_split=diff,i
    return [" ".join(words[:best_split])," ".join(words[best_split:])]

def build_karaoke_filter(segments, font, lyrics_font=None):
    if lyrics_font is None: lyrics_font = font
    if not segments: return ""
    parts=[]; FONT_SIZE=44; LINE_HEIGHT=54; MAX_CHARS=44
    for seg in segments:
        start,end,raw_text=seg["start"],seg["end"],seg["text"]
        dur=max(end-start,0.5); fade_dur=min(0.18,dur/5)
        alpha_expr=(f"if(between(t,{start},{start+fade_dur}),(t-{start})/{fade_dur},"
                    f"if(between(t,{start+fade_dur},{end-fade_dur}),1,"
                    f"if(between(t,{end-fade_dur},{end}),({end}-t)/{fade_dur},0)))")
        lines=wrap_lyric_line(raw_text,max_chars=MAX_CHARS)
        if len(lines)==1:
            parts.append(f"drawtext=fontfile={lyrics_font}:text='{ffmpeg_escape(lines[0])}':"
                         f"fontsize={FONT_SIZE}:fontcolor=white@1.0:"
                         f"borderw=4:bordercolor=black@1.0:"
                         f"shadowcolor=black@0.95:shadowx=3:shadowy=3:"
                         f"x=(w-text_w)/2:y=h*{LYRICS_Y}:alpha='{alpha_expr}'")
        else:
            base_y=LYRICS_Y-0.045
            for li,line in enumerate(lines):
                parts.append(f"drawtext=fontfile={lyrics_font}:text='{ffmpeg_escape(line)}':"
                             f"fontsize={FONT_SIZE}:fontcolor=white@1.0:"
                             f"borderw=4:bordercolor=black@1.0:"
                             f"shadowcolor=black@0.95:shadowx=3:shadowy=3:"
                             f"x=(w-text_w)/2:y=h*{base_y}+{li*LINE_HEIGHT}:alpha='{alpha_expr}'")
    return ",".join(parts)

def build_eq_bar(font):
    parts=[]; bar_count=30; bar_gap=14; half=bar_count//2; center_y=f"h*{EQ_CENTER_Y}"
    freqs=[1.3,2.1,2.7,1.9,3.1,2.4,1.7,2.9,2.2,3.5,2.0,2.8,2.1,2.8,2.0,3.5,2.2,2.9,1.7,2.4,3.1,1.9,2.7,2.1,1.3,1.8,2.5,3.0,1.6,2.3]
    phases=[0.0,0.5,1.1,1.7,0.3,0.9,1.5,0.2,0.8,1.4,0.6,1.2,0.0,1.2,0.6,1.4,0.8,0.2,1.5,0.9,0.3,1.7,1.1,0.5,0.0,0.7,1.3,0.4,1.0,1.6]
    for i in range(bar_count):
        dist=abs(i-half)/half; amplitude=int(5+36*math.exp(-2.5*dist*dist))
        alpha_up=0.90-0.25*dist; alpha_dwn=0.40-0.15*dist
        offset=(i-half)*bar_gap; bar_x=f"(w/2+({offset})-tw/2)"; fs_expr=f"4+{amplitude}*abs(sin(t*{freqs[i]}+{phases[i]}))"
        parts.append(f"drawtext=fontfile={font}:text='|':fontsize={fs_expr}:fontcolor=0xD4AF37@{alpha_up:.2f}:x={bar_x}:y=({center_y})-text_h")
        parts.append(f"drawtext=fontfile={font}:text='|':fontsize={fs_expr}:fontcolor=0xB8860B@{alpha_dwn:.2f}:x={bar_x}:y={center_y}")
    return ",".join(parts)

def build_ffmpeg_command(image_path, audio_path, output_path, duration, fps, font, font_italic, lyrics_font=None, lyrics_segments=None, artist_name="SORLUNE"):
    frames=int(duration*fps); fade_out_st=max(duration-3,duration*0.85); z_inc=0.08/max(frames,1)
    zoom_filter=(f"scale=3840:2160:flags=lanczos,zoompan=z='min(1.00+{z_inc:.8f}*on,1.08)':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':d={frames}:s=1280x720:fps={fps}")
    light_filter=(f"eq=brightness='0.03*sin(t*2.2+0.3)':contrast='1.04+0.03*sin(t*1.8+1.0)':saturation='1.06+0.08*sin(t*2.5+0.8)'")
    grade_filter="curves=r='0/0 0.5/0.53 1/1':g='0/0 0.5/0.48 1/0.95':b='0/0 0.5/0.43 1/0.86',vignette=PI/4.5,noise=alls=3:allf=t"
    fade_filter=f"fade=t=in:st=0:d=2,fade=t=out:st={fade_out_st:.2f}:d=3"
    dark_overlay=(f"drawtext=fontfile={font}:text=' ':fontsize=1:fontcolor=black@0:box=1:boxcolor=black@0.52:boxborderw=0:x=0:y=h*{DARK_START}:fix_bounds=1")
    artist_filter=build_artist_watermark(font_italic, artist_name)
    karaoke_filter=build_karaoke_filter(lyrics_segments, font, lyrics_font=lyrics_font) if lyrics_segments else ""
    eq_filter=build_eq_bar(font)
    vf_parts=[zoom_filter,light_filter,grade_filter,fade_filter,"format=yuv420p",dark_overlay,artist_filter]
    if karaoke_filter: vf_parts.append(karaoke_filter)
    vf_parts.append(eq_filter)
    vf_chain=",".join(vf_parts)
    return ['ffmpeg','-y','-loop','1','-i',image_path,'-i',audio_path,'-vf',vf_chain,'-c:v','libx264','-preset','ultrafast','-crf','20','-c:a','aac','-b:a','192k','-pix_fmt','yuv420p','-t',str(duration),'-shortest',output_path]

def generate_video_job(job_id, image_path, audio_path, output_path, lyrics_segments=None, artist_name="SORLUNE"):
    try:
        SONG_JOBS[job_id]['status']='processing'
        duration=get_audio_duration(audio_path); font=get_best_font(); font_italic=get_italic_font(); lyrics_font=get_lyrics_font()
        cmd=build_ffmpeg_command(image_path,audio_path,output_path,duration,25,font,font_italic,lyrics_font=lyrics_font,lyrics_segments=lyrics_segments,artist_name=artist_name)
        proc=subprocess.run(cmd,capture_output=True,text=True,timeout=3600)
        if proc.returncode==0 and os.path.exists(output_path):
            SONG_JOBS[job_id]['status']='completed'
        else:
            SONG_JOBS[job_id]['status']='error'; SONG_JOBS[job_id]['error']=proc.stderr[-3000:]
            print(f"[FFmpeg ERROR]\n{proc.stderr[-3000:]}")
    except Exception as e:
        SONG_JOBS[job_id]['status']='error'; SONG_JOBS[job_id]['error']=str(e)

# ══════════════════════════════════════════════════════════════════
# CHAPTERS — where each lyric section (Verse 1, Chorus...) starts in the audio
# YouTube only shows chapters when: first is 0:00, at least 3, each >= 10s.
# ══════════════════════════════════════════════════════════════════

CHAPTER_MIN_GAP = 10.0

def _section_name(label):
    s = re.sub(r'[\[\]\(\):]', '', label).strip()
    return s.title() if s else 'Part'

def parse_lyric_sections(text):
    """[{'name': 'Verse 1', 'lines': [...]}, ...] in song order."""
    sections = []
    current = None
    for raw in (text or '').replace('\r\n', '\n').replace('\r', '\n').split('\n'):
        line = raw.strip()
        if not line:
            continue
        if is_section_label(line):
            current = {'name': _section_name(line), 'lines': []}
            sections.append(current)
            continue
        if current is None:
            current = {'name': 'Verse 1', 'lines': []}
            sections.append(current)
        current['lines'].append(line)
    return [s for s in sections if s['lines']]

def _find_line_start(words, line, cursor):
    """Time the first words of `line` are sung, searching forward from `cursor`."""
    target = [normalize_word(w) for w in line.split() if normalize_word(w)][:4]
    if not target:
        return None, cursor
    norms = [w['norm'] for w in words]
    # 2 of the first 4 words is enough: sung audio is often misheard ("love" -> "glove", "2" -> "two")
    need = min(2, len(target))
    for i in range(cursor, len(norms)):
        hits = sum(1 for k in range(len(target)) if i + k < len(norms) and norms[i + k] == target[k])
        if hits >= need:
            return float(words[i]['start']), i + 1
    return None, cursor

def build_chapters(lyrics_text, words, total):
    sections = parse_lyric_sections(lyrics_text)
    if not sections or total < CHAPTER_MIN_GAP * 3:
        return []

    # unique names — the last chorus reads better as "Final Chorus"
    counts = {}
    for s in sections:
        counts[s['name']] = counts.get(s['name'], 0) + 1
    seen = {}
    for s in sections:
        n = s['name']; seen[n] = seen.get(n, 0) + 1
        if counts[n] > 1 and n.lower() == 'chorus' and seen[n] == counts[n]:
            s['name'] = 'Final Chorus'

    starts = []
    if words:
        cursor = 0
        for s in sections:
            t, cursor = _find_line_start(words, s['lines'][0], cursor)
            starts.append(t)

    # sections the transcript could not place get a share of the time by line count
    if not starts or sum(1 for t in starts if t is not None) < 2:
        lines = [len(s['lines']) for s in sections]
        acc, whole = 0.0, float(sum(lines)) or 1.0
        starts = []
        for n in lines:
            starts.append(total * acc / whole); acc += n
    else:
        for i, t in enumerate(starts):
            if t is None:
                prev = next((starts[j] for j in range(i - 1, -1, -1) if starts[j] is not None), 0.0)
                nxt = next((starts[j] for j in range(i + 1, len(starts)) if starts[j] is not None), total)
                starts[i] = (prev + nxt) / 2.0

    chapters = []
    if starts[0] >= CHAPTER_MIN_GAP:
        chapters.append({'t': 0.0, 'name': 'Intro'})
    for s, t in zip(sections, starts):
        t = 0.0 if not chapters else max(0.0, float(t))
        if chapters and t - chapters[-1]['t'] < CHAPTER_MIN_GAP:
            continue
        if total - t < CHAPTER_MIN_GAP:
            break
        chapters.append({'t': round(t, 1), 'name': s['name']})

    if chapters:
        chapters[0]['t'] = 0.0
    return chapters if len(chapters) >= 3 else []

# ══════════════════════════════════════════════════════════════════
# CAPTIONS FROM YOUR LYRICS — the on-screen text is the written lyrics
# (always English, always spelled right); Whisper only supplies the timing.
# ══════════════════════════════════════════════════════════════════

def _is_latin(text):
    """False when the text has letters outside Latin script (e.g. Sinhala, Arabic)."""
    return not re.search(r'[^\x00-ɏḀ-ỿ -⁯←-⇿\s]', text or '')

def build_lines_from_lyrics(lyrics_text, words, total):
    lines = split_lyrics_lines(lyrics_text)
    if not lines or not words:
        return []

    starts, at, cursor = [], [], 0
    for line in lines:
        t, cursor = _find_line_start(words, line, cursor)
        starts.append(t)
        at.append(cursor - 1 if t is not None else None)

    found = [i for i, t in enumerate(starts) if t is not None]
    if len(found) < max(2, len(lines) // 3):
        return []

    # lines after the last match: keep them while transcript words remain, one line per its word count
    first, last = found[0], found[-1]
    ptr = at[last] + len(lines[last].split())
    end_i = last
    for i in range(last + 1, len(lines)):
        if ptr >= len(words):
            break
        starts[i] = float(words[ptr]['start'])
        ptr += len(lines[i].split())
        end_i = i

    # lines before the first match were not sung in this audio (short clips)
    lines, starts = lines[first:end_i + 1], starts[first:end_i + 1]

    # lines between two matches share the gap
    for i, t in enumerate(starts):
        if t is None:
            prev = next(starts[j] for j in range(i - 1, -1, -1) if starts[j] is not None)
            nxt = next(starts[j] for j in range(i + 1, len(starts)) if starts[j] is not None)
            k = next(j for j in range(i + 1, len(starts)) if starts[j] is not None)
            p = next(j for j in range(i - 1, -1, -1) if starts[j] is not None)
            starts[i] = prev + (nxt - prev) * (i - p) / float(k - p)

    segs = []
    for i, (line, start) in enumerate(zip(lines, starts)):
        start = max(0.0, float(start))
        if segs and start <= segs[-1]['start']:
            start = segs[-1]['start'] + 0.3
        nxt = starts[i + 1] if i + 1 < len(starts) else total
        end = min(float(nxt) - 0.05, start + 7.0, total)
        if end - start < 0.6:
            end = min(total, start + 0.6)
        if segs and segs[-1]['end'] > start:
            segs[-1]['end'] = round(max(segs[-1]['start'] + 0.3, start - 0.03), 2)
        segs.append({'start': round(start, 2), 'end': round(end, 2), 'text': line})
    return segs

def public_base(request: Request) -> str:
    # behind Render's proxy request.base_url comes back as http://, so prefer
    # the service's own https address when Render provides it
    base = os.environ.get("PUBLIC_BASE_URL") or os.environ.get("RENDER_EXTERNAL_URL", "")
    return (base or str(request.base_url)).rstrip('/')

@router.post('/generate')
async def song_generate(request: Request):
    try:
        data = await request.json()
    except Exception:
        data = None
    if not data: return JSONResponse({'error':'No JSON data'},status_code=400)
    audio_url=data.get('audio_url'); image_url=data.get('image_url'); api_key=data.get('api_key') or 'default'
    lyrics_text=(data.get('lyrics') or '').strip(); openai_key=(data.get('openai_key') or '').strip(); artist_name=(data.get('artist') or 'SORLUNE').strip()
    if not audio_url or not image_url: return JSONResponse({'error':'Missing audio_url or image_url'},status_code=400)
    job_id=api_key; job_folder=os.path.join(UPLOAD_FOLDER,job_id); os.makedirs(job_folder,exist_ok=True)
    image_path=os.path.join(job_folder,'image.jpg'); audio_path=os.path.join(job_folder,'audio.mp3'); output_path=os.path.join(job_folder,f'{job_id}.mp4')
    SONG_JOBS[job_id]={'status':'pending','video_url':None}
    def run():
        try:
            for f in [image_path,audio_path,output_path]:
                if os.path.exists(f): os.remove(f)
            SONG_JOBS[job_id]['status']='downloading_assets'
            download_file(image_url,image_path); download_file(audio_url,audio_path)
            lyrics_segments=[]; words=[]
            if openai_key:
                try:
                    SONG_JOBS[job_id]['status']='transcribing_lyrics'
                    words=transcribe_audio_words_with_whisper(audio_path,openai_key,lyrics_text)
                    total_len=get_audio_duration(audio_path)
                    # 1) your written lyrics on Whisper's timing
                    lyrics_segments=build_lines_from_lyrics(lyrics_text,words,total_len)
                    # 2) Whisper's own lines, only if they are in Latin script
                    if not lyrics_segments:
                        lines=build_lines_from_words(words)
                        if lines and all(_is_latin(l['text']) for l in lines):
                            lyrics_segments=lines
                    SONG_JOBS[job_id]['lyrics_mode']='lyrics' if lyrics_segments and lyrics_text and lyrics_segments[0]['text'] in lyrics_text else ('whisper' if lyrics_segments else 'spread')
                except Exception as e:
                    print(f"[Lyrics] Whisper failed: {e}"); lyrics_segments=[]
            # YouTube chapters from the real audio, handed back through /song/status
            try:
                total=get_audio_duration(audio_path)
                SONG_JOBS[job_id]['duration']=round(total,1)
                SONG_JOBS[job_id]['chapters']=build_chapters(lyrics_text,words,total)
            except Exception as e:
                print(f"[Chapters] {e}")
            if not lyrics_segments and lyrics_text:
                duration=get_audio_duration(audio_path); lines=split_lyrics_lines(lyrics_text)
                if lines:
                    step=max(duration/len(lines),1.8); current=0.0
                    for line in lines:
                        lyrics_segments.append({"start":round(current,2),"end":round(min(current+step,duration),2),"text":line}); current+=step
            generate_video_job(job_id,image_path,audio_path,output_path,lyrics_segments=lyrics_segments,artist_name=artist_name)
        except Exception as e:
            SONG_JOBS[job_id]['status']='error'; SONG_JOBS[job_id]['error']=str(e)
    threading.Thread(target=run,daemon=True).start()
    return {'status':'started','job_id':job_id,'lyrics_mode':'whisper' if openai_key else 'fallback'}

@router.get('/status/{api_key}')
def song_status(api_key: str, request: Request):
    job=SONG_JOBS.get(api_key)
    if not job: return {'status':'not_found'}
    response={'status':job['status']}
    if job['status']=='completed': response['video_url']=public_base(request)+f'/song/videos/{api_key}/{api_key}.mp4'
    if 'chapters' in job: response['chapters']=job['chapters']
    if 'duration' in job: response['duration']=job['duration']
    if job.get('error'): response['error']=job['error']
    return response

@router.get('/videos/{job_id}/{filename}')
def song_video(job_id: str, filename: str):
    path=os.path.join(UPLOAD_FOLDER,os.path.basename(job_id),os.path.basename(filename))
    if not os.path.exists(path): return JSONResponse({'error':'not found'},status_code=404)
    return FileResponse(path, media_type='video/mp4')

@router.post('/clear-cache')
async def song_clear_cache(request: Request):
    try:
        data = await request.json()
    except Exception:
        data = None
    api_key=data.get('api_key') if data else None
    if api_key:
        SONG_JOBS.pop(api_key,None)
        job_folder=os.path.join(UPLOAD_FOLDER,os.path.basename(api_key))
        if os.path.exists(job_folder): shutil.rmtree(job_folder,ignore_errors=True)
    return {'status':'cleared'}

@router.post('/process-audio')
async def song_process_audio(request: Request):
    try:
        data = await request.json()
    except Exception:
        data = None
    if not data: return JSONResponse({'error':'No JSON data'},status_code=400)
    audio_url=data.get('url'); segment_duration=int(data.get('segment_duration',60))
    if not audio_url: return JSONResponse({'error':'Missing url'},status_code=400)
    # the split runs ffmpeg for minutes — keep it off the event loop
    import anyio
    return await anyio.to_thread.run_sync(_split_audio, audio_url, segment_duration)

def _split_audio(audio_url, segment_duration):
    session_id=str(uuid.uuid4())[:8]; audio_path=os.path.join(AUDIO_SEGMENTS_FOLDER,f'{session_id}_input.mp3')
    try: download_file(audio_url,audio_path)
    except Exception as e: return JSONResponse({'error':f'Download failed: {str(e)}'},status_code=500)
    result=subprocess.run(['ffprobe','-v','error','-show_entries','format=duration','-of','default=noprint_wrappers=1:nokey=1',audio_path],capture_output=True,text=True)
    try: total_duration=float(result.stdout.strip())
    except Exception: return JSONResponse({'error':'Could not read audio duration'},status_code=500)
    segments=[]; start,idx=0,0
    while start<total_duration:
        seg_fn=f'{session_id}_seg{idx:03d}.mp3'; seg_path=os.path.join(AUDIO_SEGMENTS_FOLDER,seg_fn)
        proc=subprocess.run(['ffmpeg','-y','-i',audio_path,'-ss',str(start),'-t',str(segment_duration),'-c:a','libmp3lame','-b:a','192k',seg_path],capture_output=True,timeout=120)
        if proc.returncode==0 and os.path.exists(seg_path): segments.append(seg_fn)
        start+=segment_duration; idx+=1
    os.remove(audio_path)
    return {'segments':segments}

@router.get('/audio_segments/{filename}')
def song_audio_segment(filename: str):
    path=os.path.join(AUDIO_SEGMENTS_FOLDER,os.path.basename(filename))
    if not os.path.exists(path): return JSONResponse({'error':'not found'},status_code=404)
    return FileResponse(path, media_type='audio/mpeg')

@router.get('/health')
def song_health():
    return {'status':'ok','message':'Song video engine running'}
