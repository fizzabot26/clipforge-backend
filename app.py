import os, uuid, json, time, shutil, subprocess, threading, glob
from pathlib import Path
import requests, edge_tts
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

D = Path(os.environ.get("DATA_DIR", "/tmp/data")); D.mkdir(parents=True, exist_ok=True)
K = os.environ.get("GROQ_API_KEY", "")
LLM = os.environ.get("LLM_MODEL", "llama-3.3-70b-versatile")
STT = os.environ.get("STT_MODEL", "whisper-large-v3-turbo")
TTL = 5 * 86400  # 5 din baad auto trash
LITE = os.environ.get("LITE", "1") == "1"  # 1 = halka mode (Render free), 0 = full power (bada server)
GEM_KEY = os.environ.get("GEMINI_API_KEY", "")
GEM_MODEL = os.environ.get("GEMINI_MODEL", "gemini-flash-latest")
OR_KEY = os.environ.get("OPENROUTER_API_KEY", "")
OR_MODEL = os.environ.get("OPENROUTER_MODEL", "meta-llama/llama-3.3-70b-instruct:free")
MAX_UPLOAD_MB = int(os.environ.get("MAX_UPLOAD_MB", "300" if LITE else "1500"))
G = "https://api.groq.com/openai/v1/"

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
app.mount("/files", StaticFiles(directory=D), name="files")
JOBS = {}

REGION = {
    "USA": "Friendly, direct, straightforward. Short punchy sentences, optimistic, self-improvement framing.",
    "Canada": "Polite, warm, understated humour, inclusive tone.",
    "Europe": "Thoughtful, culture-aware, less hype, quality over volume.",
    "UK": "Dry wit, understated, serious topics handled with composure.",
    "Gulf": "Respectful, family and success values, formal courtesy, aspirational luxury and faith-aware tone.",
    "China": "Collective progress, discipline, resilience, fast-paced growth stories.",
    "Japan": "Quiet determination, craftsmanship, polite and measured, gradual improvement.",
    "Korea": "Trend-driven, high-energy, career and glow-up milestones, emotional storytelling.",
    "Russia": "Pragmatic, blunt, no filler, endurance and realism.",
    "Germany": "Precise, factual, structured arguments, practical value.",
    "Scandinavia": "Calm, egalitarian, minimalist, nature and balance.",
    "Singapore": "Efficient, multicultural, practical, aspirational but grounded.",
}
LOC = {
    "USA": ["en-US"], "Canada": ["en-CA"], "Europe": ["fr-FR", "es-ES", "it-IT", "nl-NL"], "UK": ["en-GB"],
    "Gulf": ["ar-SA", "ar-AE"], "China": ["zh-CN"], "Japan": ["ja-JP"], "Korea": ["ko-KR"], "Russia": ["ru-RU"],
    "Germany": ["de-DE"], "Scandinavia": ["sv-SE", "nb-NO", "da-DK"], "Singapore": ["en-SG"],
}
MOOD = {"Horror": ("-15%", "-4Hz"), "Comedy": ("+12%", "+3Hz"), "Motivation": ("+5%", "+2Hz"),
        "Romance": ("-8%", "+1Hz"), "Commentary": ("+0%", "+0Hz"), "Fight/Argue": ("+15%", "-2Hz")}


def sh(*a):
    r = subprocess.run(a, capture_output=True, text=True)
    if r.returncode:
        raise RuntimeError((r.stderr or "command failed")[-500:])
    return r.stdout


def purge():
    for p in D.iterdir():
        if p.is_dir() and time.time() - p.stat().st_mtime > TTL:
            shutil.rmtree(p, ignore_errors=True)


def _json(txt):
    t = (txt or "").strip()
    if t.startswith("```"):
        t = t.strip("`")
        t = t[4:] if t.lower().startswith("json") else t
    return json.loads(t.strip())


def _groq(system, user):
    r = requests.post(G + "chat/completions", headers={"Authorization": "Bearer " + K}, timeout=120, json={
        "model": LLM, "response_format": {"type": "json_object"}, "temperature": 0.4,
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]})
    r.raise_for_status()
    return _json(r.json()["choices"][0]["message"]["content"])


def _gemini(system, user):
    u = f"https://generativelanguage.googleapis.com/v1beta/models/{GEM_MODEL}:generateContent"
    r = requests.post(u, headers={"x-goog-api-key": GEM_KEY}, timeout=120, json={
        "systemInstruction": {"parts": [{"text": system}]},
        "contents": [{"role": "user", "parts": [{"text": user}]}],
        "generationConfig": {"responseMimeType": "application/json", "temperature": 0.4}})
    r.raise_for_status()
    return _json(r.json()["candidates"][0]["content"]["parts"][0]["text"])


def _openrouter(system, user):
    r = requests.post("https://openrouter.ai/api/v1/chat/completions", headers={"Authorization": "Bearer " + OR_KEY}, timeout=120, json={
        "model": OR_MODEL, "temperature": 0.4,
        "messages": [{"role": "system", "content": system + " Reply with JSON only."}, {"role": "user", "content": user}]})
    r.raise_for_status()
    return _json(r.json()["choices"][0]["message"]["content"])


def llm(system, user):
    """Fallback chain: pehla provider limit/error de to agla khud chalta hai."""
    chain = [(n, f) for n, f, ok in (("groq", _groq, K), ("gemini", _gemini, GEM_KEY), ("openrouter", _openrouter, OR_KEY)) if ok]
    if not chain:
        raise RuntimeError("No AI key set: add GROQ_API_KEY (and optionally GEMINI_API_KEY / OPENROUTER_API_KEY) in the host's environment variables")
    errs = []
    for name, fn in chain:
        try:
            return fn(system, user)
        except Exception as ex:
            errs.append(f"{name}: {str(ex)[:120]}")
    raise RuntimeError("All AI providers failed or hit limits - " + " | ".join(errs))


def save_upload(up, dest):
    """Upload save karo, size cap ke saath (free server ka disk/RAM bachane ke liye)."""
    cap = MAX_UPLOAD_MB * 1024 * 1024; n = 0
    with open(dest, "wb") as f:
        while True:
            chunk = up.file.read(1024 * 1024)
            if not chunk: break
            n += len(chunk)
            if n > cap:
                f.close(); dest.unlink(missing_ok=True)
                raise HTTPException(413, f"File is larger than {MAX_UPLOAD_MB} MB limit on this server. Trim it or use a shorter video.")
            f.write(chunk)


def transcribe(src, d):
    a = d / "a.mp3"
    sh("ffmpeg", "-y", "-i", str(src), "-vn", "-ac", "1", "-ar", "16000", "-b:a", "32k", str(a))
    if a.stat().st_size > 24_000_000:
        raise RuntimeError("Audio exceeds the 25MB limit: use a shorter video or a smaller timeline range")
    with open(a, "rb") as f:
        r = requests.post(G + "audio/transcriptions", headers={"Authorization": "Bearer " + K}, timeout=900,
                          files={"file": ("a.mp3", f)}, data={"model": STT, "response_format": "verbose_json"})
    r.raise_for_status()
    return [{"s": x["start"], "e": x["end"], "t": x["text"].strip()} for x in r.json()["segments"]]


def cut(src, s, e, out, vert):
    if LITE and not vert:  # re-encode nahi: CPU almost zero, start thoda keyframe par snap ho sakta hai
        sh("ffmpeg", "-y", "-ss", f"{s:.2f}", "-i", str(src), "-t", f"{e - s:.2f}", "-c", "copy",
           "-avoid_negative_ts", "make_zero", "-movflags", "+faststart", str(out))
        return
    vf = ["-vf", "crop=ih*9/16:ih,scale=720:1280"] if vert else []
    sh("ffmpeg", "-y", "-ss", f"{s:.2f}", "-i", str(src), "-t", f"{e - s:.2f}", *vf, "-c:v", "libx264",
       "-preset", "veryfast", "-crf", "23", "-threads", "1" if LITE else "0",
       "-c:a", "aac", "-movflags", "+faststart", str(out))


def work(j, p):
    J = JOBS[j]; d = D / j
    try:
        if p["url"]:
            J["msg"] = "Downloading video"
            sh("yt-dlp", "-f", "bv*[height<=720]+ba/b[height<=720]/b", "--merge-output-format", "mp4",
               "--no-playlist", "-o", str(d / "src.%(ext)s"), p["url"])
        src = Path(glob.glob(str(d / "src.*"))[0])
        total = float(sh("ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(src)).strip())
        t1 = p["t1"] if p["t1"] > 0 else total
        cats, mix, n, T = p["cats"], p["mix"], p["count"], p["target"]
        L = T / len(cats) if mix else T
        loL = float(p.get("minL") or 0) or L * 0.4
        hiL = float(p.get("maxL") or 0) or L * 1.3
        if K:
            J["msg"] = "Transcribing audio and finding highlights"
            seg = [x for x in transcribe(src, d) if x["s"] >= p["t0"] and x["e"] <= t1]
            txt = "\n".join(f'{x["s"]:.0f}-{x["e"]:.0f}: {x["t"]}' for x in seg)[:60000]
            need = n * len(cats) if mix else n
            res = llm('Pick viral clip segments from a transcript. Return JSON {"clips":[{"start":sec,"end":sec,'
                      '"category":str,"title":str,"score":0-100}]}. No overlaps, sentence boundaries, length about '
                      + str(int(L)) + " seconds, category must be one of " + str(cats) + ".",
                      f"Need {need} clips ({'equal number per category' if mix else 'best overall'}).\n" + txt)
            clips = res.get("clips", [])
        else:  # keys ke bina bhi chale: barabar hisson me kaat do
            step = max(L, (t1 - p["t0"]) / max(1, n * (len(cats) if mix else 1)))
            clips = [{"start": p["t0"] + i * step, "end": p["t0"] + i * step + L, "category": cats[i % len(cats)],
                      "title": f"Clip {i + 1}"} for i in range(n * (len(cats) if mix else 1))]
        ok = []
        for c in clips:
            s = max(p["t0"], float(c["start"])); e = min(t1, float(c["end"]))
            if e - s > hiL: e = s + hiL
            if e - s >= loL: ok.append({**c, "start": s, "end": e})
        if not ok: raise RuntimeError("No valid clips were found for the given settings")
        groups = []
        if mix:
            by = {c: [x for x in ok if x.get("category") == c] for c in cats}
            for i in range(n):
                g = [by[c][i] for c in cats if i < len(by[c])]
                if g: groups.append(g)
        else:
            groups = [[x] for x in ok[:n]]
        J["total"] = len(groups)
        for i, g in enumerate(groups, 1):
            J["msg"] = f"Rendering clip {i}/{len(groups)}"
            out = d / f"clip{i}.mp4"
            if len(g) == 1:
                cut(src, g[0]["start"], g[0]["end"], out, p["vert"])
            else:
                parts = []
                for k, x in enumerate(g):
                    pf = d / f"p{i}_{k}.mp4"; cut(src, x["start"], x["end"], pf, p["vert"]); parts.append(pf)
                lst = d / f"l{i}.txt"; lst.write_text("".join(f"file '{x}'\n" for x in parts))
                sh("ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(lst), "-c", "copy", str(out))
            J["clips"].append({"file": f"/files/{j}/clip{i}.mp4", "title": " + ".join(str(x.get("title", "")) for x in g),
                               "category": ", ".join(sorted({str(x.get("category", "")) for x in g})),
                               "start": g[0]["start"], "end": g[-1]["end"]})
        J["status"] = "done"; J["msg"] = "Done"
    except Exception as ex:
        J["status"] = "error"; J["msg"] = str(ex)[:500]


@app.get("/health")
def health():
    return {"ok": True, "groq": bool(K), "gemini": bool(GEM_KEY), "openrouter": bool(OR_KEY), "lite": LITE, "max_upload_mb": MAX_UPLOAD_MB}


@app.post("/api/clips")
def clips(url: str = Form(""), file: UploadFile | None = File(None), kind: str = Form("short"), count: int = Form(3),
          target: int = Form(60), min_len: int = Form(0), max_len: int = Form(0), cats: str = Form("Auto"),
          mix: str = Form("single"), vertical: str = Form("false"), t0: float = Form(0), t1: float = Form(0)):
    purge()
    if not url and not file:
        raise HTTPException(400, "Provide a url or a file")
    j = uuid.uuid4().hex[:10]; d = D / j; d.mkdir()
    if file:
        try:
            save_upload(file, d / ("src" + Path(file.filename or "x.mp4").suffix))
        except HTTPException:
            shutil.rmtree(d, ignore_errors=True); raise
    cl = [c for c in cats.split(",") if c] or ["Auto"]
    p = dict(url=url, count=max(1, min(count, 4 if LITE else 12)), target=max(15, min(target, 2700)), cats=cl,
             mix=(mix == "mixed" and len(cl) > 1), vert=(vertical == "true" and kind == "short"), t0=max(0, t0), t1=t1,
             minL=max(0, min_len), maxL=max(0, max_len))
    JOBS[j] = {"status": "running", "msg": "Starting", "clips": [], "total": 0}
    threading.Thread(target=serial, args=(work, j, p), daemon=True).start()
    return {"job": j}


@app.get("/api/jobs/{j}")
def job(j: str):
    if j not in JOBS: raise HTTPException(404, "Job not found (the server may have restarted)")
    return JOBS[j]


# ===================== v3: language-independent voice & translation =====================
LANGS = [
    ("en-US", "English (US)", "a", ["en-US"]), ("en-GB", "English (UK)", "b", ["en-GB"]),
    ("en-IN", "English (India)", None, ["en-IN"]), ("hi-IN", "Hindi", None, ["hi-IN"]),
    ("ur-PK", "Urdu", None, ["ur-PK"]), ("ar-SA", "Arabic", None, ["ar-SA", "ar-AE", "ar-EG"]),
    ("es-ES", "Spanish", "e", ["es-ES", "es-MX", "es-US"]), ("fr-FR", "French", "f", ["fr-FR", "fr-CA"]),
    ("de-DE", "German", None, ["de-DE"]), ("it-IT", "Italian", "i", ["it-IT"]),
    ("pt-BR", "Portuguese", "p", ["pt-BR", "pt-PT"]), ("ru-RU", "Russian", None, ["ru-RU"]),
    ("zh-CN", "Chinese (Mandarin)", "z", ["zh-CN"]), ("ja-JP", "Japanese", "j", ["ja-JP"]),
    ("ko-KR", "Korean", None, ["ko-KR"]), ("tr-TR", "Turkish", None, ["tr-TR"]),
    ("id-ID", "Indonesian", None, ["id-ID"]), ("bn-BD", "Bengali", None, ["bn-BD", "bn-IN"]),
    ("nl-NL", "Dutch", None, ["nl-NL"]), ("sv-SE", "Swedish", None, ["sv-SE"]),
    ("nb-NO", "Norwegian", None, ["nb-NO"]), ("da-DK", "Danish", None, ["da-DK"]),
]
LANGMAP = {c: (n, k, l) for c, n, k, l in LANGS}


@app.get("/api/languages")
def languages():
    return [{"code": c, "name": n} for c, n, _, _ in LANGS]


@app.post("/api/translate")
def translate(b: dict):
    target = LANGMAP.get(b.get("target", ""), (b.get("target", "English"),))[0]
    return llm('Translate the given text into ' + target + '. Keep tone and meaning natural for spoken narration. '
               'Return JSON {"text": str}.', b.get("text", "")[:6000])


@app.get("/api/voices")
async def voices(region: str = "USA", gender: str = "Female", lang: str = ""):
    loc = LANGMAP[lang][2] if lang in LANGMAP else LOC.get(region, ["en-US"])
    return [{"name": v["ShortName"], "gender": v["Gender"], "locale": v["Locale"]}
            for v in await edge_tts.list_voices() if v["Locale"] in loc and v["Gender"] == gender]


@app.post("/api/tts")
async def tts(b: dict):
    rate, pitch = MOOD.get(b.get("category"), ("+0%", "+0Hz"))
    j = uuid.uuid4().hex[:10]; d = D / j; d.mkdir()
    await edge_tts.Communicate(b["text"][:5000], b["voice"], rate=rate, pitch=pitch).save(str(d / "v.mp3"))
    return {"file": f"/files/{j}/v.mp3"}


@app.post("/api/script")
def script(b: dict):
    return llm('Write a short-form voiceover script. Return JSON {"title":str,"script":str,"hashtags":[str]}. '
               "Punctuate for natural TTS pauses. Match this regional audience style: " + REGION.get(b.get("region"), REGION["USA"]),
               f"Category: {b.get('category', 'Commentary')}\nTopic or source text: {b.get('topic', '')[:6000]}")


REV_SYS = ("You are a social media strategist. From the metadata below (titles, views, tags, descriptions, and sometimes a transcript) "
           "reverse-engineer the channel or content. Be honest: you only see text/metadata, so mark guesses about visuals, voice and music as 'inferred'. "
           'Return JSON {"platform":str,"niche":str,"language":str,'
           '"target_audience":{"age_range":str,"gender_split":str,"regions":[str],"interests":[str]},'
           '"top_topics":[str],"high_view_videos":[{"title":str,"views":int,"why":str}],'
           '"title_formula":[str],"hashtags":[str],'
           '"videography":{"format":str,"editing_style":str,"visuals":str,"thumbnail_style":str},'
           '"voice":{"style":str,"language_or_accent":str},"music":{"style":str,"mood":str},'
           '"why_it_works":[str],"weaknesses":[str],"new_titles":[str],"new_script":str}. new_titles must contain 10 items. '
           "Regional style for the new content: ")


@app.post("/api/reverse")
def reverse(b: dict):
    url = (b.get("url") or "").strip()
    if not url: raise HTTPException(400, "Provide a channel or video link")
    try:
        info = json.loads(sh("yt-dlp", "-J", "--flat-playlist", "--playlist-end", "30", url))
    except Exception as ex:
        raise HTTPException(400, "Could not read this link (the platform may block free servers, or the link is private): " + str(ex)[:200])
    items = info.get("entries") or [info]
    items = [i for i in items if i]
    items.sort(key=lambda i: i.get("view_count") or 0, reverse=True)
    rows = []
    for i in items[:30]:
        rows.append(f'{i.get("title")} | views {i.get("view_count")} | duration {i.get("duration")}s | uploader {i.get("uploader") or i.get("channel")}'
                    + (f' | tags {",".join((i.get("tags") or [])[:12])}' if i.get("tags") else "")
                    + (f' | desc {(i.get("description") or "")[:300]}' if i.get("description") else ""))
    head = f'Source: {info.get("webpage_url") or url}\nChannel/uploader: {info.get("uploader") or info.get("channel") or ""}\n'
    return llm(REV_SYS + REGION.get(b.get("region"), REGION["USA"]), head + "\n".join(rows))


@app.post("/api/reverse_file")
def reverse_file(file: UploadFile = File(...), region: str = Form("USA")):
    purge(); j = uuid.uuid4().hex[:10]; d = D / j; d.mkdir()
    src = d / ("src" + Path(file.filename or "x.mp4").suffix)
    try:
        save_upload(file, src)
    except HTTPException:
        shutil.rmtree(d, ignore_errors=True); raise
    try:
        meta = json.loads(sh("ffprobe", "-v", "error", "-show_entries", "format=duration:stream=width,height,codec_type,r_frame_rate",
                             "-of", "json", str(src)))
    except Exception:
        meta = {}
    ctx = f"Uploaded file: {file.filename}\nTechnical info: {json.dumps(meta)[:600]}\n"
    if K:
        try:
            seg = transcribe(src, d)
            ctx += "Transcript:\n" + "\n".join(x["t"] for x in seg)[:30000]
        except Exception as ex:
            ctx += "(Transcript unavailable: " + str(ex)[:150] + ")"
    else:
        ctx += "(No GROQ_API_KEY set, so no transcript - analysis will be very limited)"
    return llm(REV_SYS + REGION.get(region, REGION["USA"]), ctx)


# ===================== v2: Kokoro TTS, age, SFX, tools, animation =====================
import math, asyncio, urllib.parse
LOCK = threading.Semaphore(1)  # free CPU par ek waqt me ek heavy job
MP = Path.home() / "models"; MP.mkdir(exist_ok=True)
KOK = None
AGE = {"Child": (1.22, 1.0), "Teen": (1.1, 1.0), "Adult": (1.0, 1.0), "Senior": (0.9, 0.95)}
LANG = {"a": "en-us", "b": "en-gb", "j": "ja", "z": "cmn", "e": "es", "f": "fr-fr", "i": "it", "p": "pt-br"}
KV = ["af_heart", "af_bella", "af_nicole", "am_adam", "am_michael", "bf_emma", "bf_isabella", "bm_george", "bm_lewis",
      "jf_alpha", "jm_kumo", "zf_xiaobei", "zm_yunxi", "ef_dora", "em_alex", "ff_siwis", "if_sara", "im_nicola", "pf_dora", "pm_alex"]
KR = {"USA": "a", "Canada": "a", "Singapore": "a", "UK": "b", "Japan": "j", "China": "z", "Europe": "efip"}


def serial(fn, *a):
    with LOCK:
        fn(*a)


def kokoro(text, voice, out):
    global KOK
    import soundfile as sf
    from kokoro_onnx import Kokoro
    if KOK is None:
        base = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/"
        for n in ("kokoro-v1.0.onnx", "voices-v1.0.bin"):
            f = MP / n
            if not f.exists():
                r = requests.get(base + n, timeout=900); r.raise_for_status(); f.write_bytes(r.content)
        KOK = Kokoro(str(MP / "kokoro-v1.0.onnx"), str(MP / "voices-v1.0.bin"))
    s, sr = KOK.create(text, voice=voice, speed=1.0, lang=LANG.get(voice[0], "en-us"))
    sf.write(str(out), s, sr)


@app.get("/api/voices2")
async def voices2(region: str = "USA", gender: str = "Female", lang: str = ""):
    g = gender[0].lower()
    kprefix = LANGMAP[lang][1] if lang in LANGMAP else KR.get(region, "")
    out = [] if LITE else [{"name": n, "engine": "kokoro"} for n in KV if n[1] == g and kprefix and n[0] in kprefix]
    return out + [{"name": v["name"], "engine": "edge"} for v in await voices(region, gender, lang)]


@app.post("/api/speak")
async def speak(b: dict):
    region, gender, lang = b.get("region", "USA"), b.get("gender", "Female"), b.get("lang", "")
    voice = b.get("voice") or ((await voices2(region, gender, lang)) or [{"name": "en-US-AriaNeural"}])[0]["name"]
    j = uuid.uuid4().hex[:10]; d = D / j; d.mkdir(); text = b["text"][:5000]; eng = "edge"; raw = d / "raw.mp3"
    if "_" in voice:
        try:
            await asyncio.to_thread(kokoro, text, voice, d / "raw.wav"); raw = d / "raw.wav"; eng = "kokoro"
        except Exception:
            voice = None
    if eng == "edge":
        if not voice or "_" in voice:
            ev = await voices(region, gender, lang); voice = ev[0]["name"] if ev else "en-US-AriaNeural"
        rate, pitch = MOOD.get(b.get("category"), ("+0%", "+0Hz"))
        await edge_tts.Communicate(text, voice, rate=rate, pitch=pitch).save(str(raw))
    p, s = AGE.get(b.get("age", "Adult"), (1, 1))
    af = ["-af", f"aresample=24000,asetrate={int(24000 * p)},aresample=24000,atempo={s / p:.3f}"] if (p, s) != (1, 1) else []
    sh("ffmpeg", "-y", "-i", str(raw), *af, str(d / "v.mp3"))
    return {"file": f"/files/{j}/v.mp3", "engine": eng, "voice": voice}


@app.get("/api/sfx")
def sfx(q: str, kind: str = "sfx"):
    k = os.environ.get("FREESOUND_KEY")
    if not k: raise HTTPException(400, "FREESOUND_KEY is not set (get a free key at freesound.org)")
    f = "duration:[0 TO 15]" if kind == "sfx" else 'duration:[30 TO 240] license:"Creative Commons 0"'
    r = requests.get("https://freesound.org/apiv2/search/text/", timeout=30, params={
        "query": q + ("" if kind == "sfx" else " music"), "filter": f, "page_size": 12, "token": k,
        "fields": "id,name,duration,license,previews"})
    r.raise_for_status()
    return [{"id": x["id"], "name": x["name"], "duration": x["duration"], "license": x["license"],
             "preview": x["previews"]["preview-hq-mp3"]} for x in r.json()["results"]]


def loc(path):
    p = (D / path.replace("/files/", "", 1)).resolve()
    if D.resolve() not in p.parents or not p.exists(): raise RuntimeError("File not found: " + path)
    return p


def dur(f):
    return float(sh("ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(f)).strip())


def img(prompt, out):
    u = "https://image.pollinations.ai/prompt/" + urllib.parse.quote(prompt) + "?nologo=true&width=720&height=1280"
    for _ in range(4):
        r = requests.get(u, timeout=180)
        if r.ok and r.headers.get("content-type", "").startswith("image"):
            out.write_bytes(r.content); return
        time.sleep(16)
    raise RuntimeError("The free image service did not return an image (anonymous rate limit) - try again shortly")


STYLE_TXT = {
    "Cinematic": "cinematic film still, dramatic lighting, shallow depth of field, anamorphic lens",
    "Photorealistic": "photorealistic, ultra detailed, natural lighting",
    "2D Animation": "2D hand-drawn animation style, flat colors, clean outlines",
    "3D Animation": "3D animated movie style, soft global illumination, stylized characters",
    "4D / Hyper-real": "hyper-realistic immersive 4D-style render, volumetric light, depth and motion feel",
    "Anime / Manga": "anime style, vibrant, detailed linework",
    "Watercolor Painting": "watercolor painting, soft washes",
    "Oil Painting": "oil painting, rich brush strokes",
    "Comic Book": "comic book art, bold ink and halftone",
    "Low-Poly 3D": "low-poly 3D render, geometric",
    "Claymation": "claymation stop-motion style, plasticine textures",
    "Fantasy Art": "epic fantasy concept art",
    "Cyberpunk": "cyberpunk neon city, rain, glow",
    "Minimalist Flat": "minimalist flat vector illustration",
}


def tool_work(j, name, p, src):
    J = JOBS[j]; d = D / j; out = d / "out.mp4"
    try:
        J["msg"] = "Processing"
        if name == "enhance":
            H = int(p.get("height", 1080))
            if H not in (720, 1080, 1440, 2160): H = 1080
            if LITE and H > 1080: H = 1080  # free halka server par 2K/4K nahi
            big = H >= 1440  # 2K/4K: lighter settings so the free CPU can finish
            J["msg"] = f"Upscaling to {H}p" + (" (2K/4K is slow on the free tier, please wait)" if big else "")
            sh("ffmpeg", "-y", "-i", str(src), "-vf",
               f"hqdn3d=2:1:3:3,scale=-2:{H}:flags=lanczos,unsharp=5:5:0.8:3:3:0.4",
               "-c:v", "libx264", "-preset", "ultrafast" if big else "veryfast", "-crf", "21" if big else "20",
               "-pix_fmt", "yuv420p", "-c:a", "copy", str(out))
        elif name == "expand" and p.get("mode") == "extend":
            sec = float(p.get("seconds", 60)); n = max(0, math.ceil(sec / dur(src)) - 1)
            sh("ffmpeg", "-y", "-stream_loop", str(n), "-i", str(src), "-t", str(sec), "-c:v", "libx264", "-preset", "veryfast", "-crf", "23", "-c:a", "aac", str(out))
        elif name == "expand":
            W, H = {"16:9": (1280, 720), "9:16": (720, 1280), "1:1": (1080, 1080), "4:5": (864, 1080)}[p.get("ratio", "16:9")]
            fc = (f"[0:v]scale={W}:{H}:force_original_aspect_ratio=increase,crop={W}:{H},boxblur=20[bg];"
                  f"[0:v]scale={W}:{H}:force_original_aspect_ratio=decrease[fg];[bg][fg]overlay=(W-w)/2:(H-h)/2")
            sh("ffmpeg", "-y", "-i", str(src), "-filter_complex", fc, "-c:v", "libx264", "-preset", "veryfast", "-crf", "23", "-c:a", "copy", str(out))
        elif name == "edit":
            s, e, sp, v = float(p.get("start", 0)), float(p.get("end", 0)), float(p.get("speed", 1)), p.get("voice")
            c = ["ffmpeg", "-y", "-ss", str(s), "-i", str(src)]
            if v: c += ["-i", str(loc(v))]
            if e > s: c += ["-t", str(e - s)]
            c += ["-vf", f"setpts=PTS/{sp}"]
            if v: c += ["-map", "0:v", "-map", "1:a", "-shortest"]
            elif p.get("mute"): c += ["-an"]
            else: c += ["-af", f"atempo={sp}"]
            sh(*c, "-c:v", "libx264", "-preset", "veryfast", "-crf", "23", "-c:a", "aac", str(out))
        elif name == "anim":
            n = max(3, min(int(p.get("scenes", 4)), 4 if LITE else 8))
            sc = llm('Split the topic into animated explainer scenes. Return JSON {"scenes":[{"narration":str,"image_prompt":str}]} with exactly '
                     + str(n) + " scenes. image_prompt: vivid prompt in English, no text inside the image, same characters and look in every scene, in this visual style: " + STYLE_TXT.get(p.get("style"), p.get("style") or "cinematic") + ". Narration style: "
                     + REGION.get(p.get("region"), REGION["USA"]), "Category: " + p.get("category", "Commentary") + "\nTopic: " + p.get("topic", ""))["scenes"][:n]
            parts = []
            for i, s in enumerate(sc):
                J["msg"] = f"Rendering scene {i + 1}/{len(sc)} (free image API can be slow)"
                im = d / f"i{i}.jpg"; img(s["image_prompt"] + ", " + STYLE_TXT.get(p.get("style"), p.get("style") or "cinematic"), im)
                a = loc(asyncio.run(speak({"text": s["narration"], "voice": p.get("voice") or None, "region": p.get("region", "USA"),
                                           "gender": p.get("gender", "Female"), "age": p.get("age", "Adult"), "category": p.get("category")}))["file"])
                t = dur(a) + 0.4; pv = d / f"s{i}.mp4"
                sh("ffmpeg", "-y", "-loop", "1", "-i", str(im), "-i", str(a), "-vf",
                   f"scale=1080:1920:force_original_aspect_ratio=increase,crop=1080:1920,zoompan=z='min(zoom+0.0012,1.25)':d={int(t * 25) + 1}:s=720x1280:fps=25",
                   "-t", f"{t:.2f}", "-c:v", "libx264", "-preset", "veryfast", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", str(pv))
                parts.append(pv)
                if i < len(sc) - 1: time.sleep(16)
            lst = d / "l.txt"; lst.write_text("".join(f"file '{x}'\n" for x in parts))
            sh("ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(lst), "-c", "copy", str(out))
        J["clips"].append({"file": f"/files/{j}/out.mp4", "title": name, "category": ""}); J["total"] = 1
        J["status"] = "done"; J["msg"] = "Done"
    except Exception as ex:
        J["status"] = "error"; J["msg"] = str(ex)[:500]


@app.post("/api/tool/{name}")
def tool(name: str, file: UploadFile | None = File(None), p: str = Form("{}")):
    if name not in ("enhance", "expand", "edit", "anim"): raise HTTPException(404, "unknown tool")
    purge(); pp = json.loads(p); j = uuid.uuid4().hex[:10]; d = D / j; d.mkdir(); src = None
    if file:
        src = d / ("src" + Path(file.filename or "x.mp4").suffix)
        with open(src, "wb") as f: shutil.copyfileobj(file.file, f)
    elif pp.get("src"):
        try: src = loc(pp["src"])
        except Exception as ex: raise HTTPException(400, str(ex))
    if name != "anim" and not src: raise HTTPException(400, "Provide a video file or pick an existing clip")
    JOBS[j] = {"status": "running", "msg": "Queued (one job runs at a time on the free tier)", "clips": [], "total": 0}
    threading.Thread(target=serial, args=(tool_work, j, name, pp, src), daemon=True).start()
    return {"job": j}
