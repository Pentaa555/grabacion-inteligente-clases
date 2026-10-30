"""Plataforma web de testeo (FastAPI) para el prototipo.

Subir/elegir un archivo -> procesar (motor + denoise + min_speakers) -> ver el
video con subtítulos y la tabla de segmentos, y descargar SRT/VTT/JSON.

Reutiliza el pipeline existente: capture / denoise / engines / subtitles.
Ejecutar:  .venv/bin/uvicorn src.webapp:app --host 127.0.0.1 --port 8000
"""
from __future__ import annotations

import threading
import uuid
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse

from . import capture
from . import denoise as denoise_mod
from . import subtitles
from . import board
from .person_segmentation import PersonSegmenter
from .engines import get_engine, available_engines

BASE_DIR = Path(__file__).resolve().parent.parent
UPLOAD_DIR = BASE_DIR / "webdata" / "uploads"
OUTPUT_DIR = BASE_DIR / "webdata" / "salida"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

AUDIO_EXT = {".flac", ".wav", ".mp3", ".m4a", ".ogg", ".opus"}

app = FastAPI(title="Prototipo — Plataforma de testeo")

# Estado de los jobs en memoria: job_id -> dict.
JOBS: dict[str, dict] = {}
_LOCK = threading.Lock()


class JobCancelled(Exception):
    """Se lanza cuando el usuario cancela un job en curso."""


def _set(job_id: str, **kw) -> None:
    with _LOCK:
        JOBS.setdefault(job_id, {}).update(kw)


def _is_cancelled(job_id: str) -> bool:
    with _LOCK:
        return bool(JOBS.get(job_id, {}).get("cancel"))


def _check_cancel(job_id: str) -> None:
    """Aborta el pipeline con JobCancelled si el usuario pidió cancelar."""
    if _is_cancelled(job_id):
        raise JobCancelled()


def _run_job(job_id: str, media_path: Path, engine_name: str, lang: str,
             model: str, denoise: bool, min_speakers: int | None,
             max_speakers: int | None, board_options: dict | None = None,
             simulate_seconds: float = 0.0) -> None:
    """Ejecuta el pipeline en un thread; actualiza JOBS[job_id]."""
    try:
        out_dir = OUTPUT_DIR / job_id
        out_dir.mkdir(parents=True, exist_ok=True)

        _check_cancel(job_id)

        # 1. Audio: extraer si es video.
        if media_path.suffix.lower() in AUDIO_EXT:
            audio = media_path
            video_for_player: Path | None = None
        else:
            _set(job_id, stage="extrayendo audio")
            audio = capture.extract_audio(media_path, out_dir / "audio.flac")
            video_for_player = media_path

        _check_cancel(job_id)

        # 2. Denoise opcional.
        if denoise:
            _set(job_id, stage="reduciendo ruido")
            audio = denoise_mod.denoise(audio, out_dir / "audio_denoised.flac")

        _check_cancel(job_id)

        # Duración total del audio: base para el % de transcripción.
        total_dur = capture.probe_duration(audio)
        _set(job_id, duration=total_dur, progress=0.0)

        # 3. Motor.
        opts: dict = {"language": lang, "model": model}
        if min_speakers is not None:
            opts["min_speakers"] = min_speakers
        if max_speakers is not None:
            opts["max_speakers"] = max_speakers
        if simulate_seconds:
            opts["simulate_seconds"] = simulate_seconds
        engine = get_engine(engine_name, **opts)
        ok, reason = engine.is_available()
        if not ok:
            _set(job_id, status="error", error=f"Motor '{engine_name}' no disponible: {reason}")
            return

        # Callback de progreso: el motor informa cuántos segundos de audio lleva
        # procesados; lo convertimos a fracción [0..1] contra la duración total.
        def _on_progress(processed_sec: float) -> None:
            _check_cancel(job_id)
            if total_dur > 0:
                frac = max(0.0, min(1.0, processed_sec / total_dur))
                _set(job_id, progress=round(frac, 4))

        _check_cancel(job_id)
        _set(job_id, stage=f"transcribiendo ({engine_name})")
        import time
        _t0 = time.perf_counter()
        result = engine.transcribe(audio, progress_cb=_on_progress)
        elapsed = time.perf_counter() - _t0
        _set(job_id, progress=1.0)

        # Registrar el tiempo de proceso y si se aplicó denoise, para comparar
        # modelos de forma reproducible (queda escrito en transcripcion.json).
        result.processing_seconds = elapsed
        result.denoise = bool(denoise)

        # 4. Subtítulos.
        paths = subtitles.write_all(result, out_dir)

        # 5. Tablero opcional: ROI + máscara de persona + alineación ECC.
        board_dict = None
        board_error = None
        board_status = "omitido"
        if board_options and board_options.get("enabled"):
            board_status = "pendiente"
            if video_for_player is None:
                board_status = "omitido"
                board_error = "La extracción del tablero requiere una entrada de video, no solo audio."
            else:
                try:
                    board_status = "procesando"
                    _set(job_id, stage="preparando tablero", progress=0.0)
                    segmenter = None
                    person_model = ""
                    if board_options.get("person_segmentation"):
                        requested = Path(board_options.get("person_model", "yolo11n-seg.pt"))
                        weights = requested if requested.is_absolute() else BASE_DIR / requested
                        segmenter = PersonSegmenter(
                            weights if weights.exists() else str(requested),
                            confidence=float(board_options.get("person_conf", 0.25)),
                        )
                        person_model = str(requested)

                    def _board_progress(frac: float) -> None:
                        _check_cancel(job_id)
                        _set(job_id, stage="analizando tablero", progress=round(frac, 4))

                    board_result = board.extract_board(
                        video_for_player,
                        out_dir / "tablero",
                        sample_fps=float(board_options.get("sample_fps", 1.0)),
                        median_window=int(board_options.get("median_window", 7)),
                        min_gap_seconds=float(board_options.get("min_gap", 20.0)),
                        roi=board_options.get("roi"),
                        person_segmenter=segmenter,
                        person_model=person_model,
                        stabilize=bool(board_options.get("stabilize")),
                        save_debug=True,
                        progress_cb=_board_progress,
                    )
                    board_dict = board_result.to_dict()
                    board_status = "completado"
                except JobCancelled:
                    raise
                except Exception as exc:  # el ASR terminado sigue siendo útil
                    board_status = "error"
                    if isinstance(exc, MemoryError):
                        board_error = (
                            "Memoria insuficiente al reconstruir el tablero. "
                            "Se debe procesar por ventanas o reducir resolución/ muestreo."
                        )
                    else:
                        board_error = f"{type(exc).__name__}: {exc}"
                    board_dir = out_dir / "tablero"
                    board_dir.mkdir(parents=True, exist_ok=True)
                    (board_dir / "error.txt").write_text(
                        f"{type(exc).__name__}: {exc}\n", encoding="utf-8",
                    )

        _set(
            job_id,
            status="done",
            stage="completado",
            result=result.to_dict(),
            outputs={k: str(v) for k, v in paths.items()},
            board=board_dict,
            board_status=board_status,
            board_error=board_error,
            video=str(video_for_player) if video_for_player else None,
            duration=capture.probe_duration(audio),
            processing_seconds=round(elapsed, 2),
        )
    except JobCancelled:
        _set(job_id, status="cancelled", stage="cancelado")
    except Exception as e:  # noqa: BLE001 — reportamos cualquier fallo al front
        import traceback
        traceback.print_exc()  # traceback completo al uvicorn.err.log
        _set(job_id, status="error", error=f"{type(e).__name__}: {e}")


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return INDEX_HTML.replace("__ENGINES__", ",".join(available_engines()))


@app.get("/api/samples")
def list_samples() -> JSONResponse:
    """Archivos de audio/video ya presentes en el proyecto, para elegir sin subir."""
    samples = []
    for p in sorted(BASE_DIR.glob("*")):
        if p.is_file() and p.suffix.lower() in AUDIO_EXT | {".mkv", ".mp4", ".webm"}:
            samples.append({"name": p.name, "size": p.stat().st_size})
    return JSONResponse(samples)


@app.post("/api/process")
async def process(
    file: UploadFile | None = File(default=None),
    sample: str | None = Form(default=None),
    engine: str = Form(default="local"),
    lang: str = Form(default="es-CO"),
    model: str = Form(default="base"),
    denoise: bool = Form(default=False),
    min_speakers: int | None = Form(default=None),
    max_speakers: int | None = Form(default=None),
    board: bool = Form(default=False),
    board_roi: str = Form(default=""),
    board_person: bool = Form(default=True),
    board_stabilize: bool = Form(default=True),
    board_sample_fps: float = Form(default=1.0),
    simulate_seconds: float = Form(default=0.0),
) -> JSONResponse:
    job_id = uuid.uuid4().hex[:12]

    # Origen del media: archivo subido o sample existente.
    if file is not None and file.filename:
        dest = UPLOAD_DIR / f"{job_id}_{Path(file.filename).name}"
        dest.write_bytes(await file.read())
        media_path = dest
    elif sample:
        media_path = BASE_DIR / Path(sample).name
        if not media_path.exists():
            raise HTTPException(404, f"Sample no encontrado: {sample}")
    else:
        raise HTTPException(400, "Debes subir un archivo o elegir un sample.")

    board_options = None
    if board:
        roi_parts = None
        if board_roi.strip():
            try:
                parsed = tuple(float(x.strip()) for x in board_roi.split(","))
                if len(parsed) != 4 or any(x < 0.0 or x > 1.0 for x in parsed):
                    raise ValueError
                if parsed[2] <= parsed[0] or parsed[3] <= parsed[1]:
                    raise ValueError
                roi_parts = parsed
            except ValueError:
                raise HTTPException(400, "ROI inválido; usa x0,y0,x1,y1 con valores entre 0 y 1.")
        board_options = {
            "enabled": True,
            "roi": roi_parts,
            "person_segmentation": bool(board_person),
            "person_model": "yolo11n-seg.pt",
            "person_conf": 0.25,
            "stabilize": bool(board_stabilize),
            "sample_fps": max(0.2, min(5.0, float(board_sample_fps))),
            "median_window": 7,
            "min_gap": 20.0,
        }

    _set(job_id, status="running", stage="en cola", engine=engine,
         source=media_path.name)
    t = threading.Thread(
        target=_run_job,
        args=(job_id, media_path, engine, lang, model, denoise,
              min_speakers, max_speakers, board_options, simulate_seconds),
        daemon=True,
    )
    t.start()
    return JSONResponse({"job_id": job_id})


@app.post("/api/cancel/{job_id}")
def cancel_job(job_id: str) -> JSONResponse:
    """Marca el job para que el pipeline lo aborte en el próximo chequeo.

    La cancelación es cooperativa: el thread de trabajo revisa la bandera
    entre etapas y en cada segmento transcrito, así que puede tardar unos
    segundos en detenerse (hasta que Whisper cierre el segmento en curso).
    """
    with _LOCK:
        job = JOBS.get(job_id)
        if not job:
            raise HTTPException(404, "job no encontrado")
        if job.get("status") == "running":
            job["cancel"] = True
            job["stage"] = "cancelando…"
    return JSONResponse({"ok": True, "job_id": job_id})


@app.get("/api/job/{job_id}")
def job_status(job_id: str) -> JSONResponse:
    with _LOCK:
        job = JOBS.get(job_id)
    if not job:
        raise HTTPException(404, "job no encontrado")
    return JSONResponse(job)


@app.get("/api/file/{job_id}/{kind}")
def get_output(job_id: str, kind: str) -> FileResponse:
    """kind: srt | vtt | json | video | audio"""
    with _LOCK:
        job = JOBS.get(job_id)
    if not job:
        raise HTTPException(404, "job no encontrado")
    out_dir = OUTPUT_DIR / job_id
    mapping = {
        "srt": (out_dir / "subtitulos.srt", "text/plain; charset=utf-8"),
        "vtt": (out_dir / "subtitulos.vtt", "text/vtt; charset=utf-8"),
        "json": (out_dir / "transcripcion.json", "application/json"),
    }
    if kind in mapping:
        path, media = mapping[kind]
        if not path.exists():
            raise HTTPException(404, f"{kind} no disponible")
        return FileResponse(path, media_type=media, filename=path.name)
    if kind == "video" and job.get("video"):
        return FileResponse(job["video"])
    raise HTTPException(404, f"salida '{kind}' no disponible")


@app.get("/api/board/{job_id}/{kind}/{index}")
def get_board_image(job_id: str, kind: str, index: int) -> FileResponse:
    """Sirve una imagen de tablero generada por un job terminado."""
    if kind not in {"clean", "binary", "mask", "raw"} or index < 0:
        raise HTTPException(400, "imagen de tablero inválida")
    with _LOCK:
        job = JOBS.get(job_id)
    if not job:
        raise HTTPException(404, "job no encontrado")
    path = OUTPUT_DIR / job_id / "tablero" / f"keyframe_{index:02d}_{kind}.png"
    if not path.exists():
        raise HTTPException(404, "imagen de tablero no disponible")
    return FileResponse(path, media_type="image/png", filename=path.name)


@app.get("/api/board/{job_id}/json")
def get_board_json(job_id: str) -> FileResponse:
    with _LOCK:
        job = JOBS.get(job_id)
    if not job:
        raise HTTPException(404, "job no encontrado")
    path = OUTPUT_DIR / job_id / "tablero" / "cuaderno.json"
    if not path.exists():
        raise HTTPException(404, "cuaderno no disponible")
    return FileResponse(path, media_type="application/json", filename=path.name)


INDEX_HTML = """<!doctype html>
<html lang="es"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Prototipo — Plataforma de testeo</title>
<style>
  :root{--bg:#0f1720;--panel:#182635;--acc:#4E9DA6;--gold:#F28F38;--txt:#EAE0CE;--mut:#8fa3b0}
  *{box-sizing:border-box} body{margin:0;font:15px/1.5 system-ui,sans-serif;background:var(--bg);color:var(--txt)}
  header{background:var(--panel);padding:16px 24px;border-bottom:2px solid var(--acc)}
  header h1{margin:0;font-size:20px} header p{margin:4px 0 0;color:var(--mut);font-size:13px}
  main{max-width:960px;margin:0 auto;padding:24px;display:grid;gap:20px}
  .card{background:var(--panel);border-radius:10px;padding:20px}
  label{display:block;font-size:13px;color:var(--mut);margin:10px 0 4px}
  select,input[type=number]{width:100%;padding:8px;border-radius:6px;border:1px solid #2a3b4d;background:#0f1720;color:var(--txt)}
  .row{display:grid;grid-template-columns:1fr 1fr;gap:14px}
  .row3{display:grid;grid-template-columns:1fr 1fr 1fr;gap:14px}
  button{background:var(--acc);color:#04141a;border:0;padding:11px 18px;border-radius:7px;font-weight:600;cursor:pointer;margin-top:14px}
  button:disabled{opacity:.5;cursor:not-allowed}
  .chk{display:flex;align-items:center;gap:8px;margin-top:16px} .chk input{width:auto}
  #status{font-size:14px;color:var(--gold)}
  video{width:100%;border-radius:8px;background:#000;margin-top:8px}
  table{width:100%;border-collapse:collapse;margin-top:12px;font-size:13px}
  th,td{text-align:left;padding:7px 9px;border-bottom:1px solid #2a3b4d;vertical-align:top}
  th{color:var(--mut);font-weight:600} .sp{color:var(--gold);font-weight:600;white-space:nowrap}
  tr.teacher{background:rgba(242,143,56,.12)} tr.teacher .sp{color:var(--gold)}
  .dl a{color:var(--acc);margin-right:14px;text-decoration:none} .hidden{display:none}
  small.hint{color:var(--mut)}
  .bar{height:14px;background:#0f1720;border:1px solid #2a3b4d;border-radius:8px;overflow:hidden;margin:6px 0}
  .bar > i{display:block;height:100%;width:0;background:linear-gradient(90deg,var(--acc),var(--gold));transition:width .25s ease}
  .barlabel{display:flex;justify-content:space-between;font-size:12px;color:var(--mut);margin-top:8px}
  .barwrap{margin-top:12px}
  .indet > i{width:35% !important;animation:slide 1.1s infinite ease-in-out}
  @keyframes slide{0%{margin-left:-35%}100%{margin-left:100%}}
  .boardbox{margin-top:18px;border-top:1px solid #2a3b4d;padding-top:16px}
  .boardgrid{display:grid;grid-template-columns:repeat(auto-fit,minmax(240px,1fr));gap:14px;margin-top:12px}
  .boardcard{background:#101a23;border:1px solid #2a3b4d;border-radius:8px;padding:10px}
  .boardcard img{width:100%;aspect-ratio:16/9;object-fit:contain;background:#000;border-radius:5px;display:block}
  .boardmeta{display:flex;justify-content:space-between;color:var(--mut);font-size:12px;margin:7px 0}
  .boardcard button{font-size:12px;padding:6px 9px;margin:0 5px 0 0}
  .boardhint{color:var(--mut);font-size:12px}
</style></head>
<body>
<header><h1>🎓 Plataforma de testeo — Grabación inteligente de clases</h1>
<p>Sube o elige un audio/video · transcribe + diariza (local / gcp / mock) · mira los subtítulos</p></header>
<main>
  <div class="card">
    <h3 style="margin-top:0">🎥 Grabar desde el navegador</h3>
    <video id="live" autoplay muted playsinline style="max-height:260px"></video>
    <div style="display:flex;gap:10px;flex-wrap:wrap">
      <button id="rec" type="button">● Grabar</button>
      <button id="stop" type="button" disabled style="background:var(--gold)">■ Detener</button>
      <span id="rectime" style="align-self:center;color:var(--mut)"></span>
    </div>
    <p id="recmsg" style="font-size:13px;min-height:18px;margin:8px 0 0"></p>
    <small class="hint">Usa la webcam + micrófono del navegador (WebM). Al detener, el clip
    queda listo abajo para procesar. Nota: el micrófono del sistema (<code>arecord</code>) por CLI
    da mejor señal de voz; esta grabación es para pruebas rápidas.</small>
  </div>

  <div class="card">
    <label>Archivo (subir o clip grabado)</label>
    <input type="file" id="file" accept="audio/*,video/*">
    <label>…o elegir uno existente del proyecto</label>
    <select id="sample"><option value="">— ninguno —</option></select>
    <div class="row3">
      <div><label>Motor</label><select id="engine"></select></div>
      <div><label>Modelo (local)</label>
        <select id="model"><option>base</option><option>small</option><option>medium</option><option>large-v3</option></select></div>
      <div><label>Idioma</label><input id="lang" value="es-CO"></div>
    </div>
    <div class="row">
      <div><label>min_speakers (opcional)</label><input type="number" id="minsp" min="1" placeholder="auto"></div>
      <div><label>max_speakers (opcional)</label><input type="number" id="maxsp" min="1" placeholder="auto"></div>
    </div>
    <div class="chk"><input type="checkbox" id="denoise"><label for="denoise" style="margin:0">Reducir ruido antes del ASR (P2)</label></div>
    <div class="boardbox">
      <div class="chk" style="margin-top:0"><input type="checkbox" id="boardchk"><label for="boardchk" style="margin:0">📓 Extraer cuaderno del tablero</label></div>
      <div class="row" id="boardopts" style="display:none">
        <div><label>Configuración de pizarra</label><small class="boardhint">La superficie se detecta automáticamente en los frames del video.</small></div>
        <div><label>Frames por segundo</label><input type="number" id="boardfps" value="1" min="0.2" max="5" step="0.2"></div>
      </div>
      <div class="chk" id="boardchecks" style="display:none;margin-top:8px">
        <input type="checkbox" id="boardperson" checked><label for="boardperson" style="margin:0">Quitar profesor con segmentación neural</label>
        <input type="checkbox" id="boardstab" checked><label for="boardstab" style="margin:0">Estabilizar/alinear frames</label>
      </div>
      <small class="boardhint">Genera páginas limpias, binario y máscara para revisar cómo se reconstruyó la pizarra.</small>
    </div>
    <button id="go">Procesar</button>
    <button id="cancel" type="button" class="hidden" style="background:var(--gold);margin-left:10px">✕ Cancelar</button>
    <p id="status"></p>

    <div id="progwrap" class="barwrap hidden">
      <div class="barlabel"><span>Subida del archivo</span><span id="uplbl">0%</span></div>
      <div class="bar"><i id="upbar"></i></div>
      <div class="barlabel"><span id="prstage">Procesando…</span><span id="prlbl"></span></div>
      <div class="bar" id="prbarbox"><i id="prbar"></i></div>
      <div class="barlabel"><span id="prtime">⏱ 0s</span><span id="prdur"></span></div>
    </div>

    <small class="hint">El modelo local grande puede tardar y consumir RAM. En este equipo usa <b>base</b>/<b>small</b>.</small>
    <div class="chk" style="margin-top:10px"><input type="checkbox" id="simchk"><label for="simchk" style="margin:0">Simular carga (motor mock) — para ver la barra sin GPU</label></div>
  </div>

  <div class="card hidden" id="results">
    <h3 style="margin-top:0">Resultado</h3>
    <div id="player"></div>
    <div id="downloads" class="dl"></div>
    <div id="boardresult" class="boardbox hidden"></div>
    <table><thead><tr><th>#</th><th>Inicio</th><th>Fin</th><th>Hablante</th><th>Texto</th></tr></thead>
    <tbody id="segs"></tbody></table>
  </div>
</main>
<script>
const engines="__ENGINES__".split(",");
const $=id=>document.getElementById(id);
engines.forEach(e=>{const o=document.createElement("option");o.textContent=e;o.value=e;if(e==="local")o.selected=true;$("engine").appendChild(o)});
fetch("/api/samples").then(r=>r.json()).then(list=>list.forEach(s=>{
  const o=document.createElement("option");o.value=s.name;o.textContent=s.name+" ("+(s.size/1024|0)+" KB)";$("sample").appendChild(o)}));
$("boardchk").onchange=()=>{
  const on=$("boardchk").checked;
  $("boardopts").style.display=on?"grid":"none";
  $("boardchecks").style.display=on?"flex":"none";
};

function fmt(t){t=Math.max(0,t);const h=t/3600|0,m=(t%3600)/60|0,s=(t%60).toFixed(1);return (h?h+":":"")+String(m).padStart(2,"0")+":"+String(s).padStart(4,"0")}

// --- Grabación desde el navegador (getUserMedia + MediaRecorder) ---
let mediaStream=null, recorder=null, chunks=[], recStart=0, recTimer=null;

function recMsg(txt,err){
  const el=$("recmsg");
  if(el){el.textContent=txt;el.style.color=err?"#ff8a80":"var(--gold)";}
  console[err?"error":"log"]("[grabador] "+txt);
}

// Diagnóstico al cargar: ¿hay API? ¿estamos en un iframe?
(function diagnose(){
  const inIframe = window.self!==window.top;
  const hasApi = !!(navigator.mediaDevices && navigator.mediaDevices.getUserMedia) && !!window.MediaRecorder;
  const secure = window.isSecureContext;
  if(!hasApi){
    $("rec").disabled=true;
    recMsg("Este navegador no expone getUserMedia/MediaRecorder"+(secure?"":" (contexto NO seguro)")+". Abre la app en http://127.0.0.1:8000 en una pestaña normal.",true);
    return;
  }
  if(inIframe){
    recMsg("⚠ La app está embebida en un panel. Si al Grabar no aparece el permiso de cámara, ábrela en una PESTAÑA propia: http://127.0.0.1:8000",false);
  }
})();

async function ensureStream(){
  if(mediaStream) return mediaStream;
  if(!(navigator.mediaDevices && navigator.mediaDevices.getUserMedia)){
    throw new Error("getUserMedia no disponible (navegador o contexto no seguro).");
  }
  mediaStream=await navigator.mediaDevices.getUserMedia({video:true,audio:true});
  $("live").srcObject=mediaStream;
  return mediaStream;
}
$("rec").onclick=async()=>{
  recMsg("Solicitando permiso de cámara/micrófono…",false);
  let s;
  try{ s=await ensureStream(); }
  catch(e){
    recMsg("No se pudo acceder a cámara/micrófono: "+(e.name||"")+" — "+e.message
      +". Revisa el permiso del navegador (icono de cámara en la barra) o abre la app en pestaña propia.",true);
    return;
  }
  try{
    chunks=[];
    const mime=MediaRecorder.isTypeSupported("video/webm;codecs=vp8,opus")?"video/webm;codecs=vp8,opus":"video/webm";
    recorder=new MediaRecorder(s,{mimeType:mime});
    recorder.ondataavailable=e=>{if(e.data.size)chunks.push(e.data)};
    recorder.onerror=e=>recMsg("Error de MediaRecorder: "+e.error,true);
    recorder.onstop=()=>{
      const blob=new Blob(chunks,{type:"video/webm"});
      if(!blob.size){recMsg("El clip salió vacío (0 bytes). Reintenta la grabación.",true);return;}
      const file=new File([blob],"grabacion_web_"+Date.now()+".webm",{type:"video/webm"});
      const dt=new DataTransfer();dt.items.add(file);$("file").files=dt.files;
      $("sample").value="";
      recMsg("✔ Clip grabado ("+(blob.size/1024|0)+" KB) listo. Pulsa Procesar abajo.",false);
    };
    recorder.start();
    recMsg("● Grabando…",false);
    recStart=Date.now();$("rec").disabled=true;$("stop").disabled=false;
    recTimer=setInterval(()=>{$("rectime").textContent="● "+((Date.now()-recStart)/1000).toFixed(0)+"s"},250);
  }catch(e){
    recMsg("No se pudo iniciar la grabación: "+e.message,true);
  }
};
$("stop").onclick=()=>{
  if(recorder&&recorder.state!=="inactive")recorder.stop();
  clearInterval(recTimer);$("rectime").textContent="";
  $("rec").disabled=false;$("stop").disabled=true;
};

let jobStart=0, jobTimer=null, currentJob=null;

$("cancel").onclick=async()=>{
  if(!currentJob)return;
  $("cancel").disabled=true;
  $("status").textContent="Cancelando…";
  try{ await fetch("/api/cancel/"+currentJob,{method:"POST"}); }
  catch(e){ $("status").textContent="Error al cancelar: "+e.message; $("cancel").disabled=false; }
};

$("go").onclick=async()=>{
  const fd=new FormData();
  const f=$("file").files[0];
  if(f)fd.append("file",f); else if($("sample").value)fd.append("sample",$("sample").value);
  else{$("status").textContent="Sube un archivo o elige un sample.";return}
  fd.append("engine",$("engine").value);fd.append("lang",$("lang").value);
  fd.append("model",$("model").value);fd.append("denoise",$("denoise").checked);
  if($("minsp").value)fd.append("min_speakers",$("minsp").value);
  if($("maxsp").value)fd.append("max_speakers",$("maxsp").value);
  if($("boardchk").checked){
    fd.append("board","true");
    fd.append("board_person",$("boardperson").checked);
    fd.append("board_stabilize",$("boardstab").checked);
    fd.append("board_sample_fps",$("boardfps").value);
  }
  if($("simchk").checked)fd.append("simulate_seconds","20");
  $("go").disabled=true;$("status").textContent="Enviando…";$("results").classList.add("hidden");
  $("cancel").classList.remove("hidden");$("cancel").disabled=false;currentJob=null;

  // Barras: mostrar y resetear.
  $("progwrap").classList.remove("hidden");
  setUp(0); setProc(null,"Subiendo…");
  $("prtime").textContent="⏱ 0s"; $("prdur").textContent="";

  // XHR para tener progreso REAL de subida (fetch no lo expone).
  const xhr=new XMLHttpRequest();
  xhr.open("POST","/api/process");
  xhr.upload.onprogress=e=>{ if(e.lengthComputable) setUp(e.loaded/e.total); };
  xhr.upload.onload=()=>{ setUp(1); setProc(null,"En cola…"); };
  xhr.onload=()=>{
    if(xhr.status<200||xhr.status>=300){
      $("status").textContent="Error: "+xhr.responseText;$("go").disabled=false;return;
    }
    const {job_id}=JSON.parse(xhr.responseText);
    currentJob=job_id;
    jobStart=Date.now();
    clearInterval(jobTimer);
    jobTimer=setInterval(()=>{$("prtime").textContent="⏱ "+((Date.now()-jobStart)/1000|0)+"s"},500);
    poll(job_id);
  };
  xhr.onerror=()=>{$("status").textContent="Error de red al subir.";$("go").disabled=false;};
  xhr.send(fd);
};

function setUp(frac){ // barra de subida
  const pct=Math.round((frac||0)*100);
  $("upbar").style.width=pct+"%"; $("uplbl").textContent=pct+"%";
}
function setProc(frac,stage){ // barra de transcripción
  if(stage!==undefined && stage!==null) $("prstage").textContent=stage;
  const box=$("prbarbox");
  if(frac===null||frac===undefined){ // sin % -> barra indeterminada animada
    box.classList.add("indet"); $("prbar").style.width=""; $("prlbl").textContent="";
  }else{
    box.classList.remove("indet");
    const pct=Math.round(frac*100);
    $("prbar").style.width=pct+"%"; $("prlbl").textContent=pct+"%";
  }
}

async function poll(id){
  const r=await fetch("/api/job/"+id);const j=await r.json();
  $("status").textContent="Estado: "+(j.stage||j.status);
  // Barra de transcripción: si el motor reporta progress numérico, lo mostramos;
  // si solo hay una etapa (extraer/denoise/gcp), barra indeterminada.
  if(typeof j.progress==="number" && (j.stage||"").startsWith("transcribiendo")){
    setProc(j.progress,"Transcribiendo");
  }else{
    setProc(null, j.stage||"Procesando…");
  }
  if(j.duration) $("prdur").textContent="audio: "+fmt(j.duration);
  if(j.status==="running"){setTimeout(()=>poll(id),800);return}
  clearInterval(jobTimer);
  $("go").disabled=false;
  $("cancel").classList.add("hidden");
  currentJob=null;
  if(j.status==="cancelled"){$("status").textContent="⛔ Cancelado por el usuario";setProc(0,"Cancelado");return}
  if(j.status==="error"){$("status").textContent="❌ "+j.error;setProc(0,"Error");return}
  setProc(1,"Completado");
  render(id,j);
}

function renderBoard(id,j){
  const box=$("boardresult"); box.innerHTML=""; box.classList.add("hidden");
  if(j.board_error){
    box.innerHTML=`<h3 style="margin:0 0 6px">📓 Cuaderno del tablero</h3><p class="boardhint">No se pudo generar: ${j.board_error}</p>`;
    box.classList.remove("hidden"); return;
  }
  const b=j.board;
  if(!b || !b.keyframes || !b.keyframes.length)return;
  const method=b.method||"board";
  const roiLabel=b.roi_source==="automatic"?" · ROI automático":(b.roi_source==="manual"?" · ROI manual":" · encuadre completo");
  const stats=`${b.num_keyframes} páginas · ${b.frames_sampled} frames · ${method}`+
    roiLabel+
    (b.stabilized?` · ${b.frames_aligned} alineados`:"")+
    (b.person_segmentation?` · personas: ${b.person_frames_detected}`:"");
  box.innerHTML=`<h3 style="margin:0">📓 Cuaderno del tablero</h3><p class="boardhint">${stats}</p><div class="boardgrid"></div><p class="boardhint"><a href="/api/board/${id}/json" download>Descargar cuaderno.json</a></p>`;
  const grid=box.querySelector(".boardgrid");
  b.keyframes.forEach((k,i)=>{
    const card=document.createElement("div");card.className="boardcard";
    card.innerHTML=`<div class="boardmeta"><b>Página ${i+1}</b><span>${fmt(k.time)}</span></div><img src="/api/board/${id}/clean/${i}" alt="Tablero reconstruido ${i+1}"><div style="margin-top:8px"><button type="button" data-view="clean">Limpio</button><button type="button" data-view="binary">Binario</button><button type="button" data-view="mask">Máscara</button></div>`;
    const image=card.querySelector("img");
    card.querySelectorAll("button").forEach(btn=>btn.onclick=()=>{
      image.src=`/api/board/${id}/${btn.dataset.view}/${i}`;
    });
    grid.appendChild(card);
  });
  box.classList.remove("hidden");
}

function render(id,j){
  let head="✅ "+(j.result.segments.length)+" segmentos · hablantes: "+j.result.speakers.join(", ");
  if(j.result.teacher){ head+=" · 🎓 docente: "+j.result.teacher; }
  const ps=j.result.processing_seconds;
  if(ps){ head+=" · ⏱ proceso: "+fmt(ps); if(j.duration){ head+=" ("+(j.duration/ps).toFixed(1)+"× tiempo real)"; } }
  if(j.result.model){ head+=" · modelo: "+j.result.model; }
  head+=" · denoise: "+(j.result.denoise?"sí":"no");
  if(j.board_status==="completado") head+=" · 📓 tablero: listo";
  else if(j.board_status==="error") head+=" · ⚠ tablero: error";
  else if(j.board_status==="omitido") head+=" · tablero: omitido";
  $("status").textContent=head;
  $("results").classList.remove("hidden");
  const p=$("player");p.innerHTML="";
  if(j.video){
    const v=document.createElement("video");v.controls=true;
    v.src="/api/file/"+id+"/video";
    const tr=document.createElement("track");tr.kind="subtitles";tr.label="es";tr.srclang="es";
    tr.src="/api/file/"+id+"/vtt";tr.default=true;v.appendChild(tr);
    p.appendChild(v);
  }else{p.innerHTML="<small class='hint'>Entrada de solo audio — sin video que reproducir. Subtítulos y tabla abajo.</small>";}
  $("downloads").innerHTML="Descargar: "+["srt","vtt","json"].map(k=>`<a href="/api/file/${id}/${k}" download>${k.toUpperCase()}</a>`).join("");
  renderBoard(id,j);
  const tb=$("segs");tb.innerHTML="";
  j.result.segments.forEach((s,i)=>{
    const tr=document.createElement("tr");
    const isT=s.is_teacher;
    const badge=isT?" 🎓":"";
    if(isT)tr.classList.add("teacher");
    tr.innerHTML=`<td>${i+1}</td><td>${fmt(s.start)}</td><td>${fmt(s.end)}</td><td class="sp">${s.speaker}${badge}</td><td>${s.text}</td>`;
    tb.appendChild(tr);
  });
}
</script>
</body></html>
"""
