
from __future__ import annotations

import asyncio
import base64
import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager

import cv2 as cv
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

import main as assistant  # noqa: E402

log = logging.getLogger("uvicorn.error")

FRAME_INTERVAL  = 0.08   
STATUS_INTERVAL = 1.0    
JPEG_QUALITY    = 80     


_latest_cue: dict[str, str] = {"text": "", "ts": ""}
_cue_lock = threading.Lock()

_orig_audio_put = assistant.audio_output.put


def _intercepting_put(item, *args, **kwargs):
    """Copy the cue into _latest_cue then pass through to pyttsx3 thread."""
    with _cue_lock:
        _latest_cue["text"] = str(item)
        _latest_cue["ts"]   = time.strftime("%H:%M:%S")
    return _orig_audio_put(item, *args, **kwargs)

assistant.audio_output.put = _intercepting_put

_executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="ws-enc")


def _draw_and_encode_frame() -> str | None:
    with assistant.ui_lock:
        raw_frame = assistant.current_display_frame
        people    = list(assistant.current_display_people)

    if raw_frame is None:
        return None

    canvas = raw_frame.copy()
    h, w   = canvas.shape[:2]

    for person in people:
        box  = person.get("box")
        name = person.get("name", "")
        if box is None:
            continue

        x1 = max(0, min(int(box[0]), w - 1))
        y1 = max(0, min(int(box[1]), h - 1))
        x2 = max(0, min(int(box[2]), w - 1))
        y2 = max(0, min(int(box[3]), h - 1))
        label_y = max(35, y1)

        cv.rectangle(canvas, (x1, y1), (x2, y2), (0, 230, 80), 2)
        cv.rectangle(canvas, (x1, label_y - 34), (x2, label_y), (0, 230, 80), cv.FILLED)
        cv.putText(
            canvas, name, (x1 + 5, label_y - 9),
            cv.FONT_HERSHEY_SIMPLEX, 0.75, (15, 15, 15), 2,
        )

    ok, buf = cv.imencode(".jpg", canvas, [cv.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    if not ok:
        return None
    return base64.b64encode(buf.tobytes()).decode("ascii")



def _run_brain_loop(brain) -> None:
    """
    Blocking LangGraph pulse loop. Runs in a daemon thread so that
    Groq API / Neo4j round-trips never stall the asyncio event loop.
    """
    last_known_names: list[str] = []

    while not assistant.stop_event.is_set():
        try:
            result = brain.invoke({"names": [], "relations": {}, "last_convos": {}})
            if result:
                last_known_names = [
                    n for n in result.get("names", [])
                    if n
                    and not n.startswith("Unknown_")
                    and n not in ("failed", "")
                ]
        except Exception as exc:
            log.error(f"Brain loop error: {exc}")

        time.sleep(assistant.BRAIN_PULSE_INTERVAL)

    with assistant.transcription_lock:
        t_len = len(assistant.transcription)
    if last_known_names and t_len > 50:
        log.info(f"Persisting final memories for: {last_known_names}")
        assistant.creategraphinfo(
            {"names": last_known_names, "relations": {}, "last_convos": {}}
        )
    assistant.neo4j_driver.close()
    log.info("Brain loop exited cleanly.")

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Start all worker threads, yield control to FastAPI, then shut down."""

    log.info("=== Alzheimer's Assistant v4.0 — starting threads ===")

    sensory = [
        threading.Thread(target=assistant.input_visual,   name="InputVisual",  daemon=True),
        threading.Thread(target=assistant.input_audio,    name="InputAudio",   daemon=True),
        threading.Thread(target=assistant.process_visual, name="ProcVisual",   daemon=True),
        threading.Thread(target=assistant.process_audio,  name="ProcAudio",    daemon=True),
        threading.Thread(target=assistant.output_audio,   name="OutputAudio",  daemon=True),
    ]
    for t in sensory:
        t.start()

    brain = assistant.workflow.compile()
    brain_thread = threading.Thread(
        target=_run_brain_loop, args=(brain,), name="BrainLoop", daemon=True
    )
    brain_thread.start()

    log.info("All threads running. Dashboard available at http://localhost:8000")
    yield

    log.info("Shutdown requested — setting stop event.")
    assistant.stop_event.set()
    _executor.shutdown(wait=False)
    log.info("Goodbye.")



app = FastAPI(
    title="Alzheimer's Cognitive Assistant",
    version="4.0.0",
    lifespan=lifespan,
)

# Serve static/  →  /static/*
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/", include_in_schema=False)
async def root():
    return FileResponse("static/index.html")


@app.get("/api/health")
async def health():
    """Lightweight health check for uptime monitors."""
    with assistant.ui_lock:
        people = list(assistant.current_display_people)
    with assistant.transcription_lock:
        t_len = len(assistant.transcription)
    return {
        "ok":             not assistant.stop_event.is_set(),
        "people_count":   len(people),
        "people":         [p.get("name", "") for p in people],
        "transcript_len": t_len,
    }


@app.post("/api/reset-diarizer")
async def reset_diarizer():
    """Clear all speaker clusters — call at the start of a new session."""
    with assistant.diarizer_lock:
        assistant.diarizer.reset()
    return {"ok": True, "message": "Speaker clusters cleared."}


@app.post("/api/clear-transcript")
async def clear_transcript():
    """Manually flush the transcription buffer."""
    global_ns = assistant.__dict__
    with assistant.transcription_lock:
        assistant.transcription = ""         
    return {"ok": True}




@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """
    Single multiplexed real-time channel.

    Outbound message envelope (JSON):
      { "type": "frame",      "data": "<base64-JPEG>" | null }
      { "type": "transcript", "data": "<full diarised string>" }
      { "type": "cue",        "text": "...", "ts": "HH:MM:SS" }
      { "type": "status",     "people": [...], "count": N, "running": bool }

    The transcript and cue messages are only sent when their value changes
    (delta-only push) to minimise bandwidth.

    The frame is sent every FRAME_INTERVAL seconds.
    The status heartbeat is sent every STATUS_INTERVAL seconds.
    """
    await websocket.accept()
    log.info(f"WebSocket client connected: {websocket.client}")

    loop             = asyncio.get_event_loop()
    last_transcript  = ""
    last_cue_text    = ""
    last_status_ts   = 0.0

    try:
        while True:
            tick = time.monotonic()

            frame_b64 = await loop.run_in_executor(_executor, _draw_and_encode_frame)
            await websocket.send_json({"type": "frame", "data": frame_b64})

            with assistant.transcription_lock:
                current_tx = assistant.transcription

            if current_tx != last_transcript:
                await websocket.send_json({"type": "transcript", "data": current_tx})
                last_transcript = current_tx
            with _cue_lock:
                cue = dict(_latest_cue)

            if cue["text"] and cue["text"] != last_cue_text:
                await websocket.send_json({"type": "cue", "text": cue["text"], "ts": cue["ts"]})
                last_cue_text = cue["text"]

            if tick - last_status_ts >= STATUS_INTERVAL:
                with assistant.ui_lock:
                    ppl = list(assistant.current_display_people)
                await websocket.send_json({
                    "type":    "status",
                    "people":  [p.get("name", "") for p in ppl],
                    "count":   len(ppl),
                    "running": not assistant.stop_event.is_set(),
                })
                last_status_ts = tick

    
            elapsed = time.monotonic() - tick
            await asyncio.sleep(max(0.0, FRAME_INTERVAL - elapsed))

    except WebSocketDisconnect:
        log.info(f"WebSocket client disconnected: {websocket.client}")
    except Exception as exc:
        log.error(f"WebSocket error ({websocket.client}): {exc}")
        try:
            await websocket.close(code=1011, reason=str(exc))
        except Exception:
            pass
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:api", host="0.0.0.0", port=8000, reload=True)
