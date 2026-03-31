# MemoryCare — Real-Time Cognitive Assistant for Alzheimer's Patients

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat-square&logo=python&logoColor=white)
![LangGraph](https://img.shields.io/badge/LangGraph-Agentic%20Workflow-blueviolet?style=flat-square)
![Groq](https://img.shields.io/badge/Groq-Llama%203.3%2070B-F55036?style=flat-square)
![Neo4j](https://img.shields.io/badge/Neo4j-Graph%20Memory-008CC1?style=flat-square&logo=neo4j&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-WebSocket-009688?style=flat-square&logo=fastapi&logoColor=white)

A wearable-class, always-on AI assistant that helps Alzheimer's patients navigate social situations in real time. It identifies faces, diarises speakers, retrieves relationship memories from a graph database, and delivers short contextual cues through text-to-speech — all running on consumer CPU hardware.

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Features](#features)
- [Tech Stack](#tech-stack)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Configuration](#configuration)
- [Running the Application](#running-the-application)
- [Project Structure](#project-structure)
- [How It Works](#how-it-works)
- [Web Dashboard](#web-dashboard)
- [Known Limitations and Roadmap](#known-limitations-and-roadmap)
- [Contributing](#contributing)

---

## Overview

MemoryCare watches through a webcam, listens via a microphone, and within seconds of detecting a face tells the patient who they are speaking with, what their relationship is, and what they last talked about. This context is delivered as a quiet 1-2 sentence audio cue so as not to overwhelm the patient.

Interaction memory is stored in a Neo4j knowledge graph, meaning the assistant learns and improves with every conversation. The entire inference pipeline — face recognition, speech transcription, speaker diarisation, LLM reasoning — runs locally on CPU, with only LLM calls routed to Groq for low-latency inference.

---

## Architecture

```
SENSOR LAYER — 5 daemon threads
--------------------------------------------------------------
 Thread 1  input_visual  ──►  visual_input  queue (maxsize=1)
 Thread 2  input_audio   ──►  audio_input   queue (maxsize=3)
 Thread 3  process_visual   InsightFace detection
               visual_input ──► visual_output queue
 Thread 4  process_audio    Whisper + resemblyzer diarisation
               audio_input  ──► transcription string (locked)
 Thread 5  output_audio     pyttsx3 TTS drain
               audio_output queue ──► speaker
--------------------------------------------------------------

BRAIN LOOP — daemon thread (BRAIN_PULSE_INTERVAL cadence)
--------------------------------------------------------------
  LangGraph StateGraph:

  START
    |
  recognize            InsightFace cosine match vs face DB
    |
    +-- (Unknown faces) --> identification   Groq structured output
    |                            |           extracts names from transcript
    +-- (Known faces) ----------+
    |
  getraginfo           Neo4j relationship + last-convo lookup
    |
  live_help            Groq: should patient get a cue?
    |                  if yes -> generate cue -> audio_output queue
    +-- (transcript > 300 chars) --> creategraphinfo
    |                                    Groq: summarise + infer relation
    |                                    Neo4j: MERGE write
    |
   END
--------------------------------------------------------------

WEB LAYER — FastAPI + WebSocket (app.py)
--------------------------------------------------------------
  GET  /                  index.html (self-contained dashboard)
  GET  /api/health
  POST /api/reset-diarizer
  POST /api/clear-transcript
  WS   /ws                multiplexed stream:
                            { type: "frame",      data: "<base64-JPEG>" }
                            { type: "transcript", data: "<diarised string>" }
                            { type: "cue",        text: "...", ts: "HH:MM:SS" }
                            { type: "status",     people: [...], count: N }
--------------------------------------------------------------
```

---

## Features

- **Multi-person face recognition** — identifies all faces in frame simultaneously using InsightFace (buffalo_s) cosine similarity; unknown faces are queued for name extraction via transcript analysis
- **Speaker diarisation** — online GE2E clustering via resemblyzer labels every Whisper segment as `[Speaker A]:`, `[Speaker B]:`, etc., keeping labels consistent across the session without requiring cloud auth or heavy model weights
- **Graph-backed long-term memory** — relationships, names, and conversation summaries are stored in Neo4j and retrieved on each new encounter; the assistant improves with use
- **Contextual audio cues** — Groq/Llama 3.3-70B generates a 1-2 sentence patient-facing cue when the transcript indicates conversational assistance is needed; delivered via pyttsx3 TTS
- **Automatic API key rotation** — cycles through a pool of Groq keys on `RateLimitError` with no restart or downtime
- **Thread-safe design** — six named locks with a strict acquisition order prevent races and deadlocks across all sensor threads; all mutable state is fully guarded
- **Web dashboard** — single-file `index.html` with three themes, live annotated video feed, diarised transcript panel, and a timestamped AI cue log

---

## Tech Stack

| Layer | Component | Notes |
|---|---|---|
| Face recognition | InsightFace `buffalo_s` | 320x320 det size, CPU; ~4x faster than 640x640 with comparable accuracy |
| Speech transcription | faster-whisper `tiny.en` | int8 quantised, beam_size=1, word timestamps enabled for accurate segment slicing |
| Speaker diarisation | resemblyzer (GE2E) | ~5 MB model, no auth required, 30-80 ms per utterance on CPU |
| AI reasoning | Groq — Llama 3.3-70B-Versatile | Structured Pydantic output for name/relation extraction; free-text for cue generation |
| Agentic workflow | LangGraph StateGraph | Conditional edges, typed dict state, multi-node memory flow |
| Graph memory | Neo4j | MERGE-based upsert via persistent pooled driver |
| TTS output | pyttsx3 | sapi5 on Windows; auto-detect on Linux/macOS |
| Backend API | FastAPI + uvicorn | Lifespan-managed thread startup; WebSocket multiplexed stream |
| Frontend | Vanilla HTML/CSS/JS | Self-contained single file, zero build step, three switchable themes |

---

## Prerequisites

- Python **3.10** or newer
- A working **webcam** and **microphone**
- A running **Neo4j** instance (local, Docker, or [AuraDB free tier](https://neo4j.com/cloud/platform/aura-graph-database/))
- One or more **Groq API keys** ([console.groq.com](https://console.groq.com) — free tier available; multiple keys recommended for sustained sessions)
- Windows, Linux, or macOS (TTS engine falls back gracefully on non-Windows)

---

## Installation

```bash
# 1. Clone the repository
git clone https://github.com/<your-username>/memorycare.git
cd memorycare

# 2. Create and activate a virtual environment
python -m venv .venv

# Windows
.venv\Scripts\activate

# Linux / macOS
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt
```

> **Linux:** PyAudio requires PortAudio headers. Install before running pip:
> ```bash
> sudo apt install portaudio19-dev
> ```

> **First run only:** resemblyzer downloads the GE2E checkpoint (~5 MB) into
> `~/.cache/torch` automatically. Subsequent startups are instant.

---

## Configuration

Create a `.env` file in the project root:

```env
# Comma-separated Groq API keys — rotated automatically on rate-limit
groq_key=gsk_key1,gsk_key2,gsk_key3

# Neo4j connection
URI=bolt://localhost:7687
Database=neo4j
pwd=your_neo4j_password
```

| Variable | Description |
|---|---|
| `groq_key` | One or more Groq API keys, comma-separated. Rotation on `RateLimitError` is automatic and requires no restart. |
| `URI` | Neo4j Bolt URI. Use `bolt://` for local instances or `neo4j+s://` for AuraDB. |
| `Database` | Neo4j database name. The default is `neo4j`. |
| `pwd` | Neo4j password. |

To change the patient's name, set `PATIENT_NAME` near the top of `main.py`:

```python
PATIENT_NAME = "Harry"   # replace with the patient's first name
```

Other tunable constants in `main.py`:

| Constant | Default | Effect |
|---|---|---|
| `BRAIN_PULSE_INTERVAL` | `2.0` s | Cadence of the LangGraph brain loop |
| `FACE_MATCH_THRESHOLD` | `0.50` | InsightFace cosine similarity cutoff for face matching |
| `TRANSCRIPTION_TRIM_THRESHOLD` | `300` chars | Transcript length that triggers a memory write to Neo4j |
| `SPEAKER_SIMILARITY_THRESHOLD` | `0.80` | GE2E cosine cutoff for assigning a segment to an existing speaker cluster |
| `MIN_SEGMENT_DURATION_SEC` | `0.50` s | Minimum segment length for GE2E embedding; shorter segments inherit the previous label |
| `MAX_SPEAKER_CLUSTERS` | `8` | Hard cap on distinct speaker labels per session to prevent noise-driven explosion |

---

## Running the Application

### With web dashboard (recommended)

```bash
uvicorn app:app --host 0.0.0.0 --port 8000
# or equivalently
python app.py
```

Open `http://localhost:8000` in a browser. The dashboard connects automatically via WebSocket and begins receiving the live video feed, transcript, and AI cues.

### Standalone mode (OpenCV window, no browser)

```bash
python main.py
```

An OpenCV window opens showing the annotated camera feed. Press **`q`** to quit cleanly. Any in-flight conversation memory is saved to Neo4j before the process exits.

---

## Project Structure

```
memorycare/
├── main.py          # v3.0 — production backend (multi-person + speaker diarisation)
├── app.py           # v4.0 — FastAPI WebSocket gateway (imports and wraps main.py)
├── index.html       # Web dashboard (self-contained, no build step required)
├── requirements.txt # Python dependencies
├── .env             # Runtime secrets (never commit this file)
├── .gitignore
├── temp_faces.pkl   # Auto-generated face embedding cache (gitignored)
└── func.py          # v1.0 — original prototype (reference only, not imported)
```

`func.py` is the original proof-of-concept and is kept for historical reference. It is not imported by any other module. All active development is in `main.py` and `app.py`.

---

## How It Works

### Sensor layer

Five daemon threads run continuously and communicate through bounded queues:

**`input_visual`** captures webcam frames and drops stale frames to keep the queue at maxsize=1, so the brain always processes the most recent face.

**`input_audio`** calibrates for ambient noise once on startup, then listens in a loop via `SpeechRecognition`. Raw WAV bytes are pushed into a 3-slot queue.

**`process_visual`** runs InsightFace detection, throttled to approximately 3 fps to remain CPU-friendly.

**`process_audio`** decodes WAV bytes with soundfile into a float32 array, transcribes with faster-whisper (word timestamps enabled), and for each segment extracts a 256-dim GE2E speaker embedding via resemblyzer. Segments are appended to the shared `transcription` string as `[Speaker A]: text` under `transcription_lock`.

**`output_audio`** drains the TTS queue and speaks cues through pyttsx3. The engine is initialised inside this thread to satisfy COM thread-safety requirements on Windows.

### LangGraph brain

A `StateGraph` is invoked on a configurable pulse interval in its own daemon thread (critical — see below). The state carries `names`, `relations`, and `last_convos` for all people currently visible.

| Node | Responsibility |
|---|---|
| `recognize` | Compares InsightFace embeddings against the local pickle DB. Matched face returns a name; unmatched returns an `Unknown_N` token and stores the pending embedding for identification. |
| `identification` | Sends the live transcript to Groq with a structured Pydantic schema (`IdentifiedPeople`) to extract names. Maps `Unknown_N` tokens to real names and persists new entries to `temp_faces.pkl`. |
| `getraginfo` | Queries Neo4j for the relationship string and last-conversation summary for each known person in the current frame. |
| `live_help` | Calls Groq once to determine whether the patient needs assistance (yes/no). If yes, a second call generates a 1-2 sentence contextual cue that is pushed to the TTS queue. The transcript is trimmed on both branches. |
| `creategraphinfo` | When the transcript exceeds `TRANSCRIPTION_TRIM_THRESHOLD`, calls Groq to extract a relationship label and summary per person. Writes or updates the Neo4j graph and clears the transcript only after a confirmed successful write. |

### Speaker diarisation

`SpeakerDiarizer` maintains an online cosine-clustering registry of `{label -> (running-mean GE2E embedding, observation count)}`. Each new Whisper segment either reinforces an existing cluster or creates a new speaker label (`A`, `B`, ... `Z`, `Spk27`, ...). The hard cluster cap prevents label explosion. Short segments below `MIN_SEGMENT_DURATION_SEC` inherit the most recent speaker label rather than triggering a new embedding call.

The transcript trim helper (`_trim_transcription_by_words`) is speaker-turn-aware: it splits on `[Speaker X]:` boundaries rather than whitespace, so trimming never leaves a dangling half-tag that would confuse the LLM.

### Memory graph

Relationships in Neo4j are stored as:

```
(Harry:Person)-[:KNOWS]->(Alice:Person {relation: "daughter", last_convo: "..."})
```

The Cypher `MERGE ... SET` pattern creates the relationship on first encounter and updates the `relation` and `last_convo` properties on all subsequent encounters.

### Lock ordering

Six locks guard all shared mutable state. The strict global acquisition order prevents deadlock cycles:

```
transcription_lock  ->  llm_lock  ->  diarizer_lock
  ->  pending_lock  ->  face_db_lock  ->  ui_lock
```

No function acquires more than two locks simultaneously.

---

## Web Dashboard

`index.html` is a fully self-contained single-file frontend. It opens a WebSocket connection to `/ws` and renders:

- **Live video panel** — annotated camera feed with per-person bounding boxes and name labels streamed at approximately 12.5 fps
- **Transcript panel** — diarised speaker turns with per-speaker colour coding and a live typing cursor on the most recent segment; auto-scrolls to the latest entry
- **AI cue log** — timestamped cues delivered to the patient, newest at the top with a flash animation on each new arrival; trimmed to the most recent 50 entries
- **Status bar** — detected face count, active speaker count, session uptime, FPS counter, and connection state

Three themes are available and persist via `localStorage`:

| Theme | Description |
|---|---|
| Clinical | Light background, blue accents — comfortable for caregiver viewing in daylight |
| Glassmorphism | Dark background with backdrop-blur effects — high contrast for low-light environments |
| Minimal | Pure black with monospace typography — minimal visual noise |

---

## Known Limitations and Roadmap

- [ ] **Voice-to-face linkage** — speaker diarisation labels and face identities are correlated only via LLM contextual reasoning. A direct linkage could be achieved by cross-referencing diariser timestamps with lip/jaw motion detection from the visual thread.

- [ ] **Retroactive cluster merging** — online clustering cannot merge clusters that were incorrectly split early in the session. An offline UMAP + agglomerative pass at end-of-session would address this without adding real-time latency.

- [ ] **Overlapping speech** — resemblyzer produces unreliable embeddings during crosstalk. A VAD-based frame selector that skips overlapping regions would improve accuracy.

- [ ] **Caregiver management UI** — extend the dashboard to allow a caregiver to rename unknown faces, correct relationship labels, and add notes to the Neo4j graph directly from the browser.

- [ ] **Dockerised deployment** — a `docker-compose.yml` bundling the Python backend and a Neo4j instance for single-command startup.

- [ ] **Edge / Raspberry Pi port** — evaluate ONNX quantisation of InsightFace and `tiny.en` for deployment on a wearable edge device.

- [ ] **Face DB management endpoint** — REST API to list, rename, and delete entries from `temp_faces.pkl` without manual file editing.

---

## Contributing

Contributions, bug reports, and feature suggestions are welcome. Please open an issue before submitting a pull request for any significant change.

```bash
# Fork, then:
git checkout -b feat/your-feature-name
git commit -m "feat: describe your change"
git push origin feat/your-feature-name
# open a pull request against main
```

When modifying `main.py`, ensure any new shared state is guarded by the appropriate lock and that the global lock acquisition order documented above is respected.