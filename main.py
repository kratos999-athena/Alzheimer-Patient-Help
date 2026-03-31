
import io
import os
import pickle
import queue
import re
import threading
import time
import logging

import cv2 as cv
import numpy as np
if not hasattr(np, 'float'):
    np.float = float
if not hasattr(np, 'int'):
    np.int = int

import soundfile as sf                                # P3: WAV -> float32 array
from resemblyzer import VoiceEncoder, preprocess_wav  # P3: GE2E embeddings

from insightface.app import FaceAnalysis
import speech_recognition as sr
from faster_whisper import WhisperModel

from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Optional

from langchain_groq import ChatGroq
from groq import RateLimitError
from neo4j import GraphDatabase
from dotenv import load_dotenv

import pyttsx3
from pydantic import BaseModel, Field


logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s][%(threadName)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

load_dotenv(override=True)

raw_keys = os.getenv("groq_key", "")
groq_key = [k.strip() for k in raw_keys.split(",") if k.strip()]
if not groq_key:
    raise EnvironmentError("No groq_key found in .env -- please set it.")

URI  = os.getenv("URI")
db   = os.getenv("Database")
pwd  = os.getenv("pwd")
AUTH = (db, pwd)

PATIENT_NAME = "Harry"


BRAIN_PULSE_INTERVAL = 2.0


FACE_MATCH_THRESHOLD = 0.50


TRANSCRIPTION_TRIM_THRESHOLD = 300


SPEAKER_SIMILARITY_THRESHOLD = 0.80

MIN_SEGMENT_DURATION_SEC = 0.50

MAX_SPEAKER_CLUSTERS = 8


stop_event = threading.Event()


visual_input  = queue.Queue(maxsize=1)
audio_input   = queue.Queue(maxsize=3)
visual_output = queue.Queue(maxsize=1)
audio_output  = queue.Queue(maxsize=3)



transcription_lock = threading.Lock()  # guards `transcription`
llm_lock           = threading.Lock()  # guards `current_key` + `llm`
face_db_lock       = threading.Lock()  # guards `global_known_faces` + pickle
ui_lock            = threading.Lock()  # guards all `current_display_*` vars
pending_lock       = threading.Lock()  # guards `_pending_embeddings`

diarizer_lock      = threading.Lock()

transcription: str = ""
current_key:   int = 0

app = FaceAnalysis(name="buffalo_s", providers=["CPUExecutionProvider"])
app.prepare(ctx_id=0, det_size=(320, 320))


model = WhisperModel("tiny.en", device="cpu", compute_type="int8")

log.info("Loading GE2E voice encoder (resemblyzer)...")
_voice_encoder = VoiceEncoder(device="cpu")
log.info("GE2E voice encoder ready.")


llm = ChatGroq(model="llama-3.3-70b-versatile", api_key=groq_key[current_key])


def rotate_llm() -> None:
    """Thread-safe Groq API-key rotation on RateLimitError."""
    global current_key, llm
    with llm_lock:
        current_key = (current_key + 1) % len(groq_key)
        llm = ChatGroq(model="llama-3.3-70b-versatile", api_key=groq_key[current_key])
        log.warning(f"Rate-limit hit -- rotated to key index {current_key}.")



neo4j_driver = GraphDatabase.driver(URI, auth=AUTH)



try:
    global_known_faces: list = pickle.load(open("temp_faces.pkl", "rb"))
    log.info(f"Loaded {len(global_known_faces)} known face(s) from disk.")
except (EOFError, FileNotFoundError):
    global_known_faces = []



current_display_frame:  Optional[np.ndarray] = None
current_display_people: list[dict]            = []   # [{name, box}, ...]

_pending_embeddings: dict[str, np.ndarray] = {}



class SpeakerDiarizer:
    """
    Online single-pass speaker diariser using GE2E cosine clustering.

    Thread-safety: all methods must be called under `diarizer_lock`.
    The object is only ever written by process_audio (single thread),
    so the lock is defensive / forward-safe rather than strictly required
    in the current architecture.
    """

    def __init__(
        self,
        similarity_threshold: float = SPEAKER_SIMILARITY_THRESHOLD,
        max_clusters: int            = MAX_SPEAKER_CLUSTERS,
    ) -> None:
        # label -> (running-mean unit embedding, observation count)
        self._clusters: dict[str, tuple[np.ndarray, int]] = {}
        self.threshold    = similarity_threshold
        self.max_clusters = max_clusters
        self._next_idx    = 0
        # Label of the most recently assigned speaker -- used as fallback
        # when a segment is too short for a reliable embedding.
        self.last_label: str = "A"

    def _make_label(self, idx: int) -> str:
        """Return 'A'...'Z' for idx 0-25, then 'Spk27', 'Spk28'..."""
        return chr(ord("A") + idx) if idx < 26 else f"Spk{idx + 1}"

    def assign(self, embedding: np.ndarray) -> str:
        """
        Match `embedding` to the closest cluster (above threshold) or
        create a new cluster.  Returns the speaker label string.

        `embedding` must already be L2-normalised (resemblyzer guarantees
        this for embed_utterance output).
        """
        norm = float(np.linalg.norm(embedding))
        if norm < 1e-6:
            return self.last_label   # degenerate embedding -- inherit

        embedding = embedding / norm   # re-normalise defensively

        best_label: Optional[str] = None
        best_sim:   float         = -1.0

        for label, (ref_emb, _) in self._clusters.items():
            sim = float(np.dot(embedding, ref_emb))
            if sim > best_sim:
                best_sim   = sim
                best_label = label

        if best_label is not None and best_sim >= self.threshold:
            # Existing speaker -- update running mean.
            ref_emb, count = self._clusters[best_label]
            new_count = count + 1
            new_emb   = (ref_emb * count + embedding) / new_count
            norm_new  = float(np.linalg.norm(new_emb))
            new_emb   = new_emb / norm_new if norm_new > 1e-6 else ref_emb
            self._clusters[best_label] = (new_emb, new_count)
            self.last_label = best_label
            return best_label

        if len(self._clusters) >= self.max_clusters:
            # Cluster cap reached -- assign to closest existing cluster to
            # prevent runaway label explosion from background noise.
            label = best_label or self._make_label(0)
            log.debug(
                f"Cluster cap ({self.max_clusters}) reached; "
                f"assigning new voice to existing label '{label}'."
            )
            self.last_label = label
            return label

        # New speaker.
        new_label = self._make_label(self._next_idx)
        self._next_idx += 1
        self._clusters[new_label] = (embedding, 1)
        log.info(f"New speaker cluster created: '{new_label}'.")
        self.last_label = new_label
        return new_label

    def reset(self) -> None:
        """Clear all clusters (call between sessions if desired)."""
        self._clusters.clear()
        self._next_idx  = 0
        self.last_label = "A"


# Module-level diariser -- persists across the entire session so speaker
# labels stay consistent as the conversation progresses.
diarizer = SpeakerDiarizer()


# ==========================================
# PHASE 3 -- SEGMENT EMBEDDING HELPER
# ==========================================
def _get_segment_embedding(
    audio_f32: np.ndarray,
    source_sr: int,
    start_sec: float,
    end_sec:   float,
) -> Optional[np.ndarray]:
    """
    Extract a GE2E speaker embedding for the sub-segment [start_sec, end_sec].

    Returns a 256-dim L2-normalised numpy array, or None if:
      - The segment is shorter than MIN_SEGMENT_DURATION_SEC.
      - resemblyzer's VAD trims away all voiced frames.
      - Any other processing error occurs.

    Called exclusively from process_audio (single thread) -- no lock needed.
    """
    duration = end_sec - start_sec
    if duration < MIN_SEGMENT_DURATION_SEC:
        return None

    i_start = max(0, int(start_sec * source_sr))
    i_end   = min(len(audio_f32), int(end_sec * source_sr))
    chunk   = audio_f32[i_start:i_end]

    min_samples = int(MIN_SEGMENT_DURATION_SEC * source_sr)
    if len(chunk) < min_samples:
        return None

    try:
        # preprocess_wav resamples to 16 kHz, normalises amplitude, and
        # applies energy-based VAD to strip silence.
        wav = preprocess_wav(chunk, source_sr=source_sr)
        # resemblyzer needs at least ~0.16 s (2560 samples @ 16 kHz).
        if len(wav) < 2560:
            return None
        embedding = _voice_encoder.embed_utterance(wav)  # (256,), L2-normed
        return embedding
    except Exception as exc:
        log.debug(f"_get_segment_embedding: resemblyzer error -- {exc}")
        return None


_SPEAKER_TAG_RE = re.compile(r"(?=\[Speaker [A-Z][A-Z0-9]*\]:)")


def _trim_transcription_by_words() -> None:
    """
    Discard the older half of the transcription.

    - Diarised transcript (contains [Speaker X]: tags):
      split on turn boundaries, keep the newer half of turns.
    - Plain transcript (no diarisation):
      split on whitespace, keep the newer half of words.

    Must be called while holding transcription_lock.
    """
    global transcription

    if "[Speaker" in transcription:
        turns = _SPEAKER_TAG_RE.split(transcription)
        turns = [t.strip() for t in turns if t.strip()]
        if len(turns) > 1:
            keep = turns[len(turns) // 2:]
            transcription = " ".join(keep).strip() + " "
        # Single turn -> leave it intact (do not trim below one turn).
    else:
        words = transcription.split()
        transcription = (
            (" ".join(words[len(words) // 2:]) + " ") if words else ""
        )



class RelationsGraph(BaseModel):
    last_convo: str = Field(description="1-2 sentence summary of the transcript.")
    relations:  str = Field(
        description=(
            "Relation of the identified person to the patient; 'Unknown' if unclear."
        )
    )


class IdentifiedPeople(BaseModel):
    """Structured LLM output for multi-name extraction."""
    names: list[str] = Field(
        description=(
            "First names of all people present in the conversation, "
            f"excluding the Alzheimer's patient ('{PATIENT_NAME}'). "
            "Return an empty list if no names can be determined."
        )
    )


class Alzheimer(TypedDict):
    """
    Extended state -- multi-person (Phase 2) + diarisation-aware (Phase 3).

    names       : All face tokens.  Known -> real name; unknown -> "Unknown_N".
    relations   : name -> relationship string.
    last_convos : name -> last-conversation summary.
    """
    names:       list[str]
    relations:   dict[str, str]
    last_convos: dict[str, str]


def input_visual() -> None:
    cap = cv.VideoCapture(0)
    if not cap.isOpened():
        log.error("Cannot open camera -- input_visual thread exiting.")
        return
    while not stop_event.is_set():
        ret, frame = cap.read()
        if not ret:
            log.warning("Camera read failed; retrying in 100 ms...")
            time.sleep(0.1)
            continue
        if visual_input.full():
            try:
                visual_input.get_nowait()
            except queue.Empty:
                pass
        visual_input.put(frame)
    cap.release()
    log.info("input_visual stopped.")


def input_audio() -> None:
    r   = sr.Recognizer()
    mic = sr.Microphone()

    with mic as source:
        log.info("Calibrating microphone for ambient noise...")
        r.adjust_for_ambient_noise(source, duration=1.0)
        log.info("Microphone ready.")

    while not stop_event.is_set():
        try:
            with mic as source:
                audio = r.listen(source, phrase_time_limit=10, timeout=5)
            raw_wav = audio.get_wav_data()
            if audio_input.full():
                try:
                    audio_input.get_nowait()
                except queue.Empty:
                    pass
            audio_input.put(raw_wav)
        except sr.WaitTimeoutError:
            continue
        except Exception as exc:
            log.error(f"input_audio error: {exc}")
            time.sleep(0.5)

    log.info("input_audio stopped.")


def process_visual() -> None:
    while not stop_event.is_set():
        try:
            frame = visual_input.get(timeout=0.5)
        except queue.Empty:
            continue

        faces = app.get(frame)

        if visual_output.full():
            try:
                visual_output.get_nowait()
            except queue.Empty:
                pass
        visual_output.put((frame, faces))
        time.sleep(0.3)

    log.info("process_visual stopped.")

    
def process_audio() -> None:
    global transcription

    while not stop_event.is_set():
        try:
            audio_item = audio_input.get(timeout=0.5)
        except queue.Empty:
            continue

        try:
            # Decode WAV bytes -> float32 mono array + native sample rate.
            audio_f32, source_sr = sf.read(io.BytesIO(audio_item), dtype="float32")
            if audio_f32.ndim > 1:
                # Multi-channel (e.g. stereo) -> mono by averaging channels.
                audio_f32 = audio_f32.mean(axis=1)

            segments_iter, _ = model.transcribe(
                io.BytesIO(audio_item),
                beam_size=1,
                word_timestamps=True,
            )
            segments = list(segments_iter)

            if not segments:
                continue

            labelled_parts: list[str] = []

            for seg in segments:
                text = seg.text.strip()
                if not text:
                    continue

                # Speaker embedding + cluster assignment.
                emb = _get_segment_embedding(
                    audio_f32, source_sr, seg.start, seg.end
                )

                with diarizer_lock:
                    if emb is not None:
                        label = diarizer.assign(emb)
                    else:
                        # Segment too short or no voiced frames -- inherit
                        # the last known speaker label.
                        label = diarizer.last_label

                labelled_parts.append(f"[Speaker {label}]: {text}")

            if labelled_parts:
                new_block = " ".join(labelled_parts)
                with transcription_lock:
                    transcription += new_block + " "
                log.info(f"Diarised tail: ...{transcription[-140:]}")

        except Exception as exc:
            log.error(f"process_audio error: {exc}")

    log.info("process_audio stopped.")


def output_audio() -> None:
    try:
        engine = pyttsx3.init("sapi5")   # Windows
    except Exception:
        engine = pyttsx3.init()          # Linux / macOS auto-detect

    engine.setProperty("rate", 160)
    engine.setProperty("volume", 1.0)
    voices = engine.getProperty("voices")
    if voices and len(voices) > 1:
        engine.setProperty("voice", voices[1].id)

    while not stop_event.is_set():
        try:
            text = audio_output.get(timeout=0.5)
        except queue.Empty:
            continue
        log.info(f"Speaking: {text[:80]}...")
        engine.say(text)
        engine.runAndWait()

    log.info("output_audio stopped.")


def recognize(state: Alzheimer) -> dict:
    global current_display_frame

    try:
        frame, faces = visual_output.get(timeout=0.5)
    except queue.Empty:
        with ui_lock:
            current_display_people.clear()
        return {"names": [], "relations": {}, "last_convos": {}}

    with ui_lock:
        current_display_frame = frame.copy()

    if not faces:
        with ui_lock:
            current_display_people.clear()
        return {"names": [], "relations": {}, "last_convos": {}}

    h, w             = frame.shape[:2]
    identified:      list[dict]             = []
    new_pending:     dict[str, np.ndarray]  = {}
    unknown_counter  = 0

    with face_db_lock:
        for face in faces:
            embedding = face.normed_embedding

            x1, y1, x2, y2 = face.bbox.astype(int).tolist()
            x1 = max(0, min(x1, w - 1))
            y1 = max(0, min(y1, h - 1))
            x2 = max(0, min(x2, w - 1))
            y2 = max(0, min(y2, h - 1))
            box = [x1, y1, x2, y2]

            matched_name: Optional[str] = None
            for entry in global_known_faces:
                if np.dot(embedding, entry["embeddings"]) > FACE_MATCH_THRESHOLD:
                    matched_name = entry["name"]
                    break

            if matched_name:
                identified.append({"name": matched_name, "box": box})
            else:
                unknown_counter += 1
                token = f"Unknown_{unknown_counter}"
                identified.append({"name": token, "box": box})
                new_pending[token] = embedding

    with ui_lock:
        current_display_people[:] = identified

    with pending_lock:
        _pending_embeddings.clear()
        _pending_embeddings.update(new_pending)

    names = [p["name"] for p in identified]
    log.info(f"recognize: frame contains {names}")
    return {"names": names, "relations": {}, "last_convos": {}}


def identification(state: Alzheimer) -> dict:
    names          = state.get("names", [])
    unknown_tokens = [n for n in names if n.startswith("Unknown_")]

    if not unknown_tokens:
        return {}

    with transcription_lock:
        snapshot = transcription

    if not snapshot.strip():
        log.warning("identification skipped -- transcription is empty.")
        return {}

    prompt = (
        f"From the conversation transcript below, extract the first names of all "
        f"people who are present and speaking "
        f"(excluding '{PATIENT_NAME}', who is the Alzheimer's patient).\n"
        f"Return an empty list if no names can be determined.\n\n"
        f"Conversation:\n{snapshot}"
    )

    try:
        with llm_lock:
            structured_llm = llm.with_structured_output(IdentifiedPeople)
            result         = structured_llm.invoke(prompt)
        extracted = [
            n.strip() for n in result.names
            if n.strip() and n.strip().lower() != PATIENT_NAME.lower()
        ]
    except RateLimitError:
        rotate_llm()
        return {}
    except Exception as exc:
        log.error(f"identification LLM error: {exc}")
        return {}

    if not extracted:
        log.info("identification: no names extracted from transcript.")
        return {}

    with pending_lock:
        pending_copy = dict(_pending_embeddings)

    updated_names = list(names)
    newly_saved:  list[str] = []

    with face_db_lock:
        for i, token in enumerate(unknown_tokens):
            if i >= len(extracted):
                break

            new_name      = extracted[i]
            already_known = any(
                e["name"].lower() == new_name.lower()
                for e in global_known_faces
            )
            if not already_known and token in pending_copy:
                global_known_faces.append(
                    {"name": new_name, "embeddings": pending_copy[token]}
                )
                newly_saved.append(new_name)
            else:
                log.info(
                    f"identification: '{new_name}' already in face DB -- skipping."
                )

            try:
                idx = updated_names.index(token)
                updated_names[idx] = new_name
            except ValueError:
                pass

        if newly_saved:
            try:
                with open("temp_faces.pkl", "wb") as fh:
                    pickle.dump(global_known_faces, fh)
                log.info(f"Face DB saved. New entries: {newly_saved}")
            except Exception as exc:
                log.error(f"Failed to persist face DB: {exc}")

    log.info(f"identification: updated names -> {updated_names}")
    return {"names": updated_names}


def getraginfo(state: Alzheimer) -> dict:
    names = state.get("names", [])
    known_names = [
        n for n in names
        if n and not n.startswith("Unknown_") and n not in ("failed",)
    ]

    if not known_names:
        return {"relations": {}, "last_convos": {}}

    relations:   dict[str, str] = {}
    last_convos: dict[str, str] = {}

    for name in known_names:
        try:
            records, _, _ = neo4j_driver.execute_query(
                "MATCH (n:Person {name:$name1})-[:KNOWS]->(m:Person {name:$name2}) RETURN m",
                name1=PATIENT_NAME, name2=name, database_="neo4j",
            )
            for record in records:
                m = record.data().get("m", {})
                relations[name]   = m.get("relation",   "")
                last_convos[name] = m.get("last_convo", "")
        except Exception as exc:
            log.error(f"getraginfo Neo4j error for '{name}': {exc}")

    log.info(f"getraginfo: loaded context for {list(relations.keys())}")
    return {"relations": relations, "last_convos": last_convos}


def live_help(state: Alzheimer) -> dict:
    global transcription

    with transcription_lock:
        snapshot = transcription

    if not snapshot.strip():
        return {}

    # P3: status-check prompt is diarisation-aware.
    status_prompt = (
        "You are an expert medical assistant for Alzheimer's patients.\n"
        "The transcript below may use [Speaker X]: tags to indicate who said what.\n"
        "Does the transcript suggest the patient needs conversational assistance?\n"
        "Reply ONLY with 'yes' or 'no'.\n\n"
        f"Transcript:\n{snapshot}"
    )

    try:
        with llm_lock:
            decision = llm.invoke(status_prompt).content.strip().lower()

        if "no" in decision:
            with transcription_lock:
                _trim_transcription_by_words()
            return {}

        # Help IS needed -- build multi-person + diarisation context block.
        names       = state.get("names", [])
        relations   = state.get("relations", {})
        last_convos = state.get("last_convos", {})
        n_people    = len(names)

        people_lines: list[str] = []
        for name in names:
            if name.startswith("Unknown_"):
                people_lines.append(f"  - {name}: an unrecognised person")
            else:
                rel = relations.get(name, "unknown relationship")
                lc  = last_convos.get(name, "no previous memory on file")
                people_lines.append(f"  - {name}: {rel}. Last spoke about: {lc}")

        people_block = (
            "\n".join(people_lines) if people_lines
            else "  - No identified people"
        )
        person_word = "person" if n_people == 1 else "people"
        is_are      = "is"     if n_people == 1 else "are"

        # P3: diarisation mapping instruction block.
        diarisation_note = (
            "SPEAKER DIARISATION:\n"
            "  The transcript uses [Speaker X]: tags assigned automatically by a\n"
            "  voice-embedding model. Speaker labels (A, B, C...) are consistent\n"
            "  within this session but are NOT yet directly linked to the named\n"
            "  people listed above.\n"
            "  Use context clues -- who is most likely speaking given what was\n"
            "  said -- and cross-reference with the names and relationships above\n"
            "  to make the best attribution you can.\n"
            f"  If uncertain, assume the patient ({PATIENT_NAME}) is one of the\n"
            "  speakers and treat the others as the people listed in the room.\n"
        )

        system_prompt = (
            f"You are an invisible cognitive assistant for an Alzheimer's patient "
            f"named {PATIENT_NAME}.\n"
            f"There {is_are} currently {n_people} {person_word} in the room:\n"
            f"{people_block}\n\n"
            f"{diarisation_note}\n"
            f"Conversation transcript:\n\"{snapshot}\"\n\n"
            f"INSTRUCTIONS:\n"
            f"1. Keep your response to 1-2 short sentences maximum.\n"
            f"2. Speak directly to {PATIENT_NAME} "
            f"   (e.g., '{PATIENT_NAME}, your daughter Alice just asked...').\n"
            f"3. Use the [Speaker X]: tags to identify WHICH person said WHAT,\n"
            f"   then craft a cue that correctly names that person.\n"
            f"4. Suggest a simple, natural reply {PATIENT_NAME} can give.\n"
            f"5. Do NOT mention that you are an AI, assistant, "
            f"   or speaker-labelling system."
        )

        with llm_lock:
            help_text = llm.invoke(system_prompt).content.strip()

        if audio_output.full():
            try:
                audio_output.get_nowait()
            except queue.Empty:
                pass
        audio_output.put(help_text)
        log.info(f"live_help queued: {help_text[:80]}...")

        with transcription_lock:
            _trim_transcription_by_words()

    except RateLimitError:
        rotate_llm()
    except Exception as exc:
        log.error(f"live_help error: {exc}")

    return {}


def creategraphinfo(state: Alzheimer) -> dict:
    global transcription

    names = state.get("names", [])
    known_names = [
        n for n in names
        if n and not n.startswith("Unknown_") and n not in ("failed",)
    ]

    if not known_names:
        return {}

    with transcription_lock:
        snapshot = transcription

    if not snapshot.strip():
        return {}

    query = """
        MERGE (e:Person {name: $name1})
        MERGE (f:Person {name: $name2})
        SET f.relation   = $relation,
            f.last_convo = $last_convo
        MERGE (e)-[:KNOWS]->(f)
    """

    any_success = False

    for name in known_names:
        # P3: diarisation-aware per-person analysis prompt.
        analysis_prompt = (
            f"You are analysing a conversation involving Alzheimer's patient "
            f"'{PATIENT_NAME}' and another person named '{name}'.\n\n"
            f"SPEAKER DIARISATION NOTE:\n"
            f"  The transcript below uses [Speaker X]: tags assigned automatically.\n"
            f"  '{name}' is one of the speakers; {PATIENT_NAME} is another.\n"
            f"  Focus on the turns most likely spoken by '{name}' to determine\n"
            f"  the relationship and produce the conversation summary.\n"
            f"  Discard turns that clearly belong to {PATIENT_NAME}.\n\n"
            f"Transcript:\n{snapshot}"
        )

        try:
            with llm_lock:
                structured_llm = llm.with_structured_output(RelationsGraph)
                res = structured_llm.invoke(analysis_prompt)
            last_convo = res.last_convo
            relation   = res.relations
        except RateLimitError:
            rotate_llm()
            continue
        except Exception as exc:
            log.error(f"creategraphinfo LLM error for '{name}': {exc}")
            continue

        try:
            neo4j_driver.execute_query(
                query,
                name1=PATIENT_NAME, name2=name,
                relation=relation, last_convo=last_convo,
                database_="neo4j",
            )
            log.info(
                f"Graph memory updated for '{name}': "
                f"relation='{relation}' | last_convo='{last_convo[:60]}...'"
            )
            any_success = True
        except Exception as exc:
            log.error(f"creategraphinfo Neo4j write error for '{name}': {exc}")
            # Do NOT clear transcription -- keep data for retry on next pulse.

    if any_success:
        with transcription_lock:
            transcription = ""
        log.info("Transcription cleared after successful memory save.")

    return {}



def condition_check_recognize(state: Alzheimer) -> str:
    names = state.get("names", [])
    if not names:
        return END
    has_unknown = any(n.startswith("Unknown_") for n in names)
    return "identification" if has_unknown else "getraginfo"


def route_after_identification(state: Alzheimer) -> str:
    names = state.get("names", [])
    has_actionable = any(
        n and not n.startswith("Unknown_") and n not in ("failed", "")
        for n in names
    )
    return "getraginfo" if has_actionable else END


def route_after_help(state: Alzheimer) -> str:
    with transcription_lock:
        t_len = len(transcription)
    return "creategraphinfo" if t_len > TRANSCRIPTION_TRIM_THRESHOLD else END


workflow = StateGraph(Alzheimer)

workflow.add_node("recognize",       recognize)
workflow.add_node("identification",  identification)
workflow.add_node("getraginfo",      getraginfo)
workflow.add_node("live_help",       live_help)
workflow.add_node("creategraphinfo", creategraphinfo)

workflow.add_edge(START, "recognize")
workflow.add_conditional_edges("recognize",      condition_check_recognize)
workflow.add_conditional_edges("identification", route_after_identification)
workflow.add_edge("getraginfo",      "live_help")
workflow.add_conditional_edges("live_help",      route_after_help)
workflow.add_edge("creategraphinfo", END)



if __name__ == "__main__":
    brain = workflow.compile()

    _brain_last_names: list[str] = []

    def _standalone_brain_loop() -> None:
        global _brain_last_names
        while not stop_event.is_set():
            try:
                result = brain.invoke(
                    {"names": [], "relations": {}, "last_convos": {}}
                )
                if result:
                    _brain_last_names = [
                        n for n in result.get("names", [])
                        if n
                        and not n.startswith("Unknown_")
                        and n not in ("failed", "")
                    ]
            except Exception as exc:
                log.error(f"Brain error: {exc}")
            time.sleep(BRAIN_PULSE_INTERVAL)

    log.info("Starting sensory threads...")
    threads = [
        threading.Thread(target=input_visual,          name="InputVisual",  daemon=True),
        threading.Thread(target=input_audio,           name="InputAudio",   daemon=True),
        threading.Thread(target=process_visual,        name="ProcVisual",   daemon=True),
        threading.Thread(target=process_audio,         name="ProcAudio",    daemon=True),
        threading.Thread(target=output_audio,          name="OutputAudio",  daemon=True),
        threading.Thread(target=_standalone_brain_loop, name="BrainLoop",   daemon=True),
    ]
    for t in threads:
        t.start()

    log.info("All threads running — press 'q' in the OpenCV window to quit.")

    try:
        while not stop_event.is_set():
            with ui_lock:
                frame_snapshot = (
                    current_display_frame.copy()
                    if current_display_frame is not None
                    else None
                )
                people_snapshot = list(current_display_people)

            if frame_snapshot is not None:
                h_frame, w_frame = frame_snapshot.shape[:2]
                for person in people_snapshot:
                    box  = person.get("box")
                    name = person.get("name", "")
                    if box is None:
                        continue

                    x1 = max(0, min(int(box[0]), w_frame - 1))
                    y1 = max(0, min(int(box[1]), h_frame - 1))
                    x2 = max(0, min(int(box[2]), w_frame - 1))
                    y2 = max(0, min(int(box[3]), h_frame - 1))
                    label_y = max(35, y1)

                    cv.rectangle(frame_snapshot, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv.rectangle(
                        frame_snapshot,
                        (x1, label_y - 35), (x2, label_y),
                        (0, 255, 0), cv.FILLED,
                    )
                    cv.putText(
                        frame_snapshot, name, (x1 + 5, label_y - 10),
                        cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2,
                    )

                cv.imshow("Alzheimer's Assistant POV", frame_snapshot)

            # ~30 fps poll; catches 'q' even while brain is mid-invoke
            if cv.waitKey(33) & 0xFF == ord("q"):
                break

    except KeyboardInterrupt:
        log.info("Manual interruption.")

    finally:
        log.info("Shutting down...")
        stop_event.set()
        cv.destroyAllWindows()

        with transcription_lock:
            t_len = len(transcription)
        if _brain_last_names and t_len > 50:
            log.info(f"Saving final memories for: {_brain_last_names}...")
            creategraphinfo(
                {"names": _brain_last_names, "relations": {}, "last_convos": {}}
            )

        neo4j_driver.close()
        log.info("Goodbye.")


