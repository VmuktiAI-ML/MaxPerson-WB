#!/usr/bin/env python3
"""
main_production_enhanced.py
═══════════════════════════════════════════════════════════════════════════════
Production-Ready 5000-Camera AI Pipeline with Dynamic Camera Addition
───────────────────────────────────────────────────────────────────────────────

CSV MONITORING (updated to match EVM pipeline):
────────────────────────────────────────────────
  - Uses watchdog FileSystemObserver for INSTANT detection when CSV is saved
  - Falls back to periodic 10-minute checks (CSV_CHECK_INTERVAL_SEC)
  - Camera ADDITIONS and REMOVALS are queued and applied at the next cycle
    boundary — so existing cameras are never interrupted mid-cycle
  - Prints clear terminal notifications on every change

Architecture
────────────
  Layer 1 — Ingestion   : asyncio event loop  +  5000 independent coroutines
                          Each camera has its OWN absolute-clock schedule.
                          Slow / timed-out cameras never delay other cameras.

  Layer 2 — Frame Queue : asyncio.Queue (bounded) with backpressure.
                          If GPU falls behind, new frames block instead of OOM.

  Layer 3 — GPU Infer   : ONE dedicated thread owns the single YOLO model.
                          Collects frames into batches, runs inference,
                          returns results via per-frame asyncio.Future.
                          No GPU context thrashing. Maximum batch efficiency.

  Layer 4 — Alert       : Per-camera cooldown (absolute clock).
                          aiohttp async POST → never blocks event loop.
                          Azure upload in thread pool → never blocks event loop.
"""

# ── stdlib env tweaks (must be before numpy/cv2 imports) ──────────────────────
import os
os.environ["OMP_NUM_THREADS"]        = "1"
os.environ["OPENBLAS_NUM_THREADS"]   = "1"
os.environ["MKL_NUM_THREADS"]        = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"]    = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import resource
# Increase the limit of open files (File Descriptors) to 1,048,576
soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (1048576, 1048576))

import sys
import asyncio

# ── Raise inotify limits so the file-system watcher can start ─────────────────
# 2000+ camera processes each open several fds; on many Linux installs the
# default max_user_watches (8192) and max_user_instances (128) are too low.
# We try to raise them via /proc; this silently no-ops if we lack permission
# (non-root), in which case the PollingObserver fallback requires no inotify.
def _raise_inotify_limits():
    targets = {
        "/proc/sys/fs/inotify/max_user_watches":   "524288",
        "/proc/sys/fs/inotify/max_user_instances":  "1024",
        "/proc/sys/fs/inotify/max_queued_events":  "131072",
    }
    for path, value in targets.items():
        try:
            with open(path, "w") as fh:
                fh.write(value + "\n")
        except OSError:
            pass   # not root — skip silently

_raise_inotify_limits()
import signal
import subprocess
import threading
import logging
import time
import queue as std_queue          # stdlib thread-safe queue for GPU bridge
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from typing import Optional, Set

import cv2
import numpy as np
import pandas as pd
import torch
import requests
from requests.adapters import HTTPAdapter

try:
    import aiohttp
    AIOHTTP_AVAILABLE = True
except ImportError:
    AIOHTTP_AVAILABLE = False

from ultralytics import YOLO

try:
    from azure.storage.blob import BlobServiceClient
    from azure.core.pipeline.transport import RequestsTransport
    AZURE_AVAILABLE = True
except ImportError:
    AZURE_AVAILABLE = False

# watchdog for instant CSV change detection.
# We prefer PollingObserver over InotifyObserver because large deployments
# (5000 cameras) easily exhaust the kernel's inotify watch limit.
# PollingObserver uses plain stat() — zero inotify watches consumed.
try:
    from watchdog.observers.polling import PollingObserver
    from watchdog.events import FileSystemEventHandler
    WATCHDOG_AVAILABLE = True
except ImportError:
    WATCHDOG_AVAILABLE = False

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION  — edit this block only
# ═══════════════════════════════════════════════════════════════════════════════

class Config:
    # Camera schedule
    POLL_INTERVAL_SEC       = 60        # 6000 cameras / 60s = 100 FPS throughput
    FFMPEG_TIMEOUT_SEC      = 45        # drop slow cams quickly to keep the flow
    CAM_BACKOFF_MAX_SEC     = 120       # retry failed cams quickly

    # Inference
    YOLO_MODEL_PATH         = "yolo11m.pt"
    YOLO_IMGSZ              = 640
    YOLO_CONF               = 0.20
    YOLO_IOU                = 0.45
    INFER_BATCH_SIZE        = 256      # optimized for H100 (80GB VRAM)
    INFER_BATCH_TIMEOUT_SEC = 0.5       # flush faster to keep 100fps movement
    INFER_FRAME_W           = 640      # resize before queuing (saves VRAM)
    INFER_FRAME_H           = 360

    # Alert
    MAX_PERSON_THRESHOLD    = 15
    ALERT_COOLDOWN_SEC      = 120       # min seconds between alerts per camera

    # Queue
    FRAME_QUEUE_MAXSIZE     = 4096      # backpressure cap

    # Thread pools
    FFMPEG_POOL_WORKERS     = 1500      # 100 FPS total flow (H100 can handle this)
    UPLOAD_POOL_WORKERS     = 128        # Azure + API upload workers

    # Paths / URLs
    CSV_FILE_PATH           = "outdoor5.csv"
    LOCAL_ALERT_DIR         = "alert"
    API_URL                 = "https://tn2023demo.vmukti.com/api/analytics"

    # Azure
    AZURE_CONNECTION_STRING = (
        "BlobEndpoint=https://nvrdatashinobi.blob.core.windows.net/;"
        "QueueEndpoint=https://nvrdatashinobi.queue.core.windows.net/;"
        "FileEndpoint=https://nvrdatashinobi.file.core.windows.net/;"
        "TableEndpoint=https://nvrdatashinobi.table.core.windows.net/;"
        "SharedAccessSignature=sv=2024-11-04&ss=bfqt&srt=sco&sp=rwdlacupiytfx"
        "&se=2026-05-30T20:16:33Z&st=2026-02-20T12:01:33Z&spr=https,http"
        "&sig=YenJbBQB3iuMwhJqtu724lm7ID%2B6L2GXpbv%2BpmhTwrk%3D"
    )
    AZURE_CONTAINER_NAME    = "nvrdatashinobi"
    AZURE_BLOB_PREFIX       = "live-record/frimages"
    AZURE_POOL_SIZE         = 64

    # Watchdog
    WATCHDOG_INTERVAL_SEC   = 60

    # CSV Monitoring
    CSV_CHECK_INTERVAL_SEC  = 600       # 10-minute periodic fallback check
    CSV_DEBOUNCE_SEC        = 2         # ignore duplicate save events within 2s

# ═══════════════════════════════════════════════════════════════════════════════
# LOGGING
# ═══════════════════════════════════════════════════════════════════════════════

class _AlertFilter(logging.Filter):
    def filter(self, record):
        return "[ALERT]" in record.getMessage()

class _CycleFilter(logging.Filter):
    def filter(self, record):
        return (
            "[WATCH]" in record.getMessage()
            or "[MAIN]"  in record.getMessage()
            or "[CSV]"   in record.getMessage()
        )

def _setup_logging():
    fmt = logging.Formatter("%(asctime)s  %(levelname)-8s  %(message)s",
                            datefmt="%Y-%m-%d %H:%M:%S")

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    root.handlers.clear()

    # main log: everything INFO+
    fh = logging.FileHandler("pipeline.log", encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(fmt)
    root.addHandler(fh)

    # alerts-only log
    ah = logging.FileHandler("alerts.log", encoding="utf-8")
    ah.setLevel(logging.INFO)
    ah.setFormatter(fmt)
    ah.addFilter(_AlertFilter())
    root.addHandler(ah)

    # debug log: everything
    dh = logging.FileHandler("debug.log", encoding="utf-8")
    dh.setLevel(logging.DEBUG)
    dh.setFormatter(fmt)
    root.addHandler(dh)

    # console: only watchdog + main + CSV monitor lines (keep terminal clean)
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter("%(asctime)s  %(message)s",
                                      datefmt="%H:%M:%S"))
    ch.addFilter(_CycleFilter())
    root.addHandler(ch)

_setup_logging()
logger = logging.getLogger(__name__)
global_cycle_num = 1

# ═══════════════════════════════════════════════════════════════════════════════
# ANNOTATION  (identical style to your original)
# ═══════════════════════════════════════════════════════════════════════════════

_BOX_COLOR   = (0, 200, 0)
_BOX_THICK   = 2
_FONT        = cv2.FONT_HERSHEY_DUPLEX
_WHITE       = (255, 255, 255)
_BLACK       = (0,   0,   0)
_RED_DARK    = (0,   0, 180)


def _rounded_rect(img, pt1, pt2, color, thickness, radius):
    x1, y1 = pt1
    x2, y2 = pt2
    r = max(1, min(radius, (x2-x1)//2, (y2-y1)//2))
    cv2.line(img, (x1+r, y1), (x2-r, y1), color, thickness)
    cv2.line(img, (x1+r, y2), (x2-r, y2), color, thickness)
    cv2.line(img, (x1, y1+r), (x1, y2-r), color, thickness)
    cv2.line(img, (x2, y1+r), (x2, y2-r), color, thickness)
    cv2.ellipse(img, (x1+r, y1+r), (r,r), 180,  0, 90, color, thickness)
    cv2.ellipse(img, (x2-r, y1+r), (r,r), 270,  0, 90, color, thickness)
    cv2.ellipse(img, (x1+r, y2-r), (r,r),  90,  0, 90, color, thickness)
    cv2.ellipse(img, (x2-r, y2-r), (r,r),   0,  0, 90, color, thickness)


def draw_annotated_frame(frame_bgr, boxes, person_count, cam_id, cam_name=""):
    img       = frame_bgr.copy()
    h, w      = img.shape[:2]
    is_alert  = person_count > Config.MAX_PERSON_THRESHOLD

    for box in boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        radius = int((y2-y1) * 0.12)
        _rounded_rect(img, (x1,y1), (x2,y2), _BOX_COLOR, _BOX_THICK, radius)

    ts_str = datetime.now().strftime("%Y/%m/%d  %H:%M:%S")

    badge = f"Outdoor (People): {person_count}"
    (bw, bh), _ = cv2.getTextSize(badge, _FONT, 0.52, 1)

    alert_txt = "ALERT: MAX PERSONS"
    aw, ah = 0, 0
    if is_alert:
        (aw, ah), _ = cv2.getTextSize(alert_txt, _FONT, 0.52, 1)

    box_w = max(bw, aw) + 16
    rx1, rx2 = w - box_w - 6, w - 6

    cv2.rectangle(img, (rx1, 6), (rx2, 6+bh+10), (210,210,210), -1)
    cv2.putText(img, badge, (rx1 + (box_w - bw)//2, 6+bh+6), _FONT, 0.52, _BLACK, 1, cv2.LINE_AA)

    if is_alert:
        ay1 = 6 + bh + 10
        ay2 = ay1 + ah + 10
        cv2.rectangle(img, (rx1, ay1), (rx2, ay2), _RED_DARK, -1)
        cv2.putText(img, alert_txt, (rx1 + (box_w - aw)//2, ay2-4), _FONT, 0.52, _WHITE, 1, cv2.LINE_AA)

    return img

# ═══════════════════════════════════════════════════════════════════════════════
# AZURE CLIENT  (one shared client, thread-safe)
# ═══════════════════════════════════════════════════════════════════════════════

def build_azure_client() -> Optional[object]:
    if not AZURE_AVAILABLE:
        logger.warning("[MAIN] azure-storage-blob not installed — uploads disabled")
        return None
    try:
        sess = requests.Session()
        adapter = HTTPAdapter(
            pool_connections=Config.AZURE_POOL_SIZE,
            pool_maxsize=Config.AZURE_POOL_SIZE,
            max_retries=3,
        )
        sess.mount("https://", adapter)
        sess.mount("http://",  adapter)
        client = BlobServiceClient.from_connection_string(
            Config.AZURE_CONNECTION_STRING,
            transport=RequestsTransport(session=sess),
            connection_timeout=15,
            read_timeout=60,
        )
        logger.info("[MAIN] Azure BlobServiceClient initialised")
        return client
    except Exception as exc:
        logger.error(f"[MAIN] Azure init failed: {exc}")
        return None

# ═══════════════════════════════════════════════════════════════════════════════
# FRAME GRABBER  (runs in thread pool — blocking ffmpeg subprocess)
# ═══════════════════════════════════════════════════════════════════════════════

def grab_single_frame(url: str) -> Optional[np.ndarray]:
    """
    Spawn ffmpeg, grab exactly 1 frame, decode to BGR numpy array.
    Returns None on any failure — never raises.
    Resize to INFER_FRAME_W × INFER_FRAME_H before returning to save VRAM.
    """
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-fflags",           "nobuffer+discardcorrupt",
        "-flags",            "low_delay",
        "-rw_timeout",       str(Config.FFMPEG_TIMEOUT_SEC * 1_000_000),
        "-analyzeduration",  "50000",         # speed up RTMP connection
        "-probesize",        "100000",        # minimum size for 1 frame
        "-threads",          "1",             # save CPU overhead
        "-i",                url,
        "-frames:v",         "1",
        "-f",                "image2",
        "-vcodec",           "mjpeg",
        "pipe:1",
    ]
    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        try:
            out, err = proc.communicate(timeout=Config.FFMPEG_TIMEOUT_SEC + 2)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.communicate()
            return None

        if proc.returncode != 0 or not out:
            msg = (err or b"").decode("utf-8", errors="ignore").strip()
            if msg:
                logger.debug(f"ffmpeg [{url[-30:]}]: {msg[:120]}")
            return None

        arr = np.frombuffer(out, np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if frame is None:
            return None

        # resize once here — not in the GPU thread
        frame = cv2.resize(
            frame,
            (Config.INFER_FRAME_W, Config.INFER_FRAME_H),
            interpolation=cv2.INTER_LINEAR,
        )
        return frame

    except Exception as exc:
        logger.debug(f"grab_single_frame exception [{url[-30:]}]: {exc}")
        return None

# ═══════════════════════════════════════════════════════════════════════════════
# GPU INFERENCE THREAD
# ═══════════════════════════════════════════════════════════════════════════════

class InferenceEngine:
    """
    Wraps the YOLO model in a dedicated thread.
    Async code submits work via submit() and awaits the returned Future.
    """

    def __init__(self, loop: asyncio.AbstractEventLoop):
        self._loop      = loop
        self._infer_q   : std_queue.Queue = std_queue.Queue(maxsize=1024)
        self._model     : Optional[YOLO]  = None
        self._device    : str             = "cpu"
        self._thread    : Optional[threading.Thread] = None
        self._stats_lock = threading.Lock()
        self._frames_inferred = 0
        self._batches_run     = 0

    def load_model(self):
        """Load YOLO model. Call from main thread before start()."""
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self._device = str(device)
        logger.info(f"[MAIN] Loading YOLO model on {self._device} ...")
        m = YOLO(Config.YOLO_MODEL_PATH)
        m.to(device)
        # warm-up: avoids first-batch latency spike
        dummy = [np.zeros((Config.INFER_FRAME_H, Config.INFER_FRAME_W, 3), dtype=np.uint8)]
        m(dummy, device=self._device, imgsz=Config.YOLO_IMGSZ,
          conf=Config.YOLO_CONF, iou=Config.YOLO_IOU,
          classes=[0], verbose=False)
        self._model = m
        logger.info("[MAIN] YOLO model loaded and warmed up")

    def start(self):
        """Start the inference thread."""
        self._thread = threading.Thread(
            target=self._run,
            name="infer-gpu",
            daemon=True,
        )
        self._thread.start()
        logger.info("[MAIN] Inference thread started")

    def submit(self, frame: np.ndarray) -> asyncio.Future:
        """
        Submit one frame for inference.
        Returns an asyncio.Future resolved with the YOLO Results object.
        Thread-safe: can be called from any async task.
        """
        fut = self._loop.create_future()
        self._infer_q.put((frame, fut))
        return fut

    def stats(self):
        with self._stats_lock:
            return self._frames_inferred, self._batches_run

    def _run(self):
        """
        Runs forever in its own thread.
        Assembles batches from infer_q, runs model(), resolves futures.
        """
        logger.info("[GPU] Inference thread running")
        while True:
            frames, futures = self._collect_batch()
            if not frames:
                continue
            try:
                results = self._model(
                    frames,
                    device=self._device,
                    imgsz=Config.YOLO_IMGSZ,
                    conf=Config.YOLO_CONF,
                    iou=Config.YOLO_IOU,
                    classes=[0],
                    verbose=False,
                )
                with self._stats_lock:
                    self._frames_inferred += len(frames)
                    self._batches_run     += 1

                for fut, result in zip(futures, results):
                    self._loop.call_soon_threadsafe(fut.set_result, result)

            except Exception as exc:
                logger.error(f"[GPU] Inference error: {exc}", exc_info=True)
                err = RuntimeError(str(exc))
                for fut in futures:
                    if not fut.done():
                        self._loop.call_soon_threadsafe(fut.set_exception, err)

    def _collect_batch(self):
        """
        Block until at least 1 item is available, then collect up to
        INFER_BATCH_SIZE items within INFER_BATCH_TIMEOUT_SEC.
        Returns (frames_list, futures_list).
        """
        frames  = []
        futures = []

        # block for first item
        try:
            frame, fut = self._infer_q.get(timeout=1.0)
            frames.append(frame)
            futures.append(fut)
        except std_queue.Empty:
            return [], []

        # drain more items up to batch limit within timeout
        deadline = time.monotonic() + Config.INFER_BATCH_TIMEOUT_SEC
        while len(frames) < Config.INFER_BATCH_SIZE:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            try:
                frame, fut = self._infer_q.get(timeout=remaining)
                frames.append(frame)
                futures.append(fut)
            except std_queue.Empty:
                break

        return frames, futures

# ═══════════════════════════════════════════════════════════════════════════════
# ALERT DISPATCHER  (async — runs in event loop)
# ═══════════════════════════════════════════════════════════════════════════════

class AlertDispatcher:
    """
    Handles cooldown tracking and fires upload + API call asynchronously.
    """

    def __init__(self, upload_executor: ThreadPoolExecutor, blob_client):
        self._last_alert  : dict = defaultdict(float)
        self._lock        = asyncio.Lock()
        self._executor    = upload_executor
        self._blob_client = blob_client
        self._loop        = asyncio.get_event_loop()

    async def maybe_alert(
        self,
        cam_id:       str,
        frame_bgr:    np.ndarray,
        yolo_result,
        person_count: int,
        stats:        dict,
    ):
        if person_count <= Config.MAX_PERSON_THRESHOLD:
            return False

        now = time.monotonic()
        async with self._lock:
            if now - self._last_alert[cam_id] < Config.ALERT_COOLDOWN_SEC:
                return False
            self._last_alert[cam_id] = now

        global global_cycle_num
        self._loop.run_in_executor(
            self._executor,
            self._upload_and_post,
            cam_id, frame_bgr.copy(), yolo_result, person_count, global_cycle_num, stats
        )
        return True

    def _upload_and_post(self, cam_id, frame_bgr, yolo_result, count, current_cycle_num, stats):
        try:
            annotated = draw_annotated_frame(
                frame_bgr, yolo_result.boxes, count, cam_id
            )
            ts       = datetime.now().strftime("%Y-%m-%d-%H-%M-%S-%f")[:-3]
            filename = f"{cam_id}_{ts}.png"

            _, buf = cv2.imencode(
                ".png", annotated, [cv2.IMWRITE_PNG_COMPRESSION, 3]
            )
            img_bytes = buf.tobytes()

            # local save
            # try:
            #     cycle_dir = os.path.join(Config.LOCAL_ALERT_DIR, f"cycle_{current_cycle_num}_alerts")
            #     os.makedirs(cycle_dir, exist_ok=True)
            #     cv2.imwrite(
            #         os.path.join(cycle_dir, filename), annotated
            #     )
            # except Exception as exc:
            #     logger.warning(f"[ALERT] local save failed cam={cam_id}: {exc}")

            # Azure upload
            img_url = f"https://nvrdatashinobi.blob.core.windows.net/{Config.AZURE_CONTAINER_NAME}/{Config.AZURE_BLOB_PREFIX}/{filename}"
            if self._blob_client:
                try:
                    blob_name = f"{Config.AZURE_BLOB_PREFIX}/{filename}"
                    bc = self._blob_client.get_blob_client(
                        container=Config.AZURE_CONTAINER_NAME, blob=blob_name
                    )
                    bc.upload_blob(img_bytes, overwrite=True)
                except Exception as exc:
                    logger.warning(f"[ALERT] Azure upload failed cam={cam_id}: {exc}")

            # API POST
            payload = {
                "cameradid":  cam_id,
                "sendtime": (datetime.now(timezone.utc) + timedelta(hours=5, minutes=30)).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
                "imgurl":     img_url,
                "an_id":      101,
                "ImgCount":   count,
                "totalcount": count,
            }
            try:
                resp = requests.post(Config.API_URL, json=payload, timeout=(5, 15))
                if resp.status_code == 200:
                    status = "OK"
                    stats["api_ok"] = stats.get("api_ok", 0) + 1
                else:
                    status = str(resp.status_code)
                    stats["api_fail"] = stats.get("api_fail", 0) + 1
                logger.info(
                    f"[ALERT] cam={cam_id}  persons={count}"
                    f"  img={filename}  api={status}"
                )
            except Exception as exc:
                stats["api_fail"] = stats.get("api_fail", 0) + 1
                logger.warning(f"[ALERT] API POST failed cam={cam_id}: {exc}")

        except Exception as exc:
            logger.error(f"[ALERT] upload_and_post exception cam={cam_id}: {exc}",
                         exc_info=True)

# ═══════════════════════════════════════════════════════════════════════════════
# CAMERA COROUTINE  (one per camera, lives forever)
# ═══════════════════════════════════════════════════════════════════════════════

def _cam_key(url: str) -> str:
    return url.rstrip("/").split("/")[-1].strip().rstrip(",")


async def camera_coroutine(
    url:        str,
    engine:     InferenceEngine,
    dispatcher: AlertDispatcher,
    executor:   ThreadPoolExecutor,
    shutdown:   asyncio.Event,
    stats:      dict,
):
    cam_id         = _cam_key(url)
    loop           = asyncio.get_event_loop()
    consecutive_fails = 0
    scheduled_at   = time.monotonic()

    logger.debug(f"[CAM] {cam_id} coroutine started")

    while not shutdown.is_set():
        # ── sleep until next scheduled time ───────────────────────────────────
        now = time.monotonic()
        sleep_for = scheduled_at - now

        # if we're already behind (e.g. overrun), skip missed cycles
        if sleep_for < 0:
            skipped = int(-sleep_for / Config.POLL_INTERVAL_SEC) + 1
            scheduled_at += skipped * Config.POLL_INTERVAL_SEC
            sleep_for = scheduled_at - time.monotonic()
            if skipped > 1:
                logger.debug(f"[CAM] {cam_id} skipped {skipped} cycles (overrun)")

        if sleep_for > 0:
            try:
                await asyncio.wait_for(
                    shutdown.wait(),
                    timeout=sleep_for,
                )
                break
            except asyncio.TimeoutError:
                pass

        if shutdown.is_set():
            break

        scheduled_at += Config.POLL_INTERVAL_SEC

        # ── grab frame in thread pool ──────────────────────────────────────────
        t_grab = time.monotonic()
        try:
            frame = await loop.run_in_executor(executor, grab_single_frame, url)
        except Exception as exc:
            logger.warning(f"[CAM] {cam_id} grab exception: {exc}")
            frame = None

        grab_ms = (time.monotonic() - t_grab) * 1000

        if frame is None:
            consecutive_fails += 1
            stats["fails"] = stats.get("fails", 0) + 1
            logger.debug(
                f"[CAM] {cam_id} grab failed  "
                f"(consecutive={consecutive_fails}  took={grab_ms:.0f}ms)"
            )
            if consecutive_fails >= 5:
                backoff_cycles = min(consecutive_fails - 4, 10)
                scheduled_at  += backoff_cycles * Config.POLL_INTERVAL_SEC
                logger.info(
                    f"[CAM] {cam_id} backing off {backoff_cycles} cycles "
                    f"(fails={consecutive_fails})"
                )
            continue

        consecutive_fails = 0
        stats["grabs"]    = stats.get("grabs", 0) + 1

        # ── submit to GPU inference engine ─────────────────────────────────────
        try:
            future = engine.submit(frame)
            result = await asyncio.wrap_future(future)
        except Exception as exc:
            logger.warning(f"[CAM] {cam_id} inference exception: {exc}")
            continue

        person_count = len(result.boxes)
        stats["inferences"] = stats.get("inferences", 0) + 1

        logger.debug(
            f"[CAM] {cam_id}  persons={person_count}"
            f"  grab={grab_ms:.0f}ms"
        )

        # ── alert dispatch ─────────────────────────────────────────────────────
        try:
            alerted = await dispatcher.maybe_alert(cam_id, frame, result, person_count, stats)
            if alerted:
                stats["alerts_sent"] = stats.get("alerts_sent", 0) + 1
        except Exception as exc:
            logger.warning(f"[CAM] {cam_id} alert exception: {exc}")

    logger.debug(f"[CAM] {cam_id} coroutine exiting")

# ═══════════════════════════════════════════════════════════════════════════════
# CSV CHANGE HANDLER  (watchdog FileSystemEventHandler)
# ═══════════════════════════════════════════════════════════════════════════════

class _CSVChangeHandler(FileSystemEventHandler if WATCHDOG_AVAILABLE else object):
    """
    Fires reload_callback when the watched CSV file is saved.
    Debounced to ignore duplicate OS events within CSV_DEBOUNCE_SEC.
    """

    def __init__(self, csv_path: str, reload_callback):
        if WATCHDOG_AVAILABLE:
            super().__init__()
        self._csv_abs       = os.path.abspath(csv_path)
        self._reload_cb     = reload_callback
        self._last_fired    = 0.0

    def on_modified(self, event):
        if getattr(event, "is_directory", False):
            return
        if os.path.abspath(event.src_path) != self._csv_abs:
            return
        now = time.time()
        if now - self._last_fired < Config.CSV_DEBOUNCE_SEC:
            return
        self._last_fired = now
        logger.info("[CSV] 📝 CSV file saved — queuing camera update at next cycle boundary")
        self._reload_cb()

# ═══════════════════════════════════════════════════════════════════════════════
# CSV MONITOR  (cycle-synchronised, supports instant + periodic reload)
# ═══════════════════════════════════════════════════════════════════════════════

async def csv_monitor(
    running_tasks:   dict,           # url -> asyncio.Task  (mutated in-place)
    engine:          InferenceEngine,
    dispatcher:      AlertDispatcher,
    executor:        ThreadPoolExecutor,
    shutdown:        asyncio.Event,
    stats:           dict,
    active_urls_ref: list,           # [set]  — read by watchdog for cam count
    pending_ref:     list,           # [list] — watchdog reads & clears additions
    removals_ref:    list,           # [set]  — watchdog reads & clears removals
):
    """
    Watches the CSV file for changes (instant via watchdog + 10-min fallback).
    Queues additions into pending_ref[0] and removals into removals_ref[0].
    The watchdog coroutine applies them at every cycle boundary.
    """
    loop = asyncio.get_event_loop()

    # ── reload trigger ─────────────────────────────────────────────────────────
    reload_requested = asyncio.Event()

    def _trigger_reload():
        loop.call_soon_threadsafe(reload_requested.set)

    # ── start file-system watcher (optional) ──────────────────────────────────
    # PollingObserver uses stat() polling — it consumes ZERO inotify watches,
    # so it works even when inotify is exhausted by 2000+ camera processes.
    # Poll interval of 5 s means changes are noticed within 5 seconds of saving.
    file_observer = None
    if WATCHDOG_AVAILABLE:
        try:
            csv_dir = os.path.dirname(os.path.abspath(Config.CSV_FILE_PATH))
            handler = _CSVChangeHandler(Config.CSV_FILE_PATH, _trigger_reload)
            file_observer = PollingObserver(timeout=5)   # poll every 5 s, zero inotify watches
            file_observer.schedule(handler, csv_dir, recursive=False)
            file_observer.start()
            logger.info(
                f"[CSV] 👁️  Watching CSV (PollingObserver 5s) — "
                f"zero inotify watches consumed: "
                f"{os.path.basename(Config.CSV_FILE_PATH)}"
            )
        except Exception as exc:
            logger.warning(
                f"[CSV] File watcher could not start: {exc}  "
                f"— falling back to {Config.CSV_CHECK_INTERVAL_SEC}s periodic check"
            )
    else:
        logger.warning(
            "[CSV] watchdog not installed — only periodic checks every "
            f"{Config.CSV_CHECK_INTERVAL_SEC}s.  pip install watchdog"
        )

    # ── pending queues (shared with watchdog via caller-supplied refs) ────────
    pending_additions: list = pending_ref[0]   # list of dicts {url, task, cam_id}
    pending_removals:  set  = removals_ref[0]  # set of urls to cancel

    last_reload      = -float("inf")
    is_initial_load  = True        # first load skips the cycle-boundary wait

    logger.info(
        f"[CSV] Monitor started — instant (watchdog) + "
        f"periodic fallback every {Config.CSV_CHECK_INTERVAL_SEC}s"
    )

    try:
        while not shutdown.is_set():
            # ── determine whether a reload is due ─────────────────────────────
            now = time.monotonic()
            should_reload = False

            if reload_requested.is_set():
                reload_requested.clear()
                should_reload = True
                logger.info("[CSV] 🔄 CSV change detected — computing diff...")
            elif now - last_reload >= Config.CSV_CHECK_INTERVAL_SEC:
                should_reload = True
                logger.debug("[CSV] 🔄 Periodic 10-min check")

            # ── perform diff and queue changes ─────────────────────────────────
            if should_reload:
                try:
                    if not os.path.exists(Config.CSV_FILE_PATH):
                        logger.warning(f"[CSV] File not found: {Config.CSV_FILE_PATH}")
                    else:
                        df = pd.read_csv(
                            Config.CSV_FILE_PATH, header=None, encoding="utf-8-sig"
                        )
                        new_urls = [
                            u.strip()
                            for u in df[0].dropna().unique().tolist()
                            if str(u).strip().startswith(("rtmp", "rtsp", "http"))
                        ]
                        new_set     = set(new_urls)
                        running_set = set(running_tasks.keys())
                        pending_set = {p["url"] for p in pending_additions}
                        all_managed = running_set | pending_set

                        # -- removals ------------------------------------------
                        to_remove = all_managed - new_set
                        if to_remove:
                            # Clear from pending_additions if they were queued but
                            # then removed before the cycle boundary arrived
                            pending_additions = [
                                p for p in pending_additions
                                if p["url"] not in to_remove
                            ]
                            pending_removals.update(to_remove)
                            logger.info(
                                f"[CSV] ⏳ Queued {len(to_remove)} removal(s) "
                                f"for next cycle boundary"
                            )

                        # -- additions ------------------------------------------
                        to_add = new_set - all_managed
                        if to_add:
                            # Un-queue any that were previously marked for removal
                            pending_removals -= to_add

                            n_adding     = max(len(to_add), 1)
                            stagger_step = Config.POLL_INTERVAL_SEC / n_adding

                            for i, url in enumerate(sorted(to_add)):
                                cam_id  = _cam_key(url)
                                stagger = i * stagger_step

                                async def _cam_wrapper(
                                    u=url, delay=stagger,
                                ):
                                    # Stagger start to avoid ffmpeg thundering-herd.
                                    # The watchdog will move this task from
                                    # pending_additions into running_tasks at the
                                    # next cycle boundary — so it is already gated.
                                    if delay > 0:
                                        try:
                                            await asyncio.wait_for(
                                                shutdown.wait(), timeout=delay
                                            )
                                            return
                                        except asyncio.TimeoutError:
                                            pass
                                    if shutdown.is_set():
                                        return
                                    await camera_coroutine(
                                        u, engine, dispatcher,
                                        executor, shutdown, stats,
                                    )

                                task = asyncio.create_task(
                                    _cam_wrapper(), name=f"cam-{cam_id}"
                                )

                                if is_initial_load:
                                    # Add immediately — no cycle-boundary wait
                                    running_tasks[url] = task
                                    #logger.debug(f"[CSV] ✅ Started cam: {cam_id}")
                                else:
                                    pending_additions.append(
                                        {"url": url, "task": task, "cam_id": cam_id}
                                    )

                            if to_add and not is_initial_load:
                                logger.info(
                                    f"[CSV] ⏳ Queued {len(to_add)} addition(s) "
                                    f"for next cycle boundary"
                                )

                        if not to_remove and not to_add:
                            logger.debug("[CSV] No camera changes detected")

                        last_reload     = time.monotonic()
                        is_initial_load = False

                        total = len(running_tasks) + len(pending_additions)
                        logger.info(
                            f"[CSV] 📊 Active: {len(running_tasks)} | "
                            f"Pending add: {len(pending_additions)} | "
                            f"Pending remove: {len(pending_removals)} | "
                            f"Total: {total}"
                        )

                except Exception as exc:
                    logger.error(f"[CSV] Error during reload: {exc}", exc_info=True)

            # cycle boundary is now applied by watchdog — nothing to do here

            # ── wait for the next trigger (1-second heartbeat) ─────────────────
            try:
                wait_shutdown = asyncio.ensure_future(shutdown.wait())
                wait_reload   = asyncio.ensure_future(reload_requested.wait())
                _done, _pend  = await asyncio.wait(
                    [wait_shutdown, wait_reload],
                    timeout=1.0,
                    return_when=asyncio.FIRST_COMPLETED,
                )
                for f in _pend:
                    f.cancel()
                if shutdown.is_set():
                    break
            except asyncio.TimeoutError:
                pass

    finally:
        if file_observer:
            try:
                file_observer.stop()
                file_observer.join(timeout=2)
            except RuntimeError:
                pass
        logger.info("[CSV] Monitor stopped")

# ═══════════════════════════════════════════════════════════════════════════════
# WATCHDOG  (logs health metrics every WATCHDOG_INTERVAL_SEC)
# ═══════════════════════════════════════════════════════════════════════════════

async def watchdog(
    engine:           InferenceEngine,
    stats:            dict,
    shutdown:         asyncio.Event,
    n_cams:           int,
    active_urls_ref:  list,    # [set_of_active_urls]  — updated by csv_monitor
    pending_ref:      list,    # [pending_additions list]  — owned by csv_monitor
    removals_ref:     list,    # [pending_removals set]    — owned by csv_monitor
    running_tasks:    dict,    # url -> Task               — shared with csv_monitor
):
    """
    Fires every WATCHDOG_INTERVAL_SEC.
    At each tick it atomically applies any camera additions/removals that
    csv_monitor has queued, then logs the health summary.
    Owning the apply step here (rather than in csv_monitor) avoids the
    asyncio event-racing problem where cycle_boundary.set()+clear() happened
    faster than csv_monitor could wake and react.
    """
    prev_inferences = 0
    prev_t          = time.monotonic()
    prev_grabs      = 0
    prev_fails      = 0
    prev_alerts     = 0
    prev_batches    = 0
    prev_api_ok     = 0
    prev_api_fail   = 0

    while not shutdown.is_set():
        try:
            await asyncio.wait_for(
                shutdown.wait(),
                timeout=Config.WATCHDOG_INTERVAL_SEC,
            )
            break
        except asyncio.TimeoutError:
            pass

        # ── apply pending camera changes (cycle boundary) ─────────────────────
        pending_additions = pending_ref[0]
        pending_removals  = removals_ref[0]

        if pending_removals:
            logger.info(
                f"[WATCH] ═══ CYCLE BOUNDARY — removing "
                f"{len(pending_removals)} camera(s) ═══"
            )
            for url in list(pending_removals):
                cam_id = _cam_key(url)
                if url in running_tasks:
                    running_tasks[url].cancel()
                    del running_tasks[url]
                    logger.info(f"[WATCH] ❌ Removed cam: {cam_id}")
            pending_removals.clear()

        if pending_additions:
            logger.info(
                f"[WATCH] ═══ CYCLE BOUNDARY — activating "
                f"{len(pending_additions)} new camera(s) ═══"
            )
            for p in list(pending_additions):
                running_tasks[p["url"]] = p["task"]
                logger.info(f"[WATCH] ✅ Started cam: {p['cam_id']}")
            pending_additions.clear()

        # refresh shared reference after mutations
        active_urls_ref[0] = set(running_tasks.keys())

        if pending_removals or pending_additions:
            logger.info(f"[WATCH] 📊 Total active cameras: {len(running_tasks)}")

        now  = time.monotonic()
        dt   = now - prev_t
        prev_t = now

        total_infer, total_batches = engine.stats()
        delta_infer = total_infer - prev_inferences
        delta_batch = total_batches - prev_batches
        fps   = delta_infer / dt if dt > 0 else 0
        prev_inferences = total_infer
        prev_batches    = total_batches

        current_grabs   = stats.get("grabs",       0)
        current_fails   = stats.get("fails",       0)
        current_alerts  = stats.get("alerts_sent", 0)
        current_api_ok  = stats.get("api_ok",      0)
        current_api_fail= stats.get("api_fail",    0)

        cycle_grabs   = current_grabs   - prev_grabs
        cycle_fails   = current_fails   - prev_fails
        cycle_alerts  = current_alerts  - prev_alerts
        cycle_api_ok  = current_api_ok  - prev_api_ok
        cycle_api_fail= current_api_fail- prev_api_fail

        prev_grabs   = current_grabs
        prev_fails   = current_fails
        prev_alerts  = current_alerts
        prev_api_ok  = current_api_ok
        prev_api_fail= current_api_fail

        current_cam_count = len(active_urls_ref[0]) if active_urls_ref[0] else n_cams

        global global_cycle_num
        logger.info(
            f"[WATCH] CYCLE {global_cycle_num} COMPLETED in {dt:.2f}s | "
            f"cameras={current_cam_count} "
            f"grabs={cycle_grabs} "
            f"fails={cycle_fails} "
            f"inferences={delta_infer} "
            f"batches={delta_batch} "
            f"fps={fps:.1f} "
            f"alerts_sent={cycle_alerts} "
            f"api_ok={cycle_api_ok} "
            f"api_fail={cycle_api_fail}"
        )
        global_cycle_num += 1

# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

async def async_main():
    logger.info("[MAIN] ═══ Pipeline starting ═══")

    # 1. Load initial cameras
    if not os.path.exists(Config.CSV_FILE_PATH):
        logger.error(f"[MAIN] CSV not found: {Config.CSV_FILE_PATH}")
        sys.exit(1)

    df   = pd.read_csv(Config.CSV_FILE_PATH, header=None, encoding="utf-8-sig")
    urls = [
        u.strip() for u in df[0].dropna().unique().tolist()
        if str(u).strip().startswith(("rtmp", "rtsp", "http"))
    ]
    if not urls:
        logger.error("[MAIN] No valid camera URLs found in CSV")
        sys.exit(1)

    logger.info(f"[MAIN] Loaded {len(urls)} initial camera URLs")

    # 2. Build shared resources
    loop        = asyncio.get_event_loop()
    blob_client = build_azure_client()
    ffmpeg_pool = ThreadPoolExecutor(
        max_workers=Config.FFMPEG_POOL_WORKERS,
        thread_name_prefix="ffmpeg",
    )
    upload_pool = ThreadPoolExecutor(
        max_workers=Config.UPLOAD_POOL_WORKERS,
        thread_name_prefix="upload",
    )

    # 3. Inference engine
    engine = InferenceEngine(loop)
    engine.load_model()
    engine.start()

    # 4. Alert dispatcher
    dispatcher = AlertDispatcher(upload_pool, blob_client)

    # 5. Shared stats dict
    stats: dict = {}

    # 6. Shutdown event + cycle boundary event
    shutdown       = asyncio.Event()
    # pending queues — csv_monitor writes, watchdog reads & applies at each tick
    _pending_additions: list = []
    _pending_removals:  set  = set()
    pending_ref  = [_pending_additions]
    removals_ref = [_pending_removals]

    def _handle_signal(signum, frame):
        logger.info(f"[MAIN] Signal {signum} received — shutting down gracefully")
        loop.call_soon_threadsafe(shutdown.set)

    signal.signal(signal.SIGINT,  _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    # 7. running_tasks dict: url -> Task  (csv_monitor mutates this)
    running_tasks: dict = {}

    # active_urls_ref: single-element list so csv_monitor & watchdog share a
    # mutable reference without needing a global.
    active_urls_ref: list = [set()]

    # 8. Spawn initial camera coroutines (staggered)
    logger.info(f"[MAIN] Spawning {len(urls)} initial camera coroutines (staggered)...")
    n = len(urls)
    for i, url in enumerate(urls):
        stagger_delay = i * (Config.POLL_INTERVAL_SEC / n)
        cam_id        = _cam_key(url)

        async def _cam_task(u=url, delay=stagger_delay):
            if delay > 0:
                try:
                    await asyncio.wait_for(shutdown.wait(), timeout=delay)
                    return
                except asyncio.TimeoutError:
                    pass
            await camera_coroutine(u, engine, dispatcher, ffmpeg_pool, shutdown, stats)

        task = asyncio.create_task(_cam_task(), name=f"cam-{cam_id}")
        running_tasks[url] = task

    active_urls_ref[0] = set(running_tasks.keys())

    # 9. Watchdog task
    watchdog_task = asyncio.create_task(
        watchdog(
            engine, stats, shutdown, n,
            active_urls_ref, pending_ref, removals_ref, running_tasks,
        ),
        name="watchdog",
    )

    # 10. CSV monitor task (manages live add/remove, shares running_tasks dict)
    csv_task = asyncio.create_task(
        csv_monitor(
            running_tasks,
            engine, dispatcher, ffmpeg_pool,
            shutdown, stats,
            active_urls_ref,
            pending_ref,
            removals_ref,
        ),
        name="csv-monitor",
    )

    logger.info(
        f"[MAIN] All {n} initial camera tasks created. Pipeline running.\n"
        f"[MAIN] Save {Config.CSV_FILE_PATH} at any time to add/remove cameras.\n"
        f"[MAIN] Changes apply at the next cycle boundary (~{Config.WATCHDOG_INTERVAL_SEC}s)."
    )

    # 11. Wait until shutdown
    await shutdown.wait()

    logger.info("[MAIN] Shutdown signal received — cancelling tasks...")
    for t in list(running_tasks.values()):
        t.cancel()
    watchdog_task.cancel()
    csv_task.cancel()

    all_tasks = list(running_tasks.values()) + [watchdog_task, csv_task]
    await asyncio.gather(*all_tasks, return_exceptions=True)

    ffmpeg_pool.shutdown(wait=False)
    upload_pool.shutdown(wait=False)

    logger.info("[MAIN] ═══ Pipeline stopped cleanly ═══")


def main():
    # High-Performance System Tweaks
    try:
        resource.setrlimit(resource.RLIMIT_NOFILE, (65535, 65535))
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
        logger.info("[MAIN] System limits optimized: NOFILE=65535, CORE=0")
    except Exception as e:
        logger.warning(f"[MAIN] Failed to optimize system limits: {e}")

    # Use uvloop if available
    try:
        import uvloop
        asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
        logger.info("[MAIN] uvloop enabled")
    except ImportError:
        pass

    asyncio.run(async_main())


if __name__ == "__main__":
    main()