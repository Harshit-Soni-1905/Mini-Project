"""Text-to-speech engine wrapper with asynchronous playback and cooldown.

Speech is executed in a background daemon thread so it never blocks
the camera capture / inference loop.  Repeated messages are suppressed
within a configurable cooldown window.
"""

from __future__ import annotations

import logging
import queue
import threading
import time

from narrator.config import SpeakerConfig

logger = logging.getLogger(__name__)


class Speaker:
    """Non-blocking text-to-speech announcer.

    Parameters
    ----------
    config : SpeakerConfig, optional
        Speech settings (rate, cooldown_seconds, enabled).
    """

    def __init__(self, config: SpeakerConfig | None = None) -> None:
        self._config = config or SpeakerConfig()
        self._queue: queue.Queue[tuple[str, float] | None] = queue.Queue(maxsize=4)
        self._last_spoken: dict[str, float] = {}
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._engine = None  # Reference to pyttsx3 engine for interruption
        self._running = False

        # Diagnostic metrics and submission tracking
        self._last_queue_latency_ms: float = 0.0
        self._last_playback_duration_ms: float = 0.0
        self._submissions_total: int = 0
        self._submissions_accepted: int = 0
        self._submissions_rejected_cooldown: int = 0
        self._submissions_rejected_queue_full: int = 0
        self._last_worker_error: str | None = None

        if self._config.enabled:
            self._start_worker()

    def _start_worker(self) -> None:
        """Start the background speech worker thread."""
        self._running = True
        self._thread = threading.Thread(
            target=self._worker_loop,
            daemon=True,
            name="SpeakerThread",
        )
        self._thread.start()

    def _worker_loop(self) -> None:
        """Background loop that drains the speech queue and speaks."""
        # Initialize the TTS engine inside the worker thread because
        # SAPI5 COM objects must be initialized and used on the same thread.
        engine = None
        com_initialized = False
        try:
            try:
                import comtypes
                comtypes.CoInitialize()
                com_initialized = True
            except Exception:
                try:
                    import pythoncom
                    pythoncom.CoInitialize()
                    com_initialized = True
                except Exception:
                    pass

            import pyttsx3

            engine = pyttsx3.init()
            engine.setProperty("rate", self._config.rate)
            self._engine = engine
        except Exception:
            logger.exception("Failed to initialize TTS engine in worker thread")

        while self._running:
            try:
                item = self._queue.get(timeout=0.2)
            except queue.Empty:
                continue

            if item is None:  # Sentinel to shut down
                logger.debug("TTS worker received shutdown sentinel")
                break

            if isinstance(item, tuple):
                text, enqueued_at = item
            else:
                text, enqueued_at = item, time.monotonic()

            queue_latency_ms = (time.monotonic() - enqueued_at) * 1000.0
            self._last_queue_latency_ms = queue_latency_ms

            logger.debug(
                "TTS worker dequeued: %r (queue_wait=%.1fms, queue_remaining=%d)",
                text,
                queue_latency_ms,
                self._queue.qsize(),
            )
            if engine is not None:
                t_play_start = time.monotonic()
                try:
                    logger.debug("TTS worker playback started: %r", text)
                    # On Windows with SAPI5, pyttsx3 sets self.proxy._busy = False upon
                    # completing runAndWait(). When the next utterance calls engine.say(),
                    # proxy._pump() pops the say command prematurely while proxy._busy is False,
                    # and subsequent runAndWait() pops engine.endLoop() immediately at the start
                    # of startLoop(), cancelling speech within ~80ms without emitting audio.
                    # Clearing any residual proxy queue and restoring proxy._busy = True ensures
                    # driver.say() is pumped cleanly by startLoop() for every utterance.
                    if hasattr(engine, "proxy"):
                        if hasattr(engine.proxy, "_queue"):
                            engine.proxy._queue.clear()
                        if hasattr(engine.proxy, "setBusy"):
                            engine.proxy.setBusy(True)
                    engine.say(text)
                    engine.runAndWait()
                    playback_dur_ms = (time.monotonic() - t_play_start) * 1000.0
                    self._last_playback_duration_ms = playback_dur_ms
                    self._last_worker_error = None
                    logger.debug(
                        "TTS worker playback finished: %r (duration=%.1fms)",
                        text,
                        playback_dur_ms,
                    )
                except Exception as exc:
                    self._last_worker_error = str(exc)
                    logger.exception("TTS playback failed for: %r", text)
                    try:
                        import pyttsx3
                        if hasattr(pyttsx3, "_activeEngines"):
                            pyttsx3._activeEngines.clear()
                        engine = pyttsx3.init()
                        engine.setProperty("rate", self._config.rate)
                        self._engine = engine
                    except Exception:
                        logger.exception("TTS engine recovery failed")
                    except Exception:
                        logger.exception("TTS engine recovery failed")

            self._queue.task_done()

        if engine is not None:
            try:
                engine.stop()
            except Exception:
                pass
            self._engine = None

        if com_initialized:
            try:
                import comtypes
                comtypes.CoUninitialize()
            except Exception:
                pass

    def clear_queue(self) -> int:
        """Clear all pending messages in the speech queue. Returns count dropped."""
        dropped = 0
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
                self._queue.task_done()
                dropped += 1
            except (queue.Empty, ValueError):
                break
        return dropped

    def interrupt(self) -> None:
        """Halt in-progress playback immediately for urgent collision alerts."""
        self.clear_queue()
        if self._engine is not None:
            try:
                self._engine.stop()
            except Exception:
                pass

    def speak(self, text: str, *, urgent: bool = False) -> bool:
        """Queue a message for speech if not on cooldown.

        Parameters
        ----------
        text : str
            Utterance text.
        urgent : bool
            If True, clears non-urgent queue, stops ongoing playback, and speaks immediately.

        Returns
        -------
        bool
            True if the message was accepted, False if suppressed
            by cooldown or if speech is disabled.
        """
        if not self._config.enabled or not text.strip():
            return False

        if urgent:
            self.interrupt()

        now = time.monotonic()
        with self._lock:
            self._submissions_total += 1
            if not urgent:
                last_time = self._last_spoken.get(text, 0.0)
                if (now - last_time) < self._config.cooldown_seconds:
                    self._submissions_rejected_cooldown += 1
                    return False  # Cooldown active, suppress duplicate

            # Non-blocking put: if the queue is full, drop the message
            # without committing cooldown timestamp so it can be retried.
            try:
                self._queue.put_nowait((text, now))
                self._last_spoken[text] = now
                self._submissions_accepted += 1
                return True
            except queue.Full:
                self._submissions_rejected_queue_full += 1
                logger.debug("Speech queue full, dropped: %r", text)
                return False

    @property
    def is_worker_alive(self) -> bool:
        """Check if background worker thread is alive."""
        return self._thread is not None and self._thread.is_alive()

    @property
    def is_healthy(self) -> bool:
        """Check if background worker thread is running and without active error."""
        return self.is_worker_alive and self._last_worker_error is None

    @property
    def queue_size(self) -> int:
        """Return the current number of messages in the speech queue."""
        return self._queue.qsize()

    @property
    def last_queue_latency_ms(self) -> float:
        """Time (ms) the last utterance waited in queue before playback started."""
        return self._last_queue_latency_ms

    @property
    def last_playback_duration_ms(self) -> float:
        """Duration (ms) of the last speech engine playback."""
        return self._last_playback_duration_ms

    @property
    def submissions_stats(self) -> dict[str, int]:
        """Summary of speech submissions."""
        return {
            "total": self._submissions_total,
            "accepted": self._submissions_accepted,
            "rejected_cooldown": self._submissions_rejected_cooldown,
            "rejected_queue_full": self._submissions_rejected_queue_full,
        }

    def is_on_cooldown(self, text: str) -> bool:
        """Check if a specific message is currently suppressed by cooldown."""
        now = time.monotonic()
        with self._lock:
            last_time = self._last_spoken.get(text, 0.0)
            return (now - last_time) < self._config.cooldown_seconds

    def stop(self) -> None:
        """Stop the speech worker thread and release resources."""
        self._running = False
        try:
            self._queue.put_nowait(None)  # shutdown sentinel
        except queue.Full:
            pass
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=1.0)
