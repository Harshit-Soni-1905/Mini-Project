"""Web-compatible speech adapter for Smart Obstacle Narrator.

Replaces server-side pyttsx3 audio playback with an in-memory, thread-safe
event queue that captures spoken alerts emitted by EventNarrator / Pipeline.
Alerts can then be displayed as live captions and synthesized client-side
by the visitor's browser via the Web Speech API (window.speechSynthesis).
"""

from __future__ import annotations

import collections
import logging
import queue
import threading
import time

logger = logging.getLogger(__name__)


class WebSpeechBridge:
    """Thread-safe speech sink implementing the Speaker interface for web sessions.

    Parameters
    ----------
    max_queue_size : int, optional
        Maximum number of unconsumed speech alerts to retain (default 10).
    history_len : int, optional
        Maximum number of recent alerts to retain in history (default 25).
    """

    def __init__(self, max_queue_size: int = 10, history_len: int = 25) -> None:
        self._queue: queue.Queue[tuple[str, float, bool]] = queue.Queue(maxsize=max_queue_size)
        self._history: collections.deque[tuple[str, float, bool]] = collections.deque(maxlen=history_len)
        self._lock = threading.Lock()
        self._latest_alert: str | None = None
        self._latest_time: float = 0.0
        self._is_alive: bool = True

    def speak(self, text: str, urgent: bool = False) -> bool:
        """Accept a speech utterance from Pipeline / EventNarrator.

        Parameters
        ----------
        text : str
            Formatted alert phrase to announce.
        urgent : bool, optional
            Whether this alert is high-priority (e.g. Near-proximity escalation).

        Returns
        -------
        bool
            True if the utterance was accepted, False if bridge is stopped or empty text.
        """
        if not self._is_alive or not text or not text.strip():
            return False

        now = time.monotonic()
        clean_text = text.strip()

        with self._lock:
            self._latest_alert = clean_text
            self._latest_time = now
            self._history.append((clean_text, now, urgent))

        try:
            self._queue.put_nowait((clean_text, now, urgent))
            return True
        except queue.Full:
            # If the queue is full (e.g. slow client polling), evict the oldest
            # alert so the user always hears/sees the freshest event.
            try:
                self._queue.get_nowait()
                self._queue.put_nowait((clean_text, now, urgent))
                return True
            except Exception:
                return False

    def pop_new_alerts(self) -> list[str]:
        """Drain and return all new unconsumed speech alerts.

        Returns
        -------
        list[str]
            List of alert strings emitted since the last call.
        """
        alerts: list[str] = []
        while not self._queue.empty():
            try:
                item = self._queue.get_nowait()
                alerts.append(item[0])
            except queue.Empty:
                break
        return alerts

    def get_latest_alert(self) -> str | None:
        """Return the most recently spoken alert text."""
        with self._lock:
            return self._latest_alert

    def get_history(self) -> list[tuple[str, float, bool]]:
        """Return a snapshot of recent alert history as (text, timestamp, urgent)."""
        with self._lock:
            return list(self._history)

    @property
    def is_worker_alive(self) -> bool:
        """True if the speech sink is active and accepting alerts."""
        return self._is_alive

    @property
    def queue_size(self) -> int:
        """Current number of unconsumed alerts in queue."""
        return self._queue.qsize()

    @property
    def last_queue_latency_ms(self) -> float:
        """Simulated queue wait duration for health monitoring compatibility."""
        return 0.0

    @property
    def last_playback_duration_ms(self) -> float:
        """Simulated playback duration for health monitoring compatibility."""
        return 0.0

    def stop(self) -> None:
        """Shut down the speech bridge and reject further inputs."""
        self._is_alive = False
