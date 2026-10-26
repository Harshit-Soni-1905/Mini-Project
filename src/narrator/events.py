"""Event-based narration manager.

Detects meaningful changes in the scene (new objects, proximity changes,
urgency escalations) and selects the single most important event to announce,
while respecting global and per-track cooldowns.
"""

from __future__ import annotations

import logging
import math
import time
from collections import deque
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import TYPE_CHECKING, Sequence

from narrator.config import EventsConfig
from narrator.direction import Direction
from narrator.distance import Proximity
from narrator.phrasing import format_spoken_alert

if TYPE_CHECKING:
    from narrator.prioritizer import PrioritizedObstacle

logger = logging.getLogger(__name__)


class EventType(str, Enum):
    """Types of events that can trigger narration."""

    NEW_OBJECT = "new_object"
    PROXIMITY_ESCALATION = "proximity_escalation"
    URGENCY_ESCALATION = "urgency_escalation"
    PERIODIC_REMINDER = "periodic_reminder"


def _current_monotonic_time() -> float:
    return time.monotonic()


@dataclass(frozen=True)
class NarrationEvent:
    """A single event ready to be announced."""

    track_id: int | None
    label: str
    direction: Direction
    proximity: Proximity
    event_type: EventType
    urgency_score: float
    speech_text: str
    created_at: float = field(default_factory=_current_monotonic_time)
    track_key: int | str | None = None


def check_trend_consistency(
    values: Sequence[float],
    *,
    min_samples: int = 3,
    min_delta: float = 0.05,
    min_r2: float = 0.60,
    min_absolute_growth: float = 0.0,
    min_relative_growth: float = 0.0,
    max_sign_reversals: int = 2,
) -> bool:
    """Check if a sequence of measurements exhibits a statistically consistent increasing trend.

    Used to detect increasing collision risk while rejecting:
    - NaN, Inf, and missing observations.
    - High-frequency sign oscillations characteristic of walking gait pitch jitter (+ - + -).
    - Minor bounding-box jitter on stationary obstacles.

    Parameters
    ----------
    values : Sequence[float]
        Sequence of relative depth or bounding-box height observations.
    min_samples : int
        Minimum number of valid samples required (default 3).
    min_delta : float
        Minimum required difference between last and first valid sample (default 0.05).
    min_r2 : float
        Minimum coefficient of determination R^2 of linear regression (default 0.60).
    min_absolute_growth : float
        Minimum absolute growth required (e.g. 6.0 pixels for bounding-box height).
    min_relative_growth : float
        Minimum relative growth required (e.g. 0.10 for 10% height increase).
    max_sign_reversals : int
        Maximum permitted sign reversals in consecutive differences (default 2).
        Excessive reversals (>= 3) indicate gait pitch oscillation rather than sustained approach.

    Returns
    -------
    bool
        True if the sequence demonstrates a consistent increasing trend.
    """
    valid = [
        float(v)
        for v in values
        if v is not None and not math.isnan(v) and not math.isinf(v)
    ]
    if len(valid) < min_samples:
        return False

    delta = valid[-1] - valid[0]
    if delta < min_delta:
        return False

    if min_absolute_growth > 0.0 and delta < min_absolute_growth:
        return False

    if min_relative_growth > 0.0:
        if valid[0] <= 0.0 or valid[-1] < valid[0] * (1.0 + min_relative_growth):
            return False

    # Gait pitch jitter rejection: count consecutive sign reversals (+ -> - or - -> +)
    reversals = 0
    prev_sign = 0
    for i in range(1, len(valid)):
        diff = valid[i] - valid[i - 1]
        if abs(diff) < 1e-6:
            continue
        sign = 1 if diff > 0 else -1
        if prev_sign != 0 and sign != prev_sign:
            reversals += 1
        prev_sign = sign

    if reversals > max_sign_reversals:
        return False

    # Linear regression slope and R^2 goodness-of-fit
    n = len(valid)
    mean_x = (n - 1) / 2.0
    mean_y = sum(valid) / float(n)

    ss_xx = 0.0
    ss_yy = 0.0
    ss_xy = 0.0
    for i, y in enumerate(valid):
        dx = float(i) - mean_x
        dy = y - mean_y
        ss_xx += dx * dx
        ss_yy += dy * dy
        ss_xy += dx * dy

    if ss_xx <= 0.0 or ss_yy <= 0.0:
        return False

    slope = ss_xy / ss_xx
    if slope <= 0.0:
        return False

    r2 = (ss_xy * ss_xy) / (ss_xx * ss_yy)
    if r2 < min_r2:
        return False

    return True


@dataclass
class TrackRecord:
    """Internal state maintained for each tracked object."""

    track_id: int | None
    label: str
    direction: Direction
    proximity: Proximity
    urgency_score: float
    first_seen: float
    last_seen: float
    last_spoken: float = -1.0  # -1.0 means "never spoken" (sentinel)
    last_spoken_proximity: Proximity | None = None
    last_spoken_direction: Direction | None = None
    last_spoken_urgency: float = 0.0
    last_offered: float = -1.0  # -1.0 means "never offered" (sentinel)
    consecutive_frames: int = 0
    frames_since_spoken: int = 0
    last_cx: float = 0.0

    # Temporal history for spatial awareness and smoothing
    recent_directions: deque[Direction] = field(default_factory=lambda: deque(maxlen=5))
    recent_proximities: deque[Proximity] = field(default_factory=lambda: deque(maxlen=5))
    recent_depths: deque[float] = field(default_factory=lambda: deque(maxlen=10))
    recent_heights: deque[float] = field(default_factory=lambda: deque(maxlen=10))
    smoothed_direction: Direction | None = None
    smoothed_proximity: Proximity | None = None

    def has_increasing_collision_risk(self) -> bool:
        """Evaluate if recent observations show a statistically robust increase in collision risk.

        IMPORTANT SENSORY LIMITATION NOTE:
        On monocular RGB cameras without ego-motion compensation (IMU or visual odometry),
        forward movement of the user produces visual expansion and downward ground-plane
        drift identical to an obstacle approaching a stationary user.
        Therefore, this assessment detects *increasing collision risk* along the user's
        heading, NOT a definitive physical claim that the external object is actively moving.

        Robustness filters:
        - Rejects NaN, Inf, and missing depth/height observations.
        - Applies linear regression slope and R^2 goodness-of-fit (>= 0.60).
        - Rejects high-frequency sign reversals characteristic of walking gait pitch jitter.
        - Enforces dual thresholding on bounding-box height growth (>= 10% AND >= 6px absolute growth)
          to prevent detector bounding-box noise on stationary objects from triggering alerts.
        """
        # 1. Depth-based trend
        if len(self.recent_depths) >= 3:
            if check_trend_consistency(
                self.recent_depths,
                min_samples=3,
                min_delta=0.05,
                min_r2=0.60,
                max_sign_reversals=2,
            ):
                return True

        # 2. Bounding-box height expansion trend
        if len(self.recent_heights) >= 3:
            if check_trend_consistency(
                self.recent_heights,
                min_samples=3,
                min_delta=6.0,
                min_absolute_growth=6.0,
                min_relative_growth=0.10,
                min_r2=0.60,
                max_sign_reversals=2,
            ):
                return True

        return False

    def is_consistently_approaching(self) -> bool:
        """Backward-compatible alias for has_increasing_collision_risk."""
        return self.has_increasing_collision_risk()


def compute_smoothed_direction(
    current_direction: Direction,
    norm_x: float,
    previous_direction: Direction | None,
    *,
    left_boundary: float = 0.33,
    right_boundary: float = 0.67,
    hysteresis: float = 0.03,
) -> Direction:
    """Compute direction with spatial hysteresis to prevent zone boundary jitter.

    Requires movement past the hysteresis margin to switch zones.
    """
    if previous_direction is None:
        return current_direction

    if previous_direction == Direction.LEFT:
        # To leave LEFT and enter CENTER, must cross boundary + hysteresis
        if norm_x >= (left_boundary + hysteresis):
            return Direction.CENTER
        return Direction.LEFT

    elif previous_direction == Direction.RIGHT:
        # To leave RIGHT and enter CENTER, must cross boundary - hysteresis
        if norm_x <= (right_boundary - hysteresis):
            return Direction.CENTER
        return Direction.RIGHT

    else:  # Direction.CENTER (ahead)
        if norm_x <= (left_boundary - hysteresis):
            return Direction.LEFT
        elif norm_x >= (right_boundary + hysteresis):
            return Direction.RIGHT
        return Direction.CENTER


def compute_smoothed_proximity(
    recent_proximities: deque[Proximity],
    fallback: Proximity,
) -> Proximity:
    """Compute smoothed proximity from recent observations to filter transient noise."""
    if not recent_proximities:
        return fallback
    if len(recent_proximities) < 2:
        return recent_proximities[-1]

    # If the last 2 frames agree, accept the state immediately
    if recent_proximities[-1] == recent_proximities[-2]:
        return recent_proximities[-1]

    # If 1-frame fluctuation (e.g. [..., A, A, B]), retain stable state A
    if len(recent_proximities) >= 3 and recent_proximities[-2] == recent_proximities[-3]:
        return recent_proximities[-2]

    return recent_proximities[-1]


class EventNarrator:
    """Manages event detection and speech triggering for tracked obstacles.

    Maintains a registry of track states, detects meaningful changes,
    and applies cooldown logic to prevent speech overload.
    """

    # Proximity order for escalation detection
    PROXIMITY_ORDER = {
        Proximity.FAR: 0,
        Proximity.MEDIUM: 1,
        Proximity.NEAR: 2,
    }

    def __init__(
        self,
        config: EventsConfig | None = None,
        *,
        tracking_enabled: bool = True,
    ) -> None:
        self._config = config or EventsConfig()
        self._tracking_enabled = tracking_enabled
        self._tracks: dict[int | str | None, TrackRecord] = {}
        self._last_global_speech: float = -float("inf")
        self._global_cooldown = self._config.global_cooldown_seconds
        self._track_cooldown = self._config.track_cooldown_seconds

    def _assign_obstacle_keys(
        self, obstacles: list[PrioritizedObstacle]
    ) -> dict[int, int | str | None]:
        """Assign unique tracking keys to all obstacles in the frame.

        For tracked detections (ByteTrack active and track_id present), uses track_id.
        For untracked detections or when tracking is disabled:
          - If a single detection exists for (label, direction), uses 'label:direction'.
          - If multiple detections exist for the same (label, direction), disambiguates
            them using horizontal centroid distance matching against existing tracks,
            or assigns indexed spatial keys ('label:direction#1', etc.) to prevent
            state overwriting and history contamination between distinct nearby objects.
        """
        mapping: dict[int, int | str | None] = {}
        untracked_groups: dict[tuple[str, Direction], list[PrioritizedObstacle]] = {}

        for obs in obstacles:
            if self._tracking_enabled and obs.detection.track_id is not None:
                mapping[id(obs)] = obs.detection.track_id
            else:
                group_key = (obs.label, obs.direction)
                untracked_groups.setdefault(group_key, []).append(obs)

        for (label, direction), group in untracked_groups.items():
            base_key = f"{label}:{direction.value}"
            if len(group) == 1:
                obs = group[0]
                matched_key = None
                closest_dist = float("inf")
                for k, rec in self._tracks.items():
                    if isinstance(k, str) and (k == base_key or k.startswith(f"{base_key}#")):
                        dist = abs(obs.detection.cx - rec.last_cx)
                        if dist < closest_dist and dist < 120.0:
                            closest_dist = dist
                            matched_key = k
                mapping[id(obs)] = matched_key if matched_key is not None else base_key
            else:
                sorted_obs = sorted(group, key=lambda o: o.detection.cx)
                candidate_keys = [
                    k for k in self._tracks.keys()
                    if isinstance(k, str) and (k == base_key or k.startswith(f"{base_key}#"))
                ]
                used_keys: set[str] = set()

                # Match to existing tracks by minimum centroid distance
                for obs in sorted_obs:
                    best_key = None
                    best_dist = float("inf")
                    for k in candidate_keys:
                        if k in used_keys:
                            continue
                        rec = self._tracks[k]
                        dist = abs(obs.detection.cx - rec.last_cx)
                        if dist < best_dist and dist < 120.0:
                            best_dist = dist
                            best_key = k
                    if best_key is not None:
                        mapping[id(obs)] = best_key
                        used_keys.add(best_key)

                # Assign indexed keys for remaining unmatched detections
                sub_idx = 0
                for obs in sorted_obs:
                    if id(obs) in mapping:
                        continue
                    while True:
                        cand = base_key if sub_idx == 0 else f"{base_key}#{sub_idx}"
                        sub_idx += 1
                        if cand not in used_keys:
                            mapping[id(obs)] = cand
                            used_keys.add(cand)
                            break

        return mapping

    def _get_event_key(self, event: NarrationEvent) -> int | str | None:
        """Resolve the track key for a narration event."""
        if event.track_key is not None:
            return event.track_key
        if self._tracking_enabled and event.track_id is not None:
            return event.track_id
        return f"{event.label}:{event.direction.value}"

    def update(
        self,
        obstacles: list[PrioritizedObstacle],
        *,
        current_time: float | None = None,
        frame_width: int = 640,
    ) -> NarrationEvent | None:
        """Process current frame's obstacles and return the highest-priority event to announce.

        Args:
            obstacles: List of prioritized obstacles from current frame.
            current_time: Optional monotonic time (default: time.monotonic()).
            frame_width: Camera frame width in pixels (for direction hysteresis).

        Returns:
            The highest-priority NarrationEvent to speak, or None if no event
            should be announced this frame.
        """
        now = current_time if current_time is not None else time.monotonic()
        events: list[NarrationEvent] = []

        obs_keys = self._assign_obstacle_keys(obstacles)
        active_keys: set[int | str | None] = set(obs_keys.values())

        # Reset consecutive_frames for tracks not seen in this frame
        for key in list(self._tracks.keys()):
            if key not in active_keys:
                self._tracks[key].consecutive_frames = 0

        # Update or create track records
        for obs in obstacles:
            key = obs_keys[id(obs)]
            record = self._tracks.get(key)

            if record is None:
                record = TrackRecord(
                    track_id=obs.detection.track_id,
                    label=obs.label,
                    direction=obs.direction,
                    proximity=obs.proximity,
                    urgency_score=obs.urgency_score,
                    first_seen=now,
                    last_seen=now,
                    consecutive_frames=1,
                    smoothed_direction=obs.direction,
                    smoothed_proximity=obs.proximity,
                    last_cx=obs.detection.cx,
                )
                record.recent_directions.append(obs.direction)
                record.recent_proximities.append(obs.proximity)
                if obs.relative_depth is not None:
                    record.recent_depths.append(obs.relative_depth)
                record.recent_heights.append(obs.detection.height)
                self._tracks[key] = record
            else:
                # Check for reacquisition gap
                time_since_last_seen = now - record.last_seen
                if time_since_last_seen > 0.4:
                    # Track reappeared after an occlusion gap exceeding 0.4s.
                    # Reset temporal trend histories to avoid computing false approach
                    # deltas across temporal discontinuities, while preserving last_spoken
                    # to prevent duplicate spoken announcements.
                    record.recent_depths.clear()
                    record.recent_heights.clear()
                    record.recent_directions.clear()
                    record.recent_proximities.clear()

                # Update observation histories
                record.recent_directions.append(obs.direction)
                record.recent_proximities.append(obs.proximity)
                if obs.relative_depth is not None:
                    record.recent_depths.append(obs.relative_depth)
                record.recent_heights.append(obs.detection.height)
                record.last_cx = obs.detection.cx

                # Compute smoothed direction with spatial hysteresis
                norm_x = (obs.detection.cx / frame_width) if (frame_width > 0) else 0.5
                record.smoothed_direction = compute_smoothed_direction(
                    current_direction=obs.direction,
                    norm_x=norm_x,
                    previous_direction=record.smoothed_direction,
                )

                # Compute smoothed proximity
                record.smoothed_proximity = compute_smoothed_proximity(
                    record.recent_proximities,
                    fallback=obs.proximity,
                )

                # Check for proximity escalation
                if self._config.announce_proximity_changes:
                    baseline_prox = (
                        record.last_spoken_proximity
                        if record.last_spoken_proximity is not None
                        else record.proximity
                    )
                    new_prox = record.smoothed_proximity or obs.proximity
                    if self.PROXIMITY_ORDER[new_prox] > self.PROXIMITY_ORDER[baseline_prox]:
                        event = self._make_event(
                            obs,
                            EventType.PROXIMITY_ESCALATION,
                            direction=record.smoothed_direction,
                            proximity=record.smoothed_proximity,
                            track_key=key,
                        )
                        if self._should_announce(event, now):
                            events.append(event)
                        else:
                            logger.debug(
                                "Proximity escalation for %s (%s -> %s) throttled/cooling down",
                                obs.label, baseline_prox.value, new_prox.value,
                            )

                # Check for direction change into path / center
                effective_dir = record.smoothed_direction or obs.direction
                if (
                    record.last_spoken_direction is not None
                    and record.last_spoken_direction != effective_dir
                    and (effective_dir == Direction.CENTER or obs.is_in_walking_path)
                ):
                    event = self._make_event(
                        obs,
                        EventType.URGENCY_ESCALATION,
                        direction=effective_dir,
                        proximity=record.smoothed_proximity,
                        track_key=key,
                    )
                    if self._should_announce(event, now):
                        events.append(event)
                    else:
                        logger.debug(
                            "Direction shift for %s (%s -> %s) throttled/cooling down",
                            obs.label, record.last_spoken_direction.value, effective_dir.value,
                        )

                # Check for urgency escalation (significant increase compared to baseline)
                baseline_urgency = (
                    record.last_spoken_urgency
                    if (record.last_spoken >= 0.0 and record.last_spoken_urgency > 0.0)
                    else record.urgency_score
                )
                if obs.urgency_score > baseline_urgency * 1.3:
                    event = self._make_event(
                        obs,
                        EventType.URGENCY_ESCALATION,
                        direction=record.smoothed_direction,
                        proximity=record.smoothed_proximity,
                        track_key=key,
                    )
                    if self._should_announce(event, now):
                        events.append(event)
                    else:
                        logger.debug(
                            "Urgency escalation for %s (%.1f -> %.1f) throttled/cooling down",
                            obs.label, baseline_urgency, obs.urgency_score,
                        )

                # Update record
                record.direction = record.smoothed_direction or obs.direction
                record.proximity = record.smoothed_proximity or obs.proximity
                record.urgency_score = obs.urgency_score
                record.last_seen = now
                record.consecutive_frames += 1

        # Check for new object confirmations (after min_frames_to_confirm)
        if self._config.announce_new_objects:
            already_event_keys = {
                self._get_event_key(e)
                for e in events
            }
            for key, record in self._tracks.items():
                if key not in active_keys or key in already_event_keys:
                    continue
                if record.last_spoken >= 0.0:
                    continue
                # Track has persisted for at least min_frames_to_confirm frames.
                # Use >= so that if a NEW_OBJECT event was blocked by global cooldown
                # on the frame it first became eligible, it remains eligible on subsequent
                # frames until the cooldown expires and the event can be emitted.
                if record.consecutive_frames >= self._config.min_frames_to_confirm:
                    # Find the obstacle matching this track
                    for obs in obstacles:
                        if obs_keys[id(obs)] == key:
                            event = self._make_event(obs, EventType.NEW_OBJECT, track_key=key)
                            if self._should_announce(event, now):
                                events.append(event)
                            break

        # Periodic reminder for critical near obstacles or persistent in-path obstacles (every ~6 seconds)
        allowed_classes = self._config.resolve_alert_classes()
        for obs in obstacles:
            key = obs_keys[id(obs)]
            record = self._tracks.get(key)
            if record and record.last_spoken >= 0.0:
                is_near = (record.proximity == Proximity.NEAR)
                is_in_path_med = (obs.is_in_walking_path and record.proximity == Proximity.MEDIUM)
                if is_near or is_in_path_med:
                    time_since_spoken = now - record.last_spoken
                    if (
                        (allowed_classes is None or obs.label in allowed_classes)
                        and time_since_spoken >= 6.0
                        and self._should_announce_global(now)
                    ):
                        event = self._make_event(obs, EventType.PERIODIC_REMINDER, track_key=key)
                        events.append(event)

        # Prune stale tracks (not seen for grace period)
        self._prune_stale_tracks(now, active_keys)

        # Select single highest-priority event
        if events:
            events.sort(key=self._event_priority_tuple, reverse=True)
            chosen = events[0]
            # Note: We do NOT record as spoken here. The pipeline will call
            # record_spoken() only after the speaker actually accepts the utterance.
            return chosen

        return None

    def _get_track_key(self, obs: PrioritizedObstacle) -> int | str | None:
        """Get the key used to track this object internally."""
        if self._tracking_enabled and obs.detection.track_id is not None:
            return obs.detection.track_id
        # Fallback: spatial key based on label + direction
        return f"{obs.label}:{obs.direction.value}"

    def _make_event(
        self,
        obs: PrioritizedObstacle,
        event_type: EventType,
        *,
        direction: Direction | None = None,
        proximity: Proximity | None = None,
        track_key: int | str | None = None,
    ) -> NarrationEvent:
        """Create a NarrationEvent from an obstacle and event type."""
        eff_dir = direction or obs.direction
        eff_prox = proximity or obs.proximity

        if eff_dir != obs.direction or eff_prox != obs.proximity:
            effective_obs = replace(obs, direction=eff_dir, proximity=eff_prox)
        else:
            effective_obs = obs

        # Use phrasing module for structured, priority-aware speech formatting
        speech = format_spoken_alert(effective_obs, event_type=event_type)

        return NarrationEvent(
            track_id=obs.detection.track_id,
            label=obs.label,
            direction=eff_dir,
            proximity=eff_prox,
            event_type=event_type,
            urgency_score=obs.urgency_score,
            speech_text=speech,
            track_key=track_key,
        )

    def _event_priority_tuple(self, event: NarrationEvent) -> tuple[int, int, float]:
        """Return a sort key that prevents person-starvation of new objects.

        Priority tiers (higher = more important):
          4: PROXIMITY_ESCALATION (dynamic approaching hazard, e.g. FAR -> MED, MED -> NEAR)
          3: NEW_OBJECT (new confirmed obstacles; unspoken items announced promptly)
          2: URGENCY_ESCALATION
          1: PERIODIC_REMINDER (routine reminders of already-known obstacles)

        Within a tier, unspoken obstacles (last_spoken < 0) take precedence over
        already-spoken obstacles, with urgency_score as the final tie-breaker.
        """
        if event.event_type == EventType.PROXIMITY_ESCALATION:
            tier = 4
        elif event.event_type == EventType.NEW_OBJECT:
            tier = 3
        elif event.event_type == EventType.URGENCY_ESCALATION:
            tier = 2
        elif event.event_type == EventType.PERIODIC_REMINDER:
            tier = 1
        else:
            tier = 0

        # Determine if this track has never been spoken
        key = self._get_event_key(event)
        record = self._tracks.get(key)
        unspoken = 1 if (record is not None and record.last_spoken < 0.0) else 0

        return (tier, unspoken, event.urgency_score)

    def _should_announce(self, event: NarrationEvent, now: float) -> bool:
        """Check if an event should be announced based on cooldowns and alert class filters."""
        # Class filter: check explicit alert_classes or active speech profile
        allowed_classes = self._config.resolve_alert_classes()
        if allowed_classes is not None and event.label not in allowed_classes:
            return False

        if not self._should_announce_global(now):
            return False

        key = self._get_event_key(event)
        record = self._tracks.get(key)
        if record is None:
            return True

        # Per-track cooldown: suppress only if BOTH proximity AND direction are unchanged
        if (
            record.last_spoken_proximity == event.proximity
            and record.last_spoken_direction == event.direction
            and (now - record.last_spoken) < self._track_cooldown
        ):
            return False

        # Don't re-offer the same event if it was recently offered but rejected
        # (e.g., by speaker cooldown or full queue). Use a short grace period.
        if record.last_offered >= 0.0 and (now - record.last_offered) < 0.5:
            return False

        return True

    def _should_announce_global(self, now: float) -> bool:
        """Check global speech cooldown."""
        return (now - self._last_global_speech) >= self._global_cooldown

    def _record_spoken(self, event: NarrationEvent, now: float) -> None:
        """Record that an event was spoken."""
        self._last_global_speech = now
        key = self._get_event_key(event)
        record = self._tracks.get(key)
        if record is not None:
            record.last_spoken = now
            record.last_spoken_proximity = event.proximity
            record.last_spoken_direction = event.direction
            record.last_spoken_urgency = event.urgency_score

    def record_offered(self, event: NarrationEvent, *, current_time: float | None = None) -> None:
        """Record that an event was offered to the speaker (whether accepted or rejected).

        This prevents the same event from being re-offered repeatedly when the speaker
        rejects it due to its own internal cooldown or queue full condition.
        """
        now = current_time if current_time is not None else time.monotonic()
        key = self._get_event_key(event)
        if not self._tracking_enabled or key is None:
            key = f"{event.label}:{event.direction.value}"
        record = self._tracks.get(key)
        if record is not None:
            record.last_offered = now

    def record_spoken(self, event: NarrationEvent, *, current_time: float | None = None) -> None:
        """Record that an event was successfully spoken (called after speaker accepts it)."""
        now = current_time if current_time is not None else time.monotonic()
        self._record_spoken(event, now)

    def _prune_stale_tracks(
        self, now: float, active_keys: set[int | str | None]
    ) -> None:
        """Remove tracks not seen for more than grace period."""
        grace_period = 1.5  # seconds
        to_remove = [
            key
            for key, record in self._tracks.items()
            if key not in active_keys and (now - record.last_seen) > grace_period
        ]
        for key in to_remove:
            del self._tracks[key]

    def reset(self) -> None:
        """Clear all track state."""
        self._tracks.clear()
        self._last_global_speech = -float("inf")