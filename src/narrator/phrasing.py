"""Low-latency, priority-aware spoken warning phrasing policy.

Provides structured phrase generation minimizing syllable count and latency
for assistive mobility under high-stress conditions.
"""

from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING

from narrator.distance import Proximity

if TYPE_CHECKING:
    from narrator.events import EventType
    from narrator.prioritizer import PrioritizedObstacle


class AlertLevel(str, Enum):
    """Urgency level governing phrase design and preemption."""

    URGENT = "urgent"    # Collision imminent (<1s): Monosyllabic, imperative ("Stop! chair ahead")
    WARNING = "warning"  # Hazard in path (1-3s): Concise spatial alert ("near chair ahead")
    ADVISORY = "advisory"# Peripheral / low hazard: Minimal informative phrase ("person left")


def format_spoken_alert(
    obstacle: PrioritizedObstacle,
    event_type: EventType | None = None,
    *,
    alert_level: AlertLevel | None = None,
) -> str:
    """Format an alert phrase tailored for low latency and cognitive clarity.

    Parameters
    ----------
    obstacle : PrioritizedObstacle
        Obstacle to announce.
    event_type : EventType, optional
        Type of event triggering the speech.
    alert_level : AlertLevel, optional
        Explicit alert level override (auto-inferred if None).

    Returns
    -------
    str
        Concise, spoken text.
    """
    from narrator.events import EventType

    # 1. Periodic Reminder always uses descriptive reminder phrasing
    if event_type == EventType.PERIODIC_REMINDER:
        return f"still {obstacle.proximity.value} {obstacle.label} {obstacle.direction.value}"

    # Auto-infer alert level if not explicitly provided
    if alert_level is None:
        if event_type is None:
            # Standalone phrasing: urgent if near in walking path
            if obstacle.proximity == Proximity.NEAR and obstacle.is_in_walking_path:
                alert_level = AlertLevel.URGENT
            elif obstacle.proximity == Proximity.NEAR or obstacle.is_in_walking_path:
                alert_level = AlertLevel.WARNING
            else:
                alert_level = AlertLevel.ADVISORY
        else:
            # Event-driven: Do not generate an urgent "Stop!" command solely because
            # a bounding box falls inside the heuristic walking corridor or is classified as NEAR.
            # Normal events (NEW_OBJECT, URGENCY_ESCALATION, PROXIMITY_ESCALATION) map to
            # WARNING or ADVISORY based on proximity/path, keeping warnings concise and calm.
            if obstacle.proximity == Proximity.NEAR or obstacle.is_in_walking_path:
                alert_level = AlertLevel.WARNING
            else:
                alert_level = AlertLevel.ADVISORY

    # 2. Urgent (Collision Imminent)
    if alert_level == AlertLevel.URGENT:
        return f"Stop! {obstacle.label} {obstacle.direction.value}"

    # 3. Warning (Hazard in Path or Near Proximity)
    if alert_level == AlertLevel.WARNING:
        if obstacle.proximity == Proximity.NEAR:
            return f"near {obstacle.label} {obstacle.direction.value}"
        return f"{obstacle.label} {obstacle.direction.value}"

    # 4. Advisory (Peripheral / Low Hazard / Medium / Far)
    return f"{obstacle.label} {obstacle.direction.value}"
