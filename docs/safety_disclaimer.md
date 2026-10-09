# Safety Disclaimer

## Research Prototype — Not a Safety Device

**Smart Obstacle Narrator** is an experimental, proof-of-concept application
developed for academic and personal exploration of assistive computer vision.

### This software is NOT:

- A medical device
- A certified assistive technology product
- A replacement for a white cane or guide dog
- A substitute for professional orientation-and-mobility training
- Safe for use as a primary navigation aid in any environment

### Known limitations:

- **Missed detections**: Object detection models routinely fail to detect
  transparent surfaces (glass doors, windows), thin objects (wires, cables,
  poles), low-contrast obstacles, and ground-level hazards (curbs, potholes,
  steps, uneven surfaces).

- **False positives**: The model may report objects that are not actually
  present, or misclassify harmless objects as hazards.

- **Imprecise distance**: Monocular distance estimation (whether from
  bounding-box heuristics or learned depth models) is inherently imprecise
  and can be wrong by large margins.  There is no calibrated depth sensor.

- **Imprecise direction**: Direction is estimated from the object's position
  in the image, not from true 3D bearing.  It is relative to the camera's
  field of view, not to the user's body orientation.

- **Latency**: There is an unavoidable delay between an obstacle appearing
  and the system announcing it.  Fast-moving hazards may not be announced
  in time.

- **Environmental conditions**: Performance degrades significantly in low
  light, rain, fog, glare, or crowded scenes.

- **Single camera**: A single forward-facing camera has a limited field of
  view.  Obstacles to the side, above, or behind the user are invisible.

- **Audio-only output**: If the user cannot hear the announcements (noisy
  environment, hearing impairment), the system provides no benefit.

### Usage responsibility:

Anyone who uses or demonstrates this software must understand that it is a
**research prototype with unreliable output**.  It must never be presented
to a visually impaired person as a dependable navigation tool without
clearly communicating its limitations and ensuring that conventional
mobility aids remain in use.

### Liability:

The authors of this software accept no responsibility for injury, damage,
or loss arising from its use.  Use entirely at your own risk.
