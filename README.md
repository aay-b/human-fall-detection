# human-fall-detection
A deep learning–powered system that detects human falls in real-time video streams using skeleton-based pose estimation and LSTM sequence modeling.

**Objective**

This project was built as a demonstration of applied computer vision, deep learning, and real-time data processing. It can analyze human motion patterns from video input and classify between normal activities (e.g., walking) and abnormal events (falls). It analyzes video streams to detect falls in real time and triggers alerts, aiming to improve safety for elderly and vulnerable individuals. This project showcases AI for social good, blending vision and sequence modeling.

---

**Features**

- Pose Estimation: Uses MediaPipe to extract 3D skeleton keypoints from human body movements.

- Temporal Modeling: An LSTM (Long Short-Term Memory) neural network analyzes sequences of skeleton frames to capture motion dynamics.

- Binary Classification: Detects whether an activity is a “fall” or “walk” with high confidence.

- Real-Time Processing: Supports live video input or pre-recorded video files.

- Visual Feedback: Overlays results on the video with banners (“WALKING” / “FALL”) for clear status updates.

- Alert System: (Optional) Plays an alert sound when a fall is detected.

- Logging: Exports per-frame metrics (probabilities, decisions, and states) to CSV for analysis.

---

**Technical Overview**

*Data Representation*

- Skeletons are represented as 25 key joints × 3 coordinates (x, y, z).
- Input sequences are normalized (hip-centered, torso-scaled) for robustness.

*Model Architecture*

- LSTM-based neural network
  - Input size: 150 (25 joints × 3 coordinates × 2 bodies)
  - Hidden size: 256
  - Layers: 3
  - Output: Probability of Fall vs. Walk

*Decision Pipeline*

1. Extract skeletons from each frame.
2. Normalize and store in a rolling sequence window.
3. Pass window through LSTM → get per-frame fall probability.
4. Apply smoothing, hysteresis, and dwell-time logic to avoid false positives.
5. Trigger alert & update HUD when a fall is confirmed.
