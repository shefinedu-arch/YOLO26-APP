# YOLO26 Object Detection Studio

A Streamlit web application powered by **Ultralytics YOLO26** for real-time object detection across images, recorded videos, and live webcam feeds.

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-FF4B4B)
![YOLO26](https://img.shields.io/badge/YOLO-v8%2Fv11-brightgreen)

---

## Key Features

- **Image Detection:** Upload images (`.jpg`, `.png`, `.jpeg`), view bounding boxes, and inspect class-level confidence scores in a table.
- **Video Detection:** Upload video files (`.mp4`, `.mov`), generate annotated previews, and view per-class confidence aggregates.
- **Live Webcam:** Real-time object detection via stream (`streamlit-webrtc`).
- **Sidebar Controls:** Dynamically adjust confidence and IoU thresholds or switch model sizes (Nano, Small, Medium).
- **Graceful Error Handling:** Handled edge cases for unsupported file types, missing webcam devices, model download failures, and memory limits.

---

## Project Structure

```text
.
├── app.py              # Main Streamlit application
├── requirements.txt    # Project dependencies
├── .gitignore          # Excluded files (weights, virtualenvs, cache)
└── README.md           # Project documentation