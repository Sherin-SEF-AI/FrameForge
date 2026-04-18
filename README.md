# FrameForge

**Auto-Labeling and Pseudo-Labeling Studio for Autonomous Driving Datasets**

FrameForge is a GPU-accelerated desktop application that turns raw dashcam footage into production-ready training datasets. Load a video, run a segmentation model, and export annotated frames in YOLO or COCO format in minutes rather than weeks.

Built specifically for the challenges of unstructured Indian road conditions: dense mixed traffic, unpredictable lane behavior, pedestrians, cyclists, livestock, and everything in between.

---

## Why FrameForge

Manual annotation is the single biggest bottleneck in autonomous driving research. A typical 60-minute GoPro recording contains over 100,000 frames. At a professional labeling speed of 2 minutes per frame, that is 3,400 hours of human work.

FrameForge cuts that number down by running state-of-the-art segmentation models directly on your footage and generating pseudo-labels automatically. An engineer then reviews and corrects rather than drawing from scratch, reducing annotation time by up to 90%.

---

## Key Features

**GPU-Optimized Inference**
Runs YOLOv11 and FastSAM in FP16 precision on CUDA, comfortably within 4 GB VRAM. Automatic CPU fallback on out-of-memory errors ensures the application never crashes mid-session.

**Seek-Based Video Access**
Frames are read on demand using OpenCV random access. The entire video never loads into RAM, making hour-long 4K recordings as easy to handle as short clips.

**Dual Export Formats**
Save annotations in YOLO segmentation format for Ultralytics training pipelines, or in standard COCO JSON for Detectron2, MMDetection, and any other COCO-compatible framework.

**Non-Blocking Architecture**
Model loading, inference, and full-video export all run in background threads. The interface stays responsive at all times.

**Real-Time Hardware Monitoring**
The status bar shows live VRAM consumption, inference FPS, and device name so you always know how hard the GPU is working.

**Dark Professional Interface**
Built with PyQt6 and a custom dark Fusion theme. Designed for long annotation sessions without eye strain.

---

## Screenshot

```
+------------------+--------------------------------------------------+
|  File            |                                                  |
|  [Open Video]    |                                                  |
|  road_01.mp4     |          [ Video Frame with Overlay ]            |
|                  |                                                  |
|  Model           |                                                  |
|  yolo11n-seg.pt  |                                                  |
|  [Load Model]    |                                                  |
|  Loaded on CUDA  +--------------------------------------------------+
|                  |  [============================-------]  Scrubber |
|  Inference       |  Frame 142 / 3600                                |
|  [x] Auto-Infer  |  [|< ] [< ] [<] [>] [>] [>|]  [> Play]         |
|  Conf: 0.35      +--------------------------------------------------+
|  IoU:  0.45      |  Device: CUDA (RTX 4050) | VRAM: 312 MB | FPS: 28|
|                  +--------------------------------------------------+
|  Export          |
|  [Save YOLO]     |
|  [Save COCO]     |
|  [Export All]    |
|                  |
|  Log             |
|  [14:03:21] ...  |
+------------------+
```

---

## Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| GPU | Any CUDA device | NVIDIA RTX 3060 or newer |
| VRAM | 4 GB | 6 GB or more |
| RAM | 8 GB | 16 GB or more |
| CPU | Quad-core | Octa-core |
| OS | Linux / Windows / macOS | Ubuntu 22.04 LTS |

CPU-only inference is supported but significantly slower. For production use a CUDA-capable GPU is strongly recommended.

---

## Installation

**Step 1. Clone the repository**

```bash
git clone https://github.com/Sherin-SEF-AI/FrameForge.git
cd FrameForge/frameforge
```

**Step 2. Create a virtual environment**

```bash
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
```

**Step 3. Install PyTorch with CUDA support**

```bash
# CUDA 12.1 (RTX 40-series, RTX 30-series)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# CPU only
pip install torch torchvision
```

**Step 4. Install remaining dependencies**

```bash
pip install -r requirements.txt
```

**Step 5. Launch**

```bash
python3 main.py
```

---

## Supported Models

| Model | Size | Speed | Best For |
|-------|------|-------|----------|
| yolo11n-seg.pt | 6 MB | Very fast | Real-time preview, low VRAM |
| yolo11s-seg.pt | 22 MB | Fast | Balanced accuracy and speed |
| FastSAM-s.pt | 23 MB | Fast | Instance segmentation focus |

Models are downloaded automatically by Ultralytics on first load. Place custom `.pt` files in the working directory and they will be detected by the model selector.

---

## Workflow

```
1. Open Video          Load any MP4, AVI, MOV, or MKV file
2. Set Output Dir      Choose where frames and labels will be saved
3. Load Model          Select a model and click Load Model
4. Preview             Scrub through frames, run inference on demand
5. Export              Save individual frames or run Export All Frames
```

**Output structure**

```
output_dir/
    images/
        frame_000000.jpg
        frame_000001.jpg
        ...
    labels/
        frame_000000.txt     (YOLO format)
        frame_000001.txt
        ...
    coco/
        annotations.json     (COCO format, accumulated)
```

---

## Export Formats

**YOLO Segmentation Format**

Each `.txt` file contains one detection per line:

```
class_id  x1 y1 x2 y2 x3 y3 ...   (polygon, normalised 0-1)
```

Falls back to bounding-box format when no mask is available:

```
class_id  cx cy width height       (normalised 0-1)
```

**COCO JSON Format**

Standard COCO object detection schema with polygon segmentation, area computed from contour, and COCO-format `[x, y, w, h]` bounding boxes. Compatible with Detectron2, MMDetection, COCO API, and LabelStudio import.

---

## GPU Memory Management

FrameForge is engineered to stay within strict VRAM budgets:

- FP16 inference via `model.half()` halves memory consumption compared to FP32
- Fixed inference resolution of 640x640 regardless of source video resolution
- `torch.cuda.empty_cache()` called after every inference pass
- Automatic fallback to CPU inference on `OutOfMemoryError` with a single retry
- CUDNN benchmark mode enabled for maximum throughput after the first run
- Warm-up forward pass on model load eliminates first-inference latency spikes

---

## Dependencies

```
PyQt6          >= 6.6.0
opencv-python  >= 4.9.0
ultralytics    >= 8.2.0
torch          >= 2.2.0
torchvision    >= 0.17.0
numpy          >= 1.26.0
Pillow         >= 10.2.0
```

---

## Roadmap

The following features are planned for upcoming releases:

- Manual correction tools: click to delete detections, redraw polygons, reassign class labels
- Frame sampling with configurable stride to avoid redundant near-identical frames
- Class filter panel to include or exclude specific YOLO categories from exports
- Confidence histogram and per-class detection statistics
- CVAT and LabelStudio XML export for annotation review pipelines
- Tracking-assisted labeling using ByteTrack for consistent object IDs across frames
- Multi-video batch export mode
- Active learning queue that flags low-confidence frames for priority human review

---

## Project Structure

```
frameforge/
    main.py               Entry point, QApplication, dark theme
    gui.py                MainWindow, FrameViewer, worker threads
    inference_engine.py   GPU inference wrapper, FP16, OOM safeguards
    video_handler.py      OpenCV seek-based frame access
    export_handler.py     YOLO and COCO annotation export
    requirements.txt      Python dependencies
```

---

## Contributing

Contributions are welcome. Please open an issue before submitting a pull request so the proposed change can be discussed first.

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Commit your changes: `git commit -m "Add your feature"`
4. Push to the branch: `git push origin feature/your-feature`
5. Open a pull request

Please follow PEP 8 style, include docstrings on all public methods, and test on both CPU and CUDA paths before submitting.

---

## Contributors

| Name | Role | Profile |
|------|------|---------|
| Sherin Joseph Roy | Creator and Lead Developer | [github.com/Sherin-SEF-AI](https://github.com/Sherin-SEF-AI) |

Contributions, bug reports, and feature requests are tracked on the [Issues](https://github.com/Sherin-SEF-AI/FrameForge/issues) page.

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

## Author

**Sherin Joseph Roy**
Principal Engineer, Safety-Enhanced AI Systems
[github.com/Sherin-SEF-AI](https://github.com/Sherin-SEF-AI)

---

*FrameForge is purpose-built for the realities of Indian road data: unstructured environments, extreme class diversity, and the need to move fast without sacrificing annotation quality.*
