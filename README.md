# FrameForge

**The annotation tool built for roads that don't follow rules.**

Most auto-labeling tools were built for clean datasets — structured intersections, painted lanes, predictable traffic. Indian roads are none of those things. FrameForge was built from the ground up for the chaos: stray cattle crossing highways, auto-rickshaws occupying three lanes at once, potholes that swallow wheels, construction debris spread across unmarked roads, and pedestrians who treat the carriageway as a footpath.

If you are building perception systems for autonomous vehicles, driver assistance, or dashcam analytics for Indian conditions — this tool was made for you.

https://github.com/user-attachments/assets/096cef8d-7759-47e9-a6f7-1dc84f92aa05

---

## What FrameForge actually does

You have dashcam footage. You need labeled training data. The gap between those two things is where most autonomous driving projects die.

Professional annotation services charge $10–30 per image for polygon-level segmentation. A single hour of 30fps footage is 108,000 frames. The math does not work.

FrameForge runs multiple state-of-the-art vision models directly on your footage, generates pseudo-labels automatically, and gives you a full correction and review interface so a single engineer can clean up what the model got wrong. The result: annotation throughput that would otherwise require a team of fifty.

---

## The pipeline

```
Raw dashcam footage
        |
        v
YOLO11-seg  -->  instant bounding boxes + polygon masks
        |
        v
SAM Box Refiner  -->  pixel-perfect mask from every YOLO box
(MobileSAM or SAM vit_b/l/h)
        |
        v
Grounded SAM  -->  zero-shot detection of Indian-specific classes
(Grounding DINO + SAM, text-prompted: "pothole . auto rickshaw . cattle .")
        |
        v
SegFormer  -->  full scene semantic segmentation, every pixel classified
        |
        v
Human review  -->  delete, redraw, reassign, undo, zoom, navigate
        |
        v
Export  -->  YOLO seg / COCO JSON / CVAT XML / LabelStudio JSON / Semantic PNG
        |
        v
Train  -->  one-click YOLO fine-tuning on your labeled data, streamed live
        |
        v
Load trained model back into FrameForge, repeat
```

This is not a labeling tool that assumes a model will get everything right. It is a complete active-learning loop.

---

## Features

### Inference

**YOLO11 + FastSAM Instance Segmentation**
Runs yolo11n-seg, yolo11s-seg, yolo11m-seg, yolo11l-seg, and FastSAM in FP16 on CUDA. Configurable confidence threshold, IoU threshold, and inference resolution (320–1280). Automatic FP32 CPU fallback on VRAM overflow — the app never crashes, it adapts.

**Grounded SAM — Zero-Shot Text-Prompted Detection**
Load Grounding DINO and SAM together. Type any class names as a text prompt ("speed bump . cattle . construction debris .") and the model finds them with no prior training. The full 16-class Indian road taxonomy is pre-loaded as the default prompt.

**SAM Box Refiner — Auto-Segment YOLO Boxes**
Run YOLO first for speed, then click "Auto-Segment Boxes" to feed every detected bounding box into SAM and get tight pixel-accurate masks in one pass. Supports both MobileSAM (10x faster, ~40 MB, auto-downloaded) and original SAM vit_b/l/h. Results are undoable.

**SegFormer Semantic Segmentation**
Full scene pixel-level classification using HuggingFace SegFormer. Every pixel in the frame gets a class label. Supports Cityscapes and ADE20K variants. Run on the current frame or queue the entire video for batch processing — results stream in frame by frame without accumulating in memory.

<img width="1920" height="1080" alt="image" src="https://github.com/user-attachments/assets/ef2287c0-bc7b-4e31-9310-d35e6f1576fd" />

**Batch Semantic Segmentation**
"Run All Frames" processes the entire video in the background. GPU cache is flushed between frames to prevent VRAM OOM. A live progress counter shows frames processed. Cancel at any time.

### Annotation and Correction

**Per-Frame Annotation Store**
Every inference result is stored in memory keyed by frame index. Navigate freely — your annotations are always there when you return to a frame.

**Undo / Redo (20 levels per frame)**
Every annotation mutation — delete, reassign, draw box, draw polygon, SAM refine — is undoable. Ctrl+Z / Ctrl+Y. Per-frame undo stacks, not a global one.

**Manual Box Drawing**
Switch to Draw Box mode and drag to create a new detection. A dialog prompts for class and confidence. SAM Refiner converts it to a tight mask immediately if loaded.

**Manual Polygon Drawing**
Click to place vertices, right-click to remove the last vertex, double-click to close. The polygon is saved as a proper segmentation mask.

**Click-to-Select and Delete**
Click any detection in the frame viewer to select it, see its class and confidence, and delete or reassign its class with one button press.

**Class Reassignment**
Select a detection, click Reassign Class, pick from the full taxonomy list. The class ID, name, and color update immediately.

**Zoom and Pan**
Scroll wheel to zoom centered on cursor. Middle-click drag to pan. Reset with a button. Essential for reviewing potholes and small obstacle annotations.

### Intelligence

**16-Class Indian Road Taxonomy**
A purpose-built class set that reflects the actual objects encountered on Indian roads:

| ID | Class | ID | Class |
|----|-------|----|-------|
| 0 | drivable_road | 8 | auto_rickshaw |
| 1 | pothole | 9 | truck |
| 2 | speed_bump | 10 | bus |
| 3 | puddle | 11 | tractor |
| 4 | pedestrian | 12 | cattle |
| 5 | cyclist | 13 | dog |
| 6 | motorcyclist | 14 | construction_debris |
| 7 | car | 15 | barrier |

**Active Learning Queue**
Scans the entire labeled dataset — both in-memory results and exported label files — and flags frames for priority human review. Configurable confidence threshold. Flags: empty detections, low average confidence, high detection count (crowded scenes), and in-memory-only frames not yet exported. One-click navigation to any flagged frame. CSV export of the flag list.

**Confidence Histogram**
Visual distribution of detection confidence scores across the current frame. Immediately shows whether the model is uncertain or confident about its predictions.

**Dataset Statistics**
Per-class detection counts across all labeled frames, visualized as a bar chart. Spots class imbalance before it becomes a training problem.

### Export

**YOLO Segmentation Format**
One `.txt` file per frame, one detection per line, polygon vertices normalized 0–1. Falls back to bounding-box format when no mask is available. Ready for `yolo train`.

**COCO JSON**
Standard COCO schema with polygon segmentation, computed area, and COCO bounding boxes. Compatible with Detectron2, MMDetection, the COCO evaluation API, and LabelStudio import.

**CVAT XML 1.1**
Annotation XML for direct import into CVAT for team review workflows.

**LabelStudio JSON**
Task-format JSON for LabelStudio. Includes polygon annotations and metadata.

**Semantic PNG**
Two files per frame: a raw label-ID grayscale PNG (each pixel value is the class ID, compatible with mmsegmentation and SegFormer training pipelines) and a BGR color visualization for human review.

**Export All Frames**
Runs YOLO inference on every Nth frame (configurable stride), exports images and labels, and applies the active class filter. Runs in a background thread with a live progress bar.

### Training Loop

**One-Click YOLO Fine-Tuning**
The single most important feature for anyone building a custom model:

1. Label frames with FrameForge
2. Set base model, epochs, batch size, image size
3. Click Start Training
4. `data.yaml` is auto-generated from the taxonomy
5. Training runs as a subprocess, output streams line by line into the app
6. On completion, FrameForge offers to load `best.pt` directly back into the inference engine

Click Start Training again while running to cancel. This is the feedback loop that turns generic YOLO weights into a model that understands Indian roads.

### Session Management

**Session Save / Load**
Serialize the entire annotation session — every frame's detections, class IDs, confidences, and masks — to a `.ffses` file. Masks are base64-encoded PNG. Resume exactly where you left off.

**Review Mode**
Table of all exported label files with frame numbers, detection counts, and one-click seek-to-frame navigation. Identifies frames that were exported with zero detections.

### Interface

**Keyboard Shortcuts**

| Key | Action |
|-----|--------|
| I | Run inference on current frame |
| S | Save YOLO label for current frame |
| Ctrl+Z | Undo |
| Ctrl+Y / Ctrl+Shift+Z | Redo |
| Scroll wheel | Zoom in/out |
| Middle-click drag | Pan |

**View Mode Toggle**
Switch between Instance view (YOLO/G-SAM overlay with per-detection colors and labels) and Semantic view (full-scene SegFormer colormap) with one click.

**Frame Sampling / Stride**
Set stride to process every 5th, 10th, or Nth frame. Avoids redundant near-identical frames in the exported dataset.

**Class Filter**
Include or exclude specific classes from inference display and export. Useful when a model detects irrelevant COCO classes that don't belong in your taxonomy.

**Non-Blocking Architecture**
Every heavy operation — model loading, inference, SAM refining, batch semantic, active learning scan, export, training — runs in a QThread. The interface is always responsive.

**Live Hardware Monitor**
Status bar shows: CUDA device name, VRAM consumption (MB), live inference FPS.

---

## Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| GPU | Any CUDA device | NVIDIA RTX 3060 or newer |
| VRAM | 4 GB | 6 GB or more |
| RAM | 8 GB | 16 GB or more |
| OS | Linux / Windows | Ubuntu 22.04 LTS |

CPU-only inference works for YOLO and FastSAM. Grounded SAM, SAM Box Refiner, and SegFormer are usable on CPU but impractically slow for batch work. A CUDA GPU is strongly recommended.

---

## Installation

**Step 1. Clone**

```bash
git clone https://github.com/Sherin-SEF-AI/FrameForge.git
cd FrameForge/frameforge
```

**Step 2. Virtual environment**

```bash
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
```

**Step 3. PyTorch with CUDA**

```bash
# CUDA 12.1 (RTX 40-series, RTX 30-series)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# CPU only
pip install torch torchvision
```

**Step 4. Core dependencies**

```bash
pip install -r requirements.txt
```

**Step 5. Optional: Grounded SAM and SegFormer**

```bash
pip install transformers>=4.38 timm pillow
pip install segment-anything
```

**Step 6. Optional: MobileSAM (recommended over SAM for speed)**

```bash
pip install git+https://github.com/ChaoningZhang/MobileSAM.git
```

**Step 7. SAM checkpoint (if using original SAM)**

```bash
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
```

MobileSAM downloads its checkpoint automatically on first load (~40 MB).

**Step 8. Launch**

```bash
python3 main.py
```

---

## Output Structure

```
output_dir/
    images/
        frame_000000.jpg
        frame_000001.jpg
    labels/
        frame_000000.txt        YOLO segmentation format
        frame_000001.txt
    coco/
        annotations.json        COCO format, all frames accumulated
    cvat/
        annotations.xml         CVAT XML 1.1
    labelstudio/
        tasks.json              LabelStudio import format
    semantic/
        frame_000000_labels.png  Raw class-ID grayscale PNG
        frame_000000_color.png   BGR color visualization
    data.yaml                   Auto-generated YOLO training config
```

---

## Supported Models

| Model | Size | Use Case |
|-------|------|----------|
| yolo11n-seg.pt | 6 MB | Real-time preview, low VRAM |
| yolo11s-seg.pt | 22 MB | Balanced accuracy and speed |
| yolo11m-seg.pt | 50 MB | Good accuracy, moderate VRAM |
| yolo11l-seg.pt | 87 MB | Best accuracy, 6+ GB VRAM |
| FastSAM-s.pt | 23 MB | Class-agnostic instance segmentation |
| Grounding DINO base | ~700 MB | Zero-shot, text-prompted detection |
| SAM vit_b | 358 MB | High-quality masks, slower |
| MobileSAM | 40 MB | Fast masks, recommended for labeling |
| SegFormer-b2 Cityscapes | ~100 MB | Full scene semantic segmentation |

---

## Project Structure

```
frameforge/
    main.py                Entry point, QApplication, dark Fusion theme
    gui.py                 MainWindow, FrameViewer, all worker threads
    inference_engine.py    InferenceEngine, GroundedSAMEngine,
                           SAMBoxRefiner, SemanticSegmentationEngine
    video_handler.py       OpenCV seek-based frame access
    export_handler.py      YOLO, COCO, CVAT, LabelStudio, Semantic export
    taxonomy.py            16-class Indian road taxonomy, ID/name/color maps
    requirements.txt       Python dependencies
```

---

## The Problem This Solves

The standard academic dataset for autonomous driving — KITTI, nuScenes, Waymo — was collected in Germany, the United States, and Singapore. The models trained on them fail in ways that are both predictable and dangerous when deployed on Indian roads.

There is no "car turns left at a red light" on a road with no traffic lights. There is no "pedestrian crosses at a crosswalk" on a road with no crosswalks. A model trained on BDD100K has never seen a bullock cart, a three-wheeled electric auto-rickshaw, or a pothole deep enough to damage a vehicle's suspension.

The only way to build a reliable perception system for these conditions is to label data from these conditions. FrameForge is the infrastructure for doing that at scale — without a commercial annotation team, without cloud costs, without waiting weeks for labeled batches to return.

---

## Contributing

Contributions are welcome. Open an issue before submitting a pull request so the proposed change can be discussed.

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Commit: `git commit -m "Add your feature"`
4. Push: `git push origin feature/your-feature`
5. Open a pull request

Please follow PEP 8, include docstrings on public methods, and test on both CPU and CUDA paths.

---

## Author

**Sherin Joseph Roy**
Principal Engineer, Safety-Enhanced AI Systems

GitHub: [github.com/Sherin-SEF-AI](https://github.com/Sherin-SEF-AI)
Email: sherin.joseph2217@gmail.com

---

## License

MIT License. See [LICENSE](LICENSE) for details.

---

*Built for the roads that trained drivers fear and autonomous systems ignore. FrameForge exists because the data problem comes before the model problem, and someone has to solve it.*
