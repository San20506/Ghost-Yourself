# 👻 Ghost Yourself

**Real-Time Object Removal Video Stream**

Make anything—or anyone—disappear from your camera feed in real time. Watch yourself vanish. Erase the clutter. Delete the past.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-4.8+-green.svg)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Segmentation-orange.svg)
![LaMa](https://img.shields.io/badge/LaMa-Inpainting-purple.svg)

## ✨ Features

- **Real-time object removal** - Click any detected object to make it disappear
- **YOLOv8 segmentation** - State-of-the-art object detection with pixel-precise masks
- **LaMa inpainting** - AI-powered background reconstruction
- **Background capture** - Blend inpainting with a clean reference frame
- **Performance options** - Configurable resolution and frame skipping

## 🎥 Demo

1. Start the application
2. Click on any detected object (yourself, a mug, your clutter)
3. Watch it vanish in real time as the AI fills in the background

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- Webcam
- NVIDIA GPU with CUDA (recommended for real-time performance)

### Installation

```bash
# Clone the repository
cd Ghost-Yourself

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

### Running

```bash
# Standard mode
python -m src.app

# Performance mode (lower resolution, faster)
python -m src.app --performance
```

## 🎮 Controls

| Key | Action |
|-----|--------|
| **Click** | Select/deselect object to remove |
| **D** | Toggle detection boxes visibility |
| **B** | Capture background reference |
| **R** | Reset object selection |
| **Q** / **ESC** | Quit application |

## ⚙️ Configuration

Edit `src/config/settings.py` to customize:

```python
# Video settings
VideoConfig(
    camera_index=0,      # Camera device index
    width=640,           # Frame width
    height=480,          # Frame height
    process_scale=1.0,   # Scale for processing (lower = faster)
)

# Detection settings
DetectionConfig(
    model_name="yolov8n-seg.pt",  # Model variant
    confidence_threshold=0.5,      # Detection confidence
)
```

## 📊 Performance

| Hardware | Expected FPS |
|----------|--------------|
| RTX 3060+ | 20-30 FPS |
| GTX 1060 | 10-15 FPS |
| CPU (modern) | 3-8 FPS |

### Tips for Better Performance

1. Use `--performance` flag for lower resolution
2. Capture background (B key) for better inpainting
3. Use GPU (CUDA) if available
4. Close other applications using GPU

## 🏗️ Architecture

```
src/
├── app.py                  # Main application
├── config/
│   └── settings.py         # Configuration dataclasses
├── features/
│   ├── detection/
│   │   ├── detector.py     # YOLOv8-seg wrapper
│   │   └── tracker.py      # Object selection & tracking
│   ├── inpainting/
│   │   └── inpainter.py    # LaMa inpainting wrapper
│   └── video/
│       └── pipeline.py     # Video capture & processing
└── common/
    └── utils.py            # Shared utilities
```

## 🔧 Technical Details

### Detection

Uses **YOLOv8n-seg** (nano segmentation model) from Ultralytics:
- Fast inference for real-time use
- Provides both bounding boxes and segmentation masks
- 80 COCO classes including people, animals, vehicles, objects

### Inpainting

Uses **LaMa** (Large Mask Inpainting) via `simple-lama-inpainting`:
- State-of-the-art inpainting quality
- Fast Fourier Convolutions for efficiency
- Handles large missing regions well

### Tracking

Simple centroid-based tracking:
- Matches selected object across frames by class + proximity
- Handles object movement smoothly

## 🤔 Philosophy

> *"What does it mean to erase something in real time?"*

This project inverts the typical computer vision paradigm. Instead of **finding** things, we make them **unfindable**. The psychological weight of watching yourself disappear—or erasing the clutter from your desk—creates a visceral, memorable experience.

## 📝 License

MIT License - See LICENSE file for details.

## 🙏 Acknowledgments

- [Ultralytics](https://github.com/ultralytics/ultralytics) for YOLOv8
- [LaMa](https://github.com/advimman/lama) for the inpainting model
- [simple-lama-inpainting](https://github.com/enesmsahin/simple-lama-inpainting) for the Python wrapper
