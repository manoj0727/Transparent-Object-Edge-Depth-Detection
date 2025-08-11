# Transparent Object Detection

**Advanced glass and transparent object detection using physics-based computer vision**

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Add your image
Place your image in this folder as:
- `input.jpg` or `input.png`
- `test.jpg` or `test.png`

### 3. Run detection
```bash
python app.py
```

## What it does

Detects transparent objects (glass, windows, bottles) using:
- Edge disruption analysis
- Refraction pattern detection  
- Gradient analysis
- Texture variance
- Gabor filters
- Phase congruency
- Depth estimation

## Output

The app creates a comprehensive visualization showing:
- Original image with detected objects (green/yellow boxes)
- Detection confidence scores
- Binary mask
- Depth estimation
- Edge analysis
- Gradient maps
- Texture analysis
- Combined features
- Detection statistics

Results are saved in the `results/` folder.

## Example

If no input image is found, the app automatically creates a demo image to show how it works.

## Requirements

- Python 3.8+
- OpenCV
- NumPy
- SciPy
- scikit-image
- Matplotlib