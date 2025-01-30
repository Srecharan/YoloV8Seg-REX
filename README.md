# YOLOv8 Leaf Segmentation (REX)

## Overview
A high-performance plant leaf segmentation system using YOLOv8, designed for robotic applications. This system provides accurate instance segmentation of individual leaves with confidence scoring, enabling precise leaf identification for robotic grasping and manipulation tasks.

## Features
- Instance segmentation of individual leaves
- High-confidence leaf detection (90%+ accuracy)
- Multi-leaf tracking with unique IDs
- Batch processing capability
- Clear visualization with distinct leaf coloring
- Confidence score reporting
- Support for various image formats

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/YoloV8Seg-REX.git
cd YoloV8Seg-REX

# Create conda environment (recommended)
conda env create -f environment.yml
conda activate yolo-v8

# Or install with pip
pip install -r requirements.txt
```

## Quick Start

```python
from src.inference.segmentation import LeafSegmentation

# Initialize segmenter
segmenter = LeafSegmentation()

# Process single image
results = segmenter.segment_image("path/to/image.jpg", visualize=True)

# Access results
masks = results['masks']           # Individual leaf masks
scores = results['scores']         # Confidence scores
viz = results['visualization']     # Visualization image
```

## Example Output
![Example Segmentation](examples/results/left_rect27_results.png)

## Performance
- Processing time: ~60ms per image
  - Preprocessing: ~8ms
  - Inference: ~52ms
  - Postprocessing: ~3-6ms
- Detection confidence threshold: 0.7
- Average confidence score: >0.90

## Applications
- Robotic leaf grasping
- Plant phenotyping
- Automated plant monitoring
- Agricultural robotics

## Model Details
- Base model: YOLOv8
- Input resolution: 832x1088
- Trained on custom leaf dataset
- Optimized for real-time performance

## Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

## License
[MIT License](LICENSE)

## Citation
If you use this work in your research, please cite:
```bibtex
@software{yolov8_leaf_segmentation_rex,
    title={YOLOv8 Leaf Segmentation for Robotic Applications},
    author={Your Name},
    year={2024},
    url={https://github.com/yourusername/YoloV8Seg-REX}
}
```