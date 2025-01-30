from setuptools import setup, find_packages

setup(
    name="yolov8-segmentation-rex",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        'ultralytics>=8.0.137',
        'torch>=2.0.1',
        'opencv-python>=4.8.0',
        'numpy>=1.25.1',
        'matplotlib>=3.7.2'
    ],
    author="Your Name",
    description="YOLOv8-based leaf segmentation package",
    python_requires=">=3.8",
)