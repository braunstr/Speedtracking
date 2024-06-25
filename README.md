# Vehicle Speed Estimation using YOLO and Supervision

This project utilizes YOLO (You Only Look Once) for object detection and the Supervision library for video processing to estimate vehicle speeds based on detected bounding boxes in a video.

## Overview

The system performs the following tasks:
- Object detection using YOLO to identify vehicles in a video stream.
- Perspective transformation to align the detected bounding boxes with a target region of interest.
- Speed estimation based on the tracked movement of vehicles across frames.

## Features

- **Object Detection**: Utilizes YOLO for real-time vehicle detection.
- **Perspective Transformation**: Transforms bounding box coordinates to a standardized output frame.
- **Speed Estimation**: Calculates vehicle speed based on tracked movement over time.
- **Dynamic Visualization**: Visualizes bounding boxes with colors indicating vehicle speed ranges (green for low, yellow for moderate, red for high speeds).
- **Video Output**: Saves annotated video with speed-annotated bounding boxes as `result.mp4`.

## Requirements

- Python 3.x
- OpenCV (cv2)
- NumPy
- Ultralytics YOLO (yolov8x.pt)
- Supervision library

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/braunstr/Speedtracking.git
   cd <repository_directory>

2. Installation dependencies:
   
   `pip install -r requirements.txt`

   Make sure to install Ultralytics YOLO according to their instructions.

## Usage

 1. Prepare your video file for analysis. Supported format: MP4
    
 2. Run the main script:
    
    `python main.py --source_video_path ./media/input.mp4`

 3. During excecution:

    Press `q` to stop the video playback and close the application.

 4. After completion:

    Find the annotated video output as result.mp4 in the project directory

## Configuration

* Adjust the SOURCE attribute with the coordinates of the desired polygonzone in your frame. If the exact coordinates are unknown, tools like https://roboflow.github.io/polygonzone/ can be used.

* Adjust TARGET_WIDTH and TARGET_HEIGHT to the actual Width and Height of the section of the Highway within your Polygonzone

## Notes
Ensure your environment supports GPU acceleration for optimal performance with YOLO.

This project requires a video file with a clear view of the vehicles to estimate speed accurately.

## Credits
YOLO by Ultralytics for object detection.

Supervision library for video processing utilities.

<p>&nbsp;</p>

*Feel free to contribute to this project by submitting issues or pull requests.*
