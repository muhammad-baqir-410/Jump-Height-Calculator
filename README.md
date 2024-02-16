# Jump Height Calculation Project

## Overview
This project is designed to calculate the jump height of individuals using video analysis. By leveraging the YOLOv8 algorithm to detect and track keypoints (specifically, hips and ankles) of a person jumping, the project calculates the jump height based on the trajectory of these keypoints. This tool is particularly useful in sports science, physical education, and personal fitness to measure and analyze jump performance.

![View Demo Video](assets/demo1.mp4)
![View Demo Video](assets/demo2.mp4)

## Features
- Uses YOLOv8 for accurate keypoint detection.
- Analyzes jumps from video input, requiring the individual to jump only once in the frame.
- Calculates jump height based on the detected keypoints' trajectory.
- Visualizes the keypoints and jump trajectory on the video for review.

## Installation

- Python 3.x
- OpenCV
- Ultralytics


## Usage

1. Place your test video(s) in the `test_videos` directory. Ensure the video contains a single jump by an individual.
2. Run the main script with the command:
   ```
   python main.py
   ```
3. The script will process the video, calculate the jump height, and display the output with keypoints and jump trajectory overlaid on the original video. The results are also saved for further analysis.

## How It Works
1. **Video Input**: A video file is input to the system, which reads the video frame by frame.
2. **Keypoint Detection**: Utilizes YOLOv8 to detect and track the keypoints of hips and ankles across all frames of the jump.
3. **Jump Analysis**: Analyzes the trajectory of the detected keypoints to calculate the jump height.
4. **Visualization**: Outputs the analyzed video with overlaid keypoints and trajectory information for review and validation.
