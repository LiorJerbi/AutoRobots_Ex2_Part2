# QR Code Detection in Video

This project focuses on detecting QR codes in a video, calculating their distance and orientation relative to the camera, and saving the results in a CSV file. Additionally, the detected QR codes are annotated in an output video file.

## Algorithm Explanation

The main steps of the algorithm are as follows:

1. **Video Capture**:
   - Open the video file using OpenCV's `VideoCapture`.

2. **ArUco Marker Detection**:
   - Use OpenCV's `aruco` module to detect QR codes in each frame.
   - The `aruco` module provides functions to detect markers, estimate their positions, and draw the detected markers on the frame.

3. **Distance Calculation**:
   - Calculate the distance of the QR code from the camera based on the size of the detected QR code.
   - A simplified method is used here, which should be calibrated with actual measurements for more accurate results.

4. **Orientation Calculation**:
   - Calculate the yaw, pitch, and roll of the QR code based on its corners.
   - These values represent the rotation of the QR code around the camera's coordinate axes.

5. **Save Results to CSV**:
   - Save the frame number, QR code ID, 2D coordinates of the QR code corners, distance, yaw, pitch, and roll to a CSV file.

6. **Annotate and Save Video**:
   - Annotate each frame with the detected QR codes and their IDs.
   - Save the annotated frames to an output video file.

## How to Run the Program

### Prerequisites

- Python 3.x
- OpenCV library
- OpenCV Contrib library

### Installation

1. Clone the repository or download the code.
2. Install the required Python packages:
   ```bash
   pip install opencv-python
   pip install opencv-contrib-python
   pip install numpy
   ```

### Running the Program
1. Place your input video file in the same directory as the code or provide the correct path to the video file.
- If you want to test the program on another video, in line 128 in main.py change the variable name to the video name.
```bash
    video_file = 'challengeB.mp4'
```
3. Run the program:
```bash
    python3 main.py
```

### Output Formats

#### CSV:
- Columns: Frame ID, QR id, QR 2D coordinates (left-up, right-up, right-down, left-down), QR 3D (distance, yaw, pitch, roll)


#### Video:
- Annotated with detected QR codes in a green rectangular frame with their IDs.

### Performance:
The code is designed to run in real-time, processing each frame in less than 30 milliseconds on a standard computer.
