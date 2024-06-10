# Aruco QR Code Detection in Video

This project focuses on detecting aruco QR codes in a video, calculating their distance and orientation relative to the camera, and saving the results in a CSV file. Additionally, the detected QR codes are annotated in an output video file.

## Algorithm Explanation

The main steps of the algorithm are as follows:

1. **Video Capture**:
   - Open the video file using OpenCV's VideoCapture.

2. **ArUco Marker Detection**:
   - Use OpenCV's aruco module to detect QR codes in each frame.
   - The aruco module provides functions to detect markers, estimate their positions, and draw the detected markers on the frame.

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

## How to Run

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
    python main.py
```
4. when running the program prints will be shown of how many aruco codes were found in each frame
```bash
Total aruco markers detected in frame 465: 2
Total aruco markers detected in frame 466: 2
Total aruco markers detected in frame 467: 3
Total aruco markers detected in frame 468: 2
Total aruco markers detected in frame 469: 1
Total aruco markers detected in frame 470: 1
Total aruco markers detected in frame 471: 2
Total aruco markers detected in frame 472: 0
Total aruco markers detected in frame 473: 0
```
### Output Formats

#### CSV:
- Columns: Frame ID, QR id, QR 2D coordinates (left-up, right-up, right-down, left-down), QR 3D (distance, yaw, pitch, roll)

this is an example on the 'challengeB.mp4' file, if more then 1 marker found in a frame, another line of frame with differnt marker id will be represented.
![alt text](https://i.imgur.com/6sBRvjA.jpeg)

#### Video:
- Annotated with detected QR codes in a green rectangular frame with their IDs.
![alt text](https://i.imgur.com/1MgWvRb.jpeg)
![alt text](https://i.imgur.com/Ne8xM8T.jpeg)
### Performance:
The code is designed to run in real-time, processing each frame in less than 30 milliseconds on a standard computer.

## Contributors

- [Lior Jerbi](https://github.com/LiorJerbi) - 314899493
- [Yael Rosen](https://github.com/yaelrosen77) - 209498211
- [Tomer Klugman](https://github.com/tomerklugman) - 312723612
- [Hadas Evers](https://github.com/hadasevers) - 206398984

