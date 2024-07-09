# Aruco QR Code Detection in Video & Live video directions 

- You can find the 1st part of the assignment and explanation about it in this link: https://github.com/tomerklugman/AutoRobots_Ex2
- In this part of the exercise we focus on live directions from pre annotated video we explain in the 1st part
- This program input is a pre-recorded video of catching QR codes in a space, and it will provide navigation directions for a live video in the same space of the previous video. 

## Setup for Using Iriun Webcam
1. Download and install the Iriun app on your computer from [here](https://iriun.com).
2. Download and install the Iriun app on your phone from the App Store or Google Play Store.
3. Ensure that both your computer and phone are connected to the same WiFi network.
4. Follow the instructions in the Iriun app to connect your phone's camera to your computer.

## Navigate Using Live Feed
The script gives navigation directions based on a live video feed and the data from the CSV file.
- When running the program it will run the 1st part on a given video, and then it will ask for a specific frame number that include QR code from the live video (exist in the outputted CSV file).
- Then, it will open your computer webcam or the Iriun webcam (if you have it running on your phone) and will direct you toward the specific frame you inputted using the data (yaw/distance/roll) from the CSV file.
- You can change the setting of the camera you want to use (Computer camera/ Iriun Webcam) by changing the input number in line 161:
```bash
    cap = cv2.VideoCapture(1)  # Use 1 for Iriun webcam, 0 for default camera
```
- After the target frame is given and directed you toward it, the program will not move to the next frames in the CSV file. It will only give navigation directions to the specified frame.


## Running the Program

1. Ensure you have all dependencies installed. If not, you can install them using:
    ```bash
    pip install opencv-python numpy
    ```

2. Modify the size of the Aruco marker according to your recording in lines 22 and 165:
    ```bash
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_100)
    ```
    Here, we used `4X4_100`.

3. In line number 292 in the `main.py` file, you can enter the desired video to work on:
    ```bash
    video_file = 'target_vid2.mp4'
    ```

4. Run the script:
    ```bash
    python main.py
    ```

5. When prompted, enter the target frame number you want to navigate towards. This should be a frame number that includes a QR code from the live video.

6. The program will then open a window with a live video feed from your computer webcam or the Iriun webcam app (if it's running on your phone). Instructions will be printed on the video feed, guiding you towards the target frame.

7. To quit the program, press the 'q' key while the live video window is open.

### Direction Example:
![alt text](https://i.imgur.com/UNMzzjD.png)


## Contributors

- [Lior Jerbi](https://github.com/LiorJerbi) - 314899493
- [Yael Rosen](https://github.com/yaelrosen77) - 209498211
- [Tomer Klugman](https://github.com/tomerklugman) - 312723612
- [Hadas Evers](https://github.com/hadasevers) - 206398984

