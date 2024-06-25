# Aruco QR Code Detection in Video & Live video directions 

- You can find the 1st part of the assignment and explanation about it in this link: https://github.com/tomerklugman/AutoRobots_Ex2
- In this part of the exercise we focus on live directions from pre annotated video we explain in the 1st part
- This program input is a pre-recorded video of catching QR codes in a space, and it will provide navigation directions for a live video in the same space of the previous video. 

## Navigate Using Live Feed
The script gives navigation directions based on a live video feed and the data from the CSV file.
- When running the program it will run the 1st part on a given video, and then it will ask for a specific frame number that include QR code from the live video (exist in the outputted CSV file).
- Then it will open your computer camera and will direct you toward the specific frame you inputted using the data (yaw/distance/roll) from the CSV file.
- After the target frame is given and directed you toward it, the program will run on the next frames in the CSV file one by one and will give you navigate direction to it using the yaw/distance/roll from the csv file. 

## Usage
- In line number 292 in the main.py file you can enter the desired video to work on
```bash
video_file = 'target_vid2.mp4'
```

- You can modify the size of the aruco marker according to you're recording in lines 22,165
```bash
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_100)
```
here we used 4X4_100.

### Direction Example:
![alt text](https://i.imgur.com/7a8yj8O.png)




## Contributors

- [Lior Jerbi](https://github.com/LiorJerbi) - 314899493
- [Yael Rosen](https://github.com/yaelrosen77) - 209498211
- [Tomer Klugman](https://github.com/tomerklugman) - 312723612
- [Hadas Evers](https://github.com/hadasevers) - 206398984

