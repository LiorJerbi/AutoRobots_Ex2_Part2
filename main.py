import math
from collections import deque
import cv2
import numpy as np
import csv
import time

# Global variables to track last instruction and time
last_instruction = None
last_instruction_time = 0

# Camera calibration parameters
camera_matrix = np.array([[920.8, 0, 620.5],
                          [0, 920.8, 360.5],
                          [0, 0, 1]])
dist_coeffs = np.array([0.115, -0.108, 0, 0, 0])  # Example distortion coefficients
aruco_marker_size = 0.05  # Actual size of the ArUco marker in meters


def detect_aruco_markers(video_file):
    cap = cv2.VideoCapture(video_file)
    if not cap.isOpened():
        print("Error: Couldn't open video file")
        return None, None

    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_100)
    detected_markers = []
    aruco_count_per_frame = []
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_markers = []
        detected_ids = set()

        corners, ids, _ = cv2.aruco.detectMarkers(frame, aruco_dict)

        if ids is not None:
            for i in range(len(ids)):
                marker_id = ids[i][0]
                if marker_id not in detected_ids:
                    marker_corners = corners[i][0]
                    distance = calculate_distance(marker_corners, camera_matrix)
                    yaw, pitch, roll = calculate_orientation(marker_corners, frame)
                    frame_markers.append((frame_count, marker_id, marker_corners, distance, yaw, pitch, roll))

                    detected_ids.add(marker_id)

        aruco_count = len(frame_markers)
        print(f"Total ArUco markers detected in frame {frame_count}: {aruco_count}")
        detected_markers.append(frame_markers)
        aruco_count_per_frame.append(aruco_count)
        frame_count += 1

    cap.release()
    return detected_markers, aruco_count_per_frame


def calculate_distance(corners, camera_matrix):
    # Calculate the distance based on the size of the detected ArUco marker
    object_points = np.array([
        [-aruco_marker_size / 2, aruco_marker_size / 2, 0],
        [aruco_marker_size / 2, aruco_marker_size / 2, 0],
        [aruco_marker_size / 2, -aruco_marker_size / 2, 0],
        [-aruco_marker_size / 2, -aruco_marker_size / 2, 0]
    ])

    ret, rvec, tvec = cv2.solvePnP(object_points, corners, camera_matrix, dist_coeffs)
    distance = np.linalg.norm(tvec)
    return distance


def save_detection_status_csv(output_file, detected_markers):
    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Frame", "Marker ID", "2D Coordinates", "Distance", "Yaw", "Pitch", "Roll"])

        for markers_in_frame in detected_markers:
            for frame_number, marker_id, marker_corners, distance, yaw, pitch, roll in markers_in_frame:
                frame_coordinates = ','.join([f'({x},{y})' for x, y in marker_corners])
                writer.writerow([frame_number, marker_id, frame_coordinates, distance, yaw, pitch, roll])


def read_csv(csv_file):
    data = []
    with open(csv_file, 'r') as file:
        reader = csv.reader(file)
        next(reader)  # Skip the header
        for row in reader:
            frame_number = int(row[0])
            marker_id = int(row[1])
            coordinates = np.array([list(map(float, coord.strip('()').split(','))) for coord in row[2].split('),(')])
            distance = float(row[3])
            yaw = float(row[4])
            pitch = float(row[5])
            roll = float(row[6])
            data.append((frame_number, marker_id, coordinates, distance, yaw, pitch, roll))
    return data


def annotate_video(video_file, output_video_file, detected_markers):
    cap = cv2.VideoCapture(video_file)
    if not cap.isOpened():
        print("Error: Couldn't open video file")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_file, fourcc, fps, (width, height))

    frame_count = 0
    for markers_in_frame in detected_markers:
        ret, frame = cap.read()
        if not ret:
            break

        for frame_number, marker_id, marker_corners, distance, yaw, pitch, roll in markers_in_frame:
            rect_color = (0, 255, 0)
            cv2.polylines(frame, [np.int32(marker_corners)], True, rect_color, 2)

            org = (int(marker_corners[0][0]), int(marker_corners[0][1]) - 20)
            cv2.putText(frame, str(marker_id), org, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

        print(f"{len(markers_in_frame)} aruco markers added to video in frame {frame_count}")

        out.write(frame)

        frame_count += 1

    cap.release()
    out.release()


def give_directions(csv_file, target_frame):
    def find_target_data(csv_file, frame_number):
        with open(csv_file, newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                frame_num = int(row['Frame'])
                if frame_num == frame_number:
                    return row
        return None

    target_data = find_target_data(csv_file, target_frame)
    if not target_data:
        print(f"Target frame {target_frame} not found in CSV file.")
        return

    target_marker_id = int(target_data['Marker ID'])
    target_yaw = float(target_data['Yaw'])
    target_pitch = float(target_data['Pitch'])
    target_roll = float(target_data['Roll'])
    target_distance = float(target_data['Distance'])  # Distance from the target frame

    cap = cv2.VideoCapture(1)  # Use 1 for Iriun webcam, 0 for default camera
    if not cap.isOpened():
        print("Error: Couldn't open live video.")
        return

    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_100)
    aruco_params = cv2.aruco.DetectorParameters()

    last_directions = deque(maxlen=5)  # Adjust the length as needed for stabilization
    current_direction = "Stay"

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=aruco_params)

        frame_center = (frame.shape[1] // 2, frame.shape[0] // 2)

        if ids is not None:
            for i in range(len(ids)):
                marker_id = ids[i][0]
                if marker_id == target_marker_id:
                    marker_corners = corners[i][0]
                    marker_center = np.mean(marker_corners, axis=0)
                    current_yaw, current_pitch, current_roll = calculate_orientation(marker_corners, frame)
                    current_distance = calculate_distance(marker_corners, camera_matrix)

                    yaw_diff = target_yaw - current_yaw
                    pitch_diff = target_pitch - current_pitch
                    distance_diff = target_distance - current_distance

                    direction_text = get_direction(yaw_diff, pitch_diff, distance_diff, marker_center, frame_center)

                    # Add the new direction to the queue
                    last_directions.append(direction_text)

                    # Update the current direction if it's consistent over the last few frames
                    if len(last_directions) == last_directions.maxlen and all(
                            d == last_directions[0] for d in last_directions):
                        current_direction = last_directions[0]

                    break
        else:
            distance_diff = None  # Set distance_diff to None when no marker is detected

        # Display distance difference instead of current distance
        distance_text = f"Distance Diff: {distance_diff:.2f} m" if isinstance(distance_diff,
                                                                              float) else "Distance Diff: NaN"
        cv2.putText(frame, distance_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        # Display direction instructions
        cv2.putText(frame, current_direction, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        cv2.imshow("Live Directions", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def get_direction(yaw_diff, pitch_diff, distance_diff, marker_center, frame_center):
    yaw_threshold = 5  # Adjust as needed based on your setup
    pitch_threshold = 5  # Adjust as needed based on your setup
    distance_threshold = 0.2  # Adjust as needed based on your setup

    yaw_direction = ""
    pitch_direction = ""
    distance_direction = ""
    move_direction = ""

    if abs(yaw_diff) > yaw_threshold:
        if yaw_diff > 0:
            yaw_direction = "turn left"
        else:
            yaw_direction = "turn right"

    if abs(pitch_diff) > pitch_threshold:
        if pitch_diff < 0:
            pitch_direction = "tilt down"
        else:
            pitch_direction = "tilt up"

    if distance_diff is not None and abs(distance_diff) > distance_threshold:
        if distance_diff < 0:
            distance_direction = "move forward"
        else:
            distance_direction = "move backward"

    # Determine move right/left based on marker position in frame
    if marker_center[0] < frame_center[0] - 20:  # Adjust threshold as needed
        move_direction = "move left"
    elif marker_center[0] > frame_center[0] + 20:  # Adjust threshold as needed
        move_direction = "move right"

    instructions = f"{move_direction} {yaw_direction} {pitch_direction} {distance_direction}".strip()

    return instructions if instructions else "Stay"


def calculate_orientation(corners, frame):
    dist_coeffs = np.zeros((4, 1))  # Assuming no lens distortion

    # SolvePnP to get the rotation and translation vectors
    object_points = np.array([
        [-0.5, -0.5, 0],  # Bottom-left
        [0.5, -0.5, 0],  # Bottom-right
        [0.5, 0.5, 0],  # Top-right
        [-0.5, 0.5, 0]  # Top-left
    ], dtype=float)

    # Convert corners to the appropriate format
    image_points = np.array(corners, dtype=float)

    success, rvec, tvec = cv2.solvePnP(object_points, image_points, camera_matrix, dist_coeffs)
    if not success:
        return 0.0, 0.0, 0.0

    # Convert rotation vector to rotation matrix
    rotation_matrix, _ = cv2.Rodrigues(rvec)
    roll, pitch, yaw = rotationMatrixToEulerAngles(rotation_matrix)

    return yaw, pitch, roll


def rotationMatrixToEulerAngles(R):
    sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])

    singular = sy < 1e-6

    if not singular:
        x = math.atan2(R[2, 1], R[2, 2])
        y = math.atan2(-R[2, 0], sy)
        z = math.atan2(R[1, 0], R[0, 0])
    else:
        x = math.atan2(-R[1, 2], R[1, 1])
        y = math.atan2(-R[2, 0], sy)
        z = 0

    return np.degrees(x), np.degrees(y), np.degrees(z)


video_file = "IMG_5567.mp4"
output_csv_file = "detection_status.csv"
output_video_file = "annotated_video.mp4"

detected_markers, aruco_count_per_frame = detect_aruco_markers(video_file)
save_detection_status_csv(output_csv_file, detected_markers)
annotate_video(video_file, output_video_file, detected_markers)

target_frame = int(input("Enter the target frame number you want to compare to: "))
give_directions(output_csv_file, target_frame)
