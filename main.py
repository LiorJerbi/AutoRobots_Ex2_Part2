import cv2
import numpy as np
import csv
import time

# Global variables to track last instruction and time
last_instruction = None
last_instruction_time = 0

# Camera calibration parameters for the Tello camera
camera_matrix = np.array([[920.8, 0, 620.5],
                          [0, 920.8, 360.5],
                          [0, 0, 1]])
dist_coeffs = np.array([0.115, -0.108, 0, 0, 0])  # Example distortion coefficients

def detect_aruco_markers(video_file):
    cap = cv2.VideoCapture(video_file)
    if not cap.isOpened():
        print("Error: Couldn't open video file")
        return None, None

    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_1000)

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
                    distance = calculate_distance(marker_corners)
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

def calculate_distance(corners):
    # Estimate the distance based on the size of the detected QR code
    side_length = np.linalg.norm(corners[0] - corners[1])
    distance = 1 / side_length
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

    def get_next_frame_number(csv_file, current_frame_number):
        with open(csv_file, newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            found_current = False
            for row in reader:
                frame_num = int(row['Frame'])
                if found_current:
                    return frame_num
                if frame_num == current_frame_number:
                    found_current = True
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

    cap = cv2.VideoCapture(1)  # Use 0 for the default camera or 1 for Iriun webcam turn your phone into webcam
    if not cap.isOpened():
        print("Error: Couldn't open live video.")
        return

    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_1000)
    aruco_params = cv2.aruco.DetectorParameters()

    last_direction = None
    last_marker_center = None
    last_print_time = 0

    aruco_detected = False
    found = False
    while not found:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=aruco_params)

        if ids is not None:
            for i in range(len(ids)):
                marker_id = ids[i][0]
                if marker_id == target_marker_id:
                    marker_corners = corners[i][0]
                    current_yaw, current_pitch, current_roll = calculate_orientation(marker_corners, frame)
                    yaw_diff = target_yaw - current_yaw
                    pitch_diff = target_pitch - current_pitch

                    direction = get_direction(yaw_diff, pitch_diff)

                    marker_center = np.mean(marker_corners, axis=0)
                    if direction != last_direction or last_marker_center is None or np.linalg.norm(
                            marker_center - last_marker_center) > 10:
                        print(direction)
                        last_direction = direction
                        last_marker_center = marker_center
                        last_print_time = time.time()  # Update last print time
                        if direction == "Hold position":
                            print("Aruco Marker Detected!")
                            while True:
                                ret, frame = cap.read()
                                if not ret:
                                    break

                                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                                corners, ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=aruco_params)

                                if ids is not None:
                                    for i in range(len(ids)):
                                        marker_id = ids[i][0]
                                        if marker_id == target_marker_id:
                                            marker_corners = corners[i][0]
                                            current_distance = calculate_distance(marker_corners)
                                            # print('current distance:' + str(current_distance))
                                            # print("target distance:" + str(target_distance))
                                            if np.isclose(current_distance, target_distance, atol=0.01):
                                                aruco_detected = True
                                                found = True
                                                break
                                            elif current_distance > target_distance:
                                                print("Move forward.")
                                                # Adjust camera position (move forward)
                                            else:
                                                print("Move backward.")
                                                # Adjust camera position (move backward)
                                            time.sleep(3)  # Adjust sleep time as needed
                                    if found:
                                        break
                        break  # Exit the loop once direction is printed for the current frame

        # Check if ArUco marker is not detected and give instructions based on target frame
        if not aruco_detected and (last_direction is None or time.time() - last_print_time >= 5):
            suggested_direction = "Rotate camera to the right."  # Initial suggestion
            if last_direction != suggested_direction:
                print(suggested_direction)
                last_direction = suggested_direction
                last_print_time = time.time()  # Update last print time

        # Interval printing logic
        if time.time() - last_print_time >= 5:
            if not aruco_detected:
                print(last_direction if last_direction else "Hold position.")
            last_print_time = time.time()

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    if found:
        next_frame_number = get_next_frame_number(csv_file, target_frame)
        if next_frame_number is not None:
            print(f"Moving to next frame: {next_frame_number}")
            give_directions(csv_file, next_frame_number)

def get_direction(yaw_diff, pitch_diff):
    yaw_threshold = 5
    pitch_threshold = 5

    if abs(yaw_diff) > yaw_threshold:
        if yaw_diff > 0:
            return "Turn left."
        else:
            return "Turn right."
    elif abs(pitch_diff) > pitch_threshold:
        if pitch_diff < 0:
            return "Tilt up."
        else:
            return "Tilt down."
    else:
        return "Hold position"

def calculate_orientation(corners, frame):
    vector1 = corners[1] - corners[0]
    vector2 = corners[3] - corners[0]

    roll = np.degrees(np.arctan2(vector1[1], vector1[0]))

    center = np.mean(corners, axis=0)
    delta_x = center[0] - frame.shape[1] / 2
    delta_y = center[1] - frame.shape[0] / 2

    yaw = np.degrees(np.arctan2(delta_x, frame.shape[1]))
    pitch = -np.degrees(np.arctan2(delta_y, frame.shape[0]))

    return yaw, pitch, roll

def main():
    video_file = 'target_vid2.mp4'
    output_file = 'detection_status.csv'
    output_video_file = 'output.mp4'

    # Detect ArUco markers
    detected_markers, _ = detect_aruco_markers(video_file)

    if detected_markers:
        # Save detection status
        save_detection_status_csv(output_file, detected_markers)

        # Annotate video with detected ArUco markers
        annotate_video(video_file, output_video_file, detected_markers)

        print("Detection status saved to", output_file)
        print("Annotated video saved to", output_video_file)
    else:
        print("No ArUco markers detected in the video")

    target_frame = ''
    # Give directions based on the current live video frame and the target frame
    while target_frame != 'q':
        target_frame = input("Enter target frame number or press q to exit: ")
        if target_frame != 'q':
            target_frame = int(target_frame)
            give_directions(output_file, target_frame)

if __name__ == "__main__":
    main()