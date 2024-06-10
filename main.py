import cv2
import numpy as np
import csv

def detect_aruco_markers(video_file):
    cap = cv2.VideoCapture(video_file)
    if not cap.isOpened():
        print("Error: Couldn't open video file")
        return None, None

    aruco_dictionaries = [cv2.aruco.DICT_4X4_100]

    detected_markers = []
    aruco_count_per_frame = []
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_markers = []
        detected_ids = set()

        for aruco_dict_id in aruco_dictionaries:
            aruco_dict = cv2.aruco.getPredefinedDictionary(aruco_dict_id)
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
    distance = 1 / side_length  # Simplified; you should calibrate with actual measurements
    return distance

def calculate_orientation(corners, frame):
    # Calculate yaw, pitch, and roll based on the corners of the QR code
    # Assuming the camera is aligned with the Z-axis perpendicular to the QR code
    # Placeholder calculations:
    # yaw: rotation around the Y-axis
    # pitch: rotation around the X-axis
    # roll: rotation around the Z-axis

    # Calculate vectors from corners
    vector1 = corners[1] - corners[0]
    vector2 = corners[3] - corners[0]

    # Calculate roll as the angle between the x-axis and vector1
    roll = np.degrees(np.arctan2(vector1[1], vector1[0]))

    # Assuming the marker is approximately flat and perpendicular to the camera's optical axis:
    center = np.mean(corners, axis=0)
    delta_x = center[0] - frame.shape[1] / 2
    delta_y = center[1] - frame.shape[0] / 2

    yaw = np.degrees(np.arctan2(delta_x, frame.shape[1]))  # Simplified approximation
    pitch = -np.degrees(np.arctan2(delta_y, frame.shape[0]))  # Simplified approximation

    return yaw, pitch, roll

def save_detection_status_csv(output_file, detected_markers):
    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Frame", "Marker ID", "2D Coordinates", "Distance", "Yaw", "Pitch", "Roll"])

        for markers_in_frame in detected_markers:
            for frame_number, marker_id, marker_corners, distance, yaw, pitch, roll in markers_in_frame:
                frame_coordinates = ','.join([f'({x},{y})' for x, y in marker_corners])
                writer.writerow([frame_number, marker_id, frame_coordinates, distance, yaw, pitch, roll])

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


def main():
    video_file = 'challengeB.mp4'
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


if __name__ == "__main__":
    main()
