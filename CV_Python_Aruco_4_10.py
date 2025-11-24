# This script is an entry point to the Aruco marker detection and pose estimation.
# It uses the camera calibration values to estimate the pose of the markers.
# The script will display the original image and the image with the detected markers and their pose.
# It is using the OpenCV library 4.10+ which has the latest Aruco functions

import cv2
import cv2.aruco as aruco
import numpy as np
import time # We will use this to ensure a steady processing rate


# Load the camera calibration values
camera_calibration = np.load('Sample_Calibration.npz')
CM=camera_calibration['CM'] #camera matrix
dist_coef=camera_calibration['dist_coef']# distortion coefficients from the camera

# Define the ArUco dictionary and parameters
marker_size = 65
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
parameters = aruco.DetectorParameters()

# Define a processing rate
processing_period = 0.25

# Create two OpenCV named windows
cv2.namedWindow("Frame", cv2.WINDOW_AUTOSIZE)
cv2.namedWindow("Gray", cv2.WINDOW_AUTOSIZE)

# Position the windows next to each other
cv2.moveWindow("Gray", 640, 100)
cv2.moveWindow("Frame", 0, 100)
# Start capturing video
cap = cv2.VideoCapture(1)

# simple in-memory tracker for pixel positions (id -> list of (x,y))
positions_history = {}
max_history = 30

# Set the starting time
start_time = time.time()
fps = 0

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cv2.imshow('gray-image', gray)

    # Detect markers
    corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)
    


    # If markers are detected
    if ids is not None:
        # Draw detected markers
        frame = aruco.drawDetectedMarkers(frame, corners, ids)

        # Estimate pose of each marker
        rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(corners, marker_size, CM, dist_coef)

        # Iterate with index so we can read corners and ids
        for i, (rvec, tvec) in enumerate(zip(rvecs, tvecs)):
            # Draw axis for each marker
            frame = cv2.drawFrameAxes(frame, CM, dist_coef, rvec, tvec, 100)

            # Compute pixel centre of the marker from its corners
            c = corners[i].reshape((4, 2))
            center_px = tuple(map(int, c.mean(axis=0)))

            # Marker id (safe cast)
            marker_id = int(ids[i][0])

            # Draw a small circle at the centre and label with id + pixel coords
            cv2.circle(frame, center_px, 6, (0, 0, 255), -1)
            cv2.putText(frame, f"ID:{marker_id} ({center_px[0]},{center_px[1]})",
                        (center_px[0] + 10, center_px[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

            # Save to history for simple tracking (keep last max_history positions)
            hist = positions_history.setdefault(marker_id, [])
            hist.append(center_px)
            if len(hist) > max_history:
                hist.pop(0)

        # Draw trails for each tracked marker
        for marker_id, hist in positions_history.items():
            for p in hist:
                cv2.circle(frame, p, 2, (0, 255, 255), -1)



    # Add the frame rate to the image
    cv2.putText(frame, f"CAMERA FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, f"PROCESSING FPS: {1/processing_period:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow('Frame', frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # Ensure a steady processing rate
    elapsed_time = time.time() - start_time
    fps = 1 / elapsed_time
    if elapsed_time < processing_period:
        time.sleep(processing_period - elapsed_time)
    start_time = time.time()



# When everything is done, release the capture and close windows
cap.release()
cv2.destroyAllWindows()