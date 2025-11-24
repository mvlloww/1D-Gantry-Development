import cv2
import numpy as np
import pandas as pd
import time

# Aruco DeltaX Test.py
# Requirements: opencv-contrib-python, pandas, numpy
# Captures from default camera, detects ArUco markers, logs ID, marker_x and deltaX (marker_x - screen_center_x).
# Prints and highlights the marker that is closest to the screen center (smallest absolute deltaX).
# Press 'c' to quit; a CSV (aruco_log.csv) will be written on exit.


def make_aruco_dict(name=cv2.aruco.DICT_4X4_50):
    try:
        return cv2.aruco.Dictionary_get(name)
    except AttributeError:
        # newer OpenCV versions keep aruco inside cv2.aruco
        return cv2.aruco.getPredefinedDictionary(name)

def main():
    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        print("ERROR: Cannot open camera")
        return

    aruco_dict = make_aruco_dict()
    parameters = cv2.aruco.DetectorParameters()

    columns = ["timestamp", "id", "marker_x", "deltaX"]
    log = pd.DataFrame(columns=columns)

    print("Running. Press 'c' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("ERROR: Frame capture failed")
            break

        h, w = frame.shape[:2]
        screen_center_x = w / 2

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

        best_id = None
        best_deltaX = None

        # if markers detected
        if ids is not None and len(ids) > 0:
            ids = ids.flatten()
            centers = []
            for c in corners:
                # corners are (1,4,2) or (4,2)
                pts = c.reshape((4, 2))
                cx = pts[:, 0].mean()
                centers.append(cx)

            # build list of (id, cx, deltaX)
            items = []
            for marker_id, cx in zip(ids, centers):
                deltaX = cx - screen_center_x  # positive if to the right of center
                items.append((int(marker_id), float(cx), float(deltaX)))
                # draw marker
            # choose marker with smallest absolute deltaX (closest to center)
            items_sorted = sorted(items, key=lambda x: abs(x[2]))
            best_id, best_cx, best_deltaX = items_sorted[0]

            # draw all markers and annotate
            cv2.aruco.drawDetectedMarkers(frame, corners, ids.reshape(-1,1))
            for marker_id, cx, deltaX in items:
                text = f"ID:{marker_id} x:{int(cx)} dX:{deltaX:.1f}"
                # find the corresponding marker corner to place text near it
                # simple approach: put text at y = 30 + 20*idx
                cv2.putText(frame, text, (10, 30 + 20*marker_id % 400), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1)

            # highlight best marker with a circle at its center
            cv2.circle(frame, (int(best_cx), int(h/2)), 8, (0, 255, 0), -1)
            cv2.putText(frame, f"BEST ID:{best_id} dX:{best_deltaX:.1f}", (10, h-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

            # log best marker
            log = log._append({
                "timestamp": time.time(),
                "id": int(best_id),
                "marker_x": float(best_cx),
                "deltaX": float(best_deltaX)
            }, ignore_index=True)

            # print to console
            print(f"BEST -> ID: {best_id}, deltaX: {best_deltaX:.1f}")

        else:
            cv2.putText(frame, "No markers detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

        # show center line
        cv2.line(frame, (int(screen_center_x), 0), (int(screen_center_x), h), (255, 0, 0), 1)

        cv2.imshow("Aruco Logger", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('c'):
            break

    cap.release()
    cv2.destroyAllWindows()

    # Save log
    if not log.empty:
        filename = "aruco_log.csv"
        log.to_csv(filename, index=False)
        print(f"Saved log to {filename}")
    else:
        print("No data to save.")

if __name__ == "__main__":
    main()