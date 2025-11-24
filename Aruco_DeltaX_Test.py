import cv2
import numpy as np
import pandas as pd
import time

def make_aruco_dict(name=cv2.aruco.DICT_4X4_50):
    try:
        return cv2.aruco.Dictionary_get(name)
    except AttributeError:
        return cv2.aruco.getPredefinedDictionary(name)

def select_targets(cap, aruco_dict):
    print("Finding ArUco markers. Press 'q' to stop.")
    found_ids = set()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("ERROR: Frame capture failed")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict)

        if ids is not None:
            ids = ids.flatten()
            found_ids.update(ids)

        cv2.aruco.drawDetectedMarkers(frame, corners)
        cv2.imshow("Select Targets", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    cv2.destroyAllWindows()
    return list(found_ids)

def main():
    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        print("ERROR: Cannot open camera")
        return

    aruco_dict = make_aruco_dict()
    parameters = cv2.aruco.DetectorParameters()

    # Target selection phase
    target_ids = select_targets(cap, aruco_dict)
    print("Found IDs:", target_ids)

    # Allow user to select targets
    selected_ids = input("Enter the IDs you want to target (comma-separated): ")
    selected_ids = [int(id.strip()) for id in selected_ids.split(",") if id.strip().isdigit()]

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

        if ids is not None and len(ids) > 0:
            ids = ids.flatten()
            centers = []
            for c in corners:
                pts = c.reshape((4, 2))
                cx = pts[:, 0].mean()
                centers.append(cx)

            items = []
            for marker_id, cx in zip(ids, centers):
                if marker_id in selected_ids:
                    deltaX = cx - screen_center_x
                    items.append((int(marker_id), float(cx), float(deltaX)))

            if items:
                items_sorted = sorted(items, key=lambda x: abs(x[2]))
                best_id, best_cx, best_deltaX = items_sorted[0]

                cv2.aruco.drawDetectedMarkers(frame, corners, ids.reshape(-1,1))
                for marker_id, cx, deltaX in items:
                    text = f"ID:{marker_id} x:{int(cx)} dX:{deltaX:.1f}"
                    cv2.putText(frame, text, (10, 30 + 20 * marker_id % 400), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

                cv2.circle(frame, (int(best_cx), int(h/2)), 8, (0, 255, 0), -1)
                cv2.putText(frame, f"BEST ID:{best_id} dX:{best_deltaX:.1f}", (10, h-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

                log = log._append({
                    "timestamp": time.time(),
                    "id": int(best_id),
                    "marker_x": float(best_cx),
                    "deltaX": float(best_deltaX)
                }, ignore_index=True)

                print(f"BEST -> ID: {best_id}, deltaX: {best_deltaX:.1f}")

        else:
            cv2.putText(frame, "No markers detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.line(frame, (int(screen_center_x), 0), (int(screen_center_x), h), (255, 0, 0), 1)

        cv2.imshow("Aruco Logger", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('c'):
            break

    cap.release()
    cv2.destroyAllWindows()

    if not log.empty:
        filename = "aruco_log.csv"
        log.to_csv(filename, index=False)
        print(f"Saved log to {filename}")
    else:
        print("No data to save.")

if __name__ == "__main__":
    main()