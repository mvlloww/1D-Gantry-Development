import cv2
import numpy as np
import pandas as pd
import time
import socket
import struct

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
    # Start in idle mode by default; all mode changes are keyboard-triggered.
    initial_mode = 'idle'

    cap = cv2.VideoCapture(0) #Change camera choice if needed (on mac webcam is 0)
    if not cap.isOpened():
        print("ERROR: Cannot open camera")
        return

    aruco_dict = make_aruco_dict()
    parameters = cv2.aruco.DetectorParameters()

    # Setup UDP socket for sending deltaX (hard-coded Raspberry Pi address)
    udp_ip = '138.38.226.213'
    #udp_ip = '172.26.236.65'
    udp_port = 50002
    send_format = 'uint8'
    min_send_interval = 0.0
    verbose = False
    last_send_time = 0.0

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    # gamemode state: idle, calibrate, finding, attack, end
    gamemode = initial_mode
    last_gamemode = None

    def send_mode(mode):
        nonlocal last_gamemode
        if mode == last_gamemode:
            return
        msg = f"MODE:{mode}".encode('utf-8')
        try:
            sock.sendto(msg, (udp_ip, udp_port))
            if verbose:
                print(f"Sent gamemode -> {mode}")
        except Exception as e:
            print("Failed to send gamemode:", e)
        last_gamemode = mode

    # Start with no selected targets; selection runs only when you press the finding key
    selected_ids = set()
    # start in the initial mode
    gamemode = initial_mode
    send_mode(gamemode)

    columns = ["timestamp", "id", "marker_x", "deltaX"]
    log = pd.DataFrame(columns=columns)

    print("Running. Press 'c' to quit.")
    print("Gamemode keys: 1=idle, 2=calibrate, 3=finding, 4=attack, 5=end")

    # NOTE: UDP socket already created above before selection

    # track dead status per target id
    dead = {tid: False for tid in selected_ids}

    while True:
        ret, frame = cap.read()
        if not ret:
            print("ERROR: Frame capture failed")
            break

        h, w = frame.shape[:2]
        screen_center_x = w / 2

        # Only run detection when not idle (finding or attack)
        ids = None
        corners = []
        if gamemode != 'idle':
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

                # update dead status for this marker
                if best_id in dead:
                    if -1.0 <= best_deltaX <= 1.0:
                        if not dead[best_id]:
                            dead[best_id] = True
                            if verbose:
                                print(f"Target {best_id} marked dead")
                    else:
                        if dead[best_id]:
                            dead[best_id] = False

                # check end condition: all selected targets dead
                if selected_ids and all(dead.get(t, False) for t in selected_ids):
                    gamemode = 'end'
                    send_mode(gamemode)

                # Send deltaX over UDP (throttle by send-interval if requested)
                now = time.time()
                if min_send_interval == 0.0 or (now - last_send_time) >= min_send_interval:
                    try:
                        if send_format == 'raw_float':
                            # pack as network-order 32-bit float
                            payload = struct.pack('!f', float(best_deltaX))
                        elif send_format == 'uint8':
                            # map deltaX to uint8: center (0) -> 128, left -> <128, right -> >128
                            # reserve 255 (0xFF) as NaN sentinel
                            half_width = w / 2.0 if w else 1.0
                            # normalize to [-1,1]
                            norm = float(best_deltaX) / half_width
                            scaled = int(round(norm * 127.0 + 128.0))
                            scaled = max(0, min(254, scaled))
                            payload = struct.pack('!B', scaled)
                        else:
                            # ASCII: timestamp,id,deltaX
                            payload = f"{time.time():.3f},{best_id},{best_deltaX:.6f}".encode('utf-8')
                        sock.sendto(payload, (udp_ip, udp_port))
                        last_send_time = now
                        if verbose:
                            print(f"UDP sent -> {payload!r} to {udp_ip}:{udp_port}")
                    except Exception as e:
                        print("UDP send error:", e)

        else:
            cv2.putText(frame, "No markers detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            # When no marker is found, send NaN indicator according to selected format
            now = time.time()
            if min_send_interval == 0.0 or (now - last_send_time) >= min_send_interval:
                try:
                    if send_format == 'raw_float':
                        payload = struct.pack('!f', float('nan'))
                    elif send_format == 'uint8':
                        # use 255 (0xFF) as NaN sentinel for uint8
                        payload = struct.pack('!B', 255)
                    else:
                        payload = b"nan"
                    sock.sendto(payload, (udp_ip, udp_port))
                    last_send_time = now
                    if verbose:
                        print(f"UDP sent (no marker) -> {payload!r} to {udp_ip}:{udp_port}")
                except Exception as e:
                    print("UDP send error:", e)

        cv2.line(frame, (int(screen_center_x), 0), (int(screen_center_x), h), (255, 0, 0), 1)
        
        # Draw current gamemode in the bottom-left corner
        try:
            mode_text = f"MODE: {gamemode.upper()}"
        except Exception:
            mode_text = "MODE: ?"
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.8
        thickness = 2
        (text_w, text_h), baseline = cv2.getTextSize(mode_text, font, scale, thickness)
        x = 10
        y = h - 10  # 10 px above bottom
        # Ensure text fits above the bottom edge
        if y < text_h + 5:
            y = text_h + 5
        # Optional background for readability
        cv2.rectangle(frame, (x - 5, y - text_h - 5), (x + text_w + 5, y + 5), (0, 0, 0), -1)
        cv2.putText(frame, mode_text, (x, y), font, scale, (0, 255, 255), thickness)

        cv2.imshow("Aruco Logger", frame)
        key = cv2.waitKey(1) & 0xFF
        # Keyboard gamemode triggers (digits 1-5)
        if key == ord('1'):
            gamemode = 'idle'
            send_mode(gamemode)
        elif key == ord('2'):
            gamemode = 'calibrate'
            send_mode(gamemode)
        elif key == ord('3'):
            # Enter finding mode and run target selection UI
            gamemode = 'finding'
            send_mode(gamemode)
            print("Entering target selection (press 'q' in the selection window to finish)")
            found = select_targets(cap, aruco_dict)
            if found:
                selected_ids = set(found)
                dead = {tid: False for tid in selected_ids}
            print("Selected IDs:", selected_ids)
        elif key == ord('4'):
            gamemode = 'attack'
            send_mode(gamemode)
        elif key == ord('5'):
            gamemode = 'end'
            send_mode(gamemode)
        elif key == ord('c'):
            break

    cap.release()
    cv2.destroyAllWindows()

    try:
        sock.close()
    except Exception:
        pass

    if not log.empty:
        filename = "aruco_log.csv"
        log.to_csv(filename, index=False)
        print(f"Saved log to {filename}")
    else:
        print("No data to save.")

if __name__ == "__main__":
    main()