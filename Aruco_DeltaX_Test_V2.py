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
    # Configuration (arguments removed, values hard-coded)
    # Destination Raspberry Pi (hard-coded)
    # Change these values here if you need a different target
    # Note: CLI arg support removed per request
    
    # nothing to parse; defaults below

    cap = cv2.VideoCapture(0) #Change camera choice if needed (on mac webcam is 0)
    if not cap.isOpened():
        print("ERROR: Cannot open camera")
        return

    aruco_dict = make_aruco_dict()
    parameters = cv2.aruco.DetectorParameters()

    # Gamemodes: start in idle. Modes can be switched with number keys (1-5)
    # 1: Idle (preview only)
    # 2: Calibrate (preview only, same as idle)
    # 3: Selection (runs selection UI immediately)
    # 4: Attack (detection + UDP sending enabled)
    # 5: End (no detection, no sending)
    mode_map = {
        1: 'idle',
        2: 'calibrate',
        3: 'selection',
        4: 'attack',
        5: 'end'
    }

    current_mode = 'idle'
    selected_ids = []

    # helper to send current mode as uint8 over UDP
    def send_mode_uint8(mode_name):
        try:
            mode_num = None
            for k, v in mode_map.items():
                if v == mode_name:
                    mode_num = int(k)
                    break
            if mode_num is None:
                return
            payload = struct.pack('!B', mode_num)
            try:
                sock.sendto(payload, (udp_ip, udp_port_mode))
                if verbose:
                    print(f"MODE sent -> {mode_name} ({mode_num}) as uint8 to {udp_ip}:{udp_port_mode}")
            except Exception as e:
                print("MODE send error:", e)
        except Exception as e:
            print("send_mode_uint8 error:", e)

    def draw_modes_overlay(img, mode_map, current_mode):
        h, w = img.shape[:2]
        x = 10
        # draw a semi-transparent background for legibility
        overlay = img.copy()
        cv2.rectangle(overlay, (5, h-120), (260, h-5), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.4, img, 0.6, 0, img)
        y = h - 100
        for key in sorted(mode_map.keys()):
            name = mode_map[key]
            prefix = f"{key}: {name}"
            color = (0, 255, 0) if name == current_mode else (200, 200, 200)
            cv2.putText(img, prefix, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            y += 22
        return img

    columns = ["timestamp", "id", "marker_x", "deltaX"]
    log = pd.DataFrame(columns=columns)

    print("Running. Press 'c' to quit. Use keys 1-5 to switch modes.")

    # Setup UDP socket for sending deltaX and mode (hard-coded Raspberry Pi address)
    # Hard-coded to Pi IP and ports per request
    #udp_ip = '138.38.226.213'
    udp_ip = '172.26.236.65' #Oskar's Laptop
    udp_port_deltax = 50002
    udp_port_mode = 50001
    send_format = 'uint8'
    min_send_interval = 0.0
    verbose = False
    last_send_time = 0.0

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("ERROR: Frame capture failed")
            break

        h, w = frame.shape[:2]
        screen_center_x = w / 2

        best_id = None
        best_deltaX = None
        best_cx = None

        # Only run detection when in selection or attack (calibrate acts like idle)
        corners = None
        ids = None
        if current_mode in ('selection', 'attack'):
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            corners, ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

        # If in finding mode show detections but do not send
        if ids is not None and len(ids) > 0:
            ids = ids.flatten()
            centers = []
            for c in corners:
                pts = c.reshape((4, 2))
                cx = pts[:, 0].mean()
                centers.append(cx)

            items = []
            for marker_id, cx in zip(ids, centers):
                deltaX = cx - screen_center_x
                # when in attack only consider selected IDs
                if current_mode == 'attack':
                    if marker_id in selected_ids:
                        items.append((int(marker_id), float(cx), float(deltaX)))
                else:
                    items.append((int(marker_id), float(cx), float(deltaX)))

            if items:
                # choose the best marker (closest to center)
                items_sorted = sorted(items, key=lambda x: abs(x[2]))
                best_id, best_cx, best_deltaX = items_sorted[0]

                cv2.aruco.drawDetectedMarkers(frame, corners)
                for idx, (marker_id, cx, deltaX) in enumerate(items):
                    text = f"ID:{marker_id} x:{int(cx)} dX:{deltaX:.1f}"
                    cv2.putText(frame, text, (10, 30 + (idx * 18)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

                if best_cx is not None:
                    cv2.circle(frame, (int(best_cx), int(h/2)), 8, (0, 255, 0), -1)
                    cv2.putText(frame, f"BEST ID:{best_id} dX:{best_deltaX:.1f}", (10, h-40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

                log = log._append({
                    "timestamp": time.time(),
                    "id": int(best_id),
                    "marker_x": float(best_cx),
                    "deltaX": float(best_deltaX)
                }, ignore_index=True)

                # Print deltaX to terminal when in attack mode
                if current_mode == 'attack':
                    try:
                        print(f"BEST -> ID: {best_id}, deltaX: {best_deltaX:.1f}")
                    except Exception:
                        print(f"BEST -> ID: {best_id}, deltaX: {best_deltaX}")

                    # If the best marker is within the 'dead' threshold, remove it from selected targets
                    try:
                        if selected_ids and int(best_id) in selected_ids and abs(float(best_deltaX)) < 5.0:
                            selected_ids = [sid for sid in selected_ids if sid != int(best_id)]
                            print(f"Removed target {int(best_id)} (deltaX {best_deltaX:.1f} < 5). Remaining: {selected_ids}")
                    except Exception as e:
                        print("Error when checking/removing target:", e)

                # Only send over UDP in attack mode
                if current_mode == 'attack':
                    now = time.time()
                    if min_send_interval == 0.0 or (now - last_send_time) >= min_send_interval:
                        try:
                            if send_format == 'raw_float':
                                payload = struct.pack('!f', float(best_deltaX))
                            elif send_format == 'uint8':
                                half_width = w / 2.0 if w else 1.0
                                norm = float(best_deltaX) / half_width
                                scaled = int(round(norm * 127.0 + 128.0))
                                scaled = max(0, min(254, scaled))
                                payload = struct.pack('!B', scaled)
                            else:
                                payload = f"{time.time():.3f},{best_id},{best_deltaX:.6f}".encode('utf-8')
                            sock.sendto(payload, (udp_ip, udp_port_deltax))
                            last_send_time = now
                            if verbose:
                                print(f"UDP sent -> {payload!r} to {udp_ip}:{udp_port_deltax}")
                        except Exception as e:
                            print("UDP send error:", e)
        else:
            if current_mode == 'attack' and (min_send_interval == 0.0 or (time.time() - last_send_time) >= min_send_interval):
                # send NaN sentinel only in attack mode when no markers found
                try:
                    if send_format == 'raw_float':
                        payload = struct.pack('!f', float('nan'))
                    elif send_format == 'uint8':
                        payload = struct.pack('!B', 255)
                    else:
                        payload = b"nan"
                    sock.sendto(payload, (udp_ip, udp_port_deltax))
                    last_send_time = time.time()
                    if verbose:
                        print(f"UDP sent (no marker) -> {payload!r} to {udp_ip}:{udp_port_deltax}")
                except Exception as e:
                    print("UDP send error:", e)

        cv2.line(frame, (int(screen_center_x), 0), (int(screen_center_x), h), (255, 0, 0), 1)

        # overlay mode options and highlight current (keys 1-5)
        frame = draw_modes_overlay(frame, mode_map, current_mode)

        cv2.imshow("Aruco Logger", frame)
        key = cv2.waitKey(1) & 0xFF

        # global quit
        if key == ord('c'):
            break

        # mode switching with number keys 1-5
        if key in [ord('1'), ord('2'), ord('3'), ord('4'), ord('5')]:
            num = int(chr(key))
            requested = mode_map.get(num, None)
            if requested:
                current_mode = requested
                print(f"Switched to mode: {current_mode}")
                # send mode as uint8 to Raspberry Pi
                send_mode_uint8(current_mode)
                # if selection mode entered explicitly, run selection UI now
                if current_mode == 'selection':
                    found = select_targets(cap, aruco_dict)
                    print("Found IDs:", found)
                    user_input = input("Enter the IDs you want to target (comma-separated): ")
                    selected_ids = [int(i.strip()) for i in user_input.split(',') if i.strip().isdigit()]
                    print("Selected IDs:", selected_ids)
                    current_mode = 'attack'
                    # announce automatic attack mode over UDP
                    send_mode_uint8(current_mode)
                    print("Automatically entering attack mode.")
        # no 'finding' mode: selection is available via key 2 or key 2->selection flow

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