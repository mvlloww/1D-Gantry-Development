import socket
import struct

UDP_IP = "0.0.0.0"
UDP_PORT = 50002  # Must match the port used for deltaX in your sender

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind((UDP_IP, UDP_PORT))
print(f"Listening for int32 deltaX on {UDP_IP}:{UDP_PORT}")

SENTINEL = 0x7FFFFFFF

try:
    while True:
        data, addr = sock.recvfrom(4096)
        if len(data) >= 4:
            val = struct.unpack('!i', data[:4])[0]
            if val == SENTINEL:
                print("Received: No marker (NaN sentinel)")
            else:
                print(f"Received deltaX (px): {val}")
        else:
            print(f"Received short/invalid packet ({len(data)} bytes): {data}")
except KeyboardInterrupt:
    print("Exiting.")
finally:
    sock.close()