#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
UDP Direction Switch sender — toggles 1 then 0 then 1 ... every 2 seconds.

This script sends a single raw byte (0x01 or 0x00) to the configured
destination every 2 seconds until interrupted (Ctrl-C).
"""

import socket
import time

# Destination (edit if needed)
UDP_IP = "138.38.226.213"
UDP_PORT = 50002

INTERVAL = 2.0  # seconds between sends

def main():
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        value = 1
        send_count = 0
        print(f"Starting toggle sender to {UDP_IP}:{UDP_PORT}, interval={INTERVAL}s. Ctrl-C to stop.")
        while True:
            payload = bytes([value])
            try:
                sent = sock.sendto(payload, (UDP_IP, UDP_PORT))
                send_count += 1
                print(f"#{send_count}: sent {payload!r} ({value}) -> {sent} bytes")
            except socket.error as e:
                print("Socket send error:", e)
            # flip value for next send
            value = 0 if value == 1 else 1
            try:
                time.sleep(INTERVAL)
            except KeyboardInterrupt:
                print("Interrupted by user")
                break
    finally:
        try:
            sock.close()
        except Exception:
            pass
        print("Sender stopped")

if __name__ == '__main__':
    main()