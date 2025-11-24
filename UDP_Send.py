#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Minimal UDP boolean sender.

Usage:
  python3 UDP_Send.py 1    # send boolean 1 (0x01)
  python3 UDP_Send.py 0    # send boolean 0 (0x00)
"""

import socket
import argparse

# Default destination (edit if needed)
UDP_IP = "138.38.226.213"
UDP_PORT = 50001

parser = argparse.ArgumentParser(description="Send a single boolean byte (0 or 1) over UDP.")
parser.add_argument('value', choices=['0', '1'], help='Boolean value to send (0 or 1)')
args = parser.parse_args()

val = 1 if args.value == '1' else 0
payload = bytes([val])  # single raw byte: 0x00 or 0x01

print(f"Sending boolean {val} to {UDP_IP}:{UDP_PORT} (raw byte: {payload!r})")

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
try:
    try:
        sent = sock.sendto(payload, (UDP_IP, UDP_PORT))
        print(f"sendto(): handed {sent} bytes to OS")
    except socket.error as e:
        print("Socket send error:", e)
        raise
finally:
    sock.close()