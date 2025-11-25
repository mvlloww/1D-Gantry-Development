#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

This is an example to setup a UDP Listener.

@author: ioannisgeorgilas
"""

# First we import our libraries
import socket   # This library will allow you to communicate over the network
import sys      # This library will give us some information about your system
import struct

# First we need to set the IP and PORT we are going to listen to
# This is the localhost IP address (this machine)
UDP_IP = "0.0.0.0"
 
# This is the LOCAL port I am expecting data (on the sending machine this is the REMOTE port)
UDP_PORT = 50002

# Create the socket for the UDP communication
sock = socket.socket(socket.AF_INET,    # Family of addresses, in this case IP (Internet Protocol) family 
                     socket.SOCK_DGRAM) # What protocol to use, in this case UDP (datagram)

# Bind to the socket and wait for data on this port
sock.bind((UDP_IP, UDP_PORT))
print("Listening on IP:", UDP_IP, "Port:", UDP_PORT)

# Wait indifenetly (you will need to use Ctrl+C to stop the program)
while True:
    #Read data
    data, addr = sock.recvfrom(1024) # buffer size is 1024 bytes
    # Try to decode as UTF-8 text first, otherwise interpret as binary numeric payload
    try:
        text = data.decode('utf-8')
        print("received message (utf-8):", text)
    except UnicodeDecodeError:
        # Binary payload: show raw bytes and attempt common numeric interpretations
        b = list(data)
        print("received bytes:", b)
        # Interpret single-byte messages as uint8
        if len(data) == 1:
            val = struct.unpack('!B', data)[0]
            print("interpreted as uint8:", val)
        # Interpret 4-byte messages as float32 (network order) or int32
        elif len(data) == 4:
            try:
                f = struct.unpack('!f', data)[0]
                print("interpreted as float32:", f)
            except Exception:
                i = struct.unpack('!i', data)[0]
                print("interpreted as int32:", i)
        else:
            print("raw binary payload (len=" + str(len(data)) + ")")

    print ("from address:", addr) # Print the address of the sender