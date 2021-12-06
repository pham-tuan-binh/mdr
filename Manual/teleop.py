'''
musv_teleop interprets SSH keyboard inputs as direction commands and sends 
the corresponding motor speed commands to the microUSV's motor controller. 

Copyright (C) 2019  CalvinGregory  cgregory@mun.ca
This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see https://www.gnu.org/licenses/gpl-3.0.html.
'''

# Based on a tutorial by Christopher Barnatt.
# https://www.explainingcomputers.com/rasp_pi_robotics.html

import sys
import curses
import struct
import os.path
from Config import Config
import socket
import json

HOST = '172.16.0.164'  # The server's hostname or IP address
PORT = 10001        # The port used by the server

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.connect((HOST, PORT))


def sendData(starboard, port):
    data = json.dumps([starboard, port])
    s.sendall(data.encode())
    s.recv(8)


def sendSpeeds(portSpeed, starboardSpeed):
    """ 
    Send formated motor speed message to Arduino

    Args:
        portSpeed (int16):      Desired port motor speed (range -127 to 127)
        starboardSpeed (int16): Desired starboard motor speed (range -127 to 127)

    Messages are prepended by two '*' characters to indicate message start.     
    """
    sendData(portSpeed, starboardSpeed)

    return


# Read config file
if (len(sys.argv) < 2):
    if (os.path.isfile('config.json')):
        config = Config('config.json')
    else:
        print('No config file path provided')
        exit()
else:
    config = Config(sys.argv[1])

# Setup terminal window for curses
screen = curses.initscr()
curses.noecho()
curses.cbreak()
screen.keypad(True)

speed = 127
port_speed = int(round(config.propSpin_port *
                 (speed - speed*(config.bias/100))))
starboard_speed = int(
    round(config.propSpin_star * (speed + speed*(config.bias/100))))
print(port_speed)
print(starboard_speed)

try:
    while True:
        msg = screen.getch()

        if msg == 27:  # if ESC key: stop motors and end program
            sendSpeeds(0, 0)
            break

        # For 1,2,3 key presses change internal motor speed to preset low, medium, or high
        elif msg == ord('1'):
            speed = 100
            port_speed = int(round(config.propSpin_port *
                             (speed - speed*(config.bias/100))))
            starboard_speed = int(
                round(config.propSpin_star * (speed + speed*(config.bias/100))))
        elif msg == ord('2'):
            speed = 200
            port_speed = int(round(config.propSpin_port *
                             (speed - speed*(config.bias/100))))
            starboard_speed = int(
                round(config.propSpin_star * (speed + speed*(config.bias/100))))
        elif msg == ord('3'):
            speed = 300
            port_speed = int(round(config.propSpin_port *
                             (speed - speed*(config.bias/100))))
            starboard_speed = int(
                round(config.propSpin_star * (speed + speed*(config.bias/100))))
        elif msg == ord('4'):
            speed = 400
            port_speed = int(round(config.propSpin_port *
                             (speed - speed*(config.bias/100))))
            starboard_speed = int(
                round(config.propSpin_star * (speed + speed*(config.bias/100))))
        elif msg == ord('5'):
            speed -= 10
            port_speed = int(round(config.propSpin_port *
                             (speed - speed*(config.bias/100))))
            starboard_speed = int(
                round(config.propSpin_star * (speed + speed*(config.bias/100))))
        elif msg == ord('6'):
            speed += 10
            port_speed = int(round(config.propSpin_port *
                             (speed - speed*(config.bias/100))))
            starboard_speed = int(
                round(config.propSpin_star * (speed + speed*(config.bias/100))))
        # For w,a,s,d and q,e,z,c key presses send motor speeds to Arduino.
        elif msg == ord('w'):
            sendSpeeds(port_speed, starboard_speed)
        elif msg == ord('a'):
            sendSpeeds(-port_speed, starboard_speed)
        elif msg == ord('s'):
            sendSpeeds(-port_speed, -starboard_speed)
        elif msg == ord('d'):
            sendSpeeds(port_speed, -starboard_speed)
        elif msg == ord('q'):
            sendSpeeds(0, starboard_speed)
        elif msg == ord('e'):
            sendSpeeds(port_speed, 0)
        elif msg == ord('z'):
            sendSpeeds(0, -starboard_speed)
        elif msg == ord('c'):
            sendSpeeds(-port_speed, 0)
        # If not a control character, set motor speeds to 0.
        else:
            sendSpeeds(0, 0)

# Reset terminal window to defaults and shutdown motors
finally:
    sendSpeeds(0, 0)
    curses.nocbreak()
    screen.keypad(False)
    curses.echo()
    curses.endwin()
