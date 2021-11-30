import socket
import json
HOST = '172.16.0.171'  # The server's hostname or IP address
PORT = 10001        # The port used by the server

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.connect((HOST, PORT))


def sendData(x, y, yaw, goalX, goalY):
    jsonString = json.dumps({"boat": [x, y, yaw], "goal": [goalX, goalY]})

    print(jsonString)
    s.sendall(jsonString.encode())
    s.recv(8)


sendData(1, 2, 3, 4, 5)
