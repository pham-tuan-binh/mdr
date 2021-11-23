import cv2
import numpy as np
import math
from dt_apriltags import Detector
from numpy.core.fromnumeric import diagonal
from pathfinding.core.diagonal_movement import DiagonalMovement
from pathfinding.core.grid import Grid
from pathfinding.finder.a_star import AStarFinder

image = cv2.imread("./resources/teset.jpg")
imageGray = cv2.imread("./resources/teset.jpg",  cv2.IMREAD_GRAYSCALE)
print(len(image.shape))


class coordinate:
    def __init__(self, x, y, yaw):
        self.x = x
        self.y = y
        self.yaw = yaw

    def updateCoordinate(self, x, y, yaw):
        self.x = x
        self.y = y
        self.yaw = yaw

    def getCoordinate(self):
        return (self.x, self.y, self.yaw)


class pathFinder:
    def __init__(self, image, threshold):
        h, w = image.shape[0], image.shape[1]
        gcd = math.gcd(w, h)

        while True:
            if gcd % threshold == 0:
                break
            threshold += 1
        matrixWidth = w//threshold
        matrixHeight = h//threshold
        matrixWC = np.zeros((matrixHeight, matrixWidth, 2))
        for countRow in range(matrixWC.shape[0]):
            for countCol in range(matrixWC.shape[1]):
                matrixWC[countRow, countCol] = (int(
                    threshold/2 - 1 + threshold*countRow), int(threshold/2 - 1 + threshold*countCol))

        self.threshold = threshold
        self.matrix = np.full((matrixHeight, matrixWidth), 1)
        self.opencvMatrix = np.zeros((matrixHeight, matrixWidth, 3))
        self.matrixWithCoordinate = matrixWC

    def convertBoxes(self, boxes):
        trash = []
        obstacles = []

        for trashBox in boxes[0]:
            trashBox = trashBox[1]
            trash.append((trashBox[1] // self.threshold,
                         trashBox[0] // self.threshold))
        for obsBox in boxes[1]:
            obsBox = obsBox[0]
            newObsBox = []
            for point in obsBox:
                newObsBox.append(
                    (point[0] // self.threshold, point[1] // self.threshold))
            obstacles.append(newObsBox)

        return (trash, obstacles)

    def path(self, robot, trash, obstacle):
        self.robot, self.trash, self.obstacle = robot, trash[0], obstacle
        for box in obstacles:
            cv2.drawContours(self.opencvMatrix, np.int0([box]),
                             0, (0, 0, 255), -1)
        for i in range(self.opencvMatrix.shape[0]):
            for j in range(self.opencvMatrix.shape[1]):
                if self.opencvMatrix[i, j, 2] == 255:
                    self.matrix[i, j] = 0

        grid = Grid(matrix=self.matrix)
        start = grid.node(robot[0], robot[1])
        end = grid.node(trash[0][0], trash[0][1])
        finder = AStarFinder(diagonal_movement=DiagonalMovement.always)
        path = finder.find_path(start, end, grid)[0]
        print(grid.grid_str(path=path, start=start, end=end))
        return path


class boat(coordinate):
    def __init__(self, id, x, y, yaw):
        super().__init__(x, y, yaw)
        self.id = id


class objectDetector:
    def distance(self, point1, point2):
        squareX = (point1[0] - point2[0])**2
        squareY = (point1[1] - point2[1])**2
        return int((squareX + squareY)**(1/2))

    def findContours(self, input):
        input = cv2.cvtColor(input, cv2.COLOR_BGR2HSV)
        hul, huh, sal, sah, val, vah, dl = 58, 96, 0, 255, 0, 255, 35
        hsvl = np.array([hul, sal, val], np.uint8)
        hsvh = np.array([huh, sah, vah], np.uint8)

        mask = cv2.inRange(input, hsvl, hsvh)
        mask = cv2.dilate(mask, np.ones((dl, dl), np.uint8), iterations=1)
        mask = cv2.bitwise_not(mask)
        cannyFrame = cv2.Canny(mask, 100, 150)
        contours, hi = cv2.findContours(
            cannyFrame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return (contours, hi)

    def calculateBoundingBox(self, contours):
        boxes = []
        for cnt in contours[0]:
            rect = cv2.minAreaRect(cnt)
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            centerX = (box[0, 0] + box[2, 0]) // 2
            centerY = (box[0, 1] + box[2, 1]) // 2
            boxes.append((box, (centerX, centerY)))
        return boxes

    def sortBoxes(self, boxes, threshold):
        trash = []
        obstacles = []
        for box, center in boxes:
            if self.distance(box[0], box[1]) > threshold or self.distance(box[1], box[2]) > threshold:
                obstacles.append((box, center))
            else:
                trash.append((box, center))
        return (trash, obstacles)

    def visualizeContours(self, input, contours):
        cv2.drawContours(input, contours, -1, (0, 0, 255), 4)
        return input


class boatDetector:
    @staticmethod
    def detectBoatCoordinate():
        at_detector = Detector(searchpath=['apriltags'],
                               families='tag36h11',
                               nthreads=1,
                               quad_decimate=1.0,
                               quad_sigma=0.0,
                               refine_edges=1,
                               decode_sharpening=0.25,
                               debug=0)

        tag = at_detector.detect(
            imageGray, estimate_tag_pose=False, camera_params=None, tag_size=None)[0]

        corners = tag.corners
        centerX = (corners[0, 0] + corners[2, 0]) // 2
        centerY = (corners[0, 1] + corners[2, 1]) // 2
        return (centerX, centerY)


class frameCapture:
    def __init__(self, id):
        self.feed = cv2.VideoCapture(id)

    def update(self):
        if not self.feed.isOpened():
            raise IOError("Cannot open webcam")
        global frame
        ret, frame = self.feed.read()

    def release(self):
        self.feed.release()


# fr = frameCapture(0)

# Check if the webcam is opened correctly
# while True:
#     fr.update()

#     cv2.imshow('Input', objectDetect.process(frame))
#     c = cv2.waitKey(1)
#     if c == 27:
#         break

objectDetect = objectDetector()
pathFind = pathFinder(image, 60)
contours = objectDetect.findContours(image)
boxes = objectDetect.calculateBoundingBox(contours)

boxes = objectDetect.sortBoxes(boxes, 50)
for box in boxes[1]:

    cv2.drawContours(image, [box[0]], 0, (255, 0, 0), 2)

trash, obstacles = pathFind.convertBoxes(boxes)

pathFind.path((14, 5), trash, obstacles)
boatDetect = boatDetector()
cors = np.int0(boatDetect.detectBoatCoordinate())
cv2.circle(image, cors, 4, (0, 0, 0), -1)
cv2.imshow("image", image)

# for cnt in contours[0]:
#     rect = cv2.minAreaRect(cnt)
#     box = cv2.boxPoints(rect)
#     box = np.int0(box)
#     # calculate center
#     centerX = (box[0, 0] + box[2, 0]) // 2
#     centerY = (box[0, 1] + box[2, 1]) // 2
#     cv2.circle(image, (604, 927), 4, (0, 255, 0), -1)
#     cv2.circle(image, (centerX, centerY), 4, (0, 255, 0), -1)
#     print((centerX, centerY))
#     threshold = 50
#     if distance(box[0], box[1]) > threshold or distance(box[1], box[2]) > threshold:
#         cv2.drawContours(image, [box], 0, (255, 0, 0), 2)
#     else:
#         cv2.drawContours(image, [box], 0, (0, 0, 255), 2)

# points = []
# for row in calculateCors(image):
#     for point in row:
#         points.append(point)
# for id, p in enumerate(points):
#     print((np.int0(p), id))
#     cv2.circle(image, np.int0(p), 10, (0, 255, 0))
#     image[np.int0(p)[0], np.int0(p)[1]] = (255, 255, 255)

# cv2.imshow("img", image)
cv2.waitKey(0)

# # # fr.release()
cv2.destroyAllWindows()
