import cv2
import dlib
import numpy as np
import os
from face_affine_utils import *
from faceBox import *


def shape_to_list(shape, dtype="int"):
    coords = []
    for i in range(0, 68):
        coords.append((shape.part(i).x, shape.part(i).y))
    return coords


def landmark_detection(face, filePath, edgeOption=None):
    landmark_txt = open('{}.txt'.format(filePath), 'w')
    # gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    gray = face
    rects = detector(gray, 1)
    for (i, rect) in enumerate(rects):
        shape = predictor(gray, rect)
        shape = shape_to_list(shape)
        if edgeOption == 'fullImg':
            shape = addEdgeLandmark(shape, face)
        elif edgeOption == 'faceArea':
            eightEdgePoints, faceRect, maskRect = faceBoundingbox(face)
            shape = shape+eightEdgePoints
        else:
            shape = shape
        for (x, y) in shape:
            landmark_txt.write(str('{} {}\n'.format(x, y)))
    landmark_txt.close()
    return shape


def mainGenerateLandmark():
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(
        '../../../tools/shape_predictor_68_face_landmarks.dat')
    edgeOption = ['fullImg', 'faceArea']
    for root, folder, files in os.walk('./data/talkingphoto/IMG_2294'):
        for file in files:
            filePath = '{}/{}'.format(root, file)
            if filePath.endswith('png') or filePath.endswith('jpg'):
                face = cv2.imread(filePath)
                landmark = landmark_detection(face, filePath, edgeOption[1])

    # filePath = '/Users/lls/Documents/face/data/talkingphoto/crop640.png'
    # face = cv2.imread(filePath)
    # landmark_detection(face, filePath, edgeOption=edgeOption[1])


if __name__ == "__main__":
    mainGenerateLandmark()
