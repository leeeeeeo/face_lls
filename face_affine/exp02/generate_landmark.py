import cv2
import dlib
import numpy as np
import os
from face_affine_utils import *


def shape_to_np(shape, dtype="int"):
    coords = np.zeros((68, 2), dtype=dtype)
    for i in range(0, 68):
        coords[i] = (shape.part(i).x, shape.part(i).y)
    return coords


def shape_to_list(shape, dtype="int"):
    coords = []
    for i in range(0, 68):
        coords.append((shape.part(i).x, shape.part(i).y))
    return coords


def landmark_detection(face, filePath):
    landmark_txt = open('{}.txt'.format(filePath), 'w')
    # gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    gray = face
    rects = detector(gray, 1)
    for (i, rect) in enumerate(rects):
        shape = predictor(gray, rect)
        shape = shape_to_list(shape)
        shape = addEdgeLandmark(shape, face)
        for (x, y) in shape:
            landmark_txt.write(str('{} {}\n'.format(x, y)))
    landmark_txt.close()
    return shape


detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(
    '../../../tools/shape_predictor_68_face_landmarks.dat')
for root, folder, files in os.walk('./data/source'):
    for file in files:
        print file
        filePath = './data/source/{}'.format(file)
        if filePath.endswith('png'):
            face = cv2.imread(filePath)
            landmark = landmark_detection(face, filePath)
            # print landmark
