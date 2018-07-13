import cv2
import dlib
import numpy as np
import os


def shape_to_np(shape, dtype="int"):
    coords = np.zeros((68, 2), dtype=dtype)
    for i in range(0, 68):
        coords[i] = (shape.part(i).x, shape.part(i).y)
    return coords


def landmark_detection(face, file):
    landmark_num = 68
    # gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    gray = face
    rects = detector(gray, 1)
    for (i, rect) in enumerate(rects):
        shape = predictor(gray, rect)
        shape = shape_to_np(shape)
        if shape.shape[0] != 0:
            landmark_txt = open('./data/source/{}.pts'.format(file), 'w')
            landmark_txt.write("version: 1" + '\n')
            landmark_txt.write("n_points: {}".format(str(landmark_num)) + '\n')
            landmark_txt.write('{' + '\n')
            for (x, y) in shape:
                landmark_txt.write(str('{}.0 {}.0\n'.format(x, y)))
            landmark_txt.write('}')
            landmark_txt.close()


detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(
    '../../../tools/shape_predictor_68_face_landmarks.dat')
for root, folder, files in os.walk('./data/source'):
    for file in files:
        if file.endswith('.png') or file.endswith('.jpg'):
            print 'processing {}'.format(file)
            face = cv2.imread('./data/source/{}'.format(file))
            landmark_detection(face, file)
