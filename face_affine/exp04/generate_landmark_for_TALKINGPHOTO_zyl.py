import cv2
import dlib
import numpy as np
import os
from face_affine_utils import *
from faceBox import *
from generate_landmark_for_TALKINGPHOTO import landmarkBuffer


def shape_to_list(shape, dtype="int"):
    coords = []
    for i in range(0, 68):
        coords.append((shape.part(i).x, shape.part(i).y))
    return coords


def landmark_detection(face, filePath, edgeOption=None):
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
            # cv2.rectangle(face, (eightEdgePoints[0][0], eightEdgePoints[0][1]), (
            #     eightEdgePoints[4][0], eightEdgePoints[4][1]), (0, 255, 0), 2)
            # cv2.imshow('a', face)
            # cv2.waitKey(0)
            # cv2.imwrite('/Users/lls/Documents/face/data/talkingphoto/IMG_2294_bbox/{}_bbox.jpg'.format(
            #     filePath.split('/')[-1].split('.')[0]), face)
            shape = shape+eightEdgePoints
        else:
            shape = shape
    return shape


def saveTxt(filePath, landmark):
    landmark_txt = open('{}.txt'.format(filePath), 'w')
    for (x, y) in landmark:
        landmark_txt.write(str('{} {}\n'.format(x, y)))
    landmark_txt.close()


def saveTxtforTXT(filePath, landmark):
    landmark_txt = open(filePath, 'w')
    for (x, y) in landmark:
        landmark_txt.write(str('{} {}\n'.format(x, y)))
    landmark_txt.close()


if __name__ == "__main__":
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(
        '../../../tools/shape_predictor_68_face_landmarks.dat')
    edgeOption = ['fullImg', 'faceArea']

    '''1. generate 68+8 landmarks for original image crop640.png'''
    filePath = "/Users/lls/Library/Containers/com.tencent.WeWorkMac/Data/Library/Application Support/WXWork/Data/1688850987389419/Cache/File/2018-07/smile 6/smile.jpg"
    face = cv2.imread(filePath)
    landmark = landmark_detection(face, filePath, edgeOption=edgeOption[1])
    saveTxt(filePath, landmark)
    eightEdgePoints, faceRect, maskRect = faceBoundingbox(face)
    '''2. add 8 landmarks of original image'''
    allLandmarkList = []
    fileList = []
    for root, folder, files in os.walk("/Users/lls/Library/Containers/com.tencent.WeWorkMac/Data/Library/Application Support/WXWork/Data/1688850987389419/Cache/File/2018-07/reconstruct 4"):
        for file in files:
            filePath = '{}/{}'.format(root, file)
            if filePath.endswith('txt'):
                fileList.append(filePath)
    fileList = natsorted(fileList)
    for filePath in fileList:
        print filePath
        landmark = readPoints(filePath)
        landmark = landmark+eightEdgePoints
        allLandmarkList.append(landmark)
        # saveTxtforTXT(filePath, landmark)
    '''3. bufferx3 landmarks, L[2]=(L[1]+L[2]+L[3])/3'''
    # landmark = landmarkBuffer(allLandmarkList)
    landmark = allLandmarkList
    '''4. save all landmark txt'''
    for i in range(len(landmark)):
        txtPath = fileList[i]
        print txtPath
        landmark_txt = open(txtPath, 'w')
        for (x, y) in landmark[i]:
            landmark_txt.write(str('{} {}\n'.format(x, y)))
        landmark_txt.close()
