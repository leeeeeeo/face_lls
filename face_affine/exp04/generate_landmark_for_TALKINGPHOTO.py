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


def saveOneLandmarkToTxt(filePath, landmark):
    landmark_txt = open('{}.txt'.format(filePath), 'w')
    for (x, y) in landmark:
        landmark_txt.write(str('{} {}\n'.format(x, y)))
    landmark_txt.close()


def landmarkBuffer(allLandmarkList):
    '''LIST TO NUMPY'''
    allLandmarkNumpy = []
    for landmark in allLandmarkList:
        allLandmarkNumpy.append(np.array(landmark))
    '''MEAN'''
    tmpMean = []
    tmpMean.append(allLandmarkNumpy[0])
    for i in range(1, len(allLandmarkList)-1):
        tmpMean.append(
            (allLandmarkNumpy[i-1]+allLandmarkNumpy[i]+allLandmarkNumpy[i+1])/3.0)
    tmpMean.append(allLandmarkNumpy[-1])
    '''FINAL LANDMARK'''
    finalLandmarkList = []
    for landmark in tmpMean:
        finalLandmarkList.append(landmark.tolist())
    return finalLandmarkList


if __name__ == "__main__":
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(
        '../../../tools/shape_predictor_68_face_landmarks.dat')
    edgeOption = ['fullImg', 'faceArea']

    '''1. generate 68+8 landmarks for original image crop640.png'''
    filePath = '/Users/lls/Documents/face/data/talkingphoto/crop640.png'
    face = cv2.imread(filePath)
    landmark = landmark_detection(face, filePath, edgeOption=edgeOption[1])
    saveOneLandmarkToTxt(filePath, landmark)
    eightEdgePoints, faceRect, maskRect = faceBoundingbox(face)
    '''2. generate 68 landmarks for target images'''
    '''3. add 8 landmarks of original image'''
    allLandmarkList = []
    fileList = []
    for root, folder, files in os.walk('./data/talkingphoto/IMG_2294_buffer'):
        for file in files:
            filePath = '{}/{}'.format(root, file)
            if filePath.endswith('png') or filePath.endswith('jpg'):
                fileList.append(filePath)
    fileList = natsorted(fileList)
    for filePath in fileList:
        print filePath
        face = cv2.imread(filePath)
        landmark = landmark_detection(face, filePath)
        landmark = landmark+eightEdgePoints
        allLandmarkList.append(landmark)
        # saveOneLandmarkToTxt(filePath, landmark)
    '''4. bufferx3 landmarks, L[2]=(L[1]+L[2]+L[3])/3'''
    landmark = landmarkBuffer(allLandmarkList)
    '''5. save all landmark txt'''
    for i in range(len(landmark)):
        txtPath = '{}.txt'.format(fileList[i])
        print txtPath
        landmark_txt = open(txtPath, 'w')
        for (x, y) in landmark[i]:
            landmark_txt.write(str('{} {}\n'.format(x, y)))
        landmark_txt.close()
