import cv2
import numpy as np
from put_face_back import getLandmark2D, imshow, drawPointsOnImg
from headpose import saveObjFile, readObj


def lr2hr(imgLR, imgHR, srcLandmarkLR):
    '''1. read LR and HR img'''
    # imgLRPath = '../../github/vrn-07231340/examples/scaled/trump-12.jpg'
    # imgLR = cv2.imread(imgLRPath)
    # imgHRPath = '../../github/vrn-07231340/examples/trump-12.jpg'
    # imgHR = cv2.imread(imgHRPath)

    '''2. face landmark detection on LR and HR img'''
    landmarkLR = getLandmark2D(imgLR)
    landmarkHR = getLandmark2D(imgHR)
    landmarkLR1 = np.float32([[landmarkLR[0][0], landmarkLR[0][1]], [landmarkLR[
        8][0], landmarkLR[8][1]], [landmarkLR[16][0], landmarkLR[16][1]]])
    landmarkHR1 = np.float32([[landmarkHR[0][0], landmarkHR[0][1]], [landmarkHR[
        8][0], landmarkHR[8][1]], [landmarkHR[16][0], landmarkHR[16][1]]])
    landmarkLR2 = np.float32([[landmarkLR[27][0], landmarkLR[27][1]], [landmarkLR[
        48][0], landmarkLR[48][1]], [landmarkLR[54][0], landmarkLR[54][1]]])
    landmarkHR2 = np.float32([[landmarkHR[27][0], landmarkHR[27][1]], [landmarkHR[
        48][0], landmarkHR[48][1]], [landmarkHR[54][0], landmarkHR[54][1]]])
    landmarkLR3 = np.float32([[landmarkLR[36][0], landmarkLR[36][1]], [landmarkLR[
        45][0], landmarkLR[45][1]], [landmarkLR[33][0], landmarkLR[33][1]]])
    landmarkHR3 = np.float32([[landmarkHR[36][0], landmarkHR[36][1]], [landmarkHR[
        45][0], landmarkHR[45][1]], [landmarkHR[33][0], landmarkHR[33][1]]])

    '''3. affine transform'''
    M1 = cv2.getAffineTransform(landmarkLR1, landmarkHR1)
    M2 = cv2.getAffineTransform(landmarkLR2, landmarkHR2)
    M3 = cv2.getAffineTransform(landmarkLR3, landmarkHR3)
    M = (M1+M2+M3)/3
    landmarkHR = []
    for landmark in srcLandmarkLR:
        l = np.array([landmark[0], landmark[1], 1])
        dst = np.dot(M, l.T)
        landmarkHR.append((dst[0], dst[1]))
    # print landmarkHR
    # drawPointsOnImg(landmarkHR, imgHR, 'r')
    return landmarkHR


def lr2hr3DModel(imgLR, imgHR,  zRatio, objPath):
    '''1. read LR and HR img'''
    # imgLRPath = '../../github/vrn-07231340/examples/scaled/trump-12.jpg'
    # imgLR = cv2.imread(imgLRPath)
    # imgHRPath = '../../github/vrn-07231340/examples/trump-12.jpg'
    # imgHR = cv2.imread(imgHRPath)

    '''2. face landmark detection on LR and HR img'''
    landmarkLR = getLandmark2D(imgLR)
    landmarkHR = getLandmark2D(imgHR)
    landmarkLR1 = np.float32([[landmarkLR[0][0], landmarkLR[0][1]], [landmarkLR[
        8][0], landmarkLR[8][1]], [landmarkLR[16][0], landmarkLR[16][1]]])
    landmarkHR1 = np.float32([[landmarkHR[0][0], landmarkHR[0][1]], [landmarkHR[
        8][0], landmarkHR[8][1]], [landmarkHR[16][0], landmarkHR[16][1]]])
    landmarkLR2 = np.float32([[landmarkLR[27][0], landmarkLR[27][1]], [landmarkLR[
        48][0], landmarkLR[48][1]], [landmarkLR[54][0], landmarkLR[54][1]]])
    landmarkHR2 = np.float32([[landmarkHR[27][0], landmarkHR[27][1]], [landmarkHR[
        48][0], landmarkHR[48][1]], [landmarkHR[54][0], landmarkHR[54][1]]])
    landmarkLR3 = np.float32([[landmarkLR[36][0], landmarkLR[36][1]], [landmarkLR[
        45][0], landmarkLR[45][1]], [landmarkLR[33][0], landmarkLR[33][1]]])
    landmarkHR3 = np.float32([[landmarkHR[36][0], landmarkHR[36][1]], [landmarkHR[
        45][0], landmarkHR[45][1]], [landmarkHR[33][0], landmarkHR[33][1]]])

    '''3. affine transform'''
    M1 = cv2.getAffineTransform(landmarkLR1, landmarkHR1)
    M2 = cv2.getAffineTransform(landmarkLR2, landmarkHR2)
    M3 = cv2.getAffineTransform(landmarkLR3, landmarkHR3)
    M = (M1+M2+M3)/3

    objLines, _, _ = readObj(objPath)

    newObjLines = []
    newVLines = []
    for objLine in objLines:
        if objLine.split()[0] == 'v':
            v, x, y, z, r, g, b = objLine.split()
            l = np.array([float(x), float(y), 1])
            dst = np.dot(M, l.T)
            newX = float(dst[0])
            newY = float(dst[1])
            newZ = float(z)*zRatio
            objLine = '{} {} {} {} {} {} {}'.format(
                v, newX, newY, newZ, r, g, b)
            newVLines.append(objLine)
        newObjLines.append(objLine)
    saveObjFile('HR', newObjLines, objPath)

    landmarkHR = []
    for landmark in landmarkLR:
        l = np.array([landmark[0], landmark[1], 1])
        dst = np.dot(M, l.T)
        landmarkHR.append((dst[0], dst[1]))

    return newObjLines, newVLines, landmarkLR, landmarkHR


def mainLR2HR():
    imgLRPath = '../../github/vrn-07231340/examples/scaled/trump-12.jpg'
    imgLR = cv2.imread(imgLRPath)
    imgHRPath = '../../github/vrn-07231340/examples/trump-12.jpg'
    imgHR = cv2.imread(imgHRPath)
    srcLandmarkLR = getLandmark2D(imgLR)
    lr2hr(imgLR, imgHR, srcLandmarkLR)


if __name__ == "__main__":
    mainLR2HR()
