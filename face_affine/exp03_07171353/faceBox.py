import cv2
import os
import dlib
from natsort import natsorted
from face_affine_utils import *
from face_affine import *
import time


detector = dlib.get_frontal_face_detector()


def rect_to_bb(rect):
    topLeftX = int(rect.left())
    topLeftY = int(rect.top())
    bottomRightX = int(rect.right())
    bottomRightY = int(rect.bottom())
    width = bottomRightX-topLeftX
    height = bottomRightY-topLeftY
    return (topLeftX, topLeftY, bottomRightX, bottomRightY, width, height)


def faceBoundingbox(faceImg):
    if faceImg.shape[2] != 1:
        gray = cv2.cvtColor(faceImg, cv2.COLOR_BGR2GRAY)
    else:
        gray = faceImg
    rects = detector(gray, 1)
    (topLeftX, topLeftY, bottomRightX, bottomRightY,
     width, height) = rect_to_bb(rects[0])
    topLeftX = int(topLeftX-width*0.2)
    topLeftY = int(topLeftY-height*0.5)
    centerX = int((topLeftX+bottomRightX)/2)
    centerY = int((topLeftY+bottomRightY)/2)
    bottomRightX = int(bottomRightX+width*0.2)
    bottomRightY = int(bottomRightY+height*0.2)
    if topLeftX < 0:
        topLeftX = 0
    if topLeftY < 0:
        topLeftY = 0
    cv2.circle(faceImg, (centerX, centerY), 3, (0, 255, 0), 1, -1)
    cv2.rectangle(faceImg, (topLeftX, topLeftY),
                  (bottomRightX, bottomRightY), (0, 255, 0), 2)
    cv2.imshow('a', faceImg)
    cv2.waitKey(0)
    edgePoints = [(topLeftX, topLeftY), (topLeftX, centerY),
                  (topLeftX, bottomRightY), (centerX, bottomRightY), (bottomRightX, bottomRightY), (bottomRightX, centerY), (bottomRightX, topLeftY), (centerX, topLeftY)]
    # cv2.imshow('b', faceImg[topLeftY: bottomRightY, topLeftX: bottomRightX, :])
    maskRect = np.zeros(faceImg.shape, dtype=np.uint8)
    maskRect[topLeftY: bottomRightY,
             topLeftX: bottomRightX, :] = (1, 1, 1)
    # cv2.imshow('c', maskRect)
    return edgePoints, faceImg[topLeftY: bottomRightY, topLeftX: bottomRightX, :], maskRect


def trumpHeadpose(imgFolder, triTxtPath, headpose):
    imgNames = []
    videoWriter = cv2.VideoWriter(
        'faceBox_{}_{}.mp4'.format(headpose, time.strftime("%M%S", time.localtime())), cv2.VideoWriter_fourcc(*'mp4v'), 25, (1920, 1080))
    for root, folder, files in os.walk(imgFolder):
        for fileName in files:
            if fileName.endswith('.png'):
                imgNames.append(fileName)
    imgNames = natsorted(imgNames)
    for imgName in imgNames:
        print imgName
        img = cv2.imread('{}/{}'.format(imgFolder, imgName))
        eightEdgePoints, faceRect, maskRect = faceBoundingbox(img)
        ptsOriginal = readPts(
            '{}/{}.pts'.format(imgFolder, imgName.split('.')[0]))
        ptsTarget = readPoints(
            '{}/{}_{}.txt'.format(imgFolder, imgName.split('.')[0], headpose))
        ptsContour = readPoints(
            '{}/{}_{}.txt'.format(imgFolder, imgName.split('.')[0], headpose), contour='FACE_CONTOUR_LANDMARKS')
        ptsOriginal = ptsOriginal+eightEdgePoints
        ptsTarget = ptsTarget+eightEdgePoints
        '''morph entire image'''
        imgMorph = morph(ptsOriginal, ptsTarget,
                         img, triTxtPath)
        imgMorph = img*(1-maskRect)+imgMorph*maskRect
        # imgMorph = drawLandmark(imgMorph)
        videoWriter.write(imgMorph)
        # cv2.imshow("imgMorph", imgMorph)
        '''morph face area'''
        # maskContour, imgRecover = recoverMask(
        #     ptsContour, img, imgMorph)
        # cv2.imshow('maskContour', maskContour)
        # imgRecover = drawLandmark(imgRecover)
        # imgRecover = drawLandmark(imgRecover, ptsTarget)
        # cv2.imshow("Recoverd Face", np.uint8(imgRecover))

        # cv2.waitKey(1)


def mainFaceBox():
    imgFolder = './data/trump/trump'
    triTxtPath = './data/source/mytri.txt'
    headpose = ['left', 'right', 'down']
    # trumpHeadpose(imgFolder, triTxtPath, headpose[1])
    '''FACE BBOX'''
    imgPath = '/Users/lls/Documents/face/data/talkingphoto/crop640.png'
    img = cv2.imread(imgPath)
    eightEdgePoints, faceRect, maskRect = faceBoundingbox(img)


if __name__ == "__main__":
    mainFaceBox()
