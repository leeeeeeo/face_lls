import cv2
import os
import dlib
from face_affine_utils import *
from face_affine import *
from faceBox import *
import time


def mainTalkingPhoto():
    imgNames = []
    outputFolder = './output/talkingphoto/talkingphoto_{}'.format(
        time.strftime("%d%H%M", time.localtime()))
    if not os.path.exists(outputFolder):
        os.mkdir(outputFolder)
    trumpImgPath = './data/talkingphoto/crop640.png'
    trumpImg = cv2.imread(trumpImgPath)
    eightEdgePoints, faceRect, maskRect = faceBoundingbox(trumpImg)
    # cv2.imshow('a', faceRect)
    # cv2.waitKey(0)
    targetFolder = './data/talkingphoto/IMG_2294_buffer'
    triTxtPath = './data/source/mytri.txt'
    videoWriter = cv2.VideoWriter(
        '{}/talkingphoto.mp4'.format(outputFolder), cv2.VideoWriter_fourcc(*'mp4v'), 25, (640, 640))
    ptsOriginal = readPoints('{}.txt'.format(trumpImgPath))
    imgNames = natsortFolder(targetFolder)
    for imgName in imgNames:
        print imgName
        ptsTarget = readPoints('{}/{}.txt'.format(targetFolder, imgName))
        imgMorph = morph(ptsOriginal, ptsTarget, trumpImg, triTxtPath)

        imgMorph = trumpImg*(1-maskRect)+imgMorph*maskRect
        drawLandmark(imgMorph)
        drawLandmark(imgMorph, targetPoints=ptsTarget)
        cv2.imwrite('{}/{}'.format(outputFolder, imgName), imgMorph)
        cv2.imshow('a', imgMorph)
        # cv2.waitKey(0)
        videoWriter.write(imgMorph)


if __name__ == "__main__":
    mainTalkingPhoto()
    # videoToImg('/Users/lls/Documents/face/data/talkingphoto/IMG_2293.mp4',
    #            '/Users/lls/Documents/face/data/talkingphoto/IMG_2293')
    # videoToImg('/Users/lls/Documents/face/data/talkingphoto/IMG_2294.mp4',
    #            '/Users/lls/Documents/face/data/talkingphoto/IMG_2294')
