import time
import sys
from TurnHR import mainTurnHR
from TurnHRHairPointsOnEllipsoid import mainTurnHRHairOnEllipsoid
sys.path.insert(0, '../../../face_affine/exp04')
from face_affine import changeExpression, readPoints, saveAnimation
import cv2
import os


def mainTurnAnimationHR():
    num = 6
    # originLandmark2DHR, leftTurnLandmark2DHR, rightTurnLandmark2DHR, _ = mainTurnHR()
    originLandmark2DHR, leftTurnLandmark2DHR, rightTurnLandmark2DHR, _ = mainTurnHRHairOnEllipsoid()
    imgHRPath = '../../github/vrn-07231340/examples/trump_12.png'
    imgHR = cv2.imread(imgHRPath)
    triTxtPath = './turnTri.txt'
    leftFrameList = changeExpression(originLandmark2DHR,
                                     leftTurnLandmark2DHR, triTxtPath, imgHR)
    reversedLeftFrameList = [frame for frame in reversed(leftFrameList)]
    rightFrameList = changeExpression(originLandmark2DHR,
                                      rightTurnLandmark2DHR, triTxtPath, imgHR)
    reversedRightFrameList = [frame for frame in reversed(rightFrameList)]
    finalFrameList = leftFrameList+reversedLeftFrameList + \
        rightFrameList+reversedRightFrameList
    videoPath = '../../output/{}/turn/turnHR_{}.mp4'.format(str(time.strftime(
        '%m%d', time.localtime())), str(time.strftime('%H%M', time.localtime())))
    if not os.path.exists(os.path.dirname(videoPath)):
        os.makedirs(os.path.dirname(videoPath))
    saveAnimation(finalFrameList, 25, videoPath,
                  imgHR, time=num, reverse=False)


if __name__ == "__main__":
    mainTurnAnimationHR()
