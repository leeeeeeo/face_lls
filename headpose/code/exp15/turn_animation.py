import time
import sys
from TurnHR import mainTurnHR
sys.path.insert(0, '../../../face_affine/exp04')
from face_affine import changeExpression, readPoints, saveAnimation
import cv2


def mainTurnAnimationHR():
    num = 6
    originLandmark2DHR, leftTurnLandmark2DHR, rightTurnLandmark2DHR, _ = mainTurnHR()
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
    saveAnimation(finalFrameList, 25, './turnHR_{}.mp4'.format(str(time.strftime('%m%d%H%M', time.localtime()))),
                  imgHR, time=num, reverse=False)


if __name__ == "__main__":
    mainTurnAnimationHR()
