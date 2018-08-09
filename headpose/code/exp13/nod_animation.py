import sys
from twoD_threeD_twoD_LR_Hair import main2D_3D_2D_LR_Hair
from twoD_threeD_twoD_HR_NodHair import main2D_3D_2D_HR_NodHair
from NodHR import mainNodHR
sys.path.insert(0, '../../../face_affine/exp04')
from face_affine import changeExpression, readPoints, saveAnimation
import cv2


def mainNodAnimationHR():
    time = 6
    # originLandmark2DHR, nodLandmark2DHR, _ = main2D_3D_2D_HR_NodHair()
    originLandmark2DHR, nodLandmark2DHR, _ = mainNodHR()
    imgHRPath = '../../github/vrn-07231340/examples/trump-12.jpg'
    imgHR = cv2.imread(imgHRPath)
    triTxtPath = './nodTri.txt'
    frameList = changeExpression(originLandmark2DHR,
                                 nodLandmark2DHR, triTxtPath, imgHR)
    saveAnimation(frameList, 25, './nodHR.mp4', imgHR, time=time, reverse=True)


def mainNodAnimation():
    HR = True
    if HR == True:
        originLandmark2DHR, nodLandmark2DHR, _ = main2D_3D_2D_LR_Hair()
        # imgPath = '../../github/vrn-07231340/examples/scaled/trump-12.jpg'
        # img = cv2.imread(imgPath)
        imgHRPath = '../../github/vrn-07231340/examples/trump-12.jpg'
        imgHR = cv2.imread(imgHRPath)

        triTxtPath = './nodTri.txt'
        frameList = changeExpression(originLandmark2DHR,
                                     nodLandmark2DHR, triTxtPath, imgHR)
        saveAnimation(frameList, 25, './nodHR.mp4', imgHR, reverse=True)
    else:
        originLandmark2D, nodLandmark2D, _ = main2D_3D_2D_LR_Hair()
        imgPath = '../../github/vrn-07231340/examples/scaled/trump-12.jpg'
        img = cv2.imread(imgPath)
        # imgHRPath = '../../github/vrn-07231340/examples/trump-12.jpg'
        # imgHR = cv2.imread(imgHRPath)

        triTxtPath = './nodTri.txt'
        frameList = changeExpression(originLandmark2D,
                                     nodLandmark2D, triTxtPath, img)
        saveAnimation(frameList, 25, './nod.mp4', img, reverse=True)


if __name__ == "__main__":
    # mainNodAnimation()
    mainNodAnimationHR()
