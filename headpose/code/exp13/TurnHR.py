# -*- coding: utf-8 -*-
import cv2
import numpy as np
import sys
from lr2hr import lr2hr3DModel
from headpose import turnHR
from meshwarp import twoPointsDistance
from skimage.draw import ellipse_perimeter
from put_face_back import getLandmark3DHR, optionChooseZMax, drawPointsOnImg, imshow
from twoD_threeD_twoD_LR_MeshHair import collarDetection, hairDetection, edgePointsOnOuterEllipse
sys.path.insert(0, '../../../face_affine/exp04')
from generate_tri_txt import generateTriList
from face_affine import morph_modify_for_2D3D2D_low_resolution


def mainTurnHR():
    '''1. read img LR and HR'''
    imgLRPath = '../../github/vrn-07231340/examples/scaled/trump_12.jpg'
    imgLR = cv2.imread(imgLRPath)
    imgHRPath = '../../github/vrn-07231340/examples/trump_12.png'
    imgHR = cv2.imread(imgHRPath)
    objPath = '../../github/vrn-07231340/obj/trump_12.obj'

    '''2. img LR --> HR'''
    zRatio = 3
    objLinesHR, vLinesHR, landmarkLR, landmarkHR = lr2hr3DModel(
        imgLR, imgHR, zRatio, objPath)

    '''3. HR: origin 2D landmark --> origin 3D landmark'''
    originLandmark3DList = getLandmark3DHR(landmarkHR, vLinesHR)
    originLandmark3DList = optionChooseZMax(originLandmark3DList)
    faceLandmarkVlines = []
    for (_, _, (_, vLine)) in originLandmark3DList:
        faceLandmarkVlines.append(vLine)

    '''4. HR: collar detection'''
    _, collarPoint = collarDetection(imgHR, minSize=1200, offset=15)

    '''5. HR: hair detection'''
    originHairPoints = hairDetection(imgHR, minSize=20000)
    # minZ = maxminXYZ(objLinesHR)['minZCoord'][2]
    minZ = 200
    hairPointsVlines = []
    for (x, y) in originHairPoints:
        hairVline = '{} {} {} {} {} {} {}'.format('v', x, y, minZ, 0, 0, 0)
        hairPointsVlines.append(hairVline)

    '''6. HR: face landmarks + hair points'''
    originVlines = faceLandmarkVlines+hairPointsVlines

    '''7. HR: turn face and hair'''
    turnAngle = 10
    turnDirection = ['left', 'right']
    turnCenterMode = 'maxY'
    leftTurnVLines = turnHR(originVlines, turnAngle,
                            turnDirection[0], turnCenterMode)
    rightTurnVLines = turnHR(originVlines, turnAngle,
                             turnDirection[1], turnCenterMode)

    assert len(originVlines) == len(leftTurnVLines) == len(rightTurnVLines)

    '''8. HR: turn 3D --> 2D'''
    originPointsHR = []
    leftTurnPointsHR = []
    rightTurnPointsHR = []
    for originVline, leftTurnVLine, rightTurnVLine in zip(originVlines, leftTurnVLines, rightTurnVLines):
        _, x, y, _, _, _, _ = originVline.split()
        originPointsHR.append((float(x), float(y)))
        _, x, y, _, _, _, _ = leftTurnVLine.split()
        leftTurnPointsHR.append((float(x), float(y)))
        _, x, y, _, _, _, _ = rightTurnVLine.split()
        rightTurnPointsHR.append((float(x), float(y)))

    assert len(originPointsHR) == len(
        leftTurnPointsHR) == len(rightTurnPointsHR)

    # drawPointsOnImg(originPointsHR, imgHR, 'g')
    # drawPointsOnImg(turnPointsHR, imgHR, 'r')

    ''''9. inner ellipse'''
    left = min(originPointsHR, key=lambda x: x[0])[0]
    right = max(originPointsHR, key=lambda x: x[0])[0]
    up = min(originPointsHR, key=lambda x: x[1])[1]
    bottom = max(originPointsHR, key=lambda x: x[1])[1]
    innerEllipseCenterX = int((left+right)/2.0)
    innerEllipseCenterY = int((up+bottom)/2.0)
    innerEllipseSemiX = int((right-left)/2.0)
    innerEllipseSemiY = int((bottom-up)/2.0)
    rr, cc = ellipse_perimeter(innerEllipseCenterX, innerEllipseCenterY,
                               innerEllipseSemiY, innerEllipseSemiX, orientation=np.pi*1.5)
    innerEllipseVerts = np.dstack([rr, cc])[0]

    '''10. outer ellipse'''
    '''10.1 ratio = outer ellipse / inner ellipse'''
    '''     椭圆心和衣领的线段  和  椭圆的交点'''
    minDistance = np.inf
    for pt in innerEllipseVerts:
        distance = twoPointsDistance(pt, collarPoint)
        if distance < minDistance:
            minDistance = distance
            ratio = twoPointsDistance((innerEllipseCenterX, innerEllipseCenterY), collarPoint) / \
                twoPointsDistance(
                    (innerEllipseCenterX, innerEllipseCenterY), pt)
    '''10.2 outer ellipse'''
    outerEllipseSemiX = int(ratio*innerEllipseSemiX)
    outerEllipseSemiY = int(ratio*innerEllipseSemiY)
    rr, cc = ellipse_perimeter(innerEllipseCenterX, innerEllipseCenterY,
                               outerEllipseSemiY, outerEllipseSemiX, orientation=np.pi*1.5)
    outerEllipseVerts = np.dstack([rr, cc])[0]

    '''11. edge points on outer ellipse'''
    edgePoints = edgePointsOnOuterEllipse(outerEllipseVerts)

    '''12. final origin and turn points'''
    originPointsHRFinal = originPointsHR+edgePoints
    leftTurnPointsHRFinal = leftTurnPointsHR+edgePoints
    rightTurnPointsHRFinal = rightTurnPointsHR+edgePoints

    '''13. tri.txt'''
    triList = generateTriList(
        originPointsHRFinal, imgHR, triTxtPath='./turnTri.txt')

    '''14. warp'''
    # imgMorph = morph_modify_for_2D3D2D_low_resolution(
    #     originPointsHRFinal, turnPointsHRFinal, imgHR, triList)
    # imshow('imgMorph', imgMorph)
    # cv2.imwrite('./imgMorph.png', imgMorph)

    return originPointsHRFinal, leftTurnPointsHRFinal, rightTurnPointsHRFinal, triList


if __name__ == "__main__":
    mainTurnHR()
