# -*- coding: utf-8 -*-
import cv2
from skimage.draw import ellipse_perimeter, ellipsoid
from put_face_back import getLandmark3DHR, optionChooseZMax, drawPointsOnImg
from lr2hr import lr2hr3DModel
from twoD_threeD_twoD_HR_NodHair import collarDetection, hairDetection, edgePointsOnOuterEllipse
from headpose import maxminXYZ, turnHR
from ellipsoidHead import ellipsoidHead
import numpy as np
from meshwarp import twoPointsDistance
import sys
sys.path.insert(0, '../../../face_affine/exp04')
from generate_tri_txt import generateTriList


def mainTurnHRHairOnEllipsoid():
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
    '''5.1 hair vLines z = -1'''
    hairPointsVlines = []
    for (x, y) in originHairPoints:
        hairVline = '{} {} {} {} {} {} {}'.format('v', x, y, -1, 0, 0, 0)
        hairPointsVlines.append(hairVline)

    '''6. HR: origin vLines = face landmark vLines + hair vLines'''
    originVlines = faceLandmarkVlines+hairPointsVlines

    '''7. HR: origin points'''
    originPointsHR = []
    for originVline in originVlines:
        _, x, y, _, _, _, _ = originVline.split()
        originPointsHR.append((float(x), float(y)))

    '''8. inner ellipse'''
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
    # drawPointsOnImg(innerEllipseVerts, imgHR, 'g')

    '''9. outer ellipse'''
    '''9.1 ratio = outer ellipse / inner ellipse'''
    '''     椭圆心和衣领的线段  和  椭圆的交点'''
    minDistance = np.inf
    for pt in innerEllipseVerts:
        distance = twoPointsDistance(pt, collarPoint)
        if distance < minDistance:
            minDistance = distance
            ratio = twoPointsDistance((innerEllipseCenterX, innerEllipseCenterY), collarPoint) / \
                twoPointsDistance(
                    (innerEllipseCenterX, innerEllipseCenterY), pt)
    '''9.2 outer ellipse'''
    outerEllipseSemiX = int(ratio*innerEllipseSemiX)
    outerEllipseSemiY = int(ratio*innerEllipseSemiY)
    rr, cc = ellipse_perimeter(innerEllipseCenterX, innerEllipseCenterY,
                               outerEllipseSemiY, outerEllipseSemiX, orientation=np.pi*1.5)
    outerEllipseVerts = np.dstack([rr, cc])[0]
    # drawPointsOnImg(outerEllipseVerts, imgHR, 'r')

    '''10. head ellipoid (outer ellipse)'''
    '''!!! F I N A L L Y !!!'''
    maxminDict = maxminXYZ(vLinesHR)
    maxZ = maxminDict['maxZCoord'][2]
    minZ = maxminDict['minZCoord'][2]
    ellipsoidHeadVLines, ellipsoidHeadXY, ellipsoidHeadXZ, ellipsoidHeadYZ, ellipsoidHeadXYZ = ellipsoidHead(
        outerEllipseSemiX, outerEllipseSemiY, maxZ - minZ, innerEllipseCenterX, innerEllipseCenterY, minZ)
    # drawPointsOnImg(ellipsoidHeadXY, imgHR, 'b')
    # drawPointsOnImg(ellipsoidHeadXZ, imgHR, 'r')
    # drawPointsOnImg(ellipsoidHeadYZ, imgHR, 'g')

    '''11. hair vLines: z = -1 --> z on ellipsoid'''
    newHairPointsVlines = []
    for hairPointsVline in hairPointsVlines:
        v, x, y, z, r, g, b = hairPointsVline.split()
        maxZ = 0
        for (x0, y0, z0) in ellipsoidHeadXYZ:
            if int(float(x)) == int(float(x0)) and int(float(y)) == int(float(y0)) and float(z0) > 0:
                # print int(float(x0)), int(float(y0)), int(float(z0))
                if int(float(z0)) > int(float(z)) and int(float(z0)) > maxZ:
                    maxZ = int(float(z0))
                    print maxZ
        newHairPointsVline = '{} {} {} {} {} {} {}'.format(
            v, x, y, maxZ, r, g, b)
        newHairPointsVlines.append(newHairPointsVline)
    # print hairPointsVlines
    # print newHairPointsVlines
    assert len(hairPointsVlines) == len(newHairPointsVlines)

    '''12. HR: origin vLines = face landmark vLines + new hair vLines'''
    originVlines = faceLandmarkVlines+newHairPointsVlines

    '''13. HR: turn face and hair'''
    turnAngle = 10
    turnDirection = ['left', 'right']
    turnCenterMode = 'maxY'
    leftTurnVLines = turnHR(originVlines, turnAngle,
                            turnDirection[0], turnCenterMode)
    rightTurnVLines = turnHR(originVlines, turnAngle,
                             turnDirection[1], turnCenterMode)

    assert len(originVlines) == len(leftTurnVLines) == len(rightTurnVLines)

    '''14. HR: turn 3D --> 2D'''
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

    '''15. edge points on outer ellipse'''
    edgePoints = edgePointsOnOuterEllipse(outerEllipseVerts)

    '''16. final origin and turn points'''
    originPointsHRFinal = originPointsHR+edgePoints
    leftTurnPointsHRFinal = leftTurnPointsHR+edgePoints
    rightTurnPointsHRFinal = rightTurnPointsHR+edgePoints

    '''17. tri.txt'''
    triList = generateTriList(
        originPointsHRFinal, imgHR, triTxtPath='./turnTri.txt')

    return originPointsHRFinal, leftTurnPointsHRFinal, rightTurnPointsHRFinal, triList


if __name__ == "__main__":
    mainTurnHRHairOnEllipsoid()
