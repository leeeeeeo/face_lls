# -*- coding: utf-8 -*-
import cv2
import sys
import numpy as np
sys.path.insert(0, '../../../face_affine/exp04')
from generate_tri_txt import generateTriList
from skimage.draw import ellipse_perimeter
from headpose import readObj, maxminXYZ, nodHR
from lr2hr import lr2hr3DModel
from meshwarp import twoPointsDistance
from face_affine import morph_modify_for_2D3D2D_low_resolution
from put_face_back import getLandmark3DHR, optionChooseZMax, getNodLandmark3D, drawPointsOnImg, imshow
from twoD_threeD_twoD_LR_MeshHair import hairDetection, collarDetection, edgePointsOnOuterEllipse


def mainNodHR():
    '''1. read img LR and HR'''
    imgLRPath = '../../github/vrn-07231340/examples/scaled/trump-12.jpg'
    imgLR = cv2.imread(imgLRPath)
    imgHRPath = '../../github/vrn-07231340/examples/trump-12.jpg'
    imgHR = cv2.imread(imgHRPath)
    objPath = '../../github/vrn-07231340/obj/trump-12.obj'

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
    minZ = 180
    hairPointsVlines = []
    for (x, y) in originHairPoints:
        hairVline = '{} {} {} {} {} {} {}'.format('v', x, y, minZ, 0, 0, 0)
        hairPointsVlines.append(hairVline)

    '''6. HR: face landmarks + hair points'''
    originVlines = faceLandmarkVlines+hairPointsVlines

    '''7. HR: nod face and hair'''
    nodAngle = 15
    nodCenterMode = 'maxY'
    nodVlines = nodHR(originVlines, nodAngle, nodCenterMode)

    assert len(originVlines) == len(nodVlines)

    '''8. HR: nod 3D --> 2D'''
    originPointsHR = []
    nodPointsHR = []
    for originVline, nodVline in zip(originVlines, nodVlines):
        _, x, y, _, _, _, _ = originVline.split()
        originPointsHR.append((float(x), float(y)))
        _, x, y, _, _, _, _ = nodVline.split()
        nodPointsHR.append((float(x), float(y)))

    assert len(originPointsHR) == len(nodPointsHR)

    # drawPointsOnImg(originPointsHR, imgHR, 'g', radius=2)
    # drawPointsOnImg(nodPointsHR, imgHR, 'r', radius=2)

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

    '''12. final origin and nod points'''
    originPointsHRFinal = originPointsHR+edgePoints
    nodPointsHRFinal = nodPointsHR+edgePoints
    assert len(originPointsHRFinal) == len(nodPointsHRFinal)

    '''13. tri.txt'''
    triList = generateTriList(
        originPointsHRFinal, imgHR, triTxtPath='./nodTri.txt')

    '''14. warp'''
    imgMorph = morph_modify_for_2D3D2D_low_resolution(
        originPointsHRFinal, nodPointsHRFinal, imgHR, triList)
    imshow('imgMorph', imgMorph)
    cv2.imwrite('./imgMorph.png', imgMorph)

    return originPointsHRFinal, nodPointsHRFinal, triList


if __name__ == "__main__":
    mainNodHR()
