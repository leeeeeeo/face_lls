# -*- coding: utf-8 -*-
import cv2
import numpy as np
from headpose import readObj, nod, nodHairHR, nodHairHR_m
from put_face_back import imshow, getLandmark2D, getLandmark3D, optionChooseZMax, objLines2vLines, getNodLandmark3D, drawPointsOnImg
from twoD_threeD_twoD_LR_MeshHair import collarDetection, hairDetection, edgePointsOnOuterEllipse
from skimage.draw import ellipse_perimeter
from meshwarp import twoPointsDistance
import sys
sys.path.insert(0, '../../../face_affine/exp04')
from generate_tri_txt import generateTriList
from face_affine import morph_modify_for_2D3D2D_low_resolution
from lr2hr import lr2hr


def main2D_3D_2D_HR_NodHair():
    '''1. read origin obj and image'''
    objPath = '../../github/vrn-07231340/obj/trump-12.obj'
    objLines, vLines, fLines = readObj(objPath)
    imgLRPath = '../../github/vrn-07231340/examples/scaled/trump-12.jpg'
    imgLR = cv2.imread(imgLRPath)
    imgHRPath = '../../github/vrn-07231340/examples/trump-12.jpg'
    imgHR = cv2.imread(imgHRPath)

    '''2. origin LR 2D landmark --> origin 3D landmark'''
    '''2.1 origin LR 2D landmark list [x, y]'''
    originLandmark2DLRList = getLandmark2D(imgLR)
    '''2.2 origin 3D landmark list [[[x, y], [(line1, vLine1), (line2, vLine2)]], ... ]'''
    originLandmark3DList = getLandmark3D(originLandmark2DLRList, vLines)
    '''2.3 OPTION 1: 相同(x, y)的情况下，选z最大的3D landmark'''
    originLandmark3DList = optionChooseZMax(originLandmark3DList)

    '''3. 3D model nod'''
    nodAngle = 10
    nodCenterMode = 'maxY'
    nodObjLines = nod(objPath, nodAngle, nodCenterMode)

    '''4. nod 3D landmark list'''
    nodVLines = objLines2vLines(nodObjLines)
    nodLandmark3DList = getNodLandmark3D(nodVLines, originLandmark3DList)

    '''5. nod 3D landmark --> nod LR 2D landmark'''
    nodLandmark2DLR = [i[2] for i in nodLandmark3DList]

    '''6. origin 2D landmark'''
    originLandmark2DLR = [(i[0][0], i[0][1]) for i in nodLandmark3DList]

    assert len(originLandmark2DLR) == len(nodLandmark2DLR)

    '''7. LR --> HR: origin and nod face landmark'''
    originLandmark2DHR = lr2hr(imgLR, imgHR, originLandmark2DLR)
    nodLandmark2DHR = lr2hr(imgLR, imgHR, nodLandmark2DLR)

    '''8. HR: collar detection'''
    _, collarPoint = collarDetection(imgHR, minSize=1200, offset=15)

    '''9. HR: hair detection'''
    originHairPoints = hairDetection(imgHR, minSize=20000)

    '''10. HR: nod hair points'''
    '''??? nod center ???'''
    nodAngleHair = 18
    # drawPointsOnImg(originHairPoints, imgHR, 'r', radius=3)
    # nodHairPoints = nodHairHR(
    # objLines, nodAngleHair, nodCenterMode, originHairPoints, minZ=100.0)
    nodHairPoints = nodHairHR_m(
        nodAngleHair, collarPoint, originHairPoints, minZ=100.0)
    # for (x, y), (i, j) in zip(nodHairPoints, nodHairPoints_m):
    #     if x == i and y == j:
    #         pass
    #     else:
    #         print 'yes'

    '''10. HR: all origin points and nod points'''
    originPoints = originLandmark2DHR+originHairPoints
    nodPoints = nodLandmark2DHR+nodHairPoints
    assert len(originPoints) == len(nodPoints)
    # drawPointsOnImg(originPoints, imgHR, 'g', radius=2)

    '''11. inner ellipse'''
    left = min(originPoints, key=lambda x: x[0])[0]
    right = max(originPoints, key=lambda x: x[0])[0]
    up = min(originPoints, key=lambda x: x[1])[1]
    bottom = max(originPoints, key=lambda x: x[1])[1]
    innerEllipseCenterX = int((left+right)/2.0)
    innerEllipseCenterY = int((up+bottom)/2.0)
    innerEllipseSemiX = int((right-left)/2.0)
    innerEllipseSemiY = int((bottom-up)/2.0)
    rr, cc = ellipse_perimeter(innerEllipseCenterX, innerEllipseCenterY,
                               innerEllipseSemiY, innerEllipseSemiX, orientation=np.pi*1.5)
    innerEllipseVerts = np.dstack([rr, cc])[0]

    '''12. outer ellipse'''
    '''12.1 ratio = outer ellipse / inner ellipse'''
    '''     椭圆心和衣领的线段  和  椭圆的交点'''
    minDistance = np.inf
    for pt in innerEllipseVerts:
        distance = twoPointsDistance(pt, collarPoint)
        if distance < minDistance:
            minDistance = distance
            ratio = twoPointsDistance((innerEllipseCenterX, innerEllipseCenterY), collarPoint) / \
                twoPointsDistance(
                    (innerEllipseCenterX, innerEllipseCenterY), pt)
    '''12.2 outer ellipse'''
    outerEllipseSemiX = int(ratio*innerEllipseSemiX)
    outerEllipseSemiY = int(ratio*innerEllipseSemiY)
    rr, cc = ellipse_perimeter(innerEllipseCenterX, innerEllipseCenterY,
                               outerEllipseSemiY, outerEllipseSemiX, orientation=np.pi*1.5)
    outerEllipseVerts = np.dstack([rr, cc])[0]

    '''13. edge points on outer ellipse'''
    edgePoints = edgePointsOnOuterEllipse(outerEllipseVerts)

    '''14. final origin and nod HR points '''
    originPointsFinal = originPoints+edgePoints
    nodPointsFinal = nodPoints+edgePoints
    assert len(originPointsFinal) == len(nodPointsFinal)
    drawPointsOnImg(originPointsFinal, imgHR, 'b', radius=2)
    drawPointsOnImg(originPoints, imgHR, 'r', radius=2)
    drawPointsOnImg(nodPointsFinal, imgHR, 'g', radius=2)

    '''15. tri.txt'''
    triList = generateTriList(originPointsFinal, imgHR,
                              triTxtPath='./nodTri.txt')

    '''16. warp'''
    imgMorph = morph_modify_for_2D3D2D_low_resolution(
        originPointsFinal, nodPointsFinal, imgHR, triList)
    imshow('imgMorph', imgMorph)
    cv2.imwrite('./imgMorph_{}2.png'.format(nodAngleHair), imgMorph)

    return originPointsFinal, nodPointsFinal, triList


if __name__ == "__main__":
    main2D_3D_2D_HR_NodHair()
