# -*- coding: utf-8 -*-
import cv2
import numpy as np
from headpose import readObj, maxminXYZ, nodHair
from put_face_back import drawPointsOnImg
import types


def main2D_3D_2D_Hair():
    '''1. read origin img and obj'''
    imgPath = '../../github/vrn-07231340/examples/scaled/trump-12.jpg'
    img = cv2.imread(imgPath)
    objPath = '../../github/vrn-07231340/obj/trump-12.obj'
    objLines, vLines, fLines = readObj(objPath)

    '''2. manually select 12 landmarks for hair region'''
    hairLandmark_0 = (33, 52)
    hairLandmark_1 = (42, 23)
    hairLandmark_2 = (91, 0)
    hairLandmark_3 = (138, 9)
    hairLandmark_4 = (149, 36)
    hairLandmark_5 = (132, 51)
    hairLandmark_6 = (119, 37)
    hairLandmark_7 = (98, 32)
    hairLandmark_8 = (68, 44)
    hairLandmark_9 = (49, 65)
    hairLandmark_10 = (76, 20)
    hairLandmark_11 = (114, 17)
    hairLandmark2DList = [hairLandmark_0, hairLandmark_1, hairLandmark_2, hairLandmark_3, hairLandmark_4,
                          hairLandmark_5, hairLandmark_6, hairLandmark_7, hairLandmark_8, hairLandmark_9, hairLandmark_10, hairLandmark_11]
    # hairLandmark = (102, 33)
    # hairLandmark2DList = [hairLandmark]

    '''3. 2D hair landmark --> 3D hair landmark'''
    '''3.1 3D hair landmark list: (x, y, z)'''
    maxminDict = maxminXYZ(objLines)
    # minZ = float(maxminDict['maxZCoord'][2])
    minZ = 50.0
    hairLandmark3DList = []
    for hairLandmark2D in hairLandmark2DList:
        hairLandmark3D = (float(hairLandmark2D[0]),
                          float(hairLandmark2D[1]), minZ)
        hairLandmark3DList.append(hairLandmark3D)
    '''3.2 hair landmark color list: (r, g, b)'''
    colorList = []
    for hairLandmark2D in hairLandmark2DList:
        r = img[hairLandmark2D[0], hairLandmark2D[1], 2]/255.0
        g = img[hairLandmark2D[0], hairLandmark2D[1], 1]/255.0
        b = img[hairLandmark2D[0], hairLandmark2D[1], 0]/255.0
        color = (r, g, b)
        colorList.append(color)
    '''3.3 3D vLines'''
    hairVLines = []
    for (x, y, z), (r, g, b) in zip(hairLandmark3DList, colorList):
        hairVLine = 'v {} {} {} {} {} {}'.format(x, y, z, r, g, b)
        hairVLines.append(hairVLine)
    '''3.4 write vLines txt'''
    hairVLinesTxtPath = './hairVLines.txt'
    hairVLinesTxt = open(hairVLinesTxtPath, 'w')
    for hairVLine in hairVLines:
        hairVLinesTxt.write(hairVLine+'\n')
    hairVLinesTxt.close()

    '''4. nod hair'''
    nodAngle = 15
    nodCenterMode = 'maxY'
    nodHairVLines = nodHair(objPath, nodAngle, nodCenterMode,
                            hairVLines, hairVLinesTxtPath)

    '''5. 2D nod hair landmark'''
    nodHairLandmark2DList = []
    for nodHairVLine in nodHairVLines:
        _, x, y, _, _, _, _ = nodHairVLine.split()
        nodHairLandmark2DList.append((float(x), float(y)))

    return hairLandmark2DList, nodHairLandmark2DList
    # drawPointsOnImg(hairLandmark2DList, img, 'g')
    # drawPointsOnImg(nodHairLandmark2DList, img, 'r')


if __name__ == "__main__":
    main2D_3D_2D_Hair()
