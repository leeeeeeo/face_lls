# -*- coding: utf-8 -*-
from headpose import nod, projection, readObj
from put_face_back import getLandmark2D, getLandmark3D, objLines2vLines, getNodedLandmark3D, findvLine, optionChooseZMax, drawPointsOnImg, imshow
import cv2
import sys
sys.path.insert(0, '../../../face_affine/exp04')
from face_affine_utils import addEdgeLandmark
from generate_tri_txt import generateTriList
from face_affine import morph_modify_for_2D3D2D_low_resolution


def main2D_3D_2D_LowResolution():
    '''1. read origin obj and image'''
    objPath = '../../github/vrn-07231340/obj/trump-12.obj'
    objLines, vLines, fLines = readObj(objPath)
    imgPath = '../../github/vrn-07231340/examples/scaled/trump-12.jpg'
    img = cv2.imread(imgPath)

    '''2. origin 2D landmark --> origin 3D landmark'''
    '''   BUT ONE 2D LANDMARK --> MORE THAN ONE 3D LANDMARK'''
    '''   HOW TO CHOOSE THE BEST ONE ?'''
    '''   OPTION 1: 选Z最大的3D landmark'''
    '''2.1 origin 2D landmark [x, y]'''
    originLandmark2DList = getLandmark2D(img)
    '''2.2 origin 3D landmark list [[[x, y], [(line1, vLine1), (line2, vLine2)]], ... ]'''
    originLandmark3DList = getLandmark3D(originLandmark2DList, vLines)
    '''2.3 OPTION 1: 选Z最大的3D landmark'''
    originLandmark3DList = optionChooseZMax(originLandmark3DList)

    '''3. nod'''
    nodAngle = 15
    nodCenterMode = 'maxY'
    nodedObjLines = nod(objPath, nodAngle, nodCenterMode)

    '''4. noded 3D landmark list'''
    '''nodedLandmark3DList: [[x0, y0], [x0, y0, z0], [x1, y1], [x1, y1, z1], (line, vLine)]'''
    '''len(nodedLandmark3DList): number of landmarks both on 2D and 3D'''
    '''[x0, y0]: original landmark on 2D img'''
    '''[x0, y0, z0]: original landmark on 3D model'''
    '''[x1, y1]: noded landmark on 2D img'''
    '''[x1, y1, z1]: noded landmark on 3D model'''
    '''(line, vLine): the line(th) noded vLine'''
    nodedVLines = objLines2vLines(nodedObjLines)
    nodedLandmark3DList = getNodedLandmark3D(nodedVLines, originLandmark3DList)

    '''5. noded 3D landmark --> noded 2D landmark'''
    nodedLandmark2D = []
    for nodedLandmark3D in nodedLandmark3DList:
        nodedLandmark2D.append(nodedLandmark3D[2])
    nodedLandmark2D = addEdgeLandmark(nodedLandmark2D, img)

    '''6. delaunay triangle for (origin landmarks which are both on 2D and 3D) + (eight edge points)'''
    originLandmark2D = []
    for nodedLandmark3D in nodedLandmark3DList:
        originLandmark2D.append((nodedLandmark3D[0][0], nodedLandmark3D[0][1]))
    originLandmark2D = addEdgeLandmark(originLandmark2D, img)
    assert len(originLandmark2D) == len(nodedLandmark2D)
    triList = generateTriList(originLandmark2D, img)
    '''7. warp'''
    imgMorph = morph_modify_for_2D3D2D_low_resolution(
        originLandmark2D, nodedLandmark2D, img, triList)
    imshow('imgMorph', imgMorph)
    cv2.imwrite('./noded.jpg', imgMorph)


if __name__ == "__main__":
    main2D_3D_2D_LowResolution()
