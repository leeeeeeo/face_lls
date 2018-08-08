import cv2
import numpy as np
from headpose import readObj, turn, projection
from put_face_back import getLandmark2D, getLandmark3D, optionChooseZMax, getTurnLandmark3D, objLines2vLines


def main2D_3D_2D_TurnFace():
    '''1. read img and obj'''
    imgPath = '../../github/vrn-07231340/examples/scaled/trump-12.jpg'
    img = cv2.imread(imgPath)
    objPath = '../../github/vrn-07231340/obj/trump-12.obj'
    objLines, vLines, fLines = readObj(objPath)

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

    '''3. turn'''
    turnAngle = 15
    turnCenterMode = 'maxY'
    turnDirection = 'right'
    turnObjLines = turn(objPath, turnAngle, turnDirection, turnCenterMode)
    # projection(turnObjLines, imgPath)

    '''4. turn 3D landmark list'''
    '''turnLandmark3DList: [[x0, y0], [x0, y0, z0], [x1, y1], [x1, y1, z1], (line, vLine)]'''
    '''len(turnLandmark3DList): number of landmarks both on 2D and 3D'''
    '''[x0, y0]: original landmark on 2D img'''
    '''[x0, y0, z0]: original landmark on 3D model'''
    '''[x1, y1]: turn landmark on 2D img'''
    '''[x1, y1, z1]: turn landmark on 3D model'''
    '''(line, vLine): the line(th) turn vLine'''
    turnVLines = objLines2vLines(turnObjLines)
    turnLandmark3DList = getTurnLandmark3D(turnVLines, originLandmark3DList)

    '''5. turn 3D landmark --> turn 2D landmark'''
    turnLandmark2D = []
    for turnLandmark3D in turnLandmark3DList:
        turnLandmark2D.append(turnLandmark3D[2])

    '''6. origin 2D landmarks'''
    originLandmark2D = []
    for turnLandmark3D in turnLandmark3DList:
        originLandmark2D.append((turnLandmark3D[0][0], turnLandmark3D[0][1]))
    assert len(originLandmark2D) == len(turnLandmark2D)

    print len(turnLandmark2D)


if __name__ == "__main__":
    main2D_3D_2D_TurnFace()
