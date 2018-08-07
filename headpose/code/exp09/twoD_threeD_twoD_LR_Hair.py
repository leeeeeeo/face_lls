# -*- coding: utf-8 -*-
from headpose import nod, projection, readObj
from put_face_back import getLandmark2D, getLandmark3D, objLines2vLines, getNodLandmark3D, findvLine, optionChooseZMax, drawPointsOnImg, imshow
import cv2
import sys
sys.path.insert(0, '../../../face_affine/exp04')
from face_affine_utils import addEdgeLandmark
from generate_tri_txt import generateTriList
from face_affine import morph_modify_for_2D3D2D_low_resolution
from twoD_threeD_twoD_LR_ManuallySelectHair import main2D_3D_2D_ManuallySelectHair
from twoD_threeD_twoD_LR_MeshHair import main2D_3D_2D_LR_MeshHair, addOuterEllipseEdgeLandmark
from LR2HR import LR2HR


def main2D_3D_2D_LR_Hair():
    '''1. read origin obj and image'''
    objPath = '../../github/vrn-07231340/obj/trump-12.obj'
    objLines, vLines, fLines = readObj(objPath)
    imgPath = '../../github/vrn-07231340/examples/scaled/trump-12.jpg'
    img = cv2.imread(imgPath)
    imgHRPath = '../../github/vrn-07231340/examples/trump-12.jpg'
    imgHR = cv2.imread(imgHRPath)

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
    nodObjLines = nod(objPath, nodAngle, nodCenterMode)

    '''4. nod 3D landmark list'''
    '''nodLandmark3DList: [[x0, y0], [x0, y0, z0], [x1, y1], [x1, y1, z1], (line, vLine)]'''
    '''len(nodLandmark3DList): number of landmarks both on 2D and 3D'''
    '''[x0, y0]: original landmark on 2D img'''
    '''[x0, y0, z0]: original landmark on 3D model'''
    '''[x1, y1]: nod landmark on 2D img'''
    '''[x1, y1, z1]: nod landmark on 3D model'''
    '''(line, vLine): the line(th) nod vLine'''
    nodVLines = objLines2vLines(nodObjLines)
    nodLandmark3DList = getNodLandmark3D(nodVLines, originLandmark3DList)

    '''5. nod 3D landmark --> nod 2D landmark'''
    nodLandmark2D = []
    for nodLandmark3D in nodLandmark3DList:
        nodLandmark2D.append(nodLandmark3D[2])

    '''6. origin 2D landmarks'''
    originLandmark2D = []
    for nodLandmark3D in nodLandmark3DList:
        originLandmark2D.append((nodLandmark3D[0][0], nodLandmark3D[0][1]))
    assert len(originLandmark2D) == len(nodLandmark2D)

    '''7. face 2D landmarks + hair 2D landmarks'''
    # hairLandmark2DList, nodHairLandmark2DList = main2D_3D_2D_ManuallySelectHair()
    hairLandmark2DList, nodHairLandmark2DList, edgePoints = main2D_3D_2D_LR_MeshHair()
    assert len(hairLandmark2DList) == len(nodHairLandmark2DList)
    '''7.1 origin face 2D landmarks + origin hair 2D landmarks'''
    originLandmark2D = originLandmark2D+hairLandmark2DList
    '''7.2 nod face 2D landmarks + nod hair 2D landmarks'''
    nodLandmark2D = nodLandmark2D+nodHairLandmark2DList
    '''7.3 save origin and nod 2D landmarks txt'''
    # originLandmark2DTxtPath = './originLandmark2D.txt'
    # originLandmark2DTxt = open(originLandmark2DTxtPath, 'w')
    # for i in originLandmark2D:
    #     originLandmark2DTxt.write('{} {}\n'.format(i[0], i[1]))
    # originLandmark2DTxt.close()
    # nodLandmark2DTxtPath = './nodLandmark2D.txt'
    # nodLandmark2DTxt = open(nodLandmark2DTxtPath, 'w')
    # for i in nodLandmark2D:
    #     nodLandmark2DTxt.write('{} {}\n'.format(i[0], i[1]))
    # nodLandmark2DTxt.close()

    '''8. add edge points'''
    '''8.1 OPTION1: 8 IMAGE EDGE POINTS'''
    '''    如果在加头发关键点之前加8个edge points，第7个edge point（图像最上面那个）会被往下warp一点'''
    '''    如果在加头发关键点之后加8个edge points，第7个edge point经过warp后仍在原处，形成一个小三角'''
    # originLandmark2D = addEdgeLandmark(originLandmark2D, img)
    # nodLandmark2D = addEdgeLandmark(nodLandmark2D, img)
    '''8.2 OPTION2: OUTER ELLIPSE'''
    originLandmark2D = originLandmark2D+edgePoints
    nodLandmark2D = nodLandmark2D+edgePoints
    drawPointsOnImg(originLandmark2D, img, 'g')
    drawPointsOnImg(nodLandmark2D, img, 'r')

    '''9. delaunay triangle for [origin landmarks (face + hair)] + (eight edge points)'''
    triList = generateTriList(originLandmark2D, img)
    '''9.1 save tri txt'''
    triTxtPath = './nodTri.txt'
    triTxt = open(triTxtPath, 'w')
    for i in triList:
        triTxt.write('{}\n'.format(i))
    triTxt.close()

    '''10. warp LR'''
    imgMorph = morph_modify_for_2D3D2D_low_resolution(
        originLandmark2D, nodLandmark2D, img, triList)
    imshow('imgMorph', imgMorph)
    cv2.imwrite('./nod.jpg', imgMorph)
    return originLandmark2D, nodLandmark2D, triList

    '''LR to HR'''
    # originLandmark2DHR = LR2HR(img, imgHR, originLandmark2D)
    # for (x, y), (xmin, ymin) in zip(originLandmark2DHR, originLandmark2D):
    #     if float(x) >= 640 or float(y) >= 640:
    #         print (int(xmin), int(ymin))
    #         cv2.circle(img, (int(xmin), int(ymin)), 3, (0, 255, 0), -1)
    #         imshow('img', img)
    # nodLandmark2DHR = LR2HR(img, imgHR, nodLandmark2D)

    '''8. add 8 egde points'''
    # nodLandmark2DHR = addEdgeLandmark(nodLandmark2DHR, imgHR)
    # originLandmark2DHR = addEdgeLandmark(originLandmark2DHR, imgHR)
    # # drawPointsOnImg(originLandmark2DHR, imgHR, 'g')
    # # drawPointsOnImg(nodLandmark2DHR, imgHR, 'r')
    # '''8.1 remove >= 640'''
    # nodLandmark2DHRFinal = []
    # originLandmark2DHRFinal = []
    # for (x, y) in nodLandmark2DHR:
    #     if float(x) < 640 and float(y) < 640:
    #         nodLandmark2DHRFinal.append((x, y))
    # for (x, y) in originLandmark2DHR:
    #     if float(x) < 640 and float(y) < 640:
    #         originLandmark2DHRFinal.append((x, y))

    '''9. delaunay triangle for [origin landmarks (face + hair)] + (eight edge points)'''
    # triList = generateTriList(originLandmark2DHRFinal, imgHR)
    # '''9.1 save tri txt'''
    # triTxtPath = './nodTri.txt'
    # triTxt = open(triTxtPath, 'w')
    # for i in triList:
    #     triTxt.write('{}\n'.format(i))
    # triTxt.close()

    '''10. warp HR'''
    # imgMorph = morph_modify_for_2D3D2D_low_resolution(
    #     originLandmark2DHRFinal, nodLandmark2DHRFinal, imgHR, triList)
    # # imshow('imgMorph', imgMorph)
    # # cv2.imwrite('./nod.jpg', imgMorph)
    # return originLandmark2DHRFinal, nodLandmark2DHRFinal, triList


if __name__ == "__main__":
    main2D_3D_2D_LR_Hair()
