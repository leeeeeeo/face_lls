import cv2
import numpy as np
import linecache
from face_affine_utils import *


def saveTriangleTxt(triangleTxtPath, imgOriginal, ptsOriginalPath):
    ptsDict = {}
    for i, pts in enumerate(readPts(ptsOriginalPath)):
        ptsDict['({}, {})'.format(int(float(pts[0])), int(float(pts[1])))] = i

    tartriList = delaunay_with_draw(imgOriginal, imgOriginal.shape, readPts(
        ptsOriginalPath), removeOutlier=True)

    myTriTxt = open(triangleTxtPath, 'w')
    for tarTri in tartriList:
        triLine = '{} {} {}'.format(
            ptsDict[str(tarTri[0])], ptsDict[str(tarTri[1])], ptsDict[str(tarTri[2])])
        myTriTxt.write(triLine+'\n')
    myTriTxt.close()


def generateTriList(pointsForDelaunay, img):
    ptsDict = {}
    triList = []
    for i, pts in enumerate(pointsForDelaunay):
        ptsDict['({}, {})'.format(int(float(pts[0])), int(float(pts[1])))] = i
    # print ptsDict
    tartriList = delaunay(img.shape, pointsForDelaunay, removeOutlier=True)
    cv2.imshow('img', img)
    cv2.waitKey(0)
    for tarTri in tartriList:
        triLine = '{} {} {}'.format(
            ptsDict[str(tarTri[0])], ptsDict[str(tarTri[1])], ptsDict[str(tarTri[2])])
        triList.append(triLine)
    return triList


# triPath = './data/'
# saveTriangleTxt('/Users/lls/Documents/face/data/talkingphoto/tri.txt',
#                 'imgOriginal', ptsOriginalPath)
