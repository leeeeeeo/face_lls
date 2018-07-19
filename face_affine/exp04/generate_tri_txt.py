import cv2
import numpy as np
import linecache
from face_affine_utils import *


def saveTriangleTxt(triangleTxtPath, imgOriginal, ptsOriginalPath):
    ptsDict = {}
    for i, pts in enumerate(readPoints(ptsOriginalPath)):
        ptsDict['({}, {})'.format(pts[0], pts[1])] = i

    tartriList = delaunay(imgOriginal.shape, readPoints(
        ptsOriginalPath), removeOutlier=True)

    myTriTxt = open(triangleTxtPath, 'w')
    for tarTri in tartriList:
        triLine = '{} {} {}'.format(
            ptsDict[str(tarTri[0])], ptsDict[str(tarTri[1])], ptsDict[str(tarTri[2])])
        myTriTxt.write(triLine+'\n')
    myTriTxt.close()


triPath = './data/'
saveTriangleTxt('/Users/lls/Documents/face/data/talkingphoto/tri.txt',
                'imgOriginal', ptsOriginalPath)
