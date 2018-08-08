# %%
import sys
import cv2
sys.path.insert(0, '../../../face_affine/exp04')
from face_affine_utils import delaunay
from skimage import data
import numpy as np


def generateMeshTriTxt(img, gridSizeHeight, gridSizeWidth, landmark):
    # img = data.astronaut()
    height, width = img.shape[0], img.shape[1]
    meshTriTxtPath = './meshTri_{}_{}.txt'.format(height, width)
    src_height = np.linspace(0, height, gridSizeHeight)
    src_width = np.linspace(0, width, gridSizeWidth)
    src_height, src_width = np.meshgrid(src_height, src_width)
    src = np.dstack([src_width.flat, src_height.flat])[0]
    src = src.astype(np.int32)
    srcList = []
    srcDict = {}
    triList = []
    count = 0
    for i, s in enumerate(src):
        s[0] = (s[0] - 1 if float(s[0]) == float(width) else s[0])
        s[1] = (s[1] - 1 if float(s[1]) == float(height) else s[1])
        srcList.append((s[0], s[1]))
        srcDict['({}, {})'.format(s[0], s[1])] = count
        count = count+1
    landmarkTuple = []
    for l in landmark:
        landmarkTuple.append((int(l[0]), int(l[1])))
        srcDict['({}, {})'.format(int(l[0]), int(l[1]))] = count
        count = count+1
    meshTriList = srcList+landmarkTuple
    meshTriTxtList = delaunay(img.shape, meshTriList, removeOutlier=True)
    # print meshTriTxtList
    # meshTriTxtFile = open(meshTriTxtPath, 'w')
    for meshTriTxt in meshTriTxtList:
        # cv2.line(img, meshTriTxt[0], meshTriTxt[1], (255, 255, 255), 1)
        # cv2.line(img, meshTriTxt[1], meshTriTxt[2], (255, 255, 255), 1)
        # cv2.line(img, meshTriTxt[2], meshTriTxt[0], (255, 255, 255), 1)
        # cv2.imshow('img', img)
        # cv2.waitKey(10)
        meshTriTxtLine = '{} {} {}'.format(srcDict[str(meshTriTxt[0])], srcDict[str(
            meshTriTxt[1])], srcDict[str(meshTriTxt[2])])
        triList.append(meshTriTxtLine)
        # meshTriTxtFile.write(meshTriTxtLine + '\n')
    # meshTriTxtFile.close()
    return triList
