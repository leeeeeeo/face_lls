# %%
import sys
import cv2
sys.path.insert(0, '../../../face_affine/exp04')
from face_affine_utils import delaunay
from skimage import data
import numpy as np


def mainMeshTriTxt():
    imgPath = '../../github/vrn-07231340/examples/trump-12.jpg'
    gridSizeHeight = 50
    gridSizeWidth = 50
    generateMeshTriTxt(imgPath, gridSizeHeight, gridSizeWidth)


def generateMeshTriTxt(imgPath, gridSizeHeight, gridSizeWidth):
    # img = data.astronaut()
    img = cv2.imread(imgPath)
    height, width = img.shape[0], img.shape[1]
    meshTriTxtPath = './meshTri_{}_{}.txt'.format(height, width)
    src_height = np.linspace(0, height, gridSizeHeight)
    src_width = np.linspace(0, width, gridSizeWidth)
    src_height, src_width = np.meshgrid(src_height, src_width)
    src = np.dstack([src_width.flat, src_height.flat])[0]
    src = src.astype(np.int32)
    srcList = []
    srcDict = {}
    for i, s in enumerate(src):
        s[0] = (s[0] - 1 if float(s[0]) == float(width) else s[0])
        s[1] = (s[1] - 1 if float(s[1]) == float(height) else s[1])
        srcList.append((s[0], s[1]))
        srcDict['({}, {})'.format(s[0], s[1])] = i

    meshTriTxtList = delaunay(img.shape, srcList, removeOutlier=True)
    # print meshTriTxtList
    meshTriTxtFile = open(meshTriTxtPath, 'w')
    for meshTriTxt in meshTriTxtList:
        meshTriTxtLine = '{} {} {}'.format(srcDict[str(meshTriTxt[0])], srcDict[str(
            meshTriTxt[1])], srcDict[str(meshTriTxt[2])])
        meshTriTxtFile.write(meshTriTxtLine + '\n')
    meshTriTxtFile.close()


if __name__ == "__main__":
    mainMeshTriTxt()
