import cv2
import os
from natsort import natsorted
from face_affine_utils import *
from face_affine import *


def addEdgeLandmark(pts, img):
    size = img.shape
    imgHeight = size[0]-1
    imgWidth = size[1]-1
    halfHeight = size[0]/2
    halfWidth = size[1]/2
    edgeLandmark = [(0, 0), (0, halfHeight), (0, imgHeight), (halfWidth, imgHeight),
                    (imgWidth, imgHeight), (imgWidth, halfHeight), (imgWidth, 0), (halfWidth, 0)]
    return pts+edgeLandmark


def mainTrump():
    imgFolder = './data/trump/trump'
    triTxtPath = './data/source/mytri.txt'
    imgNames = []
    for root, folder, files in os.walk(imgFolder):
        for fileName in files:
            if fileName.endswith('.png'):
                imgNames.append(fileName)
        imgNames = natsorted(imgNames)
        for imgName in imgNames:
            print imgName
            img = cv2.imread('{}/{}'.format(imgFolder, imgName))
            ptsOriginal = readPts(
                '{}/{}.pts'.format(imgFolder, imgName.split('.')[0]))
            ptsTarget = readPoints(
                '{}/{}_smile.txt'.format(imgFolder, imgName.split('.')[0]))
            ptsContour = readPoints(
                '{}/{}_smile.txt'.format(imgFolder, imgName.split('.')[0]), contour=True)
            ptsOriginal = addEdgeLandmark(ptsOriginal, img)
            ptsTarget = addEdgeLandmark(ptsTarget, img)
            addEdgeLandmark(ptsOriginal, img)
            '''morph entire image'''
            imgMorph = morph(ptsOriginal, ptsTarget,
                             img, triTxtPath)
            # cv2.imshow("imgMorph", imgMorph)
            '''morph face area'''
            maskContour, imgRecover = recoverMask(
                ptsContour, img, imgMorph)
            # cv2.imshow('maskContour', maskContour)
            cv2.imshow("Recoverd Face", np.uint8(imgRecover))
            cv2.waitKey(0)


if __name__ == "__main__":
    mainTrump()
