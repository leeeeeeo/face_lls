import cv2
import os
from natsort import natsorted
from face_affine_utils import *
from face_affine import *


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
                '{}/{}_smile.txt'.format(imgFolder, imgName.split('.')[0]), contour='FACE_CONTOUR_LANDMARKS')
            # print ptsOriginal
            ptsOriginal = addEdgeLandmark(ptsOriginal, img)
            # print ptsOriginal
            ptsTarget = addEdgeLandmark(ptsTarget, img)
            # addEdgeLandmark(ptsOriginal, img)
            '''morph entire image'''
            imgMorph = morph(ptsOriginal, ptsTarget,
                             img, triTxtPath)
            cv2.imshow("imgMorph", imgMorph)
            '''morph face area'''
            # maskContour, imgRecover = recoverMask(
            #     ptsContour, img, imgMorph)
            # cv2.imshow('maskContour', maskContour)
            # cv2.imshow("Recoverd Face", np.uint8(imgRecover))
            cv2.waitKey(0)


if __name__ == "__main__":
    mainTrump()
