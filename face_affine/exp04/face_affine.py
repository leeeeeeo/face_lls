import cv2
import numpy as np
import linecache
from face_affine_utils import *
import copy


def changeExpression(imgOriginalPath, ptsOriginal, ptsTarget, triTxtPath, imgOriginal):
    step = 50
    ptsOld = []
    # videoWriter = cv2.VideoWriter('./source/{}TO{}.mp4'.format(imgOriginalPath.split('.')[1].split('_')[1], ptsTargetPath.split(
    #     '.')[1].split('_')[1]), cv2.VideoWriter_fourcc(*'mp4v'), 10, (imgOriginal.shape[1], imgOriginal.shape[0]))
    for pt in ptsOriginal:
        ptsOld.append((float(pt[0]), float(pt[1])))
    for i in range(step+1):
        print 'i: {}'.format(str(i))
        ptsTmp = []
        for j in range(len(ptsOriginal)):
            stepX = (ptsTarget[j][0]-ptsOriginal[j][0])/float(step)
            stepY = (ptsTarget[j][1]-ptsOriginal[j][1])/float(step)
            tmpX = ptsOld[j][0]+stepX
            tmpY = ptsOld[j][1]+stepY
            ptsTmp.append((tmpX, tmpY))
        imgMorphTmp = morphChange(ptsOriginal, ptsTmp, imgOriginal, triTxtPath)
        ptsOld = ptsTmp
        cv2.imshow("Morphed Face Tmp", np.uint8(imgMorphTmp))
        # videoWriter.write(imgMorphTmp)
        if i == 50:
            cv2.waitKey(0)
        else:
            cv2.waitKey(10)


def recoverMask(ptsContour, imgOriginal, imgMorph):
    maskContour = np.zeros(imgOriginal.shape, dtype=imgOriginal.dtype)
    cv2.fillConvexPoly(maskContour, np.int32(ptsContour), (255, 255, 255))
    # cv2.fillConvexPoly(maskContour, np.int32(
    #     ptsContour), (1.0, 1.0, 1.0), 16, 0)
    r = cv2.boundingRect(np.float32(ptsContour))
    center = (r[0] + int(r[2] / 2), r[1] + int(r[3] / 2))
    mat = cv2.getRotationMatrix2D(center, 0, 0.95)
    maskContour = cv2.warpAffine(
        maskContour, mat, (maskContour.shape[1], maskContour.shape[0]))
    maskContour = cv2.blur(maskContour, (15, 10), center)
    imgRecover = cv2.seamlessClone(
        imgMorph, imgOriginal, maskContour, center, cv2.NORMAL_CLONE)
    # imgRecover = imgOriginal * (1 - maskContour) + imgMorph * maskContour
    return maskContour, imgRecover


def recoverMask_old(ptsContour, imgOriginal, imgMorph):
    maskContour = np.zeros(imgOriginal.shape, dtype=imgOriginal.dtype)
    cv2.fillConvexPoly(maskContour, np.int32(
        ptsContour), (1.0, 1.0, 1.0), 16, 0)
    r = cv2.boundingRect(np.float32(ptsContour))
    center = (r[0] + int(r[2] / 2), r[1] + int(r[3] / 2))
    mat = cv2.getRotationMatrix2D(center, 0, 0.95)
    maskContour = cv2.warpAffine(
        maskContour, mat, (maskContour.shape[1], maskContour.shape[0]))
    imgRecover = imgOriginal * (1 - maskContour) + imgMorph * maskContour
    return maskContour, imgRecover


def morph(ptsOriginal, ptsTarget, imgOriginal, triTxtPath):
    imgMorph = np.zeros(imgOriginal.shape, dtype=imgOriginal.dtype)
    with open(triTxtPath) as file:
        for line in file:
            x, y, z = line.split()
            x = int(x)
            y = int(y)
            z = int(z)
            t1 = [ptsOriginal[x], ptsOriginal[y], ptsOriginal[z]]
            t = [ptsTarget[x], ptsTarget[y], ptsTarget[z]]
            morphTriangle(imgOriginal, imgMorph, t1, t)
    return imgMorph


def morph_modify_for_meshwarp(ptsOriginal, ptsTarget, imgOriginal, triList):
    # imgMorph = np.zeros(imgOriginal.shape, dtype=imgOriginal.dtype)
    # imgMorph = np.empty_like(imgOriginal)
    imgMorph = copy.deepcopy(imgOriginal)
    if str(type(triList)) == "<type 'str'>":
        with open(triList) as file:
            for line in file:
                x, y, z = line.split()
                x = int(x)
                y = int(y)
                z = int(z)
                t1 = [ptsOriginal[x], ptsOriginal[y], ptsOriginal[z]]
                t = [ptsTarget[x], ptsTarget[y], ptsTarget[z]]
                t1 = np.asarray(t1)
                t = np.asarray(t)
                if (t1 == t).all():
                    pass
                else:
                    morphTriangle_modify_for_meshwarp(
                        imgOriginal, imgMorph, t1, t)
    elif str(type(triList)) == "<type 'list'>":
        for tri in triList:
            x, y, z = tri.split()
            x = int(x)
            y = int(y)
            z = int(z)
            t1 = [ptsOriginal[x], ptsOriginal[y], ptsOriginal[z]]
            t = [ptsTarget[x], ptsTarget[y], ptsTarget[z]]
            t1 = np.asarray(t1)
            t = np.asarray(t)
            if (t1 == t).all():
                pass
            else:
                morphTriangle_modify_for_meshwarp(imgOriginal, imgMorph, t1, t)
    return imgMorph


def mainAffine():
    imgOriginalPath = './data/source/S132_8.png'
    ptsOriginalPath = '{}.txt'.format(imgOriginalPath)
    ptsTargetPath = './data/source/S132_16.png.txt'
    myTriTxtPath = './data/source/mytri.txt'
    triTxtPath = myTriTxtPath  # triTxtPath = './source/tri_wo_background_w_edge.txt'
    imgOriginal = cv2.imread(imgOriginalPath)
    cv2.imshow("Original Face", np.uint8(imgOriginal))
    ptsOriginal = readPoints(ptsOriginalPath)
    ptsTarget = readPoints(ptsTargetPath)
    # ptsOriginal = addEdgeLandmark(ptsOriginal, imgOriginal)
    # ptsTarget = addEdgeLandmark(ptsTarget, imgOriginal)
    ptsContour = readPoints(ptsTargetPath, contour='FACE_CONTOUR_LANDMARKS')

    '''morph from one expression to another'''
    imgMorph = morph(ptsOriginal, ptsTarget, imgOriginal, triTxtPath)
    cv2.imshow("Morphed Face", np.uint8(imgMorph))
    cv2.waitKey(0)
    '''makeup face area mask'''
    maskContour, imgRecover = recoverMask(ptsContour, imgOriginal, imgMorph)
    # cv2.imshow('maskContour', maskContour)
    # cv2.imshow("Recoverd Face", np.uint8(imgRecover))
    # cv2.waitKey(0)
    '''change original expression to target expression in 50 iters'''
    changeExpression(imgOriginalPath, ptsOriginal,
                     ptsTarget, triTxtPath, imgOriginal)
    # cv2.waitKey(0)


if __name__ == "__main__":
    mainAffine()
