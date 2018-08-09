# %%
# -*- coding: utf-8 -*-
import copy
import cv2
import matplotlib.pyplot as plt
import numpy as np
from skimage import data
from skimage.draw import *
from skimage.measure import grid_points_in_poly, points_in_poly
from skimage.transform import PiecewiseAffineTransform, warp
from put_face_back import drawPointsOnImg, getLandmark2D, imshow
from mesh_tri import generateMeshTriTxt
import sys
sys.path.insert(0, '../../../face_affine/exp04')
from face_affine import morph_modify_for_meshwarp

NOSE_CENTER = 33
TWO_EYE_CENTER = 27
HEAD_CENTER = 28
ELLIPSE_CENTER = HEAD_CENTER


def readPoints(ptsPath):
    points = []
    with open(ptsPath) as file:
        for line in file:
            x, y = line.split()
            points.append((float(x), float(y)))
    return points


def getLandmarksInSmallGrid(meshPoint, originLandmark2D, deltaAllLandmarksList, radius=40):
    '''define a small grid'''
    meshPointX = meshPoint[0]
    meshPointY = meshPoint[1]
    landmarksInSmallGridList = []
    deltaInSmallGridList = []
    for i, (landmarkX, landmarkY) in enumerate(originLandmark2D):
        if np.sqrt((meshPointX-landmarkX)**2+(meshPointY-landmarkY)**2) < radius:
            # print [landmarkX, landmarkX], deltaAllLandmarksList[i]
            landmarksInSmallGridList.append([landmarkX, landmarkY])
            deltaInSmallGridList.append(deltaAllLandmarksList[i])

    landmarksInSmallGridArray = np.asarray(landmarksInSmallGridList)
    deltaInSmallGridArray = np.asarray(deltaInSmallGridList)

    return landmarksInSmallGridList, deltaInSmallGridList, landmarksInSmallGridArray, deltaInSmallGridArray


def getMeshPointsOnSameColumn(meshPoint, pointsInEllipseArray):
    meshPointsOnSameColumnList = []
    meshPointsIndexOnSameColumnList = []
    for i, point in enumerate(pointsInEllipseArray):
        if point[0] == meshPoint[0] and point[1] != meshPoint[1]:
            meshPointsOnSameColumnList.append(point)
            meshPointsIndexOnSameColumnList.append(i)
    meshPointsOnSameColumnArray = np.asarray(meshPointsOnSameColumnList)
    return meshPointsOnSameColumnArray, meshPointsIndexOnSameColumnList


def getMeshPointsOnSameCross(meshPoint, pointsInEllipseArray):
    meshPointsOnSameColumnList = []
    meshPointsOnSameRowList = []
    meshPointsOnSameCrossList = []
    meshPointsIndexOnSameColumnList = []
    meshPointsIndexOnSameRowList = []
    meshPointsIndexOnSameCrossList = []
    for i, point in enumerate(pointsInEllipseArray):
        if point[0] == meshPoint[0] and point[1] != meshPoint[1]:
            meshPointsOnSameColumnList.append(point)
            meshPointsIndexOnSameColumnList.append(i)
        elif point[1] == meshPoint[1] and point[0] != meshPoint[0]:
            meshPointsOnSameRowList.append(point)
            meshPointsIndexOnSameRowList.append(i)
    meshPointsOnSameCrossList = meshPointsOnSameColumnList+meshPointsOnSameRowList
    meshPointsIndexOnSameCrossList = meshPointsIndexOnSameColumnList + \
        meshPointsIndexOnSameRowList
    meshPointsOnSameCrossArray = np.asarray(meshPointsOnSameCrossList)
    meshPointsIndexOnSameCrossArray = np.asarray(
        meshPointsIndexOnSameCrossList)
    return meshPointsOnSameCrossArray, meshPointsIndexOnSameCrossArray


def twoPointsDistance(point1, point2):
    point1X, point1Y = point1[0], point1[1]
    point2X, point2Y = point2[0], point2[1]
    return np.sqrt((point1X-point2X)**2+(point1Y-point2Y)**2)


def mainMeshWarp():
    '''1. read image and load new landmarks (nod)'''
    image = cv2.imread('../../github/vrn-07231340/examples/trump-12.jpg')
    imageTmp = copy.deepcopy(image)
    rows, cols = image.shape[0], image.shape[1]

    originLandmark2D = getLandmark2D(image)

    targetLandmark2DPath = '../../data/talkingphoto/IMG_2294/IMG_2294_26.png.txt'
    targetLandmark2D = readPoints(targetLandmark2DPath)[:68]

    '''2. 新建source mesh'''
    gridSize = 50
    src_rows = np.linspace(0, rows, gridSize)  # 10 行
    src_cols = np.linspace(0, cols, gridSize)  # 20 列
    src_rows, src_cols = np.meshgrid(src_rows, src_cols)
    src = np.dstack([src_cols.flat, src_rows.flat])[0]

    '''!!! IMPORTANT DEEPCOPY!!!'''
    dst = copy.deepcopy(src)

    '''triTxt'''
    # triTxtPath = './meshTri_640_640(trump-12).txt'
    triList = generateMeshTriTxt(image, 50, 50, originLandmark2D)

    '''3. SKIMAGE画一个椭圆, 并且得到椭圆内所有的网格点'''
    rr, cc = ellipse_perimeter(originLandmark2D[ELLIPSE_CENTER][1],
                               originLandmark2D[ELLIPSE_CENTER][0], 200, 260, orientation=30)
    tmp = rr
    rr = cc
    cc = tmp
    # print rr.shape, cc.shape
    ellipseVerts = np.dstack([rr, cc])[0]  # 椭圆边长上所有点
    # print ellipseVerts
    mask = points_in_poly(src, ellipseVerts)

    pointsInEllipseList = []
    indexInEllipseList = []
    for i, (s, m) in enumerate(zip(src, mask)):
        if m == True:
            pointsInEllipseList.append(s)
            indexInEllipseList.append(i)
    # print len(indexInEllipseList)
    # x=pointsInEllipseList[i][1], y=pointsInEllipseList[i][0]
    pointsInEllipseArray = np.asarray(pointsInEllipseList)
    '''swap collums of pointsInEllipseList'''
    # x=pointsInEllipseList[i][0], y=pointsInEllipseList[i][1]
    # pointsInEllipseArray[:, [0, 1]] = pointsInEllipseArray[:, [1, 0]]
    # image[cc, rr] = 255  # draw ellipse perimeter on image
    drawPointsOnImg(pointsInEllipseArray, imageTmp, 'r')

    '''4. compute delta (68) of each new landmark X(Y) and old landmark X(Y)'''
    deltaAllLandmarksList = []
    for i in range(len(targetLandmark2D)):
        deltaAllLandmarksList.append(
            [targetLandmark2D[i][0]-originLandmark2D[i][0], targetLandmark2D[i][1]-originLandmark2D[i][1]])
    # deltaAllLandmarksArray = np.asarray(deltaAllLandmarksList)
    # print deltaAllLandmarksList

    '''5. compute delta of each point in ellipse'''
    radius = 5*rows/float(gridSize)
    targetPointsModifiedInEllipseList = []
    isPointsInEllipseModifiedList = []
    deltaOfMeshPointsInEllipseList = []
    for meshPoint in pointsInEllipseArray:
        _, _, landmarksInSmallGridArray, deltaInSmallGridArray = getLandmarksInSmallGrid(
            meshPoint, originLandmark2D, deltaAllLandmarksList, radius=radius)
        '''画smallGrid以及其内的所有landmarks的delta'''
        # imageTmp = copy.deepcopy(image)
        # cv2.circle(imageTmp, (int(meshPoint[0]),
        #                       int(meshPoint[1])), int(radius), (0, 255, 0), 2)
        # if landmarksInSmallGridArray.shape[0] != 0:
        #     for (deltaX, deltaY), (landmarkX, landmarkY) in zip(deltaInSmallGridArray, landmarksInSmallGridArray):
        #         cv2.line(imageTmp, (int(landmarkX), int(landmarkY)), (int(
        #             landmarkX+deltaX), int(landmarkY+deltaY)), (255, 0, 0), 2)
        #         cv2.circle(imageTmp, (int(landmarkX+deltaX),
        #                               int(landmarkY+deltaY)), 2, (0, 0, 255), -1)
        # cv2.imshow('imgTmp', imageTmp)
        # cv2.waitKey(50)
        deltaXOfMeshPoint = 0
        deltaYOfMeshPoint = 0
        if deltaInSmallGridArray.shape[0] != 0:
            for i, (deltaInSmallGrid, landmarkInSmallGrid) in enumerate(zip(deltaInSmallGridArray, landmarksInSmallGridArray)):
                deltaInSmallGridX = deltaInSmallGrid[0]
                deltaInSmallGridY = deltaInSmallGrid[1]
                deltaXOfMeshPoint = deltaXOfMeshPoint + deltaInSmallGridX / \
                    twoPointsDistance(meshPoint, landmarkInSmallGrid)
                deltaYOfMeshPoint = deltaYOfMeshPoint + deltaInSmallGridY / \
                    twoPointsDistance(meshPoint, landmarkInSmallGrid)
        targetMeshPointX = meshPoint[0]+deltaXOfMeshPoint
        targetMeshPointY = meshPoint[1]+deltaYOfMeshPoint
        deltaOfMeshPointsInEllipseList.append(
            [deltaXOfMeshPoint, deltaYOfMeshPoint])
        if deltaXOfMeshPoint == 0 and deltaYOfMeshPoint == 0:
            isPointsInEllipseModifiedList.append(False)
        else:
            isPointsInEllipseModifiedList.append(True)
        '''画每一个meshPoint的起点和终点'''
        # cv2.line(imageTmp, (int(meshPoint[0]), int(meshPoint[1])),
        #          (int(targetMeshPointX), int(targetMeshPointY)), (0, 255, 0), 1)
        # cv2.circle(imageTmp, (int(meshPoint[0]), int(meshPoint[1])),
        #            2, (0, 255, 0), -1)
        # cv2.circle(imageTmp, (int(targetMeshPointX), int(targetMeshPointY)),
        #            2, (255, 0, 0), -1)
        # if deltaXOfMeshPoint != 0 or deltaYOfMeshPoint != 0:
        #     cv2.imshow('imageTmp', imageTmp)
        #     cv2.waitKey(0)
        targetPointsModifiedInEllipseList.append(
            [targetMeshPointX, targetMeshPointY])
    targetPointsModifiedInEllipseArray = np.asarray(
        targetPointsModifiedInEllipseList)
    # drawPointsOnImg(pointsInEllipseArray, imageTmp, 'b')
    # drawPointsOnImg(targetPointsModifiedInEllipseArray, imageTmp, 'r')

    assert pointsInEllipseArray.shape[0] == len(isPointsInEllipseModifiedList)
    assert pointsInEllipseArray.shape[0] == len(deltaOfMeshPointsInEllipseList)
    '''5.1 compute rest points in ellipse (hair)'''
    targetPointsNotModifiedInEllipseList = []
    for i, meshPoint in enumerate(pointsInEllipseArray):
        deltaXOfMeshPointNotModified = 0
        deltaYOfMeshPointNotModified = 0
        if isPointsInEllipseModifiedList[i] == False:
            meshPointsOnSameColumnArray, meshPointsIndexOnSameColumnList = getMeshPointsOnSameColumn(
                meshPoint, pointsInEllipseArray)
            for meshPointOnSameColumn, meshPointIndexOnSameColumn in zip(meshPointsOnSameColumnArray, meshPointsIndexOnSameColumnList):
                deltaXOfMeshPointOnSameColumn = deltaOfMeshPointsInEllipseList[
                    meshPointIndexOnSameColumn][0]
                deltaYOfMeshPointOnSameColumn = deltaOfMeshPointsInEllipseList[
                    meshPointIndexOnSameColumn][1]
                deltaXOfMeshPointNotModified = deltaXOfMeshPointNotModified + \
                    deltaXOfMeshPointOnSameColumn / \
                    twoPointsDistance(meshPoint, meshPointOnSameColumn)
                deltaYOfMeshPointNotModified = deltaYOfMeshPointNotModified + \
                    deltaYOfMeshPointOnSameColumn / \
                    twoPointsDistance(meshPoint, meshPointOnSameColumn)
        targetMeshPointXNotModified = meshPoint[0] + \
            deltaXOfMeshPointNotModified
        targetMeshPointYNotModified = meshPoint[1] + \
            deltaYOfMeshPointNotModified
        targetPointsNotModifiedInEllipseList.append(
            [targetMeshPointXNotModified, targetMeshPointYNotModified])
    targetPointsNotModifiedInEllipseArray = np.asarray(
        targetPointsNotModifiedInEllipseList)

    assert targetPointsNotModifiedInEllipseArray.shape == targetPointsModifiedInEllipseArray.shape

    '''5.2 merge targetPointsModifiedInEllipseArray and targetPointsNotModifiedInEllipseArray'''
    targetPointsInEllipseList = []
    for targetPointModifiedInEllipse, targetPointNotModifiedInEllipse, isPointInEllipseModified in zip(targetPointsModifiedInEllipseArray, targetPointsNotModifiedInEllipseArray, isPointsInEllipseModifiedList):
        if isPointInEllipseModified == True:
            targetPointsInEllipseList.append(targetPointModifiedInEllipse)
        else:
            targetPointsInEllipseList.append(targetPointNotModifiedInEllipse)

    '''6. compute final target mesh'''
    # print targetPointsInEllipseList.shape
    drawPointsOnImg(targetPointsInEllipseList, imageTmp, 'b')
    # targetPointsInEllipseList[:, [0, 1]
    #                            ] = targetPointsInEllipseList[:, [1, 0]]
    dst[indexInEllipseList] = targetPointsInEllipseList
    '''draw src and dst mesh on image'''
    # drawPointsOnImg(src, imageTmp, 'g')
    # drawPointsOnImg(dst, imageTmp, 'b')
    cv2.imwrite('./tmp.png', imageTmp)

    '''7. PiecewiseAffineTransform without landmarks'''
    tform = PiecewiseAffineTransform()
    tform.estimate(dst, src)
    out = warp(image, tform)

    '''8. imshow and imwrite without landmarks for skimage'''
    imshow('out_pat_wo', out)
    cv2.imwrite('./out_pat_wo.png', out*255)

    '''7.1 mesh warp with landmarks wirtten by meself'''
    '''SEE EXP03'''
    # '''draw dst triangle'''
    # imgTmp = copy.copy(image)
    # for tri in triList:
    #     a, b, c = tri.split()
    #     a = int(a)
    #     b = int(b)
    #     c = int(c)
    #     z1 = (int(dst[a][0]), int(dst[a][1]))
    #     z2 = (int(dst[b][0]), int(dst[b][1]))
    #     z3 = (int(dst[c][0]), int(dst[c][1]))
    #     cv2.line(imgTmp, z1, z2, (255, 255, 255), 1)
    #     cv2.line(imgTmp, z2, z3, (255, 255, 255), 1)
    #     cv2.line(imgTmp, z3, z1, (255, 255, 255), 1)
    #     cv2.imshow('tmp', imgTmp)
    #     cv2.waitKey(1)
    # imgMorph = morph_modify_for_meshwarp(src, dst, image, triList)

    '''8.1 imshow and imwrite with landmarks by myself'''
    '''SEE EXP03'''
    # imshow('imgMorph', imgMorph)
    # cv2.imwrite('./out_wo.png', imgMorph)

    '''7.2 PiecewiseAffineTransform with landmarks'''
    originLandmark2DArray = np.asarray(originLandmark2D)
    src = np.concatenate((src, originLandmark2DArray), axis=0)
    targetLandmark2DArray = np.asarray(targetLandmark2D)
    dst = np.concatenate((dst, targetLandmark2DArray), axis=0)
    tform = PiecewiseAffineTransform()
    tform.estimate(dst, src)
    out = warp(image, tform)

    '''8.2 imshow and imwrite with landmarks for skimage'''
    imshow('out_pat_w', out)
    cv2.imwrite('./out_pat_w.png', out*255)
    # cv2.imwrite('./ori.jpg', image)
    # # fig, ax = plt.subplots()
    # # ax.imshow(out)
    # # ax.scatter(pointsInEllipseArray[:, 0], pointsInEllipseArray[:, 1],
    # #            marker='+', color='b', s=5)
    # # ax.plot(tform.inverse(src)[:, 0], tform.inverse(src)[:, 1], '.r')
    # # plt.show()

    '''7.3 mesh warp with landmarks wirtten by meself'''
    originLandmark2DArray = np.asarray(originLandmark2D)
    src = np.concatenate((src, originLandmark2DArray), axis=0)
    targetLandmark2DArray = np.asarray(targetLandmark2D)
    dst = np.concatenate((dst, targetLandmark2DArray), axis=0)
    '''draw dst triangle'''
    # imgTmp = copy.copy(image)
    # for tri in triList:
    #     a, b, c = tri.split()
    #     a = int(a)
    #     b = int(b)
    #     c = int(c)
    #     z1 = (int(dst[a][0]), int(dst[a][1]))
    #     z2 = (int(dst[b][0]), int(dst[b][1]))
    #     z3 = (int(dst[c][0]), int(dst[c][1]))
    #     cv2.line(imgTmp, z1, z2, (255, 255, 255), 1)
    #     cv2.line(imgTmp, z2, z3, (255, 255, 255), 1)
    #     cv2.line(imgTmp, z3, z1, (255, 255, 255), 1)
    #     cv2.imshow('tmp', imgTmp)
    #     cv2.waitKey(1)
    imgMorph = morph_modify_for_meshwarp(src, dst, image, triList)

    '''8.3 imshow and imwrite with landmarks by myself'''
    imshow('imgMorph', imgMorph)
    cv2.imwrite('./out_w.png', imgMorph)


if __name__ == "__main__":
    mainMeshWarp()
