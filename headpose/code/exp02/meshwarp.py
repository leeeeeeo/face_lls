# %%
# -*- coding: utf-8 -*-
import cv2
import copy
import matplotlib.pyplot as plt
import numpy as np
from skimage import data
from skimage.transform import PiecewiseAffineTransform, warp
from skimage.draw import *
from skimage.measure import grid_points_in_poly, points_in_poly
from put_face_back import getLandmark2D, drawPointsOnImg


NOSE_CENTER = 33


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


def twoPointsDistance(point1, point2):
    point1X, point1Y = point1[0], point1[1]
    point2X, point2Y = point2[0], point2[1]
    return np.sqrt((point1X-point2X)**2+(point1Y-point2Y)**2)

# %%


def mainMeshWarp():
    '''1. read image and load new landmarks (nod)'''
    image = cv2.imread('../../github/vrn-07231340/examples/trump-12.jpg')
    rows, cols = image.shape[0], image.shape[1]

    originLandmark2D = getLandmark2D(image)
    targetLandmark2DPath = '../../data/talkingphoto/IMG_2294/IMG_2294_74.png.txt'
    targetLandmark2D = readPoints(targetLandmark2DPath)[:68]

    '''2. 新建source mesh'''
    gridSize = 500
    src_rows = np.linspace(0, rows, gridSize)  # 10 行
    src_cols = np.linspace(0, cols, gridSize)  # 20 列
    src_rows, src_cols = np.meshgrid(src_rows, src_cols)
    src = np.dstack([src_cols.flat, src_rows.flat])[0]
    '''!!! IMPORTANT DEEPCOPY!!!'''
    dst = copy.deepcopy(src)

    '''3. SKIMAGE画一个椭圆, 并且得到椭圆内所有的网格点'''
    rr, cc = ellipse_perimeter(originLandmark2D[NOSE_CENTER][1],
                               originLandmark2D[NOSE_CENTER][0], 180, 300, orientation=30)
    # print rr.shape, cc.shape
    ellipseVerts = np.dstack([rr, cc])[0]  # 椭圆边长上所有点
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
    pointsInEllipseArray[:, [0, 1]] = pointsInEllipseArray[:, [1, 0]]
    # image[rr, cc] = 255  # draw ellipse perimeter on image
    # drawPointsOnImg(pointsInEllipseArray,image,'r')

    '''4. compute delta (68) of each new landmark X(Y) and old landmark X(Y)'''
    deltaAllLandmarksList = []
    for i in range(len(targetLandmark2D)):
        deltaAllLandmarksList.append(
            [targetLandmark2D[i][0]-originLandmark2D[i][0], targetLandmark2D[i][1]-originLandmark2D[i][1]])
    # deltaAllLandmarksArray = np.asarray(deltaAllLandmarksList)
    # print deltaAllLandmarksList

    '''5. compute delta of each point in ellipse'''
    radius = 5*rows/float(gridSize)
    targetPointsInEllipseList = []
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
        # cv2.imshow('imgTmp', imageTmp)
        # cv2.waitKey(0)
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
        targetPointsInEllipseList.append([targetMeshPointX, targetMeshPointY])
    targetPointsInEllipseArray = np.asarray(targetPointsInEllipseList)
    # drawPointsOnImg(pointsInEllipseArray, image, 'b')
    # drawPointsOnImg(targetPointsInEllipseArray, image, 'r')

    '''6. compute final target mesh'''
    targetPointsInEllipseArray[:, [0, 1]
                               ] = targetPointsInEllipseArray[:, [1, 0]]
    dst[indexInEllipseList] = targetPointsInEllipseArray

    # src = np.concatenate((src, np.asarray(originLandmark2D)), axis=0)
    # dst = np.concatenate((dst, np.asarray(targetLandmark2D)), axis=0)

    '''7. PiecewiseAffineTransform'''
    tform = PiecewiseAffineTransform()
    # for i, s, d in zip(indexInEllipseList, src, dst):
    #     print src[i], dst[i]
    tform.estimate(src, dst)
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    out = warp(image, tform)
    cv2.imwrite('./out.jpg', out*255)
    cv2.imwrite('./ori.jpg', image)
    # fig, ax = plt.subplots()
    # ax.imshow(out)
    # ax.scatter(pointsInEllipseArray[:, 0], pointsInEllipseArray[:, 1],
    #            marker='+', color='b', s=5)
    # ax.plot(tform.inverse(src)[:, 0], tform.inverse(src)[:, 1], '.r')
    # plt.show()


if __name__ == "__main__":
    mainMeshWarp()
