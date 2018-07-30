# -*- coding: utf-8 -*-
import cv2
import copy
import matplotlib.pyplot as plt
import numpy as np
from skimage import data
from skimage.transform import PiecewiseAffineTransform, warp
from skimage.draw import *
from skimage.measure import grid_points_in_poly, points_in_poly
from put_face_back import getLandmark2D


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
            landmarksInSmallGridList.append([landmarkX, landmarkX])
            deltaInSmallGridList.append(deltaAllLandmarksList[i])

    landmarksInSmallGridArray = np.asarray(landmarksInSmallGridList)
    deltaInSmallGridArray = np.asarray(deltaInSmallGridList)

    return landmarksInSmallGridList, deltaInSmallGridList, landmarksInSmallGridArray, deltaInSmallGridArray


def twoPointsDistance(point1, point2):
    return np.sqrt((point1[0]-point2[0])**2+(point1[1]-point2[1])**2)


def mainMeshWarp():
    '''1. read image and load new landmarks (nod)'''
    image = cv2.imread('../../github/vrn-07231340/examples/trump-12.jpg')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    rows, cols = image.shape[0], image.shape[1]

    originLandmark2D = getLandmark2D(image)
    newLandmarkPath = '../../data/talkingphoto/IMG_2294/IMG_2294_26.png.txt'
    newLandmark = readPoints(newLandmarkPath)[:68]

    '''2. 新建source mesh'''
    gridSize = 50
    src_rows = np.linspace(0, rows, gridSize)  # 10 行
    src_cols = np.linspace(0, cols, gridSize)  # 20 列
    src_rows, src_cols = np.meshgrid(src_rows, src_cols)
    src = np.dstack([src_cols.flat, src_rows.flat])[0]
    '''!!! IMPORTANT DEEPCOPY!!!'''
    dst = copy.deepcopy(src)

    '''3. SKIMAGE画一个椭圆, 并且得到椭圆内所有的网格点'''
    rr, cc = ellipse_perimeter(originLandmark2D[NOSE_CENTER][1],
                               originLandmark2D[NOSE_CENTER][0], 180, 300, orientation=30)
    print rr.shape, cc.shape
    ellipseVerts = np.dstack([rr, cc])[0]
    mask = points_in_poly(src, ellipseVerts)

    pointsInEllipseList = []
    indexInEllipseList = []
    for i, (s, m) in enumerate(zip(src, mask)):
        if m == True:
            pointsInEllipseList.append(s)
            indexInEllipseList.append(i)
    print len(indexInEllipseList)
    # x=pointsInEllipseList[i][1], y=pointsInEllipseList[i][0]
    pointsInEllipseArray = np.asarray(pointsInEllipseList)
    '''swap collums of pointsInEllipseList'''
    # x=pointsInEllipseList[i][0], y=pointsInEllipseList[i][1]
    pointsInEllipseArray[:, [0, 1]] = pointsInEllipseArray[:, [1, 0]]
    # image[rr, cc] = 255  # draw ellipse perimeter on image

    # '''4. compute distance between (each point in ellipse) and (each point in landmark)'''
    # distanceArray = np.zeros((pointsInEllipseArray.shape[0], 68))
    # for (i, (landmarkX, landmarkY)) in enumerate(originLandmark2D):
    #     for (j, (meshPointX, meshPointY)) in enumerate(pointsInEllipseArray):
    #         distance = np.sqrt((landmarkX-meshPointX)**2 +
    #                            (landmarkY-meshPointY)**2)
    #         distanceArray[j, i] = distance
    # '''concate pointsInEllipseArray (100,2) and distanceArray (100,68)'''
    # # pointsInEllipseArray_distanceArray = np.concatenate(
    # #     (pointsInEllipseArray, distanceArray), axis=1)

    '''4. compute delta (68) of each new landmark X(Y) and old landmark X(Y)'''
    deltaAllLandmarksList = []
    for i in range(len(newLandmark)):
        deltaAllLandmarksList.append(
            [newLandmark[i][0]-originLandmark2D[i][0], newLandmark[i][1]-originLandmark2D[i][1]])
    # deltaAllLandmarksArray = np.asarray(deltaAllLandmarksList)
    # print len(deltaAllLandmarksList)

    '''5. compute delta of each point in ellipse'''
    radius = 3*rows/float(gridSize)
    targetPointsInEllipseList = []
    for meshPoint in pointsInEllipseArray:
        _, _, landmarksInSmallGridArray, deltaInSmallGridArray = getLandmarksInSmallGrid(
            meshPoint, originLandmark2D, deltaAllLandmarksList, radius=radius)
        deltaXOfMeshPoint = 0
        deltaYOfMeshPoint = 0
        if deltaInSmallGridArray.shape[0] != 0:
            for i, deltaInSmallGrid in enumerate(deltaInSmallGridArray):
                deltaInSmallGridX = deltaInSmallGrid[0]
                deltaInSmallGridY = deltaInSmallGrid[1]
                deltaXOfMeshPoint = deltaXOfMeshPoint + deltaInSmallGridX / \
                    twoPointsDistance(meshPoint, landmarksInSmallGridArray[i])
                deltaYOfMeshPoint = deltaYOfMeshPoint + deltaInSmallGridY / \
                    twoPointsDistance(meshPoint, landmarksInSmallGridArray[i])
        targetMeshPointX = meshPoint[0]+deltaXOfMeshPoint
        targetMeshPointY = meshPoint[1]+deltaYOfMeshPoint
        targetPointsInEllipseList.append([targetMeshPointX, targetMeshPointY])
    targetPointsInEllipseArray = np.asarray(targetPointsInEllipseList)

    '''6. compute final target points mesh'''
    targetPointsInEllipseArray[:, [0, 1]
                               ] = targetPointsInEllipseArray[:, [1, 0]]
    dst[indexInEllipseList] = targetPointsInEllipseArray

    '''7. '''
    tform = PiecewiseAffineTransform()
    tform.estimate(src, dst)
    out = warp(image, tform)

    fig, ax = plt.subplots()
    ax.imshow(out)
    # ax.scatter(pointsInEllipseArray[:, 0], pointsInEllipseArray[:, 1],
    #            marker='+', color='b', s=5)
    # ax.plot(tform.inverse(src)[:, 0], tform.inverse(src)[:, 1], '.r')
    plt.show()


if __name__ == "__main__":
    mainMeshWarp()
