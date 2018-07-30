# -*- coding: utf-8 -*-
import dlib
import cv2
import os
from headpose import *
import math
from skimage.draw import ellipsoid
from mpl_toolkits.mplot3d import Axes3D


def drawPointsOnImg(points, img, color):
    for point in points:
        if color == 'r':
            cv2.circle(img, (int(point[0]), int(point[1])), 1, (0, 0, 255), -1)
        elif color == 'g':
            cv2.circle(img, (int(point[0]), int(point[1])), 1, (0, 255, 0), -1)
        else:
            cv2.circle(img, (int(point[0]), int(point[1])), 1, (255, 0, 0), -1)
    cv2.imshow('drawPointsOnImg', img)
    cv2.waitKey(0)


def getAllPoints(vLines):
    pointsXY = []
    pointsXZ = []
    pointsYZ = []
    pointsXYZ = []
    for vLine in vLines:
        _, x, y, z, _, _, _ = vLine.split()
        pointsXY.append([float(x), float(y)])
        pointsXZ.append([float(x), float(z)])
        pointsYZ.append([float(y), float(z)])
        pointsXYZ.append([float(x), float(y), float(z)])
    return pointsXY, pointsXZ, pointsYZ, pointsXYZ


def dlibShape2List(landmark2D, dtype="int"):
    coords = []
    for i in range(0, 68):
        coords.append([landmark2D.part(i).x, landmark2D.part(i).y])
    return coords


def getLandmark2D(img):
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(
        '../../../../tools/shape_predictor_68_face_landmarks.dat')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 1)
    landmark2D = predictor(gray, rects[0])
    landmark2D = dlibShape2List(landmark2D)
    # for (x, y) in landmark2D:
    #     cv2.circle(img, (x, y), 1, (0, 255, 0), -1)
    # cv2.imshow('img', img)
    # cv2.waitKey(0)
    return landmark2D


def getLandmark3D(landmark2D, vLines):
    originLandmark3D = []
    originLandmark3DLines = []
    '''exactly match'''
    for (xLandmark, yLandmark) in landmark2D:
        for (line, vLine) in enumerate(vLines):
            v, x, y, z, r, g, b = vLine.split()
            if float(x) == float(xLandmark) and float(y) == float(yLandmark) and (float(xLandmark), float(yLandmark)) not in originLandmark3D:
                originLandmark3D.append((float(xLandmark), float(yLandmark)))
                originLandmark3DLines.append((line, vLine))
                break
    '''nearest match'''
    # xyz = []
    # xy = []
    # for (xLandmark, yLandmark) in landmark2D:
    #     for (line, vLine) in enumerate(vLines):
    #         v, x, y, z, r, g, b = vLine.split()
    #         xy.append((float(x), float(y)))
    #         xyz.append((float(x), float(y), float(z)))
    # for (xLandmark, yLandmark) in landmark2D:
    #     if (xLandmark, yLandmark) not in originLandmark3D:
    #         xLandmark = xLandmark+0.01
    #         if (xLandmark, yLandmark) in xy:
    #             originLandmark3D.append((float(xLandmark), float(yLandmark)))
    #             continue

    print len(originLandmark3D), len(originLandmark3DLines)
    return originLandmark3D, originLandmark3DLines


def getEllipse(points, img):
    points = np.array(points)
    points = points.astype(np.int32)
    ellipse = cv2.fitEllipse(points)
    cv2.ellipse(img, ellipse, (0, 255, 0), 1)
    cv2.imshow('PIC', img)
    cv2.waitKey(0)


def mainPutFaceBack():
    # imgPath = './tmp.jpg'
    imgPath = '../../github/vrn-07231340/examples/scaled/trump-12.jpg'
    objPath = '../../github/vrn-07231340/obj/trump-12.obj'
    img = cv2.imread(imgPath)
    '''1. detect landmarks for origin image'''
    originLandmark2D = getLandmark2D(img)
    # for (x, y) in originLandmark2D:
    #     cv2.circle(img, (x, y), 1, (0, 255, 0), -1)
    '''2. read origin 3d model'''
    objLines, vLines, _ = readObj(objPath)
    '''3. (OPENCV) fit 3d ellipsoid '''
    '''想用XY, YZ, XZ三个面分别拟合三个椭圆，取最大的长短轴拟合椭球体，但是效果不好。'''
    '''画出来的椭圆不能包围所有的点'''
    pointsXY, pointsXZ, pointsYZ, pointsXYZ = getAllPoints(vLines)
    '''   3.1 fit XY front face ellipse'''
    # getEllipse(pointsXY, img)
    '''   3.2 fit XZ front face ellipse'''
    # getEllipse(pointsXZ, img)
    '''   3.3 fit YZ front face ellipse'''
    # for i in range(len(pointsYZ)):
    #     pointsYZ[i] = [pointsYZ[i][0]+20, pointsYZ[i][1]+20]
    # getEllipse(pointsYZ, img)

    '''3. (SKIMAGE) fit 3d ellipsoid '''
    '''由三个轴的半长轴求一个椭圆体'''
    '''首先求3d model 分别在三个轴上的最大最小值'''
    maxminDict = maxminXYZ(objLines)
    maxX = maxminDict['maxXCoord'][0]
    minX = maxminDict['minXCoord'][0]
    maxY = maxminDict['maxYCoord'][1]
    minY = maxminDict['minYCoord'][1]
    maxZ = maxminDict['maxZCoord'][2]
    minZ = maxminDict['minZCoord'][2]
    semiMajorX = (maxX-minX)/2
    semiMajorY = (maxY-minY)/2
    semiMajorZ = (maxZ-minZ)
    print semiMajorX, semiMajorY, semiMajorZ
    ellipsoid_base = ellipsoid(semiMajorX, semiMajorY, semiMajorZ)
    '''ellipsoid_base就是求得的椭球体，print出来都是False'''
    print ellipsoid_base
    '''下面这段代码想把椭球画出来看看长什么样子'''
    data = ellipsoid_base
    print data.shape
    x, y, z = data[0], data[1], data[2]
    print x.shape, y.shape, z.shape
    ax = plt.subplot(111, projection='3d')  # 创建一个三维的绘图工程
    ax.scatter(x, y, z, c='y')  # 绘制数据点

    ax.set_zlabel('Z')  # 坐标轴
    ax.set_ylabel('Y')
    ax.set_xlabel('X')
    plt.show()


if __name__ == "__main__":
    mainPutFaceBack()
