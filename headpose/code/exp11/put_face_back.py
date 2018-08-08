# -*- coding: utf-8 -*-
import dlib
import cv2
import os
from headpose import *
import math
from skimage.draw import ellipsoid
from mpl_toolkits.mplot3d import Axes3D
import types
import copy


def optionChooseZMax(originLandmark3DList):
    maxZLandmark3D = []
    for originLandmark3D in originLandmark3DList:
        originLandmark3DXY = originLandmark3D[0]
        originLandmark3DXYZ = originLandmark3D[1]
        zList = []
        for i, (line, vLine) in enumerate(originLandmark3D[2]):
            _, _, _, z, _, _, _ = vLine.split()
            zList.append(float(z))
        maxZLandmark3D.append(
            [originLandmark3DXY, originLandmark3DXYZ[zList.index(max((zList)))], originLandmark3D[2][zList.index(max(zList))]])
    return maxZLandmark3D


def getNodLandmark3D(nodedVLines, originLandmark3DList):
    nodedLandmark3DList = []
    for originLandmark3D in originLandmark3DList:
        originLandmark3DPointXY = originLandmark3D[0]
        originLandmark3DPointXYZ = originLandmark3D[1]
        originLandmark3DLine = originLandmark3D[2]
        vID = int(originLandmark3DLine[0])
        vLine = originLandmark3DLine[1]
        nodedVLine = findvLine(vID, nodedVLines)
        nodedXY = findXY(nodedVLine)
        nodedXYZ = findXYZ(nodedVLine)
        nodedLandmark3DList.append(
            [originLandmark3DPointXY, originLandmark3DPointXYZ, nodedXY, nodedXYZ, (vID, nodedVLine)])
    return nodedLandmark3DList


def getTurnLandmark3D(turnedVLines, originLandmark3DList):
    turnedLandmark3DList = []
    for originLandmark3D in originLandmark3DList:
        originLandmark3DPointXY = originLandmark3D[0]
        originLandmark3DPointXYZ = originLandmark3D[1]
        originLandmark3DLine = originLandmark3D[2]
        vID = int(originLandmark3DLine[0])
        vLine = originLandmark3DLine[1]
        turnedVLine = findvLine(vID, turnedVLines)
        turnedXY = findXY(turnedVLine)
        turnedXYZ = findXYZ(turnedVLine)
        turnedLandmark3DList.append(
            [originLandmark3DPointXY, originLandmark3DPointXYZ, turnedXY, turnedXYZ, (vID, turnedVLine)])
    return turnedLandmark3DList


def findXY(vLine):
    _, x, y, _, _, _, _ = vLine.split()
    return [float(x), float(y)]


def findXYZ(vLine):
    _, x, y, z, _, _, _ = vLine.split()
    return [float(x), float(y), float(z)]


def findvLine(vID, vLines):
    for i, vLine in enumerate(vLines):
        if int(i) == int(vID):
            return vLine


def objLines2vLines(objLines):
    vLines = [objLine for objLine in objLines if objLine.split()[0] == 'v']
    return vLines


def imshow(title, image):
    cv2.imshow(title, image)
    cv2.waitKey(0)


def drawPointsOnImg(points, img, color, radius=1, cover=True):
    if cover == False:
        imgTmp = copy.copy(img)
    else:
        imgTmp = img
    if type(points) == types.ListType:
        for point in points:
            if color == 'r':
                cv2.circle(imgTmp, (int(point[0]), int(
                    point[1])), radius, (0, 0, 255), -1)
            elif color == 'g':
                cv2.circle(imgTmp, (int(point[0]), int(
                    point[1])), radius, (0, 255, 0), -1)
            else:
                cv2.circle(imgTmp, (int(point[0]), int(
                    point[1])), radius, (255, 0, 0), -1)
    elif type(points) == types.TupleType:
        if color == 'r':
            cv2.circle(
                imgTmp, (int(points[0]), int(points[1])), radius, (0, 0, 255), -1)
        elif color == 'g':
            cv2.circle(
                imgTmp, (int(points[0]), int(points[1])), radius, (0, 255, 0), -1)
        else:
            cv2.circle(
                imgTmp, (int(points[0]), int(points[1])), radius, (255, 0, 0), -1)
    elif type(points) == type(np.array([])):
        if points.shape[0] == 1 or points.shape[1] == 1:
            if color == 'r':
                cv2.circle(imgTmp, (int(points[0]), int(
                    points[1])), radius, (0, 0, 255), -1)
            elif color == 'g':
                cv2.circle(imgTmp, (int(points[0]), int(
                    points[1])), radius, (0, 255, 0), -1)
            else:
                cv2.circle(imgTmp, (int(points[0]), int(
                    points[1])), radius, (255, 0, 0), -1)
        else:
            for point in points:
                if color == 'r':
                    cv2.circle(imgTmp, (int(point[0]), int(
                        point[1])), radius, (0, 0, 255), -1)
                elif color == 'g':
                    cv2.circle(imgTmp, (int(point[0]), int(
                        point[1])), radius, (0, 255, 0), -1)
                else:
                    cv2.circle(imgTmp, (int(point[0]), int(
                        point[1])), radius, (255, 0, 0), -1)
    cv2.imshow('drawPointsOnImg', imgTmp)
    cv2.waitKey(0)
    return imgTmp


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
        coords.append([int(landmark2D.part(i).x), int(landmark2D.part(i).y)])
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
    originLandmark3DList = []
    '''all matches'''
    '''[[[x, y], [(line1, vLine1), (line2, vLine2)]], ... ]'''
    for (xLandmark, yLandmark) in landmark2D:
        originLandmark3DPointsXY = []
        originLandmark3DPointsXYZ = []
        originLandmark3DLines = []
        for (line, vLine) in enumerate(vLines):
            v, x, y, z, r, g, b = vLine.split()
            if float(x) == float(xLandmark) and float(y) == float(yLandmark):
                if [float(xLandmark), float(yLandmark)] not in originLandmark3DPointsXY:
                    originLandmark3DPointsXY.append(
                        [float(xLandmark), float(yLandmark)])
                originLandmark3DPointsXYZ.append(
                    [float(x), float(y), float(z)])
                originLandmark3DLines.append((line, vLine))
        if len(originLandmark3DPointsXY) != 0:
            originLandmark3DList.append(
                [originLandmark3DPointsXY[0], originLandmark3DPointsXYZ, originLandmark3DLines])
    return originLandmark3DList


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
