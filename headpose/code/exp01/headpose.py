# -*- coding: utf-8 -*-
import os
import cv2
import numpy as np
import math
from scipy import interpolate
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
import pylab as pl
import matplotlib as mpl


def findTurnCenter(objLines, turnCenterMode):
    maxminDict = maxminXYZ(objLines)
    if turnCenterMode == 'maxY':
        '''a. max y as turn center'''
        maxYCoord = maxminDict['maxYCoord']
        turnCenter = (maxYCoord[0], maxYCoord[2])
    elif turnCenterMode == 'midX':
        '''b. minpoint of x as turn center'''
        maxXCoord = maxminDict['maxXCoord']
        minXCoord = maxminDict['minXCoord']
        midpointX = (maxXCoord[0]+minXCoord[0])/2
        allX = []
        allY = []
        allZ = []
        for objLine in objLines:
            if objLine.split()[0] == 'v':
                v, x, y, z, r, g, b = objLine.split()
                if float(x) == midpointX:
                    allX.append(float(x))
                    allY.append(float(y))
                    allZ.append(float(z))
        turnCenter = (allX[allY.index(max(allY))],
                      allZ[allY.index(max(allY))])
    return turnCenter


def findShakeCenter(objLines, shakeCenterMode):
    maxminDict = maxminXYZ(objLines)
    if shakeCenterMode == 'maxY':
        '''a. max y as shake center'''
        maxYCoord = maxminDict['maxYCoord']
        shakeCenter = (maxYCoord[0], maxYCoord[1])
    elif shakeCenterMode == 'midX':
        '''b. minpoint of x as shake center'''
        maxXCoord = maxminDict['maxXCoord']
        minXCoord = maxminDict['minXCoord']
        midpointX = (maxXCoord[0]+minXCoord[0])/2
        allX = []
        allY = []
        for objLine in objLines:
            if objLine.split()[0] == 'v':
                v, x, y, z, r, g, b = objLine.split()
                if float(x) == midpointX:
                    allX.append(float(x))
                    allY.append(float(y))
        shakeCenter = (allX[allY.index(max(allY))], max(allY))
    return shakeCenter


def findNodCenter(objLines, nodCenterMode):
    maxminDict = maxminXYZ(objLines)
    if nodCenterMode == 'maxY':
        '''a. max y as nod center'''
        maxYCoord = maxminDict['maxYCoord']
        nodCenter = (maxYCoord[1], maxYCoord[2])
    elif nodCenterMode == 'midX':
        '''b. minpoint of x as nod center'''
        maxXCoord = maxminDict['maxXCoord']
        minXCoord = maxminDict['minXCoord']
        midpointX = (maxXCoord[0]+minXCoord[0])/2
        allY = []
        allZ = []
        for objLine in objLines:
            if objLine.split()[0] == 'v':
                v, x, y, z, r, g, b = objLine.split()
                if float(x) == midpointX:
                    allY.append(float(y))
                    allZ.append(float(z))
        nodCenter = (max(allY), allZ[allY.index(max(allY))])
    return nodCenter


def turn3DModel(objLines, turnCenter, turnAngle):
    turnedObjLines = []
    for objLine in objLines:
        if objLine.split()[0] == 'v':
            v, x, y, z, r, g, b = objLine.split()
            ptX = float(x)
            ptZ = float(z)
            turnedX, turnedZ = rotateYZPlane(ptX, ptZ, turnCenter, turnAngle)
            objLine = '{} {} {} {} {} {} {}'.format(
                v, turnedX, y, turnedZ, r, g, b)
        turnedObjLines.append(objLine)
    return turnedObjLines


def shake3DModel(objLines, shakeCenter, shakeAngle):
    shakedObjLines = []
    for objLine in objLines:
        if objLine.split()[0] == 'v':
            v, x, y, z, r, g, b = objLine.split()
            ptX = float(x)
            ptY = float(y)
            shakedX, shakedY = rotateYZPlane(ptX, ptY, shakeCenter, shakeAngle)
            objLine = '{} {} {} {} {} {} {}'.format(
                v, shakedX, shakedY, z, r, g, b)
        shakedObjLines.append(objLine)
    return shakedObjLines


def nod3DModel(objLines, nodCenter, nodAngle):
    nodedObjLines = []
    for objLine in objLines:
        if objLine.split()[0] == 'v':
            v, x, y, z, r, g, b = objLine.split()
            ptY = float(y)
            ptZ = float(z)
            nodedY, nodedZ = rotateYZPlane(ptY, ptZ, nodCenter, nodAngle)
            objLine = '{} {} {} {} {} {} {}'.format(
                v, x, nodedY, nodedZ, r, g, b)
        nodedObjLines.append(objLine)
    return nodedObjLines


def saveObjFile(objFileName, objLines):
    newObjFilePath = '{}-{}.obj'.format(
        os.path.splitext(objFilePath)[0], objFileName)
    newObjFile = open(newObjFilePath, 'w')
    for objLine in objLines:
        newObjFile.write('{}\n'.format(objLine))
    newObjFile.close()


def objFloat2Int(objLines):
    newObjLines = []
    for objLine in objLines:
        if objLine.split()[0] == 'v':
            v, x, y, z, r, g, b = objLine.split()
            x = int(float(x))
            y = int(float(y))
            z = int(float(z))
            objLine = '{} {} {} {} {} {} {}'.format(v, x, y, z, r, g, b)
        newObjLines.append(objLine)
    return newObjLines


def readObj(objFilePath):
    '''read obj file'''
    objFile = open(objFilePath, 'r')
    objLines = [line.strip() for line in objFile]
    vLines = [objLine for objLine in objLines if objLine.split()[0] == 'v']
    fLines = [objLine for objLine in objLines if objLine.split()[0] == 'f']
    return objLines, vLines, fLines


def testObj(objFilePath):
    '''read origin obj file'''
    objLines, vLines, fLines = readObj(objFilePath)
    '''modify obj lines'''
    newObjLines = []
    for objLine in objLines:
        if objLine.split()[0] == 'v':
            v, x, y, z, r, g, b = objLine.split()
            if float(x) > 80 and float(x) < 90:
                r = 1
                g = 0
                b = 0
                objLine = '{} {} {} {} {} {} {}'.format(v, x, y, z, r, g, b)
            if float(x) > 90 and float(x) < 100:
                r = 0
                g = 1
                b = 0
                objLine = '{} {} {} {} {} {} {}'.format(v, x, y, z, r, g, b)
            if float(y) > 120:
                r = 0
                g = 1
                b = 0
                objLine = '{} {} {} {} {} {} {}'.format(v, x, y, z, r, g, b)
            if float(z) > 85:
                r = 0
                g = 0
                b = 1
                objLine = '{} {} {} {} {} {} {}'.format(v, x, y, z, r, g, b)
        newObjLines.append(objLine)
    '''save new obj file'''
    saveObjFile('new', newObjLines)


def testImg(imgPath):
    '''read same image'''
    img = cv2.imread(imgPath)
    imgHeight = img.shape[0]
    imgWidth = img.shape[1]
    for x in range(imgHeight):
        for y in range(imgWidth):
            if y > 80 and y < 90:
                img[x, y, 0] = 0
                img[x, y, 1] = 0
                img[x, y, 2] = 255
            if x > 120:
                img[x, y, 0] = 0
                img[x, y, 1] = 255
                img[x, y, 2] = 0
    cv2.imshow('1', img)
    cv2.waitKey(0)


def projection(objFilePath_OR_objLines, imgPath):
    if os.path.isfile(str(objFilePath_OR_objLines)):
        '''read origin obj file'''
        objLines, vLines, fLines = readObj(objFilePath)
    else:
        objLines = objFilePath_OR_objLines
    '''read same image'''
    img = cv2.imread(imgPath)
    '''create projected image'''
    projectedImg = np.zeros(img.shape)
    for objLine in objLines:
        if objLine.split()[0] == 'v':
            v, x, y, z, r, g, b = objLine.split()
            projectedImg[int(float(y)), int(float(x)), 0] = b
            projectedImg[int(float(y)), int(float(x)), 1] = g
            projectedImg[int(float(y)), int(float(x)), 2] = r
    cv2.imshow('projectedImg', projectedImg)
    cv2.waitKey(0)
    return projectedImg


def maxminXYZ(objLines):
    allX = []
    allY = []
    allZ = []
    maxminDict = {}
    for objLine in objLines:
        if objLine.split()[0] == 'v':
            v, x, y, z, r, g, b = objLine.split()
            allX.append(float(x))
            allY.append(float(y))
            allZ.append(float(z))
    maxX = max(allX)
    minX = min(allX)
    maxY = max(allY)
    minY = min(allY)
    maxZ = max(allZ)
    minZ = min(allZ)
    # print maxX, minX, maxY, minY, maxZ, minZ
    for objLine in objLines:
        if objLine.split()[0] == 'v':
            v, x, y, z, r, g, b = objLine.split()
            if float(x) == maxX:
                maxminDict['maxXCoord'] = [float(x), float(y), float(z)]
            if float(x) == minX:
                maxminDict['minXCoord'] = [float(x), float(y), float(z)]
            if float(y) == maxY:
                maxminDict['maxYCoord'] = [float(x), float(y), float(z)]
            if float(y) == minY:
                maxminDict['minYCoord'] = [float(x), float(y), float(z)]
            if float(z) == maxZ:
                maxminDict['maxZCoord'] = [float(x), float(y), float(z)]
            if float(z) == minZ:
                maxminDict['minZCoord'] = [float(x), float(y), float(z)]
    return maxminDict


def rotateYZPlane(ptY, ptZ, center, angle):
    imgWidth = 192
    imgHeight = 192
    ptY = ptY
    ptZ = imgHeight-ptZ
    centerY = center[0]
    centerZ = imgHeight-center[1]
    newY = (ptY-centerY)*math.cos(math.pi/180.0*angle) - \
        (ptZ-centerZ)*math.sin(math.pi/180.0*angle)+centerY
    newZ = (ptY-centerY)*math.sin(math.pi/180.0*angle) + \
        (ptZ-centerZ)*math.cos(math.pi/180.0*angle)+centerZ
    newY = newY
    newZ = imgHeight-newZ
    return round(newY, 1), round(newZ, 1)


def nod(objFilePath, nodAngle, nodCenterMode):
    '''read origin obj file'''
    objLines, vLines, fLines = readObj(objFilePath)
    '''1. find nod center'''
    nodCenter = findNodCenter(objLines, nodCenterMode)
    '''2. nod every point in YZ plane around nodCenter'''
    nodedObjLines = nod3DModel(objLines, nodCenter, nodAngle)
    '''3. save noded obj file'''
    saveObjFile('noded', nodedObjLines)
    return nodedObjLines


def shake(objFilePath, shakeAngle, shakeDirection, shakeCenterMode):
    '''read origin obj file'''
    objLines, vLines, fLines = readObj(objFilePath)
    '''1. find shake center'''
    shakeCenter = findShakeCenter(objLines, shakeCenterMode)
    '''2. shake every point in XY plane around shakeCenter'''
    shakedObjLines = shake3DModel(objLines, shakeCenter, shakeAngle)
    '''3. save shaked obj file'''
    saveObjFile('shaked', shakedObjLines)
    return shakedObjLines


def turn(objFilePath, turnAngle, turnDirection, turnCenterMode):
    '''read origin obj file'''
    objLines, vLines, fLines = readObj(objFilePath)
    '''1. find turn center'''
    turnCenter = findTurnCenter(objLines, turnCenterMode)
    '''2. turn every point in XZ plane around turnCenter'''
    if turnDirection == 'left':
        turnedObjLines = turn3DModel(objLines, turnCenter, turnAngle)
    elif turnDirection == 'right':
        turnedObjLines = turn3DModel(objLines, turnCenter, 0-turnAngle)
    '''3. save turned obj file'''
    saveObjFile('turned', turnedObjLines)
    return turnedObjLines


def resizeObj(objFilePath):
    '''read origin obj file'''
    objLines, vLines, fLines = readObj(objFilePath)
    '''modify obj lines'''
    newObjLines = []
    for objLine in objLines:
        if objLine.split()[0] == 'v':
            v, x, y, z, r, g, b = objLine.split()
            x = float(x)*3
            y = float(y)*3
            z = float(z)*3
            objLine = '{} {} {} {} {} {} {}'.format(
                v, x, y, z, r, g, b)
        newObjLines.append(objLine)
    '''save new obj file'''
    saveObjFile('resize', newObjLines)
    return newObjLines


def projectionResize(objFilePath_OR_objLines, imgPath):
    if os.path.isfile(str(objFilePath_OR_objLines)):
        '''read origin obj file'''
        objLines, vLines, fLines = readObj(objFilePath)
    else:
        objLines = objFilePath_OR_objLines
    '''read same image'''
    img = cv2.imread(imgPath)
    '''create projected image'''
    projectedImg = np.zeros(img.shape)
    for objLine in objLines:
        if objLine.split()[0] == 'v':
            v, x, y, z, r, g, b = objLine.split()
            projectedImg[int(float(y)), int(float(x)), 0] = b
            projectedImg[int(float(y)), int(float(x)), 1] = g
            projectedImg[int(float(y)), int(float(x)), 2] = r
    cv2.imshow('projectedImg', projectedImg)
    resizedImg = cv2.resize(projectedImg, (500, 500))
    cv2.imshow('resizedImg', resizedImg)
    # resizedImg = cv2.resize(projectedImg, (500, 500),
    #                         interpolation=cv2.INTER_AREA)
    # cv2.imshow('resizedImgA', resizedImg)
    # resizedImg = cv2.resize(projectedImg, (500, 500),
    #                         interpolation=cv2.INTER_CUBIC)
    # cv2.imshow('resizedImgC', resizedImg)
    # resizedImg = cv2.resize(projectedImg, (500, 500),
    #                         interpolation=cv2.INTER_LANCZOS4)
    # cv2.imshow('resizedImgL', resizedImg)
    cv2.waitKey(0)
    return projectedImg


def projectionResize1(objFilePath_OR_objLines):
    if os.path.isfile(str(objFilePath_OR_objLines)):
        '''read origin obj file'''
        objLines, vLines, fLines = readObj(objFilePath)
    else:
        objLines = objFilePath_OR_objLines
    '''read same image'''
    '''create projected image'''
    projectedImg = np.zeros((500, 500, 3))
    for objLine in objLines:
        if objLine.split()[0] == 'v':
            v, x, y, z, r, g, b = objLine.split()
            projectedImg[int(float(y)), int(float(x)), 0] = b
            projectedImg[int(float(y)), int(float(x)), 1] = g
            projectedImg[int(float(y)), int(float(x)), 2] = r
    cv2.imshow('projectedImg', projectedImg)
    cv2.waitKey(0)
    return projectedImg


def scipyInterpolate(objFilePath):
    '''origin plt'''
    grid_x, grid_y = np.mgrid[0:200:200j, 0:200:200j]
    objLines, vLines, fLines = readObj(objFilePath)
    origin = np.zeros(grid_x.shape)
    for objLine in objLines:
        if objLine.split()[0] == 'v':
            v, x, y, z, r, g, b = objLine.split()
            origin[int(float(y)), int(float(x))] = z
    plt.subplot(121)
    plt.imshow(origin)
    plt.title('original size')
    # plt.show()
    '''points & values'''
    grid_x, grid_y = np.mgrid[0:200:200j, 0:200:200j]
    objLines, vLines, fLines = readObj(objFilePath)
    xArray = np.array([])
    yArray = np.array([])
    zArray = np.array([])
    for objLine in objLines:
        if objLine.split()[0] == 'v':
            v, x, y, z, r, g, b = objLine.split()
            xArray = np.append(xArray, float(x))
            yArray = np.append(yArray, float(y))
            zArray = np.append(zArray, float(z))
    points = np.zeros((xArray.shape[0], 2))
    for i in range(xArray.shape[0]):
        points[i, :] = [yArray[i], xArray[i]]
    values = zArray
    grid = griddata(points, values, (grid_x, grid_y), method='cubic')
    plt.subplot(122)
    plt.imshow(grid)
    plt.title('after interpolate')
    plt.show()


if __name__ == "__main__":
    objFilePath = '../../github/vrn-07231340/obj/trump-12.obj'
    imgPath = '../../github/vrn-07231340/examples/scaled/trump-12.jpg'
    # testObj(objFilePath)
    # testImg(imgPath)
    '''3d project to 2d'''
    # projection(objFilePath, imgPath)
    '''nod'''
    '''there are 2 options for nod center:'''
    '''a. 'maxY': Y axis maximum point'''
    '''b. 'midX': midpoint of X axis with maximum of Y axis'''
    # nodAngle = 30
    # nodCenterMode = ['maxY', 'midX']
    # nodedObjLines = nod(objFilePath, nodAngle, nodCenterMode[1])
    # projection(nodedObjLines, imgPath)
    '''shake'''
    # shakeAngle = 30
    # shakeCenterMode = ['maxY', 'midX']
    # shakeDirection = ['left', 'right']
    # shakedObjLines = shake(objFilePath, shakeAngle,
    #                        shakeDirection[0], shakeCenterMode[0])
    # projection(shakedObjLines, imgPath)
    '''turn'''
    # turnAngle = 30
    # turnCenterMode = ['maxY', 'midX']
    # turnDirection = ['left', 'right']
    # turnedObjLines = turn(objFilePath, turnAngle,
    #                       turnDirection[0], turnCenterMode[0])
    # projection(turnedObjLines, imgPath)

    '''resize obj'''
    '''a. 直接把obj里面的所有点的坐标x, y, z放大3倍（x3），重新生成的3d模型看起来正常'''
    '''   但是由于点的间距变大，投影到2d会出现黑色点'''
    resizeObjLines = resizeObj(objFilePath)
    projectionResize1(resizeObjLines)
    '''b. 先把3d投影回2d，然后直接对2d做resize'''
    objLines = readObj(objFilePath)
    projectionResize(objFilePath, imgPath)
    '''c. 用scipy.interpolate.griddata进行插值'''
    '''   左图是原始x, y, z. 右图是插值后的x, y, z.'''
    scipyInterpolate(objFilePath)
