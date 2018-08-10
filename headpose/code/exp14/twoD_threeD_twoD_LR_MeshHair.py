# -*- coding: utf-8 -*-
import cv2
from headpose import readObj, maxminXYZ
import numpy as np
np.set_printoptions(threshold=np.inf)
from skimage.draw import ellipse_perimeter, ellipse, circle_perimeter
from skimage.measure import points_in_poly
from put_face_back import getLandmark2D, drawPointsOnImg, imshow, nodHair


ELLIPSE_CENTER = 27
FACE_CONTOUR_INDICES = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,
                        12, 13, 14, 15, 16, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17]


def edgePointsOnOuterEllipse(outerEllipseVerts):
    edgePoints = []
    '''13.1 (2) 椭圆和图像上边界两个交点'''
    UpPoints = [(x, y) for (x, y) in outerEllipseVerts if y == 0]
    leftUpPoint = min(UpPoints, key=lambda x: x[0])
    rightUpPoint = max(UpPoints, key=lambda x: x[0])
    edgePoints = edgePoints+[leftUpPoint, rightUpPoint]
    '''13.2 (2) 椭圆最左最右两点'''
    leftX = min(outerEllipseVerts, key=lambda x: x[0])[0]
    leftPoints = [(x, y) for (x, y) in outerEllipseVerts if x == leftX]
    leftPoint = min(leftPoints, key=lambda x: x[1])
    rightX = max(outerEllipseVerts, key=lambda x: x[0])[0]
    rightPoints = [(x, y) for (x, y) in outerEllipseVerts if x == rightX]
    rightPoint = min(rightPoints, key=lambda x: x[1])
    edgePoints = edgePoints+[leftPoint, rightPoint]
    '''13.3 (2) 椭圆和图像上边界的交点 和 椭圆最左点 的中点'''
    upMidPoints = [(x, y) for (x, y) in outerEllipseVerts if int(
        y) == int((leftPoint[1]+leftUpPoint[1])/2.0)]
    leftUpMidPoint = min(upMidPoints, key=lambda x: x[0])
    rightUpMidPoint = max(upMidPoints, key=lambda x: x[0])
    edgePoints = edgePoints+[leftUpMidPoint, rightUpMidPoint]
    '''13.4 (2) 13.3 和 13.1 的中点'''
    pt1 = (leftUpMidPoint[0], leftUpMidPoint[1]/2.0)
    pt2 = (rightUpMidPoint[0], rightUpMidPoint[1]/2.0)
    edgePoints = edgePoints+[pt1, pt2]
    '''13.5 (1) 椭圆最底部'''
    '''    OPTION 1: 衣领点'''
    # bottomPoint=collarPoint
    '''    OPTION 2: 椭圆最底点'''
    bottomY = max(outerEllipseVerts, key=lambda x: x[1])[1]
    bottomPoints = [(x, y) for (x, y) in outerEllipseVerts if y == bottomY]
    leftBottomX = min(bottomPoints, key=lambda x: x[0])[0]
    rightBottomX = max(bottomPoints, key=lambda x: x[0])[0]
    bottomPoint = ((leftBottomX+rightBottomX)/2.0, bottomY)
    edgePoints = edgePoints+[bottomPoint]
    '''13.6 (2) 椭圆最底部 和 椭圆最左点 的中点'''
    bottomMidPoints = [(x, y) for (x, y) in outerEllipseVerts if int(
        y) == int((rightPoint[1]+bottomY)/2.0)]
    leftBottomMidPoint = min(bottomMidPoints, key=lambda x: x[0])
    rightBottomMidPoint = max(bottomMidPoints, key=lambda x: x[0])
    edgePoints = edgePoints+[leftBottomMidPoint, rightBottomMidPoint]
    '''13.7 (2) 椭圆最底点左右两点'''
    leftBottomPoint = ((leftBottomMidPoint[0]+bottomPoint[0])/2.0, bottomY)
    rightBottomPoint = ((rightBottomMidPoint[0]+bottomPoint[0])/2.0, bottomY)
    edgePoints = edgePoints+[leftBottomPoint, rightBottomPoint]
    return edgePoints


def minYInLandmark(landmark):
    yList = []
    for (x, y) in landmark:
        yList.append(float(y))
    minY = min(yList)
    return minY


def points_in_ellipse(ellipseCenterX, ellipseCenterY, ellipseSemiX, ellipseSemiY, points):
    pointsInEllipseList = []
    for (x, y) in points:
        if ((x-ellipseCenterX)/ellipseSemiX)**2+((y-ellipseCenterY)/ellipseSemiY)**2 < 1:
            pointsInEllipseList.append((x, y))
    return pointsInEllipseList


def hairDetection(img, minSize=800):
    hairColor = ([50, 50, 120], [120, 110, 160])
    lowerTh = np.array(hairColor[0], dtype="uint8")
    upperTh = np.array(hairColor[1], dtype="uint8")
    mask = cv2.inRange(img, lowerTh, upperTh)
    output = cv2.bitwise_and(img, img, mask=mask)
    # cv2.imshow("images", np.hstack([img, output]))
    # cv2.waitKey(0)
    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(
        mask, connectivity=8)
    sizes = stats[1:, -1]
    sizes = list(sizes)
    nb_components = nb_components - 1
    min_size = minSize
    maskHair = np.zeros((output.shape[0], output.shape[1], 3))
    maskTmp = np.zeros((output.shape), dtype=np.uint8)
    # ret, maskTmp = cv2.threshold(maskTmp,127,255,cv2.THRESH_BINARY)
    pIndex = sizes.index(max(sizes))
    # 对于面积最大的头发区域，找到上下左右四个极点：
    maskHair[output == pIndex+1] = 100
    maskTmp[output == pIndex+1] = 255
    maxi = 0
    maxj = 0
    mini = maskHair.shape[0]
    minj = maskHair.shape[1]
    for i in range(0, maskTmp.shape[0]):
        for j in range(0, maskTmp.shape[1]):
            if maskTmp[i, j] == 255:
                if i < mini:
                    mini = i
                elif i > maxi:
                    maxi = i
                if j < minj:
                    minj = j
                elif j > maxj:
                    maxj = j
    for j in range(0, maskTmp.shape[1]):
        if maskTmp[maxi, j] == 255:
            maskHair[maxi, j] = [0, 0, 255]
            bottomPoint = (j, maxi)
        if maskTmp[mini, j] == 255:
            maskHair[mini, j] = [0, 255, 0]
            upPoint = (j, mini)
    for i in range(0, maskTmp.shape[0]):
        maskHair[i, minj] = [0, 0, 255]
        if maskTmp[i, maxj] == 255:
            maskHair[i, maxj] = [255, 0, 0]
            rightPoint = (maxj, i)
        if maskTmp[i, minj] == 255:
            maskHair[i, minj] = [255, 255, 0]
            leftPoint = (minj, i)
    cornerPoint = [upPoint, bottomPoint, leftPoint, rightPoint]
    # 左右两个极点画圆
    circleCenter = ((leftPoint[0]+rightPoint[0]) /
                    2.0, (leftPoint[1]+rightPoint[1])/2.0)
    circleRadius = np.sqrt(
        (leftPoint[0]-circleCenter[0])**2+(leftPoint[1]-circleCenter[1])**2)
    rr, cc = circle_perimeter(int(circleCenter[0]), int(
        circleCenter[1]), int(circleRadius))
    circle = np.dstack([rr, cc])[0]
    cornerPoint = cornerPoint+[circleCenter]
    # 四个极点等间隔，在圆上找几个点
    jiange = 5
    points = []
    for i in range(0, jiange):
        leftUpPoint = min(circle[circle[:, 1] == upPoint[1]+i *
                                 (leftPoint[1]-upPoint[1])/jiange], key=lambda x: x[0])
        leftUpPoint = (leftUpPoint[0], leftUpPoint[1])
        points.append(leftUpPoint)
        rightUpPoint = max(circle[circle[:, 1] == upPoint[1]+i *
                                  (rightPoint[1]-upPoint[1])/jiange], key=lambda x: x[0])
        rightUpPoint = (rightUpPoint[0], rightUpPoint[1])
        points.append(rightUpPoint)
        for j in range(1, jiange):
            jiangeX = (rightUpPoint[0]-leftUpPoint[0])/jiange
            jiangeY = (rightUpPoint[1]-leftUpPoint[1])/jiange
            points.append(
                ((leftUpPoint[0]+j*jiangeX), (leftUpPoint[1]+j*jiangeY)))

        leftBottomPoint = min(circle[circle[:, 1] == leftPoint[1]+i *
                                     (bottomPoint[1]-leftPoint[1])/jiange], key=lambda x: x[0])
        leftBottomPoint = (leftBottomPoint[0], leftBottomPoint[1])
        points.append(leftBottomPoint)
        rightBottomPoint = max(circle[circle[:, 1] == rightPoint[1]+i *
                                      (bottomPoint[1]-rightPoint[1])/jiange], key=lambda x: x[0])
        rightBottomPoint = (rightBottomPoint[0], rightBottomPoint[1])
        points.append(rightBottomPoint)

        # if i<2:
        #     for j in range(1, jiange):
        #         jiangeX = (rightBottomPoint[0]-leftBottomPoint[0])/jiange
        #         jiangeY = (rightBottomPoint[1]-leftBottomPoint[1])/jiange
        #         points.append(
        #             ((leftBottomPoint[0]+j*jiangeX), (leftBottomPoint[1]+j*jiangeY)))

    # 删除距离太近的点
    pointsToDel = list(set([(x, y) for (x, y) in points for (i, j)
                            in cornerPoint if abs(x-i) < 5 and abs(y-j) < 5]))
    points = [point for point in points if point not in pointsToDel]
    # 头发所有关键点=圆上的几个点+头发区域上下左右四个极点
    points = points+cornerPoint
    points = list(set(points))

    return points


def collarDetection(img, minSize=800, offset=5):
    collarColor = ([185, 175, 185], [230, 220, 230])
    lowerTh = np.array(collarColor[0], dtype="uint8")
    upperTh = np.array(collarColor[1], dtype="uint8")
    mask = cv2.inRange(img, lowerTh, upperTh)
    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(
        mask, connectivity=8)
    sizes = stats[1:, -1]
    nb_components = nb_components - 1
    min_size = minSize
    maskCollar = np.zeros((output.shape[0], output.shape[1], 3))
    maskTmp = np.zeros((output.shape))
    cornerPoints = []
    for p in range(0, nb_components):
        maxi = 0
        maxj = 0
        mini = maskCollar.shape[0]
        minj = maskCollar.shape[1]
        if sizes[p] >= min_size:
            maskCollar[output == p + 1] = 100
            maskTmp[output == p + 1] = 255
            for i in range(0, maskTmp.shape[0]):
                for j in range(0, maskTmp.shape[1]):
                    if maskTmp[i, j] == 255:
                        if i < mini:
                            mini = i
                        elif i > maxi:
                            maxi = i
                        if j < minj:
                            minj = j
                        elif j > maxj:
                            maxj = j
            for j in range(0, maskTmp.shape[1]):
                if maskTmp[maxi, j] == 255:
                    maskCollar[maxi, j] = [0, 0, 255]
                    bottomPoint = (j, maxi)
                if maskTmp[mini, j] == 255:
                    maskCollar[mini, j] = [0, 255, 0]
                    upPoint = (j, mini)
            for i in range(0, maskTmp.shape[0]):
                maskCollar[i, minj] = [0, 0, 255]
                if maskTmp[i, maxj] == 255:
                    maskCollar[i, maxj] = [255, 0, 0]
                    rightPoint = (maxj, i)
                if maskTmp[i, minj] == 255:
                    maskCollar[i, minj] = [255, 255, 0]
                    leftPoint = (minj, i)
            cornerPoint = [upPoint, bottomPoint, leftPoint, rightPoint]
            upPoint = (-1, -1)
            bottomPoint = (-1, -1)
            leftPoint = (-1, -1)
            rightPoint = (-1, -1)
            cornerPoints.append(cornerPoint)
            maskTmp = np.zeros((output.shape))
    # 左衣领的最右点  和  右衣领的最左点  的中间点
    leftCollarPoints = cornerPoints[0]
    rightCollarPoints = cornerPoints[1]
    collarPoint = ((leftCollarPoints[3][0]+rightCollarPoints[2][0]) /
                   2.0, (leftCollarPoints[3][1]+rightCollarPoints[2][1])/2.0)
    collarPoint = (collarPoint[0], collarPoint[1]+offset)  # 两个领子中间那个点往下挪一点
    return maskCollar, collarPoint


def addOuterEllipseEdgeLandmark(innerEllipseParam, img):
    outerEllipseVerts, edgePoints = outerEllipse(innerEllipseParam)
    '''collar detection'''
    maskCollar, collarPoint = collarDetection(
        img, minSize=200, offset=5)

    points = [i for i in outerEllipseVerts if i[1]
              == collarPoint[1]]  # 领子左右两个点
    points = [points[0], points[-1]]
    collarPoint = [collarPoint]

    edgePoints = edgePoints+points+collarPoint
    edgePoints = [(x, y) for (x, y) in edgePoints]
    # print len(edgePoints)
    return edgePoints, outerEllipseVerts


def outerEllipse(innerEllipseParam):
    '''outer ellipse'''
    outerBoundary = 5
    outerEllipseParam = innerEllipseParam
    outerEllipseParam['ellipseSemiY'] = innerEllipseParam['ellipseSemiY']+outerBoundary
    outerEllipseParam['ellipseSemiX'] = innerEllipseParam['ellipseSemiX']+outerBoundary
    rrOuter, ccOuter = ellipse_perimeter(outerEllipseParam['ellipseCenterY'], outerEllipseParam['ellipseCenterX'],
                                         outerEllipseParam['ellipseSemiY'], outerEllipseParam['ellipseSemiX'], orientation=outerEllipseParam['orientation'])
    outerEllipseVerts = np.dstack([rrOuter, ccOuter])[0]
    edgePoints = []

    points = [i for i in outerEllipseVerts if i[1] == 0]  # y=0的两个点

    maxXOuterEllipse = max(outerEllipseVerts, key=lambda x: x[0])[0]
    minXOuterEllipse = min(outerEllipseVerts, key=lambda x: x[0])[0]

    rightPoints = [i for i in outerEllipseVerts if i[0] == maxXOuterEllipse]
    leftPoints = [i for i in outerEllipseVerts if i[0] == minXOuterEllipse]
    rightPoints = np.asarray(rightPoints)
    leftPoints = np.asarray(leftPoints)
    edgePoints.append(np.mean(rightPoints, axis=0))  # 最左边和最右边的两个点
    edgePoints.append(np.mean(leftPoints, axis=0))
    edgePoints = edgePoints+points

    return outerEllipseVerts, edgePoints


def main2D_3D_2D_LR_MeshHair():
    '''1. read obj and img'''
    objPath = '../../github/vrn-07231340/obj/trump-12.obj'
    objLines, vLines, fLines = readObj(objPath)
    imgPath = '../../github/vrn-07231340/examples/scaled/trump-12.jpg'
    img = cv2.imread(imgPath)

    '''2. 2D face landmark'''
    oriFaceLandmark2DList = getLandmark2D(img)

    '''3. create source hair region mesh'''
    gridSize = 15
    rows, cols = img.shape[0], img.shape[1]
    src_rows = np.linspace(0, rows, gridSize)  # 10 rows
    src_cols = np.linspace(0, cols, gridSize)  # 20 columns
    src_rows, src_cols = np.meshgrid(src_rows, src_cols)
    src = np.dstack([src_cols.flat, src_rows.flat])[0]

    '''3.1 create head region ellipse'''
    '''!!!手动修改了椭圆中心点，因为川普本来就是侧脸!!!'''
    ellipseParam = {'ellipseCenterX': oriFaceLandmark2DList[ELLIPSE_CENTER][1], 'ellipseCenterY': oriFaceLandmark2DList[
        ELLIPSE_CENTER][0]-10, 'ellipseSemiX': 70, 'ellipseSemiY': 90, 'orientation': np.pi*1.5}
    rr, cc = ellipse_perimeter(
        ellipseParam['ellipseCenterY'], ellipseParam['ellipseCenterX'], ellipseParam['ellipseSemiY'], ellipseParam['ellipseSemiX'], orientation=ellipseParam['orientation'])
    '''3.2 mesh points in ellipse'''
    ellipseVerts = np.dstack([rr, cc])[0]
    '''3.3 add outer ellipse edge points'''
    edgePoints, outerEllipseVerts = addOuterEllipseEdgeLandmark(
        ellipseParam, img)
    # drawPointsOnImg(ellipseVerts, img, 'g', cover=True)
    # drawPointsOnImg(edgePoints, img, 'r')
    mask = points_in_poly(src, ellipseVerts)
    pointsInEllipseList = []
    indexInEllipseList = []
    for i, (s, m) in enumerate(zip(src, mask)):
        if m == True:
            pointsInEllipseList.append(s)
            indexInEllipseList.append(i)
    assert len(pointsInEllipseList) == len(indexInEllipseList)

    '''3.3 mesh points in face region'''
    faceVerts = []
    for FACE_CONTOUR_INDICE in FACE_CONTOUR_INDICES:
        faceVerts.append([oriFaceLandmark2DList[FACE_CONTOUR_INDICE]
                          [0], oriFaceLandmark2DList[FACE_CONTOUR_INDICE][1]])
    mask = points_in_poly(src, faceVerts)
    pointsInFaceList = []
    indexInFaceList = []
    for i, (s, m) in enumerate(zip(src, mask)):
        if m == True:
            pointsInFaceList.append(s)
            indexInFaceList.append(i)
    assert len(pointsInFaceList) == len(indexInFaceList)

    '''3.4 mesh points between (face region) and (ellipse)'''
    for i, ((pointInEllipseX, pointInEllipseY), indexInEllise) in enumerate(zip(pointsInEllipseList, indexInEllipseList)):
        for (pointInFaceX, pointInFaceY) in pointsInFaceList:
            if pointInEllipseX == pointInFaceX and pointInEllipseY == pointInFaceY:
                pointsInEllipseList[i] = np.array([-1, -1])
                indexInEllipseList[i] = -1
    pointsInHairList = []
    for (pointInEllipseX, pointInEllipseY) in pointsInEllipseList:
        if pointInEllipseX != -1 and pointInEllipseY != -1:
            pointsInHairList.append([pointInEllipseX, pointInEllipseY])
    indexInHairList = [x for x in indexInEllipseList if x != -1]
    assert len(pointsInHairList) == len(indexInHairList)

    '''3.5 mesh points above minY'''
    minY = minYInLandmark(oriFaceLandmark2DList)
    hairMeshPointsInEllipseList = []
    hairMeshIndexInEllipseList = []
    for (s, i) in zip(pointsInHairList, indexInHairList):
        srcY = s[1]
        srcX = s[0]
        if srcY < minY and srcY >= 0:
            hairMeshPointsInEllipseList.append((srcX, srcY))
            hairMeshIndexInEllipseList.append(i)
    assert len(hairMeshPointsInEllipseList) == len(hairMeshIndexInEllipseList)
    # drawPointsOnImg(hairMeshPointsInEllipseList, img, 'r')

    '''4. 2D hair mesh points --> 3D hair mesh points'''
    '''4.1 3D hair landmark list: (x, y, z)'''
    # maxminDict = maxminXYZ(objLines)
    # minZ = float(maxminDict['maxZCoord'][2])
    minZ = 60.0
    hairLandmark3DList = []
    for hairLandmark2D in hairMeshPointsInEllipseList:
        hairLandmark3D = (float(hairLandmark2D[0]),
                          float(hairLandmark2D[1]), minZ)
        hairLandmark3DList.append(hairLandmark3D)
    '''4.2 hair landmark color list: (r, g, b)'''
    colorList = []
    for hairLandmark2D in hairMeshPointsInEllipseList:
        r = img[int(hairLandmark2D[0]), int(hairLandmark2D[1]), 2]/255.0
        g = img[int(hairLandmark2D[0]), int(hairLandmark2D[1]), 1]/255.0
        b = img[int(hairLandmark2D[0]), int(hairLandmark2D[1]), 0]/255.0
        color = (r, g, b)
        colorList.append(color)
    '''4.3 3D vLines'''
    hairVLines = []
    for (x, y, z), (r, g, b) in zip(hairLandmark3DList, colorList):
        hairVLine = 'v {} {} {} {} {} {}'.format(x, y, z, r, g, b)
        hairVLines.append(hairVLine)
    '''4.4 write vLines txt'''
    hairVLinesTxtPath = './hairVLines.txt'
    hairVLinesTxt = open(hairVLinesTxtPath, 'w')
    for hairVLine in hairVLines:
        hairVLinesTxt.write(hairVLine+'\n')
    hairVLinesTxt.close()

    '''5. nod hair'''
    nodAngle = 15
    nodCenterMode = 'maxY'
    nodHairVLines = nodHair(objPath, nodAngle, nodCenterMode,
                            hairVLines, hairVLinesTxtPath)

    '''6. 2D nod hair landmark'''
    nodHairLandmark2DList = []
    for nodHairVLine in nodHairVLines:
        _, x, y, _, _, _, _ = nodHairVLine.split()
        nodHairLandmark2DList.append((float(x), float(y)))
    assert len(hairMeshPointsInEllipseList) == len(nodHairLandmark2DList)

    # print hairMeshPointsInEllipseList
    # print nodHairLandmark2DList
    return hairMeshPointsInEllipseList, nodHairLandmark2DList, edgePoints, ellipseVerts, outerEllipseVerts


if __name__ == "__main__":
    main2D_3D_2D_LR_MeshHair()
