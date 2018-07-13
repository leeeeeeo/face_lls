import cv2
import numpy as np
import linecache


def changeExpression():
    ptsOriginal = readPoints(ptsOriginalPath)
    ptsTarget = readPoints(ptsTargetPath)
    step = 50
    ptsOld = []
    videoWriter = cv2.VideoWriter('./source/{}TO{}.mp4'.format(imgOriginalPath.split('.')[1].split('_')[1], ptsTargetPath.split(
        '.')[1].split('_')[1]), cv2.VideoWriter_fourcc(*'mp4v'), 10, (imgOriginal.shape[1], imgOriginal.shape[0]))
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
        imgMorphTmp = morphChange(ptsOriginal, ptsTmp, imgOriginal)
        ptsOld = ptsTmp
        cv2.imshow("Morphed Face Tmp", np.uint8(imgMorphTmp))
        videoWriter.write(imgMorphTmp)
        if i == 50:
            cv2.waitKey(0)
        else:
            cv2.waitKey(30)


def recoverMask():
    ptsContour = readPoints(ptsTargetPath, contour=True)
    maskContour = np.zeros(imgOriginal.shape, dtype=np.float32)
    cv2.fillConvexPoly(maskContour, np.int32(
        ptsContour), (1.0, 1.0, 1.0), 16, 0)
    r = cv2.boundingRect(np.int32(ptsContour))
    center = (r[0] + int(r[2] / 2), r[1] + int(r[3] / 2))
    mat = cv2.getRotationMatrix2D(center, 0, 0.95)
    maskContour = cv2.warpAffine(
        maskContour, mat, (maskContour.shape[1], maskContour.shape[0]))
    cv2.imshow('maskContour', maskContour)
    imgRecover = imgOriginal * (1 - maskContour) + imgMorph * maskContour
    cv2.imshow("Recoverd Face", np.uint8(imgRecover))
    return imgRecover


def morph():
    ptsOriginal = readPoints(ptsOriginalPath)
    ptsTarget = readPoints(ptsTargetPath)
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
    cv2.imshow("Morphed Face", np.uint8(imgMorph))
    return imgMorph


def morphChange(ptsOriginal, ptsTmp, imgOriginal):
    imgMorphTmp = np.zeros(imgOriginal.shape, dtype=imgOriginal.dtype)
    with open(triTxtPath) as file:
        for line in file:
            x, y, z = line.split()
            x = int(x)
            y = int(y)
            z = int(z)
            t1 = [ptsOriginal[x], ptsOriginal[y], ptsOriginal[z]]
            t = [ptsTmp[x], ptsTmp[y], ptsTmp[z]]
            morphTriangle(imgOriginal, imgMorphTmp, t1, t)
    return imgMorphTmp


def saveTriangleTxt(triangleTxtPath):
    ptsDict = {}
    for i, pts in enumerate(readPoints(ptsOriginalPath)):
        ptsDict['({}, {})'.format(pts[0], pts[1])] = i

    tartriList = delaunay(imgOriginal.shape, readPoints(
        ptsOriginalPath), removeOutlier=True)

    myTriTxt = open(myTriTxtPath, 'w')
    for tarTri in tartriList:
        triLine = '{} {} {}'.format(
            ptsDict[str(tarTri[0])], ptsDict[str(tarTri[1])], ptsDict[str(tarTri[2])])
        myTriTxt.write(triLine+'\n')
    myTriTxt.close()


def rect_contains(rect, point):
    if point[0] < rect[0]:
        return False
    elif point[1] < rect[1]:
        return False
    elif point[0] > rect[2]:
        return False
    elif point[1] > rect[3]:
        return False
    return True


def readPoints(ptsPath, contour=False):
    points = []
    if contour == True:
        count = 0
        for landmark in FACE_CONTOUR_LANDMARKS:
            x, y = linecache.getline(ptsPath, landmark+1).split()
            points.append((int(x), int(y)))
    else:
        with open(ptsPath) as file:
            for line in file:
                x, y = line.split()
                points.append((int(x), int(y)))
    return points


def delaunay(size, points, removeOutlier=False):
    triList = []
    rect = (0, 0, size[1], size[0])
    subdivOriginal = cv2.Subdiv2D(rect)
    for p in points:
        subdivOriginal.insert(p)
    triangleList = subdivOriginal.getTriangleList()
    triangleList = triangleList.astype(np.int32)
    if removeOutlier == True:
        for triangle in triangleList:
            pt1 = (triangle[0], triangle[1])
            pt2 = (triangle[2], triangle[3])
            pt3 = (triangle[4], triangle[5])
            if rect_contains(rect, pt1) and rect_contains(rect, pt2) and rect_contains(rect, pt3):
                triList.append((pt1, pt2, pt3))
    else:
        for triangle in triangleList:
            pt1 = (triangle[0], triangle[1])
            pt2 = (triangle[2], triangle[3])
            pt3 = (triangle[4], triangle[5])
            triList.append((pt1, pt2, pt3))
    return triList


def applyAffineTransform(src, srcTri, dstTri, size):
    warpMat = cv2.getAffineTransform(np.float32(srcTri), np.float32(dstTri))
    dst = cv2.warpAffine(src, warpMat, (size[0], size[1]), None,
                         flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
    return dst


def morphTriangle(img1, img, t1,  t):
    r1 = cv2.boundingRect(np.float32([t1]))
    r = cv2.boundingRect(np.float32([t]))

    t1Rect = []
    tRect = []

    for i in xrange(0, 3):
        tRect.append(((t[i][0] - r[0]), (t[i][1] - r[1])))
        t1Rect.append(((t1[i][0] - r1[0]), (t1[i][1] - r1[1])))

    maskTriangle = np.zeros((r[3], r[2], 3), dtype=np.float32)
    cv2.fillConvexPoly(maskTriangle, np.int32(tRect), (1.0, 1.0, 1.0), 16, 0)

    img1Rect = img1[r1[1]:r1[1] + r1[3], r1[0]:r1[0] + r1[2]]

    size = (r[2], r[3])
    warpImage1 = applyAffineTransform(img1Rect, t1Rect, tRect, size)

    imgRect = warpImage1
    img[r[1]:r[1] + r[3], r[0]:r[0] + r[2]] = img[r[1]:r[1] +
                                                  r[3], r[0]:r[0] + r[2]] * (1 - maskTriangle) + imgRect * maskTriangle


FACE_CONTOUR_LANDMARKS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,
                          12, 13, 14, 15, 16, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17]
imgOriginalPath = './source/S132_8.png'
ptsOriginalPath = '{}.txt'.format(imgOriginalPath)
ptsTargetPath = './source/S132_16.png.txt'
myTriTxtPath = './source/mytri.txt'
triTxtPath = myTriTxtPath  # triTxtPath = './source/tri_wo_background_w_edge.txt'
imgOriginal = cv2.imread(imgOriginalPath)
cv2.imshow("Original Face", np.uint8(imgOriginal))
"""
save mytri.txt
"""
# saveTriangleTxt(myTriTxtPath)
"""
morph from one expression to another
"""
imgMorph = morph()
"""
makeup face area mask
"""
imgRecover = recoverMask()
"""
change original expression to target expression in 50 iterations
"""
changeExpression()

cv2.waitKey(0)
