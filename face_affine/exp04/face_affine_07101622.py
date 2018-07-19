import cv2
import numpy as np


def recoverMask():
    ptsContour = readPoints(ptsTargetPath, contour=True)
    maskContour = np.zeros(imgOriginal.shape, dtype=np.float32)
    cv2.fillConvexPoly(maskContour, np.int32(
        ptsContour), (1.0, 1.0, 1.0), 16, 0)
    imgRecover = imgOriginal*(1-maskContour)+imgMorph*maskContour
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
    return imgMorph


def saveTriangleTxt(triangleTxtPath, trianglePointsList):
    triangleTxt = open(triangleTxtPath, 'w')
    for i in trianglePointsList:
        triangleTxt.write(str(i)+'\n')
    triangleTxt.close()


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
        with open(ptsPath) as file:
            for line in file:
                if count in FACE_CONTOUR_LANDMARKS:
                    x, y = line.split()
                    points.append((int(x), int(y)))
                count = count+1
    else:
        with open(ptsPath) as file:
            for line in file:
                x, y = line.split()
                points.append((int(x), int(y)))
    return points


def delaunay(size, points):
    ptsList = []
    rect = (0, 0, size[1], size[0])
    subdivOriginal = cv2.Subdiv2D(rect)
    for p in points:
        subdivOriginal.insert(p)
    triangleList = subdivOriginal.getTriangleList()
    triangleList = triangleList.astype(np.int32)
    for triangle in triangleList:
        pt1 = (triangle[0], triangle[1])
        pt2 = (triangle[2], triangle[3])
        pt3 = (triangle[4], triangle[5])
        # if rect_contains(rect, pt1) and rect_contains(rect, pt2) and rect_contains(rect, pt3):
        #     ptsList.append((pt1, pt2, pt3))
        ptsList.append((pt1, pt2, pt3))
    print len(ptsList)
    return ptsList


def applyAffineTransform(src, srcTri, dstTri, size):

    # Given a pair of triangles, find the affine transform.
    warpMat = cv2.getAffineTransform(np.float32(srcTri), np.float32(dstTri))
    # print warpMat
    # Apply the Affine Transform just found to the src image
    dst = cv2.warpAffine(src, warpMat, (size[0], size[1]), None,
                         flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
    return dst

# Warps and alpha blends triangular regions from img1 and img2 to img


def morphTriangle(img1, img, t1,  t):

    # Find bounding rectangle for each triangle
    r1 = cv2.boundingRect(np.float32([t1]))
    r = cv2.boundingRect(np.float32([t]))

    # Offset points by left top corner of the respective rectangles
    t1Rect = []
    tRect = []

    for i in xrange(0, 3):
        tRect.append(((t[i][0] - r[0]), (t[i][1] - r[1])))
        t1Rect.append(((t1[i][0] - r1[0]), (t1[i][1] - r1[1])))
    # print tRect

    # Get mask by filling triangle
    maskTriangle = np.zeros((r[3], r[2], 3), dtype=np.float32)
    cv2.fillConvexPoly(maskTriangle, np.int32(tRect), (1.0, 1.0, 1.0), 16, 0)

    # Apply warpImage to small rectangular patches
    img1Rect = img1[r1[1]:r1[1] + r1[3], r1[0]:r1[0] + r1[2]]

    size = (r[2], r[3])
    warpImage1 = applyAffineTransform(img1Rect, t1Rect, tRect, size)

    # Alpha blend rectangular patches
    imgRect = warpImage1
    # Copy triangular region of the rectangular patch to the output image
    img[r[1]:r[1] + r[3], r[0]:r[0] + r[2]] = img[r[1]:r[1] +
                                                  r[3], r[0]:r[0] + r[2]] * (1 - maskTriangle) + imgRect * maskTriangle


FACE_CONTOUR_LANDMARKS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,
                          12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26]
imgOriginalPath = './source/S132_8.png'
ptsOriginalPath = '{}.txt'.format(imgOriginalPath)
ptsTargetPath = './source/S132_16.png.txt'
triTxtPath = './source/tri_wo_background.txt'
imgOriginal = cv2.imread(imgOriginalPath)
# morph
imgMorph = morph()
# makeup mask
imgRecover = recoverMask()

cv2.imshow("Morphed Face", np.uint8(imgMorph))
cv2.imshow("Recoverd Face", np.uint8(imgRecover))
cv2.waitKey(0)
