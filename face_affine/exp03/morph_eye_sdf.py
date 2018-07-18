#!/usr/bin/env python

import numpy as np
import cv2
import linecache


LEFTEYE_ALL_LANDMARK = [17, 18, 19, 20, 21, 39, 40, 41, 36, 37, 38]
RIGHTEYE_ALL_LANDMARK = [22, 23, 24, 25, 26, 45, 46, 47, 42, 43, 44]
LEFTEYE_CONTOUR_LANDMARK = [17, 18, 19, 20, 21, 39, 40, 41, 36]
# Read points from text file


def addEdgeLandmark(pts, img):
    size = img.shape
    imgHeight = size[0]-1
    imgWidth = size[1]-1
    halfHeight = size[0]/2
    halfWidth = size[1]/2
    edgeLandmark = [(0, 0), (0, halfHeight), (0, imgHeight), (halfWidth, imgHeight),
                    (imgWidth, imgHeight), (imgWidth, halfHeight), (imgWidth, 0), (halfWidth, 0)]
    return pts+edgeLandmark


def recoverMask(ptsContour, imgOriginal, imgMorph):
    print imgOriginal.dtype
    maskContour = np.zeros(imgOriginal.shape, dtype=np.uint8)
    cv2.fillConvexPoly(maskContour, np.int32(ptsContour), (255, 255, 255))
    # cv2.fillConvexPoly(maskContour, np.int32(
    #     ptsContour), (1.0, 1.0, 1.0), 16, 0)
    r = cv2.boundingRect(np.float32(ptsContour))
    center = (r[0] + int(r[2] / 2), r[1] + int(r[3] / 2))
    mat = cv2.getRotationMatrix2D(center, 0, 0.95)
    maskContour = cv2.warpAffine(
        maskContour, mat, (maskContour.shape[1], maskContour.shape[0]))
    # maskContour = cv2.blur(maskContour, (15, 10), center)
    cv2.imshow('1', imgMorph)
    cv2.imshow('2', imgOriginal)
    cv2.imshow('3', maskContour)
    # cv2.waitKey(0)
    imgRecover = cv2.seamlessClone(
        imgMorph, imgOriginal, maskContour, center, cv2.NORMAL_CLONE)
    # imgRecover = imgOriginal * (1 - maskContour) + imgMorph * maskContour
    return maskContour, imgRecover


def readPoints(ptsPath, contour=None):
    points = []
    if contour == 'LEFTEYE_CONTOUR_LANDMARK':
        count = 0
        for landmark in LEFTEYE_CONTOUR_LANDMARK:
            x, y = linecache.getline(ptsPath, landmark+1).split()
            points.append((float(x), float(y)))
    else:
        with open(ptsPath) as file:
            for line in file:
                x, y = line.split()
                points.append((float(x), float(y)))
    return points

# Apply affine transform calculated using srcTri and dstTri to src and
# output an image of size.


def applyAffineTransform(src, srcTri, dstTri, size):

    # Given a pair of triangles, find the affine transform.
    warpMat = cv2.getAffineTransform(np.float32(srcTri), np.float32(dstTri))
    # print warpMat
    # Apply the Affine Transform just found to the src image
    dst = cv2.warpAffine(src, warpMat, (size[0], size[1]), None,
                         flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
    return dst


# Warps and alpha blends triangular regions from img1 and img2 to img
def morphTriangle(img1, img2, img, t1, t2, t, alpha):

    # Find bounding rectangle for each triangle
    r1 = cv2.boundingRect(np.float32([t1]))
    r2 = cv2.boundingRect(np.float32([t2]))
    r = cv2.boundingRect(np.float32([t]))

    # Offset points by left top corner of the respective rectangles
    t1Rect = []
    t2Rect = []
    tRect = []

    for i in xrange(0, 3):
        tRect.append(((t[i][0] - r[0]), (t[i][1] - r[1])))
        t1Rect.append(((t1[i][0] - r1[0]), (t1[i][1] - r1[1])))
        t2Rect.append(((t2[i][0] - r2[0]), (t2[i][1] - r2[1])))
    print 'tRect: {}'.format(str(tRect))

    # Get mask by filling triangle
    mask = np.zeros((r[3], r[2], 3), dtype=np.float32)
    # print mask
    cv2.fillConvexPoly(mask, np.int32(tRect), (1.0, 1.0, 1.0), 16, 0)

    # Apply warpImage to small rectangular patches
    img1Rect = img1[r1[1]:r1[1] + r1[3], r1[0]:r1[0] + r1[2]]
    img2Rect = img2[r2[1]:r2[1] + r2[3], r2[0]:r2[0] + r2[2]]

    size = (r[2], r[3])
    warpImage1 = applyAffineTransform(img1Rect, t1Rect, tRect, size)
    warpImage2 = applyAffineTransform(img2Rect, t2Rect, tRect, size)

    # Alpha blend rectangular patches
    imgRect = (1.0 - alpha) * warpImage1 + alpha * warpImage2

    # Copy triangular region of the rectangular patch to the output image
    img[r[1]:r[1] + r[3], r[0]:r[0] + r[2]] = img[r[1]:r[1] +
                                                  r[3], r[0]:r[0] + r[2]] * (1 - mask) + imgRect * mask


if __name__ == '__main__':

    filename1 = '/Users/lls/Desktop/hillary/hillary_clinton.jpg'
    filename2 = '/Users/lls/Desktop/hillary/hillary_clinton_closeeye.jpg'
    alpha = 1

    # Read images
    img1 = cv2.imread(filename1)
    img2 = cv2.imread(filename2)

    # Convert Mat to float data type
    img1 = np.float32(img1)
    img2 = np.float32(img2)

    # Read array of corresponding points
    points1 = readPoints(filename1 + '.txt')
    points2 = readPoints(filename2 + '.txt')
    points1 = addEdgeLandmark(points1, img1)
    points2 = addEdgeLandmark(points2, img2)
    points = []

    # Compute weighted average point coordinates
    for i in xrange(0, len(points1)):
        x = (1 - alpha) * points1[i][0] + alpha * points2[i][0]
        y = (1 - alpha) * points1[i][1] + alpha * points2[i][1]
        points.append((x, y))

    # Allocate space for final output
    imgMorph = np.zeros(img1.shape, dtype=img1.dtype)

    # Read triangles from tri.txt
    with open("/Users/lls/Documents/face/data/source/mytri.txt") as file:
        for line in file:
            x, y, z = line.split()

            x = int(x)
            y = int(y)
            z = int(z)
            if (x in LEFTEYE_ALL_LANDMARK) and (y in LEFTEYE_ALL_LANDMARK) and (z in LEFTEYE_ALL_LANDMARK):

                t1 = [points1[x], points1[y], points1[z]]
                t2 = [points2[x], points2[y], points2[z]]
                t = [points[x], points[y], points[z]]
                # print t1, t2, t

                # Morph one triangle at a time.
                morphTriangle(img1, img2, imgMorph, t1, t2, t, alpha)

    cv2.imshow("Morphed Face", np.uint8(imgMorph))

    ptsContour = readPoints(
        filename2 + '.txt', contour='LEFTEYE_CONTOUR_LANDMARK')
    maskContour, imgRecover = recoverMask(
        ptsContour, np.uint8(img1), np.uint8(imgMorph))
    cv2.imshow('recover', imgRecover)
    cv2.waitKey(0)
