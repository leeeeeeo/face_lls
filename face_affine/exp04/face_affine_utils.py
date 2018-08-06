import cv2
import dlib
import numpy as np
import os
import logging
import linecache
from natsort import natsorted

FACE_CONTOUR_LANDMARKS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,
                          12, 13, 14, 15, 16, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17]
LEFTEYE_CONTOUR_LANDMARK = [18, 19, 20, 21, 22, 40, 41, 42, 37]
RIGHTEYE_CONTOUR_LANDMARK = [23, 24, 25, 26, 27, 46, 47, 48, 43]
LEFT_FACE_WIDTH_INDICES = [0, 1, 2]
RIGHT_FACE_WIDTH_INDICES = [14, 15, 16]
# center eye point + left + right eye
UPPER_FACE_HEIGHT_INDICES = [27, 36, 37,
                             38, 39, 40, 41, 42, 43, 44, 45, 46, 47]
BOTTOM_FACE_HEIGHT_INDICES = [31, 32, 33, 34, 35]
CENTER_FACE_INDICES = [27, 28, 29, 30, 31, 32, 33, 34, 35]
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(
    '../../../tools/shape_predictor_68_face_landmarks.dat')

# the general flow is to first rotate around the center of eyes
# then normalize according face dimensions


def calculateInvariantDims(shapeRotated):
    pointsLeft = extractPolygonCenter(shapeRotated, LEFT_FACE_WIDTH_INDICES)
    pointsRight = extractPolygonCenter(shapeRotated, RIGHT_FACE_WIDTH_INDICES)
    pointsUpper = extractPolygonCenter(shapeRotated, UPPER_FACE_HEIGHT_INDICES)
    pointsBottom = extractPolygonCenter(
        shapeRotated, BOTTOM_FACE_HEIGHT_INDICES)
    widthFace = pointsRight[0] - pointsLeft[0]
    heightFace = (pointsBottom[1] - pointsUpper[1]) * 2.0    # 2 x nose height
    ratio = float(widthFace) / float(heightFace)
    centerX, centerY = extractPolygonCenter(shapeRotated, CENTER_FACE_INDICES)
    faceInvariantParam = (ratio, centerX, centerY, widthFace, heightFace)
    return faceInvariantParam


def getNormedFaceSize(widthList, heightList):
    widthNormed = sum(widthList)/len(widthList)
    heightNormed = sum(heightList)/len(heightList)
    return (widthNormed, heightNormed)


def rotateShape(rotateMat, shapeNumpy):
    r'''step 1: rotate the shape from dlib into an np array shape
    rotateMat can be feed in with a (or an inverse) rotate matrix
    shapeNumpy can be norm/unnormed all/only-lip shapes'''
    rowNumber, columnNumber = shapeNumpy.shape
    shapeNumpyMatForRotate = np.vstack(
        [shapeNumpy, np.full(columnNumber, 1.0)])
    shapeRotated = np.dot(rotateMat, shapeNumpyMatForRotate)
    return shapeRotated


def extractPolygon(shapeNumpy, polygonIndices):
    points = map(lambda i: shapeNumpy[:, i], polygonIndices)
    return list(points)


def extractPolygonCenter(shapeNumpy, polygonIndices):
    points = extractPolygon(shapeNumpy, polygonIndices)
    centerX = map(lambda p: p[0], points)
    centerY = map(lambda p: p[1], points)
    return float(sum(centerX)) / len(centerX), float(sum(centerY)) / len(centerY)


def natsortFolder(folder):
    fileNames = []
    for root, folder, files in os.walk(folder):
        for fileName in files:
            if fileName.endswith('.png') or fileName.endswith('.jpg'):
                fileNames.append(fileName)
    fileNames = natsorted(fileNames)
    return fileNames


def shape_to_np(shape, dtype="int"):
    coords = np.zeros((68, 2), dtype=dtype)
    for i in range(0, 68):
        coords[i] = (shape.part(i).x, shape.part(i).y)
    return coords


def drawLandmark(img, targetPoints=None):
    '''green: target points'''
    '''blue: detected points'''
    if targetPoints:
        for point in targetPoints:
            cv2.circle(img, (int(point[0]), int(point[1])), 3, (0, 255, 0), -1)
    else:
        if img.shape[2] != 1:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img
        rects = detector(gray, 1)
        shape = predictor(gray, rects[0])
        shape = shape_to_np(shape)
        for (x, y) in shape:
            cv2.circle(img, (x, y), 3, (255, 0, 0), -1)
    return img


def addEdgeLandmark(pts, img):
    size = img.shape
    imgHeight = size[0]-1
    imgWidth = size[1]-1
    halfHeight = size[0]/2
    halfWidth = size[1]/2
    edgeLandmark = [(0, 0), (0, halfHeight), (0, imgHeight), (halfWidth, imgHeight),
                    (imgWidth, imgHeight), (imgWidth, halfHeight), (imgWidth, 0), (halfWidth, 0)]
    # edgeLandmark = [[0, 0], [0, halfHeight], [0, imgHeight], [halfWidth, imgHeight],
    # [imgWidth, imgHeight], [imgWidth, halfHeight], [imgWidth, 0], [halfWidth, 0]]
    return pts+edgeLandmark


def readPts(ptsFilePath):
    with open(ptsFilePath) as ptsFile:
        txtFile = []
        for line in ptsFile:
            txtFile.append(line[:-1])
    txtFile = txtFile[3:-1]
    tmpFile = []
    for line in txtFile:
        tmpLine = (float(line.split()[0]), float(line.split()[1]))
        tmpFile.append(tmpLine)
    return tmpFile


def videoToImg(videoPath, imgFolder):
    if not os.path.exists(imgFolder):
        os.mkdir(imgFolder)
    videoCapture = cv2.VideoCapture(videoPath)
    count = 0
    while True:
        success, frame = videoCapture.read()
        # print success
        if success == False:
            break
        imgPath = '{}/{}_{}.jpg'.format(imgFolder,
                                        videoPath.split('/')[-1].split('.')[0], count)
        print 'processing {}'.format(imgPath)
        cv2.imwrite(imgPath, frame)
        count = count+1
        # if cv2.waitKey(100) & 0xFF == ord('q'):
        #     break


def video_to_image(mp4_file, mp4_dir, trans_file, param):
    r'''extract image from the mp4_file, dump into mp4_dir'''
    cap = cv2.VideoCapture(mp4_file)
    count = 1
    img_list = []
    while count <= param.video_frame_num:
        if (count - 1) % 1000 == 0:
            div = int((count - 1) / 1000)
            mp4_sub_dir = os.path.join(
                mp4_dir, str(div).zfill(param.zfill_num - 3))
            if not os.path.isdir(mp4_sub_dir):
                os.mkdir(mp4_sub_dir)
                logging.info("writing into %s" % mp4_sub_dir)
        img_path = os.path.join(mp4_sub_dir, str(
            count).zfill(param.zfill_num) + ".png")
        os.path.join(mp4_dir, str(count).zfill(param.zfill_num) + ".png")
        ret, img = cap.read()
        if ret == 0:
            logging.warning("%s cannot be read." % img)
        if not os.path.isfile(img_path):
            cv2.imwrite(img_path, img)
        else:
            logging.info("skipping %s" % img_path)
        img_list.append(img_path)
        count = count + 1
        with open(trans_file, 'a') as f:
            f.write("#")
            f.write("%d,%s" % (param.video_frame_num, mp4_dir))


def morphChange(ptsOriginal, ptsTmp, imgOriginal, triTxtPath):
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


def morphTriangle(img1, img, t1,  t):
    '''
    img1: imgOriginal
    img: imgMorph
    '''
    r1 = cv2.boundingRect(np.float32([t1]))
    r = cv2.boundingRect(np.float32([t]))

    t1Rect = []
    tRect = []

    for i in xrange(0, 3):
        tRect.append(((t[i][0] - r[0]), (t[i][1] - r[1])))
        t1Rect.append(((t1[i][0] - r1[0]), (t1[i][1] - r1[1])))

    maskTriangle = np.zeros((r[3], r[2], 3), dtype=np.float32)
    cv2.fillConvexPoly(maskTriangle, np.int32(tRect), (1.0, 1.0, 1.0), 16, 0)
    # cv2.imshow('maskTriangle', maskTriangle)
    img1Rect = img1[r1[1]:r1[1] + r1[3], r1[0]:r1[0] + r1[2]]
    # print (r1[1], r1[1] + r1[3], r1[0], r1[0] + r1[2])
    # cv2.imshow('img1Rect', img1Rect)
    # cv2.waitKey(0)

    size = (r[2], r[3])
    warpImage1 = applyAffineTransform(img1Rect, t1Rect, tRect, size)

    imgRect = warpImage1
    img[r[1]:r[1] + r[3], r[0]:r[0] + r[2]] = img[r[1]:r[1] +
                                                  r[3], r[0]:r[0] + r[2]] * (1 - maskTriangle) + imgRect * maskTriangle
    # cv2.imshow('img', img)
    # cv2.waitKey(0)


def morphTriangle_modify_for_meshwarp(img1, img, t1,  t):
    '''
    img1: imgOriginal
    img: imgMorph
    '''
    r1 = cv2.boundingRect(np.float32([t1]))
    r = cv2.boundingRect(np.float32([t]))

    t1Rect = []
    tRect = []

    for i in xrange(0, 3):
        tRect.append(((t[i][0] - r[0]), (t[i][1] - r[1])))
        t1Rect.append(((t1[i][0] - r1[0]), (t1[i][1] - r1[1])))

    maskTriangle = np.zeros((r[3], r[2], 3), dtype=np.float32)
    cv2.fillConvexPoly(maskTriangle, np.int32(tRect), (1.0, 1.0, 1.0), 16, 0)
    # cv2.imshow('maskTriangle', maskTriangle)
    img1Rect = img1[r1[1]:r1[1] + r1[3], r1[0]:r1[0] + r1[2]]
    # print (r1[1], r1[1] + r1[3], r1[0], r1[0] + r1[2])
    # cv2.imshow('img1Rect', img1Rect)
    # cv2.waitKey(0)

    size = (r[2], r[3])
    warpImage1 = applyAffineTransform(img1Rect, t1Rect, tRect, size)

    imgRect = warpImage1
    img[r[1]:r[1] + r[3], r[0]:r[0] + r[2]] = img[r[1]:r[1] +
                                                  r[3], r[0]:r[0] + r[2]] * (1 - maskTriangle) + imgRect * maskTriangle
    # cv2.imshow('img', img)
    # cv2.waitKey(0)


def applyAffineTransform(src, srcTri, dstTri, size):

    warpMat = cv2.getAffineTransform(np.float32(srcTri), np.float32(dstTri))
    dst = cv2.warpAffine(src, warpMat, (size[0], size[1]), None,
                         flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
    return dst


def readPoints(ptsPath, contour=None):
    points = []
    if contour == 'FACE_CONTOUR_LANDMARKS':
        count = 0
        for landmark in FACE_CONTOUR_LANDMARKS:
            x, y = linecache.getline(ptsPath, landmark+1).split()
            points.append((float(x), float(y)))
    else:
        with open(ptsPath) as file:
            for line in file:
                x, y = line.split()
                points.append((float(x), float(y)))
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


def delaunay_with_draw(img, size, points, removeOutlier=False):
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
                cv2.line(img, pt1, pt2, (0, 0, 255), 1, cv2.LINE_AA, 0)
                cv2.line(img, pt2, pt3, (0, 0, 255), 1, cv2.LINE_AA, 0)
                cv2.line(img, pt3, pt1, (0, 0, 255), 1, cv2.LINE_AA, 0)
                triList.append((pt1, pt2, pt3))
    else:
        for triangle in triangleList:
            pt1 = (triangle[0], triangle[1])
            pt2 = (triangle[2], triangle[3])
            pt3 = (triangle[4], triangle[5])
            triList.append((pt1, pt2, pt3))

    cv2.imshow('img', img)
    cv2.waitKey(0)
    return triList


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
