import cv2
import numpy as np


def save_triangle_txt(triangle_txt_path, triangle_pointsList):
    triangle_txt = open(triangle_txt_path, 'w')
    for i in triangle_pointsList:
        triangle_txt.write(str(i)+'\n')
    triangle_txt.close()


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


def read_points(points_path):
    points = []
    with open(points_path) as file:
        for line in file:
            x, y = line.split()
            points.append((int(x), int(y)))
    return points


def delaunay(size, points):
    points_list = []
    rect = (0, 0, size[1], size[0])
    subdiv_original = cv2.Subdiv2D(rect)
    for p in points:
        subdiv_original.insert(p)
    triangleList = subdiv_original.getTriangleList()
    triangleList = triangleList.astype(np.int32)
    for triangle in triangleList:
        pt1 = (triangle[0], triangle[1])
        pt2 = (triangle[2], triangle[3])
        pt3 = (triangle[4], triangle[5])
        # if rect_contains(rect, pt1) and rect_contains(rect, pt2) and rect_contains(rect, pt3):
        #     points_list.append((pt1, pt2, pt3))
        points_list.append((pt1, pt2, pt3))
    print len(points_list)
    return points_list


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
    mask = np.zeros((r[3], r[2], 3), dtype=np.float32)
    cv2.fillConvexPoly(mask, np.int32(tRect), (1.0, 1.0, 1.0), 16, 0)

    # Apply warpImage to small rectangular patches
    img1Rect = img1[r1[1]:r1[1] + r1[3], r1[0]:r1[0] + r1[2]]

    size = (r[2], r[3])
    warpImage1 = applyAffineTransform(img1Rect, t1Rect, tRect, size)

    # Alpha blend rectangular patches
    imgRect = warpImage1
    # Copy triangular region of the rectangular patch to the output image
    img[r[1]:r[1] + r[3], r[0]:r[0] + r[2]] = img[r[1]:r[1] +
                                                  r[3], r[0]:r[0] + r[2]] * (1 - mask) + imgRect * mask


image_original_path = './source/S132_8.png'
image_original = cv2.imread(image_original_path)
imgMorph = np.zeros(image_original.shape, dtype=image_original.dtype)
points_original_path = '{}.txt'.format(image_original_path)
points_target_path = './source/S132_16.png.txt'
points_original = read_points(points_original_path)
points_target = read_points(points_target_path)


with open("./source/tri.txt") as file:
    for line in file:
        x, y, z = line.split()
        x = int(x)
        y = int(y)
        z = int(z)
        t1 = [points_original[x], points_original[y], points_original[z]]
        t = [points_target[x], points_target[y], points_target[z]]
        morphTriangle(image_original, imgMorph, t1, t)

cv2.imshow("Morphed Face", np.uint8(imgMorph))
cv2.waitKey(0)
