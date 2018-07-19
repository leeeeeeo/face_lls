# # import cv2
# # import numpy as np
# # t1 = [(599, 400), (477, 345), (599, 0)]
# # t2 = [(599, 400), (464, 329), (599, 0)]
# # t = [(599.0, 400.0), (470.5, 337.0), (599.0, 0.0)]
# # r1 = cv2.boundingRect(np.float32([t1]))
# # r2 = cv2.boundingRect(np.float32([t2]))
# # r = cv2.boundingRect(np.float32([t]))
# # print t1[0]
# # print r1
# # img = np.zeros((700, 700, 3), dtype=np.uint8)
# # for i in range(img.shape[0]):
# #     for j in range(img.shape[1]):
# #         img[i, j] = (255, 255, 255)
# # cv2.circle(img, t1[0], 5, (0, 0, 255), -1)
# # cv2.circle(img, t1[1], 5, (0, 0, 255), -1)
# # cv2.circle(img, t1[2], 5, (0, 0, 255), -1)
# # cv2.rectangle(img, (r1[0], r1[1]), (r1[0] + r1[2],
# #                                     r1[1] + r1[3]), (0, 255, 0), 1)
# # t1Rect = []
# # t2Rect = []
# # tRect = []
# # for i in xrange(0, 3):
# #     tRect.append(((t[i][0] - r[0]), (t[i][1] - r[1])))
# #     t1Rect.append(((t1[i][0] - r1[0]), (t1[i][1] - r1[1])))
# #     t2Rect.append(((t2[i][0] - r2[0]), (t2[i][1] - r2[1])))

# # cv2.circle(img, t1Rect[0], 5, (255, 0, 0), -1)
# # cv2.circle(img, t1Rect[1], 5, (255, 0, 0), -1)
# # cv2.circle(img, t1Rect[2], 5, (255, 0, 0), -1)
# # cv2.imshow("img", img)

# # mask = np.zeros((r[3], r[2], 3), dtype=np.float32)
# # cv2.fillConvexPoly(mask, np.int32(tRect), (1.0, 1.0, 1.0), 16, 0)
# # cv2.imshow("mask", mask)

# # img1Rect = img[r1[1]:r1[1] + r1[3], r1[0]:r1[0] + r1[2]]
# # cv2.imshow("img1Rect", img1Rect)


# # cv2.waitKey(0)
# # print 5 / 50
# # print a
# # a = 50
# # a = [(1, 2), (2, 3)]
# # b = [(4, 5), (6, 7)]
# # print a + b

# # import os
# # from natsort import natsorted
# # imgNames = []
# # imgFolder = '/Users/lls/Documents/trump/trump'
# # for root, folder, files in os.walk(imgFolder):
# #     for fileName in files:
# #         if fileName.endswith('.png'):
# #             imgNames.append(fileName)
# # imgNames = natsorted(imgNames)
# # natsorted(imgNames)
# # print imgNames

# # import copy
# # a = [1, 2, 3]
# # b = copy.deepcopy(a)
# # print b
# # b.append(4)
# # print a, b

# import os
# root = '/Users/lls/Documents/face/data/headpose/HeadPoseImageDatabase_2'
# folderss = os.listdir(root)
# for folder in folderss:
#     if os.path.isdir(os.path.join(root, folder)):
#         newFolder = os.path.join(root, folder)
#         print newFolder
#         for a, b, c in os.walk(newFolder):
#             for file in c:
#                 try:
#                         # UP DOWN
#                     if file[12] == '9' or file[12] == '6':
#                         os.remove(os.path.join(newFolder, file))
#                     # LEFT RIGHT
#                     elif file[14] == '9' or file[14] == '7' or file[14] == '6' or file[14] == '4' or file[14] == '3' or file[14] == '1':
#                         os.remove(os.path.join(newFolder, file))
#                     elif file[15] == '9' or file[15] == '7' or file[15] == '6' or file[15] == '4' or file[15] == '3' or file[15] == '1':
#                         os.remove(os.path.join(newFolder, file))
#                 except:
#                     print os.path.join(newFolder, file)


# import os
# import shutil
# root = '/Users/lls/Documents/face/data/headpose/HeadPoseImageDatabase_2'
# folderss = os.listdir(root)
# for folder in folderss:
#     if os.path.isdir(os.path.join(root, folder)):
#         newFolder = os.path.join(root, folder)
#         for a, b, c in os.walk(newFolder):
#             for file in c:
#                 try:
#                     folderName = file.split('.')[0][11:14]
#                     folderPath = '/Users/lls/Documents/face/data/headpose/Angle/{}'.format(
#                         folderName)
#                     if not os.path.exists(folderPath):
#                         os.mkdir(folderPath)
#                     shutil.copy(os.path.join(newFolder, file), folderPath)
#                 except:
#                     print os.path.join(newFolder, file)


# import os
# import shutil
# for root, dirs, files in os.walk('/Users/lls/Documents/face/data/headpose/Angle/neutral0'):
#     for file in files:
#         folderName = file.split('.')[0][11:14]
#         print folderName
#         folderPath = '/Users/lls/Documents/face/data/headpose/Angle/{}'.format(
#             folderName)
#         if not os.path.exists(folderPath):
#             os.mkdir(folderPath)
#         try:
#             shutil.copy(os.path.join(root, file), folderPath)
#         except:
#             print os.path.join(root, file)


# import os
# import shutil
# for root, dirs, files in os.walk('/Users/lls/Documents/face/data/headpose/Angle'):
#     for file in files:
#         if file.endswith('.txt'):
#             os.remove(os.path.join(root, file))

# from face_affine_utils import *
# videoPath = "/Users/lls/Library/Containers/com.tencent.xinWeChat/Data/Library/Application Support/com.tencent.xinWeChat/2.0b4.0.9/8dcbd411b740144829123f3edb690a08/Message/MessageTemp/c74fbd5bf21b7a9dd96a14ddfee6e079/Video/01-01-03-01-01-01-01_1531532564906779.mp4"
# imgFolder = "/Users/lls/Desktop/01-01-03-01-01-01-01"
# videoCapture = cv2.VideoCapture(videoPath)
# videoToImg(videoPath, imgFolder)

# import cv2
# img = cv2.imread('/Users/lls/Documents/face/data/trump/trump/trump_12.png')
# newWidth = int(img.shape[0]*0.6)
# newHeight = int(img.shape[1]*0.6)
# img = cv2.resize(img, (newHeight, newWidth))
# topLeftX = 0
# topLeftY = 250
# bottomRightX = topLeftX+640
# bottomRightY = topLeftY+640
# img = img[topLeftX:bottomRightX, topLeftY:bottomRightY, :]
# cv2.imshow('img', img)
# # cv2.waitKey(0)
# cv2.imwrite('/Users/lls/Documents/face/data/talkingphoto/crop640.png', img)

import numpy as np
a = [[1, 2, 3], [7, 4, 9], [3, 7, 4], [1, 8, 3]]
'''to np'''
b = []
for a_i in a:
    b.append(np.array(a_i))

print b
'''mean'''
c = []
c.append(b[0])
for i in range(1, len(a)-1):
    c.append((b[i-1]+b[i]+b[i+1])/3.0)
c.append(b[-1])
print c
d = []
for c_i in c:
    d.append(c_i.tolist())
print d
