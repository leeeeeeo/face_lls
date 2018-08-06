import sys
sys.path.insert(0, '../../../face_affine/exp04')
from face_affine import changeExpression, readPoints, saveAnimation
import cv2

imgPath = '../../github/vrn-07231340/examples/scaled/trump-12.jpg'
img = cv2.imread(imgPath)
originLandmark2DTxtPath = './originLandmark2D.txt'
originLandmark2DTxt = readPoints(originLandmark2DTxtPath)
nodLandmark2DTxtPath = './nodLandmark2D.txt'
nodLandmark2DTxt = readPoints(nodLandmark2DTxtPath)
triTxtPath = './nodTri.txt'
frameList = changeExpression(imgPath, originLandmark2DTxt,
                             nodLandmark2DTxt, triTxtPath, img)
saveAnimation(frameList, 25, './nod.mp4', img, reverse=True)
