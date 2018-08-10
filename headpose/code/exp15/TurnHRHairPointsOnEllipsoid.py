# -*- coding: utf-8 -*-
import cv2
from skimage.draw import ellipse_perimeter, ellipsoid


def mainTurnHRHairOnEllipsoid():
    '''1. read img LR and HR'''
    imgLRPath = '../../github/vrn-07231340/examples/scaled/trump_12.jpg'
    imgLR = cv2.imread(imgLRPath)
    imgHRPath = '../../github/vrn-07231340/examples/trump_12.png'
    imgHR = cv2.imread(imgHRPath)
    objPath = '../../github/vrn-07231340/obj/trump_12.obj'


if __name__ == "__main__":
    mainTurnHRHairOnEllipsoid()
