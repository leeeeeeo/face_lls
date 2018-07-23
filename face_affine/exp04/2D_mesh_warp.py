import cv2
import os
from natsort import natsorted
import numpy as np
from face_affine_utils import *


def chooseTrainingset():
    folder = './data/talkingphoto/IMG_2294_nod'
    imgList = natsorted(
        [file for file in os.listdir(folder) if file.endswith('.png')])
    imgPathList = [os.path.join(folder, imgFile) for imgFile in imgList]
    for imgPath in imgPathList:
        img = cv2.imread(imgPath)
        cv2.imshow('img', img)
        cv2.waitKey(5)


def get_outerlip_size(np_outerlip_shape):
    right = np_outerlip_shape[RIGHT_MOST_LIP - START_OUTER_LIP]
    left = np_outerlip_shape[LEFT_MOST_LIP - START_OUTER_LIP]
    upper = np_outerlip_shape[UP_MOST_LIP - START_OUTER_LIP]
    bottom = np_outerlip_shape[BOTTOM_MOST_LIP - START_OUTER_LIP]
    width = right[0] - left[0]
    height = bottom[1] - upper[1]
    return width, height


def get_normed_outerlip_size(np_outer_lip_shape, widthFace, heightFace):
    r'''normalize outerlip'''
    width, height = get_outerlip_size(np_outer_lip_shape)
    return width / widthFace, height / heightFace


def main2DMeshWarp():
    '''1. rotate training set landmarks to front face'''
    calculateInvariantDims()
    rotateShape()
    '''2. normalize training set landmarks to standard size'''
    '''3. input a new image'''
    '''4. rotate input image to front face'''
    '''5. normalize input landmark to standard size'''
    '''6. add deltaX and deltaY'''
    '''7. multiply 1/alpha'''


if __name__ == "__main__":
    # chooseTrainingset()
    main2DMeshWarp()
