import numpy as np
import cv2,re
import logging
import os, math
import pickle, shutil

# https://www.pyimagesearch.com/2017/04/03/facial-landmarks-dlib-opencv-python/
LEFT_EYE_INDICES = [36, 37, 38, 39, 40, 41]
RIGHT_EYE_INDICES = [42, 43, 44, 45, 46, 47]
OUTER_LIP_INDICES = [48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59]
INNER_LIP_INDICES = [60, 61, 62, 63, 64, 65, 66, 67]
LEFT_FACE_WIDTH_INDICES = [0, 1, 2]
RIGHT_FACE_WIDTH_INDICES = [14, 15, 16]
UPPER_FACE_HEIGHT_INDICES = [27, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47]  # center eye point + left + right eye
BOTTOM_FACE_HEIGHT_INDICES = [31, 32, 33, 34, 35]
CENTER_FACE_INDICES = [27, 28, 29, 30, 31, 32, 33, 34, 35]
ALL_LIP_INDICES = [48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67]
CHEEK_INDICES = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]

LEFT_FACE_INDICES  = [0, 1,  2,  3,  4,  5,  6,  7,  17, 18, 19, 20, 21, 36, 37, 38, 39, 40, 41, 31, 32, 48, 49, 50, 58, 59, 60, 61, 67]
RIGHT_FACE_INDICES = [9, 10, 11, 12, 13, 14, 15, 16, 22, 23, 24, 25, 26, 42, 43, 44, 45, 46, 47, 34, 35, 52, 53, 54, 55, 56, 63, 64, 65]


g_face_landmark_keys = ['np_shape', 'img_path', 'rotate_center', 'rotate_mat', 'rotated_img_path',
                'rotated_shape', 'rotated_outerlip_shape', 'rotated_outerlip_image',
                'rotated_face_invariant_param', 'rotated_lip_center']


# invariant points
LEFT_MOST_CHEEK = 0
RIGHT_MOST_CHEEK = 16
BOTTOM_NOSE = 33
MIDDLE_EYE = 27
LEFT_MOST_LIP = 48
START_OUTER_LIP = 48
END_OUTER_LIP = 60 #not include
RIGHT_MOST_LIP = 54
UP_MOST_LIP = 51
BOTTOM_MOST_LIP = 57

######################################################################
#
#    Basic Ops
#
######################################################################
def rect_to_tuple(rect):
    left = rect.left()
    right = rect.right()
    top = rect.top()
    bottom = rect.bottom()
    return left, top, right, bottom

def rect_to_points(rect):
    left = rect.left()
    right = rect.right()
    top = rect.top()
    bottom = rect.bottom()
    return (left, top), (right, bottom)

def extract_polygon(np_shape, polygon_indices):
    points = map(lambda i: np_shape[:, i], polygon_indices)
    return list(points)

def extract_polygon_center(np_shape, polygon_indices):
    points = extract_polygon(np_shape, polygon_indices)
    xs = map(lambda p: p[0], points)
    ys = map(lambda p: p[1], points)
    return float(sum(xs)) / len(xs), float(sum(ys)) / len(ys)

def extract_eyes_center(np_shape):
    r'''return eyes center point'''
    le_x, le_y =  extract_polygon_center(np_shape, LEFT_EYE_INDICES)
    re_x, re_y = extract_polygon_center(np_shape, RIGHT_EYE_INDICES)
    return 0.5 * (le_x + re_x), 0.5 * (le_y + re_y)

def crop_image(image, det):
    left, top, right, bottom = rect_to_tuple(det)
    return image[top:bottom, left:right]


def dlib_shape_to_np_shape(dlib_shape):
    # switch x, y, for opencv coordinate
    np_shape = np.zeros((2, dlib_shape.num_parts), dtype=np.float32)
    for i in range(dlib_shape.num_parts):
        np_shape[0][i] = dlib_shape.part(i).x
        np_shape[1][i] = dlib_shape.part(i).y
    return np_shape



###################################################################
#
#  Extract feature points
#
###################################################################
def get_lip_and_cheek_point(in_vector):
    r'''reord dlib output of lip and cheek points into a whole vector '''
    # total: 37
    # 0:17: CHEEK_INDICES = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
    # 17:29: OUTER_LIP_INDICES = [48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59]
    # 29:37  INNER_LIP_INDICES = [60, 61, 62, 63, 64, 65, 66, 67]
    out_vector = np.zeros(74, dtype=np.float)
    # cheek x
    out_vector[0:17] = in_vector[0, 0:17]
    # cheek y
    out_vector[17:34] = in_vector[1, 0:17]
    # lip x
    out_vector[34:54] = in_vector[0, 48:68]
    # lip y
    out_vector[54:74] = in_vector[1, 48:68]
    return out_vector

def get_outerlip_shape(np_shape):
    r'''shape 2x68 full size'''
    np_outerlip_shape = np.zeros((END_OUTER_LIP - START_OUTER_LIP, 2), dtype=(np.int32))
    for j in range(START_OUTER_LIP, END_OUTER_LIP):
        np_outerlip_shape[j - START_OUTER_LIP, :] = np.around(np_shape[:, j]).astype(np.int32)
    return np_outerlip_shape

def get_outer_lip_upperleft_point(np_outerlip_shape):
    r'''x_upper_left, and y_upper_right'''
    # left = np_outerlip_shape[:, LEFT_MOST_LIP - START_OUTER_LIP]
    # upper = np_outerlip_shape[:, UP_MOST_LIP - START_OUTER_LIP]
    xmin = min(np_outerlip_shape[0, :])
    ymin = min(np_outerlip_shape[1, :])
    # return left[0], upper[1]
    return xmin, ymin

def get_outerlip_size(np_outerlip_shape):
    right = np_outerlip_shape[RIGHT_MOST_LIP - START_OUTER_LIP]
    left = np_outerlip_shape[LEFT_MOST_LIP - START_OUTER_LIP]
    upper = np_outerlip_shape[UP_MOST_LIP - START_OUTER_LIP]
    bottom = np_outerlip_shape[BOTTOM_MOST_LIP - START_OUTER_LIP]
    width = right[0] - left[0]
    height = bottom[1] - upper[1]
    return width, height

def get_normed_outerlip_size(np_outer_lip_shape, face_width, face_height):
    r'''normalize outerlip'''
    width, height = get_outerlip_size(np_outer_lip_shape)
    return width / face_width, height / face_height

def get_basic_vector_from_lip_and_cheek_vector(in_vector):
    r'''get lip with, height, and center
    use the out_vector from get_lip_and_cheek_point
    in_vector is the output vector
    is it being used? probably should stop'''

    # 48, 54
    id = 34 + RIGHT_MOST_LIP - LEFT_MOST_LIP
    lip_width = in_vector[id] - in_vector[34]

    # 48, 54
    id = 54 + RIGHT_MOST_LIP - LEFT_MOST_LIP
    lip_center_y = 0.5 * (in_vector[id] + in_vector[54])
    # 51 and 57
    # in y axis
    id = 57 + BOTTOM_MOST_LIP - UP_MOST_LIP
    lip_height = in_vector[id] - in_vector[57]

    out_vector = np.zeros(3, dtype=np.float)
    out_vector[0] = lip_width
    out_vector[1] = lip_height
    out_vector[2] = lip_center_y
    return out_vector

def revert_lip_and_cheek_point_to_shape(in_vector, shape):
    # in_vector was obtained from lip and cheeck point, fill into the shapes
    # shape has some invariant variables which are copied from rotated_shape
    shape[0, 0:17] = in_vector[0:17]
    shape[1, 0:17] = in_vector[17:34]
    shape[0, 48:68] = in_vector[34:54]
    shape[1, 48:68] = in_vector[54:74]
    return shape

def get_lip_location(np_shape):
    r'''suppose x is aligned, only the difference of y matters'''
    lip_center_x, lip_center_y = extract_polygon_center(np_shape, ALL_LIP_INDICES)
    return (lip_center_x, lip_center_y)



################################################################################
#
#     Showing and drawing image
#
################################################################################
def show_image(img, rotated_img, rotated_shape, det):
    '''where is it used? need to be refactored later on'''
    upper_left, bottom_right = rect_to_points(det)

    cv2.rectangle(img, upper_left, bottom_right, (0, 255, 0), 2)
    for col in rotated_shape.T:
        lm_point = (int(col[0]), int(col[1]))
        cv2.circle(img, lm_point, 1, (0, 0, 255), -1)
    # draw on
    cv2.imshow("not rotated", img)

    cv2.rectangle(rotated_img, upper_left, bottom_right, (0, 255, 0), 2)
    for col in rotated_shape.T:
        lm_point = (int(col[0]), int(col[1]))
        cv2.circle(rotated_img, lm_point, 1, (0, 0, 255), -1)
    # draw on
    cv2.imshow("rotated", rotated_img)
    cv2.waitKey(0)

# should shaped should be rotated back?
def write_predict_flm_on_image(img, rotated_shape):
    for col in rotated_shape.T:
        lm_point = (int(col[0]), int(col[1]))
        cv2.circle(img, lm_point, 1, (0, 0, 255), -1)


####################################################
#
#    This triangulation and blending
#
###################################################
def load_triangular_point(tri_conf):
    r'''each line has a set of triangular point on the face landmarks'''
    with open(tri_conf, 'r') as f:
        content = f.readlines()

        str_list = [x.strip().split() for x in content]
        tri_point_list = []
        for str_tri in str_list:
            tri = [int(x) for x in str_tri]
            tri_point_list.append(tri)
    return tri_point_list


def generate_converted_lip(base_rotated_image, base_rotated_face_shape, base_rotated_face_invariant_param,
                           base_rotate_mat,
                           base_rotated_lip_center, src_lip_param):
    r'''given a set face landmarks, replace the moving lip '''

    base_inv_rotate_mat = cv2.invertAffineTransform(base_rotate_mat)
    src_rotated_outerlip_shape = src_lip_param['rotated_outerlip_shape']
    src_rotated_face_invariant_param = src_lip_param['rotated_face_invariant_param']
    src_rotated_lip_image = src_lip_param['rotated_outerlip_image']
    src_rotated_lip_center = src_lip_param['rotated_lip_center']

    # new lip location, and new ratio
    (base_ratio, base_xc, base_yc, base_face_width, base_face_height) = base_rotated_face_invariant_param
    (src_ratio, src_xc, src_yc, src_face_width, src_face_height) = src_rotated_face_invariant_param
    base_to_src_x_ratio = float(base_face_width) / float(src_face_width)
    base_to_src_y_ratio = float(base_face_height) / float(src_face_height)
    (base_lc_x, base_lc_y) = base_rotated_lip_center
    (src_lc_x, src_lc_y) = src_rotated_lip_center

    # face and lip shape have to be resized
    # need to reset the row and col src_rotated_outerlip_shape correctly
    src_resized_rotated_outerlip_shape = np.empty_like(src_rotated_outerlip_shape.T)
    converted_rotated_face_shape = np.copy(base_rotated_face_shape)
    for i, (src_x, src_y) in enumerate(src_rotated_outerlip_shape):
        base_x = (src_x - src_lc_x) * base_to_src_x_ratio + base_lc_x
        base_y = (src_y - src_lc_y) * base_to_src_y_ratio + base_lc_y
        # print i
        j = OUTER_LIP_INDICES[i]
        src_resized_rotated_outerlip_shape[0, i] = base_x
        src_resized_rotated_outerlip_shape[1, i] = base_y
        converted_rotated_face_shape[0, j] = base_x
        converted_rotated_face_shape[1, j] = base_y

    # resized image
    resized_rotated_lip_image = cv2.resize(src_rotated_lip_image, None, fx=base_to_src_x_ratio, fy=base_to_src_y_ratio)
    cv2.imwrite('resized_rotated_lip_image.png', resized_rotated_lip_image)

    # paste the resized  image on to a big blank canvas
    base_lul_x, base_lul_y = get_outer_lip_upperleft_point(src_resized_rotated_outerlip_shape)

    # should not use this
    # base_lul_x = int(src_lul_x - src_lc_x + base_lc_x)
    # base_lul_y = int(src_lul_y - src_lc_y + base_lc_y)

    # paste from upper left
    # create a canvas, then put it to the right location
    height, width, _ = base_rotated_image.shape
    rz_height, rz_width, _ = resized_rotated_lip_image.shape
    resized_rotated_lip_on_blank_image = np.zeros((height, width, 3), np.uint8)
    resized_rotated_lip_on_blank_image[base_lul_y:base_lul_y + rz_height,
    base_lul_x:base_lul_x + rz_width] = resized_rotated_lip_image

    # rotate
    converted_face_shape = rotate_shape(base_inv_rotate_mat, converted_rotated_face_shape)
    # src_resized_outerlip_shape = rotate_shape(base_inv_rotate_mat, src_resized_rotated_outerlip_shape)
    # resized_lip_on_blank_image = cv2.warpAffine(resized_rotated_lip_on_blank_image, base_inv_rotate_mat, (height, width), flags=cv2.INTER_CUBIC)

    cv2.imwrite('resized_rotated_lip_on_blank_image.png', resized_rotated_lip_on_blank_image)
    resized_lip_on_blank_image = cv2.warpAffine(resized_rotated_lip_on_blank_image, base_inv_rotate_mat,
                                                (width, height))
    cv2.imwrite('resized_lip_on_blank_image.png', resized_lip_on_blank_image)

    return converted_face_shape, resized_lip_on_blank_image


def apply_affine_transform(src_image, src_rect, tgt_rect, size):
    warp_mat = cv2.getAffineTransform(np.float32(src_rect), np.float32(tgt_rect))

    # Apply the Affine Transform just found to the src image
    tgt_image = cv2.warpAffine(src_image, warp_mat, (size[0], size[1]), None, flags=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_REFLECT_101)

    return tgt_image


def morph_triangle(src_img, tgt_img, draw_src_img, src_tri, tgt_tri):
    r'''from a src (overwrite) to a tgt image
    tgt_image is not empty, initialized as
    src_tri is a list of 3 points, each point has two items
    tgt_img: i/o
    br: boundr ect, x ,y, width(2), height(3)'''
    sbr = cv2.boundingRect(np.float32([src_tri]))
    tbr = cv2.boundingRect(np.float32([tgt_tri]))

    # Offset points by left top corner of the respective rectangles
    src_rect = []
    tgt_rect = []
    for i in xrange(0, 3):
        # src_rect.append((int(round(src_tri[i][0] - sbr[0])), int(round(src_tri[i][1] - sbr[1]))))
        src_rect.append((src_tri[i][0] - sbr[0], src_tri[i][1] - sbr[1]))
        # tgt_rect.append((int(round(tgt_tri[i][0] - tbr[0])), int(round(tgt_tri[i][1] - tbr[1]))))
        tgt_rect.append((tgt_tri[i][0] - tbr[0], tgt_tri[i][1] - tbr[1]))
    for i in xrange(0, 3):
        pt1 = (int(round(src_tri[i][0])), int(round(src_tri[i][1])))
        if i == 2:
            pt2 = (int(round(src_tri[0][0])), int(round(src_tri[0][1])))
        else:
            pt2 = (int(round(src_tri[i + 1][0])), int(round(src_tri[i + 1][1])))
        cv2.line(draw_src_img, pt1, pt2, (0, 255, 0), thickness=1)

    # Get mask by filling triangle, first dim is height, then 2nd dim is width
    tgt_mask = np.zeros((tbr[3], tbr[2], 3), dtype=np.float32)
    cv2.fillConvexPoly(tgt_mask, np.int32(tgt_rect), (1.0, 1.0, 1.0), 16, 0)

    # Apply warpImage to small rectangular patches
    src_img_rect = src_img[sbr[1]:sbr[1] + sbr[3], sbr[0]:sbr[0] + sbr[2]]

    size = (tbr[2], tbr[3])  # width, height
    tgt_img_rect = apply_affine_transform(src_img_rect, src_rect, tgt_rect, size)

    # Copy triangular region of the rectangular patch to the output image
    # tgt_img[tbr[1]:tbr[1] + tbr[3], tbr[0]:tbr[0] + tbr[2]] = tgt_img[tbr[1]:tbr[1] + tbr[3], tbr[0]:tbr[0] + tbr[2]] * (1 - mask) + tgt_img_rect * mask
    # tgt_img[tbr[1]:tbr[1] + tbr[3], tbr[0]:tbr[0] + tbr[2]] = tgt_img[tbr[1]:tbr[1] + tbr[3], tbr[0]:tbr[0] + tbr[2]] * (1 - mask) + tgt_img_rect * mask
    bg_img = tgt_img[tbr[1]:tbr[1] + tbr[3], tbr[0]:tbr[0] + tbr[2]]
    tgt_img[tbr[1]:tbr[1] + tbr[3], tbr[0]:tbr[0] + tbr[2]] = bg_img * (1 - tgt_mask) + tgt_img_rect * tgt_mask
    # drawgrid(img1_bg)
    # cv2.imwrite('img1_bg.png', img1_bg)
    # drawgrid(img2_fg)
    # cv2.imwrite('img2_fg.png', img2_fg)
    # Put logo in ROI and modify the main image

def morph_image(base_face_param, src_lip_param, tri_point_list):
    r''' scale the lip shape and image, rotate the shape and image,
        for rest of the face, replace the shape param in the lip, then do morph'''

    # np_shape img rotate_center rotate_mat rotated_img rotated_shape rotated_outerlip_shape rotated_outerlip_image
    # rotated_face_invariant_param rotated_lip_center

    base_np_shape = base_face_param['np_shape']
    base_rotated_shape = base_face_param['rotated_shape']
    base_rotated_face_invariant_param = base_face_param['rotated_face_invariant_param']
    base_rotated_lip_center = base_face_param['rotated_lip_center']
    base_img = cv2.imread(base_face_param['img_path'])
    base_rotated_image = cv2.imread(base_face_param['rotated_img_path'])
    base_rotate_mat = base_face_param['rotate_mat']


    # 'rotated_outerlip_shape', 'rotated_outerlip_image', 'rotated_face_invariant_param', 'rotated_lip_center'
    # 1. scale lip,  2 replace lip points 3. then rotate whole shape
    # c_lip_on_base_img only contain lip, other region black (0,0,0)
    c_shp_on_base_img, c_lip_on_base_img = generate_converted_lip(base_rotated_image, base_rotated_shape, base_rotated_face_invariant_param,

                                                                  base_rotate_mat, base_rotated_lip_center, src_lip_param)
    # step1: first for mouth region other than lip
    chg_cheeck_img = np.copy(base_img)
    triline_base_img = np.copy(base_img)
    for i, (id_1, id_2, id_3) in enumerate(tri_point_list):
        # print i, id_1, id_2, id_3
        pt_1 = (base_np_shape[0, id_1], base_np_shape[1, id_1])
        pt_2 = (base_np_shape[0, id_2], base_np_shape[1, id_2])
        pt_3 = (base_np_shape[0, id_3], base_np_shape[1, id_3])
        src_tri = [pt_1, pt_2, pt_3]
        pt_1 = (c_shp_on_base_img[0, id_1], c_shp_on_base_img[1, id_1])
        pt_2 = (c_shp_on_base_img[0, id_2], c_shp_on_base_img[1, id_2])
        pt_3 = (c_shp_on_base_img[0, id_3], c_shp_on_base_img[1, id_3])
        tgt_tri = [pt_1, pt_2, pt_3]
        morph_triangle(base_img, chg_cheeck_img, triline_base_img, src_tri, tgt_tri)

    '''
    # drawgrid(chg_cheeck_img)
    cv2.imwrite('chg_cheeck_img.png', chg_cheeck_img)
    # drawgrid(triline_base_img)
    cv2.imwrite('triline_base_img.png', triline_base_img)
    '''

    # step2: then past the new lip on it
    converted_img = add_foreground_mouth(c_lip_on_base_img, c_shp_on_base_img, chg_cheeck_img)

    # printing debugging
    '''
    src_img = cv2.imread(src_lip_param['img_path'])
    # drawgrid(src_img)
    cv2.imwrite('src.png', src_img)
    # drawgrid(c_lip_on_base_img)
    cv2.imwrite('lip_on_base.png', c_lip_on_base_img)
    # drawgrid(base_img)
    cv2.imwrite('base.png', base_img)
    # drawgrid(converted_img)
    cv2.imwrite('conv.png', converted_img)
    cv2.imwrite('src_rotated_lip.png', src_lip_param['rotated_outerlip_image'])
    '''

    return converted_img
