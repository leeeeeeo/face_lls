import numpy as np
import cv2, re
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

LEFT_FACE_INDICES = [0, 1, 2, 3, 4, 5, 6, 7, 17, 18, 19, 20, 21, 36, 37, 38, 39, 40, 41, 31, 32, 48, 49, 50, 58, 59, 60,
                     61, 67]
RIGHT_FACE_INDICES = [9, 10, 11, 12, 13, 14, 15, 16, 22, 23, 24, 25, 26, 42, 43, 44, 45, 46, 47, 34, 35, 52, 53, 54, 55,
                      56, 63, 64, 65]

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
END_OUTER_LIP = 60  # not include
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
    le_x, le_y = extract_polygon_center(np_shape, LEFT_EYE_INDICES)
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


############################################################
#
#    Rotation
#
############################################################
def angle_between_2_points(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    tan = (float(y2) - float(y1)) / (float(x2) - float(x1))
    return np.degrees(np.arctan(tan))


def get_shape_rotation_matrix(np_shape):
    r''' get rotate matrix to have angle after rotation to be 0
    return rotate_mat and rotation center'''

    p1 = extract_polygon_center(np_shape, LEFT_FACE_INDICES)
    p2 = extract_polygon_center(np_shape, RIGHT_FACE_INDICES)
    angle = angle_between_2_points(p1, p2)
    x1, y1 = p1
    x2, y2 = p2
    xc = (x1 + x2) * 0.5
    yc = (y1 + y2) * 0.5
    rotate_mat = cv2.getRotationMatrix2D((xc, yc), angle, 1)
    # inv_rotate_mat = cv2.invertAfflineTransform(rotate_mat)
    return rotate_mat, (xc, yc)


# the general flow is to first rotate around the center of eyes
# then normalize according face dimensions
def calculate_invariant_dims(rotated_shape):
    left_pt = extract_polygon_center(rotated_shape, LEFT_FACE_WIDTH_INDICES)
    right_pt = extract_polygon_center(rotated_shape, RIGHT_FACE_WIDTH_INDICES)
    upper_pt = extract_polygon_center(rotated_shape, UPPER_FACE_HEIGHT_INDICES)
    bottom_pt = extract_polygon_center(rotated_shape, BOTTOM_FACE_HEIGHT_INDICES)
    face_width = right_pt[0] - left_pt[0]
    face_height = (bottom_pt[1] - upper_pt[1]) * 2.0  # 2 x nose height
    ratio = float(face_width) / float(face_height)

    xc, yc = extract_polygon_center(rotated_shape, CENTER_FACE_INDICES)
    face_invariant_param = (ratio, xc, yc, face_width, face_height)
    return face_invariant_param


def rotate_shape(rotate_mat, np_shape):
    r'''step 1: rotate the shape from dlib into an np array shape
    rotate_mat can be feed in with a (or an inverse) rotate matrix
    np_shape can be norm/unnormed all/only-lip shapes'''
    n_row, n_col = np_shape.shape
    np_shape_mat_for_rotate = np.vstack([np_shape, np.full(n_col, 1.0)])
    rotated_shape = np.dot(rotate_mat, np_shape_mat_for_rotate)
    return rotated_shape


def normalize_shape(np_shape, invariant_param):
    r'''np arrayed shape normalization'''
    ratio, xc, yc, face_width, face_height = invariant_param
    if ratio < 1:
        return None
    norm_shape = np.empty_like(np_shape)
    for i in range(0, norm_shape.shape[1]):
        norm_shape[0][i] = (np_shape[0][i] - xc) / face_width
        norm_shape[1][i] = (np_shape[1][i] - yc) / face_height
    return norm_shape


def unnormalize_shape(np_norm_shape, invariant_param):
    r'''reverse of normalize_rotated_shape'''
    ratio, xc, yc, face_width, twice_nose_height = invariant_param
    if ratio < 1:
        return None
    np_shape = np.empty_like(np_norm_shape)
    for i in range(0, np_shape.shape[1]):
        np_shape[0][i] = np_norm_shape[0][i] * face_width + xc
        np_shape[1][i] = np_norm_shape[1][i] * twice_nose_height + yc
    return np_shape


def is_gray(img):
    # img = cv2.imread(img_path)
    # a bit loose
    w, h, _ = img.shape
    sum = [0.0, 0.0, 0.0]
    for i in range(w):
        for j in range(h):
            r, g, b = img[i, j]
            sum[0] += r
            sum[1] += g
            sum[2] += b

    if abs(sum[0]) < 0.1:
        return True
    r1 = abs((sum[1] - sum[0]) / sum[0])
    r2 = abs((sum[2] - sum[0]) / sum[0])
    if r1 < 0.05 and r2 < 0.05:
        return True
    else:
        return False


'''
def plugin_in_flm(key_shape, np_shape):
    # this is to put the predicted FLMs back into shape -- is useful?
    # this 74 lip + cheek flms
    src_np_shape = np.copy(np_shape)
    for i in range(0, src_np_shape.shape[1]):
        src_np_shape[0][i] = key_shape[0][i]
        src_np_shape[1][i] = key_shape[1][i]

    return src_np_shape
'''

'''
def unnormalize_vector(normed_vector, invariant_param):
    # reverse of noamized of normed_vector -- useful?
    normed_lip_width, normed_lip_height, normed_lip_center_y = normed_key_box_vector
    lip_width = normed_lip_width * face_width
    lip_height = normed_lip_height * twice_nose_height
    lip_center_y = normed_lip_center_y * twice_nose_height

    top_left_x = int(xc - 0.5 * lip_width)
    top_left_y = int(yc + lip_center_y - 0.4 * lip_height)
    bottom_right_x = int(xc + 0.5 * lip_width)
    bottom_right_y = int(yc + lip_center_y + 0.6 * lip_height)
    return (top_left_x, top_left_y), (bottom_right_x, bottom_right_y)
'''


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

    # draw on
    # cv2.imwrite(predict_img_path, img)


# mouth box
def write_predict_box_on_image(img, top_left, bottom_right, box_vector, style='solid', color=(0, 255, 0),
                               location='BR'):
    # cv2.rectangle(img, top_left, bottom_right, )
    # lip_width, lip_height, lip_loc_y = box_vector
    drawrect(img, top_left, bottom_right, color, 2, style)
    text = "w=%.2f, h=%.2f" % (box_vector[0], box_vector[1])
    if location is 'BR':
        cv2.putText(img, text, bottom_right, cv2.FONT_HERSHEY_SIMPLEX, 2, color, 2, cv2.LINE_AA)
    else:
        top_right = (bottom_right[0], top_left[1])
        cv2.putText(img, text, top_right, cv2.FONT_HERSHEY_SIMPLEX, 2, color, 2, cv2.LINE_AA)


def write_text_on_image(img, style='solid', color=(255, 255, 255)):
    height, width, channels = img.shape
    text = "red dots: facial landmarks"
    cv2.putText(img, text, (40, height - 40), cv2.FONT_HERSHEY_SIMPLEX, 2, color, 2, cv2.LINE_AA)
    text = "w: normlized mouth width, h: normlized mouth height"
    cv2.putText(img, text, (40, height - 100), cv2.FONT_HERSHEY_SIMPLEX, 2, color, 2, cv2.LINE_AA)
    text = "white box: ground truth"
    cv2.putText(img, text, (40, height - 220), cv2.FONT_HERSHEY_SIMPLEX, 2, color, 2, cv2.LINE_AA)
    text = "green box: predicted from speech"
    cv2.putText(img, text, (40, height - 160), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2, cv2.LINE_AA)


def ploygon_to_rect(int32_crop_shape, max_shape):
    r'''crop_shape x, y; max_shape: y, x '''
    xmin = max(np.amin(int32_crop_shape[:, 0]), 0)
    ymin = max(np.amin(int32_crop_shape[:, 1]), 0)
    xmax = min(np.amax(int32_crop_shape[:, 0]), max_shape[1])
    ymax = min(np.amax(int32_crop_shape[:, 1]), max_shape[0])

    return (xmin, xmax, ymin, ymax)


def crop_polygon_in_image(image, crop_shape):
    r'''this will crop a masked image from the whole image
    mask defaulting to black for 3-channel and transparent for 4-channel'''
    mask = np.zeros(image.shape, dtype=np.uint8)
    xmin, xmax, ymin, ymax = ploygon_to_rect(crop_shape, image.shape)
    channel_count = image.shape[2]  # i.e. 3 or 4 depending on your image
    ignore_mask_color = (255,) * channel_count
    # from Masterfool: use cv2.fillConvexPoly if you know it's convex
    cv2.fillPoly(mask, [crop_shape], ignore_mask_color)
    # apply the mask
    masked_image = cv2.bitwise_and(image, mask)
    cropped_image_with_mask = masked_image[ymin:ymax, xmin:xmax, :]

    cv2.imwrite("rotated.png", image)
    cv2.imwrite("mask.png", mask)
    cv2.imwrite("masked_image.png", masked_image)
    cv2.imwrite("test.png", cropped_image_with_mask)
    cv2.waitKey(0)

    return cropped_image_with_mask


def show_predict_flm_on_image(self, img, shape, rotated_shape):
    count = 1
    for k, col in enumerate(rotated_shape.T):
        lm_point = (int(col[0]), int(col[1]))
        rot_lm_point = (int(rotated_shape[0][k]), int(rotated_shape[1][k]))
        cv2.circle(img, lm_point, 1, (0, 0, 255), -1)
        cv2.circle(img, rot_lm_point, 1, (0, 0, 255), 2)
        # cv2.putText(img, str(count), lm_point, cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        count = count + 1

    cv2.imshow("img with flm", img)
    cv2.waitKey(0)


def interpolate_face_landmarks(floor_shape, ceil_shape, alpha):
    r'''shape interpolation'''
    if floor_shape.shape != ceil_shape.shape:
        logging.error("shape_interpolating: shape of x and y not equal.")
    shape = np.empty_like(floor_shape)
    row_num, col_num = shape.shape
    if row_num != 2:
        logging.error("shape_interpolating: row should be 2.")
    for j in range(0, col_num):
        for i in range(0, row_num):
            shape[i][j] = (1.0 - alpha) * floor_shape[i][j] + alpha * ceil_shape[i][j]
    return shape


def non_lazy_np_load_flm(fh):
    param = {}
    for fhkey in g_face_landmark_keys:
        param[fhkey] = fh[fhkey]
    return param


def load_facelandmarks(face_landmarks_path, face_param_dict, frame_id):
    r'''load one image frame's face landmarks
    return valid flag and normalized rotated shape, need to be refactored
    npz_file['rotated_outerlip_shape'], npz_file['rotated_outerlip_image'], npz_file['rotated_face_invariant_param'], npz_file['rotated_lip_center']'''
    if not os.path.isfile(face_landmarks_path):
        logging.error("not valid path %s" % face_landmarks_path)

    # face_landmarks_path.seek(0)
    with np.load(face_landmarks_path, 'r') as face_param:
        if face_param['ret'] == 1:
            face_param_dict[frame_id] = non_lazy_np_load_flm(face_param)
            face_param_dict[frame_id]['img_path'] = re.sub(r'\.flm(.*)\.(.*)$', '.png', face_landmarks_path)
            face_param_dict[frame_id]['rotated_img_path'] = re.sub(r'\.flm(.*)\.(.*)$', '.rotated.png',
                                                                   face_landmarks_path)


'''
def load_predicted_flm_from_file(feature_path):
    # load f in_feature out_feature predicted_featuer , currently not used
    if not os.path.isfile(feature_path):
        logging.error("not valid path %s" % feature_path)
    with open(feature_path, "rb") as f:
        raw_feat = pickle.load(f, fix_imports=True, encoding="bytes")
    return raw_feat
'''


def extract_facelandmarks(face_detector, face_predictor, img_path, face_landmarks_path):
    r'''norm_rotate_shape 2x68 np array
    rotate_shape 2x68 np array
    saves: normed_rotated_outerlip (points & shape)
    no need to save rotation matrix'''
    img = cv2.imread(img_path)
    rotated_img_path = re.sub(r'.png', '.rotated.png', img_path)
    dets_path = re.sub(r'.png', '.dlib.dets', img_path)
    dets = None
    dlib_shape = None
    if not os.path.isfile(dets_path):
        dets = face_detector(img, 1)
    else:
        with open(dets_path, 'rb') as f:
            # dlib_shape = pickle.load(f, fix_imports=True, encoding="bytes")
            dlib_shape = pickle.load(f)

    # only detect head and that is just one head
    if len(dets) == 1 or dlib_shape != None:
        det = dets[0]
        dlib_shape = face_predictor(img, det)
        np_shape = dlib_shape_to_np_shape(dlib_shape)
        # rotate
        rotate_mat, rotate_center = get_shape_rotation_matrix(np_shape)
        rotated_shape = rotate_shape(rotate_mat, np_shape)
        # (ratio, xc, yc, face_width, twice_nose_height)

        # now crop image on the rotated image
        rows, cols, _ = img.shape
        # img, M, (width, height), flags = cv2.INTER_CUBIC)
        rotated_img = cv2.warpAffine(img, rotate_mat, (cols, rows), flags=cv2.INTER_CUBIC)
        rotated_outerlip_shape = get_outerlip_shape(rotated_shape)
        # note this image is dependent of the invariant dims
        rotated_outerlip_image = crop_polygon_in_image(rotated_img, rotated_outerlip_shape)
        # cv2.imshow('lip', rotated_outerlip_image)
        # cv2.waitKey(0)
        rotated_face_invariant_param = calculate_invariant_dims(rotated_shape)

        # normalize, is it necessary
        # normed_rotated_shape = normalize_shape(rotated_shape, rotated_face_invariant_param)
        rotated_lip_center = get_lip_location(rotated_shape)

        # no need to save rotation matrix
        np.savez(face_landmarks_path, ret=[1],
                 np_shape=np_shape,
                 img_path=img_path,
                 rotated_img_path=rotated_img_path,
                 rotate_center=rotate_center,
                 rotate_mat=rotate_mat,
                 rotated_shape=rotated_shape,
                 rotated_outerlip_shape=rotated_outerlip_shape,
                 rotated_outerlip_image=rotated_outerlip_image,
                 rotated_face_invariant_param=rotated_face_invariant_param,
                 rotated_lip_center=rotated_lip_center, fmt='%1.6e')

        cv2.imwrite(rotated_img_path, rotated_img)
    elif len(dets) == 0:
        # no head
        np.savez(face_landmarks_path, ret=[-1], fmt='%1.6e')
    else:
        np.savez(face_landmarks_path, ret=[len(dets)], fmt='%1.6e')

    if not os.path.isfile(dets_path):
        # save
        with open(dets_path, 'wb') as f:
            pickle.dump(dlib_shape, f, pickle.HIGHEST_PROTOCOL)


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


def add_foreground_mouth(fg_img, fg_face_shape, bg_img):
    r'''mask is for foreground image'''
    # Now create a mask of logo and create its inverse mask also
    fg_poly = []
    for i in OUTER_LIP_INDICES:
        fg_poly.append((fg_face_shape[0, i], fg_face_shape[1, i]))

    mask = np.zeros((fg_img.shape[0], fg_img.shape[1], 3), dtype=np.float32)
    int_mask = np.zeros((fg_img.shape[0], fg_img.shape[1], 3), dtype=np.uint8)
    # float_band_mask = np.zeros((fg_img.shape[0], fg_img.shape[1], 3), dtype=np.float32)
    # cv2.fillConvexPoly(mask, np.int32(fg_poly), (1.0, 1.0, 1.0), 16, 0)
    cv2.fillPoly(mask, [np.int32(fg_poly)], (1.0, 1.0, 1.0))
    cv2.fillPoly(int_mask, [np.int32(fg_poly)], (255, 255, 255))

    raw_img = fg_img * mask + bg_img * (1 - mask)
    cv2.imwrite('raw_img.png', raw_img)

    sz = 6
    kernel = np.ones((sz, sz), np.uint8)
    # diliate mask
    erode_mask = cv2.erode(int_mask, kernel)
    cv2.imwrite('erode_mask.png', erode_mask)
    dilate_mask = cv2.dilate(int_mask, kernel)
    cv2.imwrite('dilate_mask.png', dilate_mask)
    band_mask = cv2.bitwise_xor(erode_mask, dilate_mask)
    cv2.imwrite('int_band_mask.png', band_mask)

    blur_img = cv2.blur(raw_img, (sz, sz))
    cv2.imwrite('blur_img.png', blur_img)
    float_band_mask = np.asfarray(band_mask) / 255.0
    img = blur_img * float_band_mask + raw_img * (1 - float_band_mask)

    return img


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
    c_shp_on_base_img, c_lip_on_base_img = generate_converted_lip(base_rotated_image, base_rotated_shape,
                                                                  base_rotated_face_invariant_param,

                                                                  base_rotate_mat, base_rotated_lip_center,
                                                                  src_lip_param)
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


####################################################
#
#   Distance between lips
#
###################################################
def dist_between_lip(np_lip1_size, np_lip2_size):
    r'''version 1, right now just the size
    later should add the distance between image, '''
    (w1, h1) = np_lip1_size
    (w2, h2) = np_lip2_size
    dw = w1 - w2
    dh = h1 - h2
    dist = 0.3 * (pow(dw / w1, 2), pow(dw / w2, 2)) + 0.7 * (pow(dh / h1, 2), pow(dh / h2, 2))


#####################################################
#
#    This is to make face changes
#
#####################################################
# rect is an image saving the mouth region
#
'''
def mouth_region_wrap(image, scale_x, scale_y):
    (x1, y1, x2, y2) = rect
    img = cv2.resize(image, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_AREA)
    return src_rect
'''


def drawline(img, pt1, pt2, color, thickness=1, style='dotted', gap=20):
    dist = ((pt1[0] - pt2[0]) ** 2 + (pt1[1] - pt2[1]) ** 2) ** .5
    pts = []
    for i in np.arange(0, dist, gap):
        r = i / dist
        x = int((pt1[0] * (1 - r) + pt2[0] * r) + .5)
        y = int((pt1[1] * (1 - r) + pt2[1] * r) + .5)
        p = (x, y)
        pts.append(p)

    if style == 'dotted':
        for p in pts:
            cv2.circle(img, p, thickness, color, -1)
    else:
        s = pts[0]
        e = pts[0]
        i = 0
        for p in pts:
            s = e
            e = p
            if i % 2 == 1:
                cv2.line(img, s, e, color, thickness)
            i += 1


def drawpoly(img, pts, color, thickness=1, style='dotted', ):
    s = pts[0]
    e = pts[0]
    pts.append(pts.pop(0))
    for p in pts:
        s = e
        e = p
        drawline(img, s, e, color, thickness, style)


def drawrect(img, pt1, pt2, color, thickness=1, style='dotted'):
    pts = [pt1, (pt2[0], pt1[1]), pt2, (pt1[0], pt2[1])]
    drawpoly(img, pts, color, thickness, style)


def drawgrid(img, xstep=100, ystep=100, color=(0, 255, 0)):
    # horizontal the vertical
    h, w, _ = img.shape
    x = 0
    y = 0
    while x < w:
        x = x + xstep
        drawline(img, (x, 0), (x, w), color, 1, style='solid')
        cv2.putText(img, str(x), (x, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 1, cv2.LINE_AA)
    while y < h:
        y = y + ystep
        drawline(img, (0, y), (h, y), color, 1, style='solid')
        cv2.putText(img, str(y), (100, y), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 1, cv2.LINE_AA)


def blend_transparent(face_img, overlay_t_img):
    # Split out the transparency mask from the colour info
    # ==============================================================================

    # We load the images
    # face_img = cv2.imread("field.jpg")
    # overlay_t_img = cv2.imread("dice.png", -1)  # Load with transparency

    # result_2 = blend_transparent(face_img, overlay_t_img)
    # cv2.imwrite("merged_transparent.png", result_2)
    overlay_img = overlay_t_img[:, :, :3]  # Grab the BRG planes
    overlay_mask = overlay_t_img[:, :, 3:]  # And the alpha plane

    # Again calculate the inverse mask
    background_mask = 255 - overlay_mask

    # Turn the masks into three channel, so we can use them as weights
    overlay_mask = cv2.cvtColor(overlay_mask, cv2.COLOR_GRAY2BGR)
    background_mask = cv2.cvtColor(background_mask, cv2.COLOR_GRAY2BGR)

    # Create a masked out face image, and masked out overlay
    # We convert the images to floating point in range 0.0 - 1.0
    face_part = (face_img * (1 / 255.0)) * (background_mask * (1 / 255.0))
    overlay_part = (overlay_img * (1 / 255.0)) * (overlay_mask * (1 / 255.0))

    # And finally just add them together, and rescale it back to an 8bit integer image
    return np.uint8(cv2.addWeighted(face_part, 255.0, overlay_part, 255.0, 0.0))


##################################################################
##
##
## feature, not *.mouth_fea, but load *.fea
SRC_LABEL_WIN_NUM = 5
TGT_LABEL_WIN_NUM = 1  # extract info from the source labels
BASIC_FEATURE_DIM = 3
FLM_FEATURE_DIM = 74


# lip_width, and size of 3
# only pick the middle frame then put in the target label
def extract_basic_labels(np_labels):
    row_num, col_num = np_labels.shape
    np_basic_labels = np.empty([row_num, TGT_LABEL_WIN_NUM * BASIC_FEATURE_DIM], np.float32)
    for i in range(0, row_num):
        for j in range(0, TGT_LABEL_WIN_NUM):
            src_s = 2 * FLM_FEATURE_DIM
            src_e = 3 * FLM_FEATURE_DIM
            tgt_s = j * BASIC_FEATURE_DIM
            tgt_e = (j + 1) * BASIC_FEATURE_DIM
            np_basic_labels[i][tgt_s:tgt_e] = get_basic_vector_from_lip_and_cheek_vector(np_labels[i][src_s:src_e])
            # for k,v in enumerate(vector):
            #    np_basic_labels[i][tgt_s+k] = v
    return np_basic_labels


def extract_norm_basic_labels(np_basic_labels):
    row_num, col_num = np_basic_labels.shape
    np_norm_basic_labels = np.empty([row_num, BASIC_FEATURE_DIM], np.float32)
    for i in range(0, BASIC_FEATURE_DIM):
        mu = np.mean(np_basic_labels[:, i])
        std = np.std(np_basic_labels[:, i])
        np_norm_basic_labels[:, i] = (np_basic_labels[:, i] - mu) / std
    return np_norm_basic_labels


def onehot_phone_feature_to_phone(in_vector, phone_size=40):
    win_size = int(len(in_vector) / phone_size)
    out_vector = []
    for i in range(0, win_size):
        for j in range(i * phone_size, (i + 1) * phone_size):
            if in_vector[j] == 1:
                out_vector.append(j)
                break
    return out_vector


# feature loading
# extract output label from saved date
def load_feature_extract_label_from_a_file(fea_path, basic_flag=None, phone_size=40):
    with open(fea_path.strip(), "rb") as ff:
        onehot_features = []
        labels = []
        features = []
        raw_feat = pickle.load(ff, fix_imports=True, encoding="bytes")
        for i, in_vector, out_vector in raw_feat:
            onehot_features.append(in_vector)
            labels.append(out_vector)
            if not basic_flag is None:
                vec = onehot_phone_feature_to_phone(in_vector)
                features.append(vec)

        np_onehot_features = np.array(onehot_features).astype(np.float32)
        np_features = np.array(features).astype(np.float32)
        np_labels = np.array(labels).astype(np.float32)
        np_basic_labels = extract_basic_labels(np_labels)

        return np_onehot_features, np_features, np_basic_labels, raw_feat


def load_feature_extract_label(list, basic_flag=None, phone_size=40):
    np_onehot_features = np.array([])
    np_basic_labels = np.array([])
    np_features = np.array([])  ## phones --> onehot --> features
    with open(list) as f:
        for fea_path in f.readlines():
            print(fea_path)
            part_np_onehot_features, part_np_features, part_np_basic_labels, raw_feat = load_feature_extract_label_from_a_file(
                fea_path)
            np_onehot_features = np.vstack(
                [np_onehot_features, part_np_onehot_features]) if np_onehot_features.size else part_np_onehot_features
            np_features = np.vstack([np_features, part_np_features]) if np_features.size else part_np_features
            np_basic_labels = np.vstack(
                [np_basic_labels, part_np_basic_labels]) if np_basic_labels.size else part_np_basic_labels

    if basic_flag is None:
        return np_onehot_features, np_basic_labels
    else:
        return np_features, np_basic_labels


# no basic label extraction
def load_feature(list, basic_flag=None, output_type=np.float32, mix_num=16):
    if basic_flag is None:
        # onehot features
        return load_feature_extract_label(list)
    elif basic_flag == 'phone':
        # not onehot featueres
        return load_feature_extract_label(list, basic_flag)
    else:
        features = []
        labels = []
        with open(list) as f:
            for fea_path in f.readlines():
                print(fea_path)
                with open(fea_path.strip(), "rb") as ff:
                    raw_feat = pickle.load(ff, fix_imports=True, encoding="bytes")
                    for i, in_vector, out_vector in raw_feat:
                        features.append(in_vector)
                        # onehot_out_vector = np.zeros(mix_num, dtype=np.int32)
                        # onehot_out_vector[out_vector] = 1
                        labels.append(out_vector)
                # temp
        np_features = np.array(features).astype(np.float32)
        np_labels = np.array(labels).astype(output_type)

        return np_features, np_labels

# convert gmm weights to value
# given gmm, and weight, calculate the
# def convert_gmm_weights_to_value(gmm, weights)
