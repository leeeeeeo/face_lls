from pathlib import Path
import menpo.io as mio
from menpo.visualize import print_progress
from menpo.landmark import labeller, face_ibug_68_to_face_ibug_68_trimesh
from menpofit.aam import HolisticAAM
from menpo.feature import fast_dsift
#from menpodetect import load_dlib_frontal_face_detector
from menpofit.aam import LucasKanadeAAMFitter, WibergInverseCompositional
import matplotlib.pyplot as plt
from menpo.transform import AlignmentAffine
from menpofit.modelinstance import OrthoPDM
import cv2
import numpy as np


def PDMModel(path, max_components=None):
    training_shapes = []
    for lg in print_progress(mio.import_landmark_files(path / '*.pts', verbose=True)):
        training_shapes.append(lg['all'])
    # train source PDM model
    shape_model = OrthoPDM(training_shapes, max_n_components=max_components)
    return shape_model, training_shapes


def project(target, source_model):
    # align the source and target face
    transform = AlignmentAffine(target, source_model.model.mean())
    normalized_target = transform.apply(target)
    weights = source_model.model.project(normalized_target)
    return weights


def AAMModel(path, max_shape=None, max_appearance=None):
    training_images = []
    for img in print_progress(mio.import_images(path, verbose=True)):
        labeller(img, 'PTS', face_ibug_68_to_face_ibug_68_trimesh)
        training_images.append(img)
    aam_model = HolisticAAM(training_images, group='face_ibug_68_trimesh', scales=(
        0.5, 1.0), holistic_features=fast_dsift, verbose=True, max_shape_components=max_shape, max_appearance_components=max_appearance)
    return aam_model, training_images


def main():
    path_to_neutral = Path(
        '/Users/lls/Documents/face/data/headpose/Angle/neutral0/')
    path_to_smile = Path(
        '/Users/lls/Documents/face/data/headpose/Angle/down30')
    path_to_source = Path('/Users/lls/Documents/face/data/trump/trump')

    # PDM shape
    neutral = PDMModel(path_to_neutral, 20)[0].model.mean()
    smile = PDMModel(path_to_smile, 20)[0].model.mean()
    source_shape, source_img = PDMModel(path_to_source, 20)
    p_smile = project(smile, source_shape)
    p_neutral = project(neutral, source_shape)
    delta = (p_smile - p_neutral) * 1.5
    ptsPath = '/Users/lls/Documents/face/data/trump/trump/trump_13.pts'
    trumpShape = mio.import_landmark_file(ptsPath).lms
    p_i = project(trumpShape, source_shape)
    new_p_i = p_i + delta
    reconstructed_img_i = source_shape.model.instance(new_p_i)
    trans_reconstructed_img_i = AlignmentAffine(
        reconstructed_img_i, trumpShape)
    reconstructed_img_i_pc = trans_reconstructed_img_i.apply(
        reconstructed_img_i)
    plt.subplot(241)
    reconstructed_img_i_pc.view()
    plt.gca().set_title('reconstructed_img_i_pc')
    plt.subplot(242)
    trumpShape.view()
    plt.gca().set_title('trumpShape')
    plt.show()

    '''
    neutral_aam_model = AAMModel(path_to_neutral, 20, 150)[0]
    neutral_aam = neutral_aam_model.instance()
    smile_aam = AAMModel(path_to_smile, 20, 150)[0].instance()
    source_model, source_images = AAMModel(path_to_source, 20, 150)
    p_smile = project(smile, source_shape)
    p_neutral = project(neutral, source_shape)
    delta = (p_smile - p_neutral) * 1.5
    '''

    # for i in range(len(source_img)):
    #     img_i = source_img[i]
    #     p_i = project(img_i, source_shape)
    #     new_p_i = p_i + delta
    #     reconstructed_img_i = source_shape.model.instance(new_p_i)
    #     trans_reconstructed_img_i = AlignmentAffine(reconstructed_img_i, img_i)
    #     reconstructed_img_i_pc = trans_reconstructed_img_i.apply(
    #         reconstructed_img_i)
    # plt.subplot(111)
    # reconstructed_img_i_pc.view()
    # plt.gca().set_title('reconstructed_img_i_pc')
    # plt.show()


if __name__ == "__main__":
    main()
#target_model, training_shapes_target= PDMModel(path_to_target)
