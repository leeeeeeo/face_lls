from pathlib import Path
import menpo.io as mio
from menpo.visualize import print_progress
from menpofit.modelinstance import OrthoPDM
import matplotlib.pyplot as plt
from menpo.transform import AlignmentAffine
import os


def PointDistributionModel(imgFolder):
    '''LOAD IMAGES'''
    path_to_lfpw = Path(imgFolder)
    training_shapes = []
    for lg in print_progress(mio.import_landmark_files(path_to_lfpw/'*.pts', verbose=True)):
        training_shapes.append(lg['all'])
    '''TRAIN PDM MODEL'''
    shape_model = OrthoPDM(training_shapes, max_n_components=None)
    '''MODIFY PARAMETERS'''
    shape_model.n_active_components = 40
    # shape_model.n_active_components = 0.95
    return shape_model


def Projection(trumpShapeModel, headShape):
    '''ALIGN HEADSHAPE TO TRUMP'''
    transform = AlignmentAffine(headShape, trumpShapeModel.model.mean())
    normalized_shape = transform.apply(headShape)
    '''HEAD P'''
    headWeights = trumpShapeModel.model.project(normalized_shape)
    return headWeights, transform


def deltaWeights(headWeights, neutralWeights):
    return headWeights-neutralWeights


def Reconstruction(trumpShapeModel, deltaWeights, trumpShape):
    trumpWeights, transform = Projection(trumpShapeModel, trumpShape)
    trumpWeights = trumpWeights+deltaWeights
    # print trumpWeights
    reconstructed_normalized_shape = trumpShapeModel.model.instance(
        trumpWeights)
    # reconstructed_shape = transform.pseudoinverse().apply(
    # reconstructed_normalized_shape)
    trans_reconstructed_shape = AlignmentAffine(
        reconstructed_normalized_shape, trumpShape)
    reconstructed_img_shape = trans_reconstructed_shape.apply(
        reconstructed_normalized_shape)
    return reconstructed_img_shape


def writeHeadposeTxt(txtPath, headShape):
    headShape_np = headShape.h_points()[:2, ::].T
    headTxt = open(txtPath, 'w')
    for shape_np in headShape_np:
        x = str(int(shape_np[1]))
        y = str(int(shape_np[0]))
        headTxt.write('{} {}\n'.format(x, y))
    headTxt.close()


def mainHeadpose():
    trumpFolder = './data/trump/trump'
    leftFolder = './data/headpose/Angle/left15'
    rightFolder = './data/headpose/Angle/right15'
    neutralFolder = './data/headpose/Angle/neutral0'
    downFolder = './data/headpose/Angle/down30'
    '''SHAPE MODEL'''
    trumpShapeModel = PointDistributionModel(trumpFolder)
    leftShapeModel = PointDistributionModel(leftFolder)
    rightShapeModel = PointDistributionModel(rightFolder)
    neutralShapeModel = PointDistributionModel(neutralFolder)
    downShapeModel = PointDistributionModel(downFolder)
    '''MEAN SHAPE'''
    trumpMeanShape = trumpShapeModel.model.mean()
    leftMeanShape = leftShapeModel.model.mean()
    rightMeanShape = rightShapeModel.model.mean()
    neutralMeanShape = neutralShapeModel.model.mean()
    downMeanShape = downShapeModel.model.mean()
    '''WEIGHTS'''
    leftWeights, _ = Projection(trumpShapeModel, leftMeanShape)
    # print leftWeights
    rightWeights, _ = Projection(trumpShapeModel, rightMeanShape)
    neutralWeights, _ = Projection(trumpShapeModel, neutralMeanShape)
    downWeights, _ = Projection(trumpShapeModel, downMeanShape)
    '''DELTA'''
    leftDeltaWeights = deltaWeights(leftWeights, neutralWeights)
    # print leftDeltaWeights
    rightDeltaWeights = deltaWeights(rightWeights, neutralWeights)
    downDeltaWeights = deltaWeights(downWeights, neutralWeights)
    '''RECONSTRUCTION'''
    ptsPath = '/Users/lls/Documents/face/data/trump/trump/trump_13.pts'
    trumpShape = mio.import_landmark_file(ptsPath).lms
    leftTrumpShape = Reconstruction(
        trumpShapeModel, leftDeltaWeights, trumpShape)
    rightTrumpShape = Reconstruction(
        trumpShapeModel, rightDeltaWeights, trumpShape)
    downTrumpShape = Reconstruction(
        trumpShapeModel, downDeltaWeights, trumpShape)
    '''PLOT'''
    # plt.subplot(241)
    # trumpMeanShape.view()
    # plt.gca().set_title('trumpMeanShape')
    # plt.subplot(242)
    # leftMeanShape.view()
    # plt.gca().set_title('leftMeanShape')
    # plt.subplot(243)
    # rightMeanShape.view()
    # plt.gca().set_title('rightMeanShape')
    # plt.subplot(244)
    # downMeanShape.view()
    # plt.gca().set_title('downMeanShape')
    # plt.subplot(245)
    # leftTrumpShape.view()
    # plt.gca().set_title('leftTrumpShape')
    # plt.subplot(246)
    # rightTrumpShape.view()
    # plt.gca().set_title('rightTrumpShape')
    # plt.subplot(247)
    # downTrumpShape.view()
    # plt.gca().set_title('downTrumpShape')
    # plt.show()

    '''PRODUCE TRUMP IMAGES'''
    for root, dirs, files in os.walk(trumpFolder):
        for file in files:
            if os.path.join(root, file).endswith('.pts'):
                '''READ ONE TRUMP IMAGE/PTS'''
                ptsPath = os.path.join(root, file)
                trumpShape = mio.import_landmark_file(ptsPath).lms
                leftTrumpShape = Reconstruction(
                    trumpShapeModel, leftDeltaWeights, trumpShape)
                rightTrumpShape = Reconstruction(
                    trumpShapeModel, rightDeltaWeights, trumpShape)
                downTrumpShape = Reconstruction(
                    trumpShapeModel, downDeltaWeights, trumpShape)
                leftTxtPath = '{}_left.txt'.format(
                    os.path.join(root, file)[:-4])
                rightTxtPath = '{}_right.txt'.format(
                    os.path.join(root, file)[:-4])
                downTxtPath = '{}_down.txt'.format(
                    os.path.join(root, file)[:-4])
                writeHeadposeTxt(leftTxtPath, leftTrumpShape)
                writeHeadposeTxt(rightTxtPath, rightTrumpShape)
                writeHeadposeTxt(downTxtPath, downTrumpShape)


if __name__ == "__main__":
    mainHeadpose()
