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
    shape_model.n_active_components = 20
    shape_model.n_active_components = 0.95
    return shape_model


def ProjectionAndReconstruction(trumpShapeModel, headMeanShape):
    '''ALIGN HEADSHAPE TO TRUMP'''
    transform = AlignmentAffine(headMeanShape, trumpShapeModel.model.mean())
    normalized_shape = transform.apply(headMeanShape)
    '''HEAD P'''
    headWeights = trumpShapeModel.model.project(normalized_shape)
    '''RECONSTRUCTION'''
    reconstructed_normalized_shape = trumpShapeModel.model.instance(
        headWeights)
    reconstructed_shape = transform.pseudoinverse().apply(
        reconstructed_normalized_shape)
    return reconstructed_shape, headWeights


def Projection(trumpShapeModel, headMeanShape):
    '''ALIGN HEADSHAPE TO TRUMP'''
    transform = AlignmentAffine(headMeanShape, trumpShapeModel.model.mean())
    normalized_shape = transform.apply(headMeanShape)
    '''HEAD P'''
    headWeights = trumpShapeModel.model.project(normalized_shape)
    return headWeights


def deltaWeights(headWeights, neutralWeights):
    return headWeights-neutralWeights


def Reconstruction(trumpShapeModel, deltaWeights, trumpShape):
    transform = AlignmentAffine(trumpShape, trumpShapeModel.model.mean())
    trumpWeights = trumpShapeModel.model.project(trumpShape)
    trumpWeights = trumpWeights+deltaWeights
    reconstructed_normalized_shape = trumpShapeModel.model.instance(
        trumpWeights)
    reconstructed_shape = transform.pseudoinverse().apply(
        reconstructed_normalized_shape)
    return reconstructed_shape


def main():
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
    leftWeights = Projection(trumpShapeModel, leftMeanShape)
    rightWeights = Projection(trumpShapeModel, rightMeanShape)
    neutralWeights = Projection(trumpShapeModel, neutralMeanShape)
    downWeights = Projection(trumpShapeModel, downMeanShape)
    '''DELTA'''
    leftDeltaWeights = deltaWeights(leftWeights, neutralWeights)
    rightDeltaWeights = deltaWeights(rightWeights, neutralWeights)
    downDeltaWeights = deltaWeights(downWeights, neutralWeights)
    '''RECONSTRUCTION'''

    '''PRODUCE TRUMP IMAGES'''
    for root, dirs, files in os.walk(trumpFolder):
        for file in files:
            if os.path.join(root, file).endswith('.pts'):
                '''READ ONE TRUMP IMAGE/PTS'''
                ptsPath = os.path.join(root, file)
                trumpShape = mio.import_landmark_file(ptsPath)
                headTrumpShape = Reconstruction(
                    trumpShapeModel, leftDeltaWeights, trumpShape)


if __name__ == "__main__":
    main()
