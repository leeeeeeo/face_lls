import time
import os
import matplotlib.pyplot as plt
import numpy as np
import visvis as vv
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from skimage import measure
from skimage.draw import ellipsoid


def saveEllipsoidObj(objPath, verts, faces):
    os.makedirs(os.path.dirname(objPath)) if not os.path.exists(
        os.path.dirname(objPath)) else False
    objFile = open(objPath, 'w')
    objLines = []
    for vert in verts:
        objLines.append('{} {} {} {} {} {} {}'.format(
            'v', vert[0], vert[1], vert[2], 0.5, 0.3, 0.1))
    for face in faces:
        objLines.append('{} {} {} {}'.format('f', face[0], face[1], face[2]))
    for objLine in objLines:
        objFile.write(objLine+'\n')
    objFile.close()


def ellipsoidHead(semiX, semiY, semiZ, centerX, centerY, centerZ):
    ellip = ellipsoid(semiX, semiY, semiZ, spacing=(
        1, 1, 1), levelset=True)
    verts, faces, normals, values = measure.marching_cubes_lewiner(ellip, 0)
    verts[:, 0] = verts[:, 0]+centerX-np.mean(verts, axis=0)[0]
    verts[:, 1] = verts[:, 1]+centerY-np.mean(verts, axis=0)[1]
    verts[:, 2] = verts[:, 2]+centerZ-np.mean(verts, axis=0)[2]
    ellipsoidHeadVLines = []
    ellipsoidHeadXY = []
    ellipsoidHeadXZ = []
    ellipsoidHeadYZ = []
    ellipsoidHeadXYZ = []
    for vert in verts:
        ellipsoidHeadVLines.append('{} {} {} {} {} {} {}'.format(
            'v', vert[0], vert[1], vert[2], 0, 0, 0))
        ellipsoidHeadXY.append((vert[0], vert[1]))
        ellipsoidHeadXZ.append((vert[0], vert[2]))
        ellipsoidHeadYZ.append((vert[1], vert[2]))
        ellipsoidHeadXYZ.append((vert[0], vert[1], vert[2]))
    objPath = '../../output/{}/ellipsoid/ellipsoid.obj'.format(
        time.strftime('%m%d', time.localtime()))
    saveEllipsoidObj(objPath, verts, faces)

    return ellipsoidHeadVLines, ellipsoidHeadXY, ellipsoidHeadXZ, ellipsoidHeadYZ, ellipsoidHeadXYZ


def mainEllipseHead():
    '''1. create ellipsoid'''
    ellip = ellipsoid(3, 5, 8, spacing=(1, 1, 1), levelset=True)
    verts, faces, normals, values = measure.marching_cubes_lewiner(ellip, 0)
    verts[:, 0] = verts[:, 0]+100
    print verts
    print verts.shape

    '''2. save obj'''
    objPath = '../../output/{}/ellipsoid/ellipsoid_{}.obj'.format(time.strftime(
        '%m%d', time.localtime()), time.strftime('%H%M', time.localtime()))
    os.makedirs(objPath) if not os.path.exists(
        os.path.dirname(objPath)[0]) else False
    objFile = open(objPath, 'w')
    objLines = []
    for vert in verts:
        objLines.append('{} {} {} {} {} {} {}'.format(
            'v', vert[0], vert[1], vert[2], 0.5, 0.3, 0.1))
    for face in faces:
        objLines.append('{} {} {} {}'.format('f', face[0], face[1], face[2]))
    for objLine in objLines:
        objFile.write(objLine+'\n')
    objFile.close()

    '''3. display'''
    '''3.1 plt'''
    # fig = plt.figure(figsize=(10, 10))
    # ax = fig.add_subplot(111, projection='3d')
    # mesh = Poly3DCollection(verts[faces])
    # mesh.set_edgecolor('k')
    # ax.add_collection3d(mesh)
    # ax.set_xlabel("x-axis: a = 6 per ellipsoid")
    # ax.set_ylabel("y-axis: b = 10")
    # ax.set_zlabel("z-axis: c = 16")
    # ax.set_xlim(0, 24)  # a = 6 (times two for 2nd ellipsoid)
    # ax.set_ylim(0, 20)  # b = 10
    # ax.set_zlim(0, 32)  # c = 16
    # plt.tight_layout()
    # plt.show()

    '''3.2 visvis'''
    # vv.mesh(np.fliplr(verts), faces, normals, values)
    # vv.use().Run()


if __name__ == "__main__":
    mainEllipseHead()
