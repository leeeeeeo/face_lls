# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import PiecewiseAffineTransform, warp
from skimage import data

'''read image'''
image = data.astronaut()
rows, cols = image.shape[0], image.shape[1]

'''新建source mesh'''
src_cols = np.linspace(0, cols, 20)
src_rows = np.linspace(0, rows, 10)
src_rows, src_cols = np.meshgrid(src_rows, src_cols)
src = np.dstack([src_cols.flat, src_rows.flat])[0]


'''修改source mesh得到destination mesh'''
'''dst_rows是example原始的的'''
'''dst_rows1是我写的'''
'''我想把row的前200个做修改，后面的保持不变，看看会有什么效果，destination mesh的确变化了，但是图片并没有按照destination mesh进行morph'''
'''以下是原始example'''
dst_rows = src[:, 1] - np.sin(np.linspace(0, 3 * np.pi, src.shape[0])) * 50
dst_cols = src[:, 0]
dst_rows *= 1.5
dst_rows -= 1.5 * 50
dst = np.vstack([dst_cols, dst_rows]).T

tform = PiecewiseAffineTransform()  # skimage 的 mesh warp
tform.estimate(src, dst)

out_rows = image.shape[0] - 1.5 * 50
out_cols = cols
out = warp(image, tform, output_shape=(out_rows, out_cols))

fig, ax = plt.subplots()
ax.imshow(out)
ax.plot(tform.inverse(src)[:, 0], tform.inverse(src)[:, 1], '.b')
ax.axis((0, out_cols, out_rows, 0))
plt.show()

'''以下是我的修改，只改动了dst_row'''
'''1. size=src.shape[0]，希望应该出现和example同样的结果，但是并没有。'''
'''   就连网格的变化也不一样，图像也没有进行morph。'''
'''2. size=100，希望左边100个会有变化，而且图片左边一部分会随着网格的变化进行morph'''
dst_rows1 = src[:, 1]
size = src.shape[0]
# size = 100
dst_rows1[:size] = src[:, 1][:size] - \
    np.sin(np.linspace(0, 3 * np.pi, size)) * 50

dst_cols = src[:, 0]
dst_rows1 *= 1.5
dst_rows1 -= 1.5 * 50
dst = np.vstack([dst_cols, dst_rows1]).T

tform = PiecewiseAffineTransform()  # skimage 的 mesh warp
tform.estimate(src, dst)

out_rows = image.shape[0] - 1.5 * 50
out_cols = cols
out = warp(image, tform, output_shape=(out_rows, out_cols))

fig, ax = plt.subplots()
ax.imshow(out)
ax.plot(tform.inverse(src)[:, 0], tform.inverse(src)[:, 1], '.b')
ax.axis((0, out_cols, out_rows, 0))
plt.show()


'''3. 只改变mesh中一个点的坐标，希望得到图片根据修改的点进行morph'''
src_cols = np.linspace(0, cols, 20)
src_rows = np.linspace(0, rows, 10)
src_rows, src_cols = np.meshgrid(src_rows, src_cols)
src = np.dstack([src_cols.flat, src_rows.flat])[0]


dst_rows2 = src[:, 1]
dst_cols = src[:, 0]
dst_rows2 *= 1.5
dst_rows2 -= 1.5 * 50

dst_rows2[50] = 300
dst_cols[50] = 10
dst = np.vstack([dst_cols, dst_rows2]).T

tform = PiecewiseAffineTransform()  # skimage 的 mesh warp
tform.estimate(src, dst)

out_rows = image.shape[0] - 1.5 * 50
out_cols = cols
out = warp(image, tform, output_shape=(out_rows, out_cols))

fig, ax = plt.subplots()
ax.imshow(out)
ax.plot(tform.inverse(src)[:, 0], tform.inverse(src)[:, 1], '.b')
ax.axis((0, out_cols, out_rows, 0))
plt.show()
