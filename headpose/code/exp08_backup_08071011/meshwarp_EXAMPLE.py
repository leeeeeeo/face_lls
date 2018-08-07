# -*- coding: utf-8 -*-
import copy
import matplotlib.pyplot as plt
import numpy as np
from skimage import data
from skimage.transform import PiecewiseAffineTransform, warp

'''read image'''
image = data.astronaut()
rows, cols = image.shape[0], image.shape[1]

'''新建source mesh'''
src_rows = np.linspace(0, rows, 10)  # 10 行
src_cols = np.linspace(0, cols, 20)  # 20 列


# 生成src_rows和src_cols两个
# 都是src_cols行 src_rows列的矩阵
# 都是20行 10列的矩阵
# src_rows 20行 10列 每一行都是
# [  0.          56.88888889 113.77777778 170.66666667 227.55555556 284.44444444 341.33333333 398.22222222 455.11111111 512.        ]
# src_cols 20行 10列 每一行是相同的数字
# [0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
# [26.94736842 26.94736842 26.94736842 26.94736842 26.94736842 26.94736842 26.94736842 26.94736842 26.94736842 26.94736842]
src_rows, src_cols = np.meshgrid(src_rows, src_cols)


# src: [src_col, src_row]
# src: [0, 0], [0, 56], [0, 113], [0, 170], [0, 227], [0, 284], [0, 341], [0, 398], [0, 455], [0, 512]
# ⬆️ 共10个
# src: [26, 0], [26, 56], [26, 113], [26, 170], [26, 227], [26, 284], [26, 341], [26, 398], [26, 455], [26, 512]
# ⬆️ 共10个
# ............
src = np.dstack([src_cols.flat, src_rows.flat])[0]
'''!!! IMPORTANT !!!'''
dst = copy.deepcopy(src)
print src[92]
dst[92] = [250, 100]
tform = PiecewiseAffineTransform()
tform.estimate(src, dst)
dst = tform(src)
out = warp(image, tform)

fig, ax = plt.subplots()
ax.imshow(out)
ax.plot(dst[:, 0], dst[:, 1], '.b')
# ax.plot(tform.inverse(src)[:, 0], tform.inverse(src)[:, 1], '.r')
plt.show()
