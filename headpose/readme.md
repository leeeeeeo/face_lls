# code/exp02/
## meshwarp_EXAMPLE.py
对<a href="http://scikit-image.org/docs/dev/auto_examples/transform/plot_piecewise_affine.html?highlight=piecewise#piecewise-affine-transformation" target="_blank">skimage mesh warp example</a>成功修改的小例子

## meshwarp.py
正在写椭圆面内的2D mesh warp


---
---
---
---
---

# code/exp01/
## headpose.py
1. 解析obj文件
2. 将3D模型投影为2D
3. 点头, 转头, 摇头
4. 三种插值：
  - 直接把obj里面的所有点的坐标x, y, z放大3倍(x3), 重新生成的3d模型看起来正常, 但是由于点的间距变大, 投影到2d会出现黑色点
  - 先把3d投影回2d, 然后直接对2d做resize
  - 用scipy.interpolate.griddata进行插值

## put\_face\_back.py
1. 对2D原始图片做关键点检测
2. 读3D model
3. `拟合椭球体：`
  - `109-120 行是使用OPENCV拟合三个面的椭圆, 椭圆不能完全包围住`
  - `122-150 行是使用`<a href="http://scikit-image.org/docs/dev/api/skimage.draw.html?highlight=ellipse#ellipsoid" target="_blank">skimage ellipsoid</a>`拟合椭球体`

## meshwarp.py
尝试对<a href="http://scikit-image.org/docs/dev/auto_examples/transform/plot_piecewise_affine.html?highlight=piecewise#piecewise-affine-transformation" target="_blank">skimage mesh warp example</a>进行修改，但是失败了。原因是deepcopy！


---
---
---
---
---


# github/
## ellipsoid\_fit\_python/plot_ellipsoid.py
`开源代码拟合3D model的椭球体, 输入是3D model的所有点坐标, 但是输出的椭球很奇怪`

## vrn-07231340
可运行的VRN 2D-->3D  

1. bash run.sh
  - 修改CUDA_VISIBLE_DEVICES
  - 将examples/里的图片生成raw文件, 存在output/里
2. py raw2obj.py --image examples/scaled/trump-12.jpg --volume output/trump-12.raw --obj obj/trump-12.obj
  - 将raw文件转为obj文件

