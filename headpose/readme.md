# code/exp07
## twoD\_threeD\_twoD\_LR\_MeshHair.py
1. 整个头部椭圆取 mesh
2. 找到face landmark的minY
3. 取mesh中y<minY的网格，作为头发区域的关键点（有点像是个帽子）
4. 头发2D关键点 --> 3D
5. 头发3D关键点 nod
6. 头发3D关键点 --> 2D nod 头发关键点

## twoD\_threeD\_twoD\_LR\_ManuallySelectHair.py
1. 头发区域手动选择12个关键点
2. 头发2D关键点 --> 3D
3. 头发3D关键点 nod
4. 头发3D关键点 --> 2D nod 头发关键点

## twoD\_threeD\_twoD\_LR\_Hair.py
1. 2D face landmark --> 3D face landmark
2. 3D model nod
3. 3D nod face landmark --> 2D nod face landmark
4. 从 twoD\_threeD\_twoD\_LR\_MeshHair.py 或者 twoD\_threeD\_twoD\_LR\_ManuallySelectHair.py 得到origin hair landmark 和 nod hair landmark
5. 源图 + (2D face landmark + origin hair landmark) + (2D nod face landmark + nod hair landmark)   
  =(warp)=> ^_^ 

---
---
---
---
---

# code/exp06
## twoD\_threeD\_twoD\_LowResolutionManuallySelectHair.py
在 exp05/2D-3D-2D_low-resolution.py 的基础上，**手动** 选取了头发上12个关键点。  
源图 +（2D face源图关键点+2D hair源图关键点）+（2D face点头关键点+2D hair点头关键点）=(warp)=> 2D点头图像

---
---
---
---
---

# code/exp05
## 2D-3D-2D_low-resolution.py
1. 源图2D关键点检测
2. 2D源图关键点 --> 3D源model关键点
3. 3D点头
4. 3D点头关键点 --> 2D点头关键点
5. 源图+2D源图关键点+2D点头关键点 =(warp)=> 2D点头图像

---
---
---
---
---

# code/exp04
## mesh_tri.py
对网格点+68关键点生成delaunay三角形。

## meshwarp.py
  - 用skimage的PiecewiseAffineTransform，做网格点的mesh warp。（out_pat_wo.png）
  - 用skimage的PiecewiseAffineTransform，做网格点+68关键点的mesh warp。（out_pat_w.png）
  - 用之前face affine里的warp，做网格点+68关键点的mesh warp。（out_w.png）
  - 现在发现的问题是：1. 如果只用网格点进行affine，有些网格点（eg，脸颊）周围没有关键点。2. 周围有关键点的部分目标网格点构成的三角形扭曲太大，这种现象在加入关键点后更明显，导致图像有明显的分块的现象。
  - 解决方法：调整影响网格点的关键点数量。增大关键点数量，问题2有减轻，但是仍然存在，而且会导致目标点的变化幅度减小，动作幅度减小。另外，对于椭圆区域内没有受到关键点的影响产生位移的关键点（头发），把所在列的其他有位移的网格点按照delta/distance叠加给它，头发会有一丢丢的移动。所以我觉得现在的问题就是如何更好地确定网格点的位移。所以还会想办法取合适的网格目标点。

---
---
---
---
---

# code/exp03
## meshwarp.py
用之前face affine里的warp，做网格点的mesh warp。（out_wo.png）

---
---
---
---
---

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
  - 修改CUDA\_VISIBLE_DEVICES
  - 将examples/里的图片生成raw文件, 存在output/里
2. py raw2obj.py --image examples/scaled/trump-12.jpg --volume output/trump-12.raw --obj obj/trump-12.obj
  - 将raw文件转为obj文件

