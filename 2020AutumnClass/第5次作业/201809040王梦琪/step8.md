# 【Step8】

>>>># 17 卷积神经网络 

>>### 17.0 卷积神经网络原理
#### 卷积神经网络的能力
卷积神经网络（CNN，Convolutional Neural Net)是神经网络的类型之一，在图像识别和分类领域中取得了非常好的效果，比如识别人脸、物体、交通标识等，这就为机器人、自动驾驶等应用提供了坚实的技术基础。

#### 卷积神经网络的典型结构

一个典型的卷积神经网络的结构如图17-5所示。

<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/17/conv_net.png" />

图17-5 卷积神经网络的典型结构图

我们分析一下它的层级结构：

1. 原始的输入是一张图片，可以是彩色的，也可以是灰度的或黑白的。这里假设是只有一个通道的图片，目的是识别0~9的手写体数字；
2. 第一层卷积，我们使用了4个卷积核，得到了4张feature map；激活函数层没有单独画出来，这里我们紧接着卷积操作使用了Relu激活函数；
3. 第二层是池化，使用了Max Pooling方式，把图片的高宽各缩小一倍，但仍然是4个feature map；
4. 第三层卷积，我们使用了4x6个卷积核，其中4对应着输入通道，6对应着输出通道，从而得到了6张feature map，当然也使用了Relu激活函数；
5. 第四层再次做一次池化，现在得到的图片尺寸只是原始尺寸的四分之一左右；
6. 第五层把第四层的6个图片展平成一维，成为一个fully connected层；
7. 第六层再接一个小一些的fully connected层；
8. 最后接一个softmax函数，判别10个分类。

所以，在一个典型的卷积神经网络中，会至少包含以下几个层：

- 卷积层
- 激活函数层
- 池化层
- 全连接分类层

#### 卷积核的作用
下面说明9个卷积核的作用：

|序号|名称|说明|
|---|---|---|
|1|锐化|如果一个像素点比周围像素点亮，则此算子会令其更亮|
|2|检测竖边|检测出了十字线中的竖线，由于是左侧和右侧分别检查一次，所以得到两条颜色不一样的竖线|
|3|周边|把周边增强，把同色的区域变弱，形成大色块|
|4|Sobel-Y|纵向亮度差分可以检测出横边，与横边检测不同的是，它可以使得两条横线具有相同的颜色，具有分割线的效果|
|5|Identity|中心为1四周为0的过滤器，卷积后与原图相同|
|6|横边检测|检测出了十字线中的横线，由于是上侧和下侧分别检查一次，所以得到两条颜色不一样的横线|
|7|模糊|通过把周围的点做平均值计算而“杀富济贫”造成模糊效果|
|8|Sobel-X|横向亮度差分可以检测出竖边，与竖边检测不同的是，它可以使得两条竖线具有相同的颜色，具有分割线的效果|
|9|浮雕|形成大理石浮雕般的效果|

#### 卷积后续的运算

前面我们认识到了卷积核的强大能力，卷积神经网络通过反向传播而令卷积核自我学习，找到分布在图片中的不同的feature，最后形成的卷积核中的数据。但是如果想达到这种效果，只有卷积层的话是不够的，还需要激活函数、池化等操作的配合。

图17-7中的四个子图，依次展示了：

1. 原图
2. 卷积结果
3. 激活结果
4. 池化结果

<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/17/circle_conv_relu_pool.png" ch="500" />

图17-7 原图经过卷积-激活-池化操作后的效果

1. 注意图一是原始图片，用cv2读取出来的图片，其顺序是反向的，即：

- 第一维是高度
- 第二维是宽度
- 第三维是彩色通道数，但是其顺序为BGR，而不是常用的RGB

1. 我们对原始图片使用了一个3x1x3x3的卷积核，因为原始图片为彩色图片，所以第一个维度是3，对应RGB三个彩色通道；我们希望只输出一张feature map，以便于说明，所以第二维是1；我们使用了3x3的卷积核，用的是sobel x算子。所以图二是卷积后的结果。

2. 图三做了一层Relu激活计算，把小于0的值都去掉了，只留下了一些边的特征。

3. 图四是图三的四分之一大小，虽然图片缩小了，但是特征都没有丢失，反而因为图像尺寸变小而变得密集，亮点的密度要比图三大而粗。

#### 卷积神经网络的学习
回忆一下MNIST数据集，所有的样本数据都是处于28x28方形区域的中间地带，如下图中的左上角的图片A所示。

<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/17/8.png" ch="500" />

图17-8 同一个背景下数字8的大小、位置、形状的不同

我们的问题是：

1. 如果这个“8”的位置很大地偏移到了右下角，使得左侧留出来一大片空白，即发生了平移，如上图右上角子图B
2. “8”做了一些旋转或者翻转，即发生了旋转视角，如上图左下角子图C
3. “8”缩小了很多或放大了很多，即发生了尺寸变化，如上图右下角子图D

尽管发生了变化，但是对于人类的视觉系统来说都可以轻松应对，即平移不变性、旋转视角不变性、尺度不变性。那么卷积神经网络网络如何处理呢？

- 平移不变性
  
  对于原始图A，平移后得到图B，对于同一个卷积核来说，都会得到相同的特征，这就是卷积核的权值共享。但是特征处于不同的位置，由于距离差距较大，即使经过多层池化后，也不能处于近似的位置。此时，后续的全连接层会通过权重值的调整，把这两个相同的特征看作同一类的分类标准之一。如果是小距离的平移，通过池化层就可以处理了。

- 旋转不变性

  对于原始图A，有小角度的旋转得到C，卷积层在A图上得到特征a，在C图上得到特征c，可以想象a与c的位置间的距离不是很远，在经过两层池化以后，基本可以重合。所以卷积网络对于小角度旋转是可以容忍的，但是对于较大的旋转，需要使用数据增强来增加训练样本。一个极端的例子是当6旋转90度时，谁也不能确定它到底是6还是9。

- 尺度不变性

  对于原始图A和缩小的图D，人类可以毫不费力地辨别出它们是同一个东西。池化在这里是不是有帮助呢？没有！因为神经网络对A做池化的同时，也会用相同的方法对D做池化，这样池化的次数一致，最终D还是比A小。如果我们有多个卷积视野，相当于从两米远的地方看图A，从一米远的地方看图D，那么A和D就可以很相近似了。这就是Inception的想法，用不同尺寸的卷积核去同时寻找同一张图片上的特征。

>>### 17.1 卷积的前向计算原理
### 卷积的数学定义
#### 连续定义

$$h(x)=(f*g)(x) = \int_{-\infty}^{\infty} f(t)g(x-t)dt \tag{1}$$

卷积与傅里叶变换有着密切的关系。利用这点性质，即两函数的傅里叶变换的乘积等于它们卷积后的傅里叶变换，能使傅里叶分析中许多问题的处理得到简化。

#### 离散定义

$$h(x) = (f*g)(x) = \sum^{\infty}_{t=-\infty} f(t)g(x-t) \tag{2}$$

### 单入单出的二维卷积
1. 我们实现的卷积操作不是原始数学含义的卷积，而是工程上的卷积，可以简称为卷积
2. 在实现卷积操作时，并不会反转卷积核

在传统的图像处理中，卷积操作多用来进行滤波，锐化或者边缘检测啥的。我们可以认为卷积是利用某些设计好的参数组合（卷积核）去提取图像空域上相邻的信息。

### 单入多出的升维卷积

原始输入是一维的图片，但是我们可以用多个卷积核分别对其计算，从而得到多个特征输出。如图17-13所示。

<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/17/conv_2w3.png" ch="500" />

图17-13 单入多出的升维卷积

一张4x4的图片，用两个卷积核并行地处理，输出为2个2x2的图片。在训练过程中，这两个卷积核会完成不同的特征学习。

### 多入单出的降维卷积

一张图片，通常是彩色的，具有红绿蓝三个通道。我们可以有两个选择来处理：

1. 变成灰度的，每个像素只剩下一个值，就可以用二维卷积
2. 对于三个通道，每个通道都使用一个卷积核，分别处理红绿蓝三种颜色的信息

### 多入多出的同维卷积

<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/17/conv3dp.png" ch="500" />

图17-15 多入多出的卷积运算

第一个过滤器Filter-1为棕色所示，它有三卷积核(Kernal)，命名为Kernal-1，Keanrl-2，Kernal-3，分别在红绿蓝三个输入通道上进行卷积操作，生成三个2x2的输出Feature-1,n。然后三个Feature-1,n相加，并再加上b1偏移值，形成最后的棕色输出Result-1。

对于灰色的过滤器Filter-2也是一样，先生成三个Feature-2,n，然后相加再加b2，最后得到Result-2。

之所以Feature-m,n还用红绿蓝三色表示，是因为在此时，它们还保留着红绿蓝三种色彩的各自的信息，一旦相加后得到Result，这种信息就丢失了。

### 卷积编程模型

上图侧重于解释数值计算过程，而图17-16侧重于解释五个概念的关系：

- 输入 Input Channel
- 卷积核组 WeightsBias
- 过滤器 Filter
- 卷积核 kernal
- 输出 Feature Map

对于三维卷积，有以下特点：

1. 预先定义输出的feature map的数量，而不是根据前向计算自动计算出来，此例中为2，这样就会有两组WeightsBias
2. 对于每个输出，都有一个对应的过滤器Filter，此例中Feature Map-1对应Filter-1
3. 每个Filter内都有一个或多个卷积核Kernal，对应每个输入通道(Input Channel)，此例为3，对应输入的红绿蓝三个通道
4. 每个Filter只有一个Bias值，Filter-1对应b1，Filter-2对应b2
5. 卷积核Kernal的大小一般是奇数如：1x1, 3x3, 5x5, 7x7等，此例为5x5

对于上图，我们可以用在全连接神经网络中的学到的知识来理解：

1. 每个Input Channel就是特征输入，在上图中是3个
2. 卷积层的卷积核相当于隐层的神经元，上图中隐层有2个神经元
3. $W(m,n), m=[1,2], n=[1,3]$相当于隐层的权重矩阵$w_{11},w_{12},......$
4. 每个卷积核（神经元）有1个偏移值

### 步长 stride

前面的例子中，每次计算后，卷积核会向右或者向下移动一个单元，即步长stride = 1。而在图17-17这个卷积操作中，卷积核每次向右或向下移动两个单元，即stride = 2。

<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/17/Stride2.png" />

图17-17 步长为2的卷积

在后续的步骤中，由于每次移动两格，所以最终得到一个2x2的图片。

### 填充 padding

如果原始图为4x4，用3x3的卷积核进行卷积后，目标图片变成了2x2。如果我们想保持目标图片和原始图片为同样大小，该怎么办呢？一般我们会向原始图片周围填充一圈0，然后再做卷积。如图17-18。

<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/17/padding.png" ch="500" />

图17-18 带填充的卷积

>>### 17.3 卷积的反向传播原理

#### 计算反向传播的梯度矩阵

正向公式：

$$Z = W*A+b \tag{0}$$

其中，W是卷积核，*表示卷积（互相关）计算，A为当前层的输入项，b是偏移（未在图中画出），Z为当前层的输出项，但尚未经过激活函数处理。

#### 步长不为1时的梯度矩阵还原

我们先观察一下stride=1和2时，卷积结果的差异如图17-23。

<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/17/stride_1_2.png"/>

图17-23 步长为1和步长为2的卷积结果的比较

二者的差别就是中间那个结果图的灰色部分。如果反向传播时，传入的误差矩阵是stride=2时的2x2的形状，那么我们只需要把它补上一个十字，变成3x3的误差矩阵，就可以用步长为1的算法了。

以此类推，如果步长为3时，需要补一个双线的十字。所以，当知道当前的卷积层步长为S（S>1）时：

1. 得到从上层回传的误差矩阵形状，假设为$M \times N$
2. 初始化一个$(M \cdot S) \times (N \cdot S)$的零矩阵
3. 把传入的误差矩阵的第一行值放到零矩阵第0行的0,S,2S,3S...位置
4. 然后把误差矩阵的第二行的值放到零矩阵第S行的0,S,2S,3S...位置
5. ......

步长为2时，用实例表示就是这样：

$$
\begin{bmatrix}
  \delta_{11} & 0 & \delta_{12} & 0 & \delta_{13}\\\\
  0 & 0 & 0 & 0 & 0\\\\
  \delta_{21} & 0 & \delta_{22} & 0 & \delta_{23}\\\\
\end{bmatrix}
$$ 

步长为3时，用实例表示就是这样：

$$
\begin{bmatrix}
  \delta_{11} & 0 & 0 & \delta_{12} & 0 & 0 & \delta_{13}\\\\
  0 & 0 & 0 & 0 & 0 & 0 & 0\\\\
  0 & 0 & 0 & 0 & 0 & 0 & 0\\\\
  \delta_{21} & 0 & 0 & \delta_{22} & 0 & 0 & \delta_{23}\\\\
\end{bmatrix}
$$ 

#### 有多个卷积核时的梯度计算
正向公式：

$$z_{111} = w_{111} \cdot a_{11} + w_{112} \cdot a_{12} + w_{121} \cdot a_{21} + w_{122} \cdot a_{22}$$
$$z_{112} = w_{111} \cdot a_{12} + w_{112} \cdot a_{13} + w_{121} \cdot a_{22} + w_{122} \cdot a_{23}$$
$$z_{121} = w_{111} \cdot a_{21} + w_{112} \cdot a_{22} + w_{121} \cdot a_{31} + w_{122} \cdot a_{32}$$
$$z_{122} = w_{111} \cdot a_{22} + w_{112} \cdot a_{23} + w_{121} \cdot a_{32} + w_{122} \cdot a_{33}$$

$$z_{211} = w_{211} \cdot a_{11} + w_{212} \cdot a_{12} + w_{221} \cdot a_{21} + w_{222} \cdot a_{22}$$
$$z_{212} = w_{211} \cdot a_{12} + w_{212} \cdot a_{13} + w_{221} \cdot a_{22} + w_{222} \cdot a_{23}$$
$$z_{221} = w_{211} \cdot a_{21} + w_{212} \cdot a_{22} + w_{221} \cdot a_{31} + w_{222} \cdot a_{32}$$
$$z_{222} = w_{211} \cdot a_{22} + w_{212} \cdot a_{23} + w_{221} \cdot a_{32} + w_{222} \cdot a_{33}$$

#### 有多个输入时的梯度计算

当输入层是多个图层时，每个图层必须对应一个卷积核，如图17-25。

<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/17/conv_1W222.png" ch="500" />

图17-25 多个图层的卷积必须有一一对应的卷积核

所以有前向公式：

$$
\begin{aligned}
z_{11} &= w_{111} \cdot a_{111} + w_{112} \cdot a_{112} + w_{121} \cdot a_{121} + w_{122} \cdot a_{122}
\\\\
&+ w_{211} \cdot a_{211} + w_{212} \cdot a_{212} + w_{221} \cdot a_{221} + w_{222} \cdot a_{222} 
\end{aligned}
\tag{10}
$$
$$
\begin{aligned}
z_{12} &= w_{111} \cdot a_{112} + w_{112} \cdot a_{113} + w_{121} \cdot a_{122} + w_{122} \cdot a_{123} \\\\
&+ w_{211} \cdot a_{212} + w_{212} \cdot a_{213} + w_{221} \cdot a_{222} + w_{222} \cdot a_{223} 
\end{aligned}\tag{11}$$
$$
\begin{aligned}
z_{21} &= w_{111} \cdot a_{121} + w_{112} \cdot a_{122} + w_{121} \cdot a_{131} + w_{122} \cdot a_{132} \\\\
&+ w_{211} \cdot a_{221} + w_{212} \cdot a_{222} + w_{221} \cdot a_{231} + w_{222} \cdot a_{232} 
\end{aligned}\tag{12}$$
$$
\begin{aligned}
z_{22} &= w_{111} \cdot a_{122} + w_{112} \cdot a_{123} + w_{121} \cdot a_{132} + w_{122} \cdot a_{133} \\\\
&+ w_{211} \cdot a_{222} + w_{212} \cdot a_{223} + w_{221} \cdot a_{232} + w_{222} \cdot a_{233} 
\end{aligned}\tag{13}$$

最复杂的情况，求$J$对$a_{122}$的梯度：

$$
\begin{aligned}
\frac{\partial J}{\partial a_{111}}&=\frac{\partial J}{\partial z_{11}}\frac{\partial z_{11}}{\partial a_{122}} + \frac{\partial J}{\partial z_{12}}\frac{\partial z_{12}}{\partial a_{122}} + \frac{\partial J}{\partial z_{21}}\frac{\partial z_{21}}{\partial a_{122}} + \frac{\partial J}{\partial z_{22}}\frac{\partial z_{22}}{\partial a_{122}}
\\\\
&=\delta_{z_{11}} \cdot w_{122} + \delta_{z_{12}} \cdot w_{121} + \delta_{z_{21}} \cdot w_{112} + \delta_{z_{22}} \cdot w_{111} 
\end{aligned}
$$

泛化以后得到：

$$\delta_{out1} = \delta_{in} * W_1^{rot180} \tag{14}$$

求$J$对$a_{222}$的梯度：

$$
\begin{aligned}
\frac{\partial J}{\partial a_{211}}&=\frac{\partial J}{\partial z_{11}}\frac{\partial z_{11}}{\partial a_{222}} + \frac{\partial J}{\partial z_{12}}\frac{\partial z_{12}}{\partial a_{222}} + \frac{\partial J}{\partial z_{21}}\frac{\partial z_{21}}{\partial a_{222}} + \frac{\partial J}{\partial z_{22}}\frac{\partial z_{22}}{\partial a_{222}} \\\\
&=\delta_{z_{11}} \cdot w_{222} + \delta_{z_{12}} \cdot w_{221} + \delta_{z_{21}} \cdot w_{212} + \delta_{z_{22}} \cdot w_{211} 
\end{aligned}
$$

泛化以后得到：

$$\delta_{out2} = \delta_{in} * W_2^{rot180} \tag{15}$$

#### 权重（卷积核）梯度计算

图17-26展示了我们已经熟悉的卷积正向运算。

<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/17/conv_forward.png" />

图17-26 卷积正向计算

要求J对w11的梯度，从正向公式可以看到，w11对所有的z都有贡献，所以：

$$
\begin{aligned}
\frac{\partial J}{\partial w_{11}} &= \frac{\partial J}{\partial z_{11}}\frac{\partial z_{11}}{\partial w_{11}} + \frac{\partial J}{\partial z_{12}}\frac{\partial z_{12}}{\partial w_{11}} + \frac{\partial J}{\partial z_{21}}\frac{\partial z_{21}}{\partial w_{11}} + \frac{\partial J}{\partial z_{22}}\frac{\partial z_{22}}{\partial w_{11}}
\\\\
&=\delta_{z11} \cdot a_{11} + \delta_{z12} \cdot a_{12} + \delta_{z21} \cdot a_{21} + \delta_{z22} \cdot a_{22} 
\end{aligned}
\tag{9}
$$

对W22也是一样的：

$$
\begin{aligned}
\frac{\partial J}{\partial w_{12}} &= \frac{\partial J}{\partial z_{11}}\frac{\partial z_{11}}{\partial w_{12}} + \frac{\partial J}{\partial z_{12}}\frac{\partial z_{12}}{\partial w_{12}} + \frac{\partial J}{\partial z_{21}}\frac{\partial z_{21}}{\partial w_{12}} + \frac{\partial J}{\partial z_{22}}\frac{\partial z_{22}}{\partial w_{12}}
\\\\
&=\delta_{z11} \cdot a_{12} + \delta_{z12} \cdot a_{13} + \delta_{z21} \cdot a_{22} + \delta_{z22} \cdot a_{23} 
\end{aligned}
\tag{10}
$$

观察公式8和公式9，其实也是一个标准的卷积（互相关）操作过程，因此，可以把这个过程看成图17-27。

<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/17/conv_delta_w.png" />

图17-27 卷积核的梯度计算

总结成一个公式：

$$\delta_w = A * \delta_{in} \tag{11}$$

#### 偏移的梯度计算

根据前向计算公式1，2，3，4，可以得到：

$$
\begin{aligned}
\frac{\partial J}{\partial b} &= \frac{\partial J}{\partial z_{11}}\frac{\partial z_{11}}{\partial b} + \frac{\partial J}{\partial z_{12}}\frac{\partial z_{12}}{\partial b} + \frac{\partial J}{\partial z_{21}}\frac{\partial z_{21}}{\partial b} + \frac{\partial J}{\partial z_{22}}\frac{\partial z_{22}}{\partial b}
\\\\
&=\delta_{z11} + \delta_{z12}  + \delta_{z21} + \delta_{z22} 
\end{aligned}
\tag{12}
$$

所以：

$$
\delta_b = \delta_{in} \tag{13}
$$

每个卷积核W可能会有多个filter，或者叫子核，但是一个卷积核只有一个偏移，无论有多少子核。


>>### 17.5 池化的前向计算与反向传播
### 池化层

#### 常用池化方法

池化 pooling，又称为下采样，downstream sampling or sub-sampling。

池化方法分为两种，一种是最大值池化 Max Pooling，一种是平均值池化 Mean/Average Pooling。如图17-32所示。

<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/17/pooling.png" />

图17-32 池化

- 最大值池化，是取当前池化视野中所有元素的最大值，输出到下一层特征图中。
- 平均值池化，是取当前池化视野中所有元素的平均值，输出到下一层特征图中。

其目的是：

- 扩大视野：就如同先从近处看一张图片，然后离远一些再看同一张图片，有些细节就会被忽略
- 降维：在保留图片局部特征的前提下，使得图片更小，更易于计算
- 平移不变性，轻微扰动不会影响输出：比如上图中最大值池化的4，即使向右偏一个像素，其输出值仍为4
- 维持同尺寸图片，便于后端处理：假设输入的图片不是一样大小的，就需要用池化来转换成同尺寸图片

一般我们都使用最大值池化。

#### 池化的其它方式

在上面的例子中，我们使用了size=2x2，stride=2的模式，这是常用的模式，即步长与池化尺寸相同。

我们很少使用步长值与池化尺寸不同的配置。

#### 池化层的训练

我们假设图17-34中，$[[1,2],[3,4]]$是上一层网络回传的残差，那么：

- 对于最大值池化，残差值会回传到当初最大值的位置上，而其它三个位置的残差都是0。
- 对于平均值池化，残差值会平均到原始的4个位置上。

<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/17/pooling_backward.png" />

图17-34 平均池化与最大池化

<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/17/pooling_backward_max.png" />

图17-35 池化层反向传播的示例

#### Max Pooling

严格的数学推导过程以图17-35为例进行。

正向公式：

$$
w = max(a,b,e,f)
$$

反向公式（假设Input Layer中的最大值是b）：

$$
{\partial w \over \partial a} = 0, \quad {\partial w \over \partial b} = 1
$$

$$
{\partial w \over \partial e} = 0, \quad {\partial w \over \partial f} = 0
$$

因为a,e,f对w都没有贡献，所以偏导数为0，只有b有贡献，偏导数为1。

$$
\delta_a = {\partial J \over \partial a} = {\partial J \over \partial w} {\partial w \over \partial a} = 0
$$

$$
\delta_b = {\partial J \over \partial b} = {\partial J \over \partial w} {\partial w \over \partial b} = \delta_w \cdot 1 = \delta_w
$$

$$
\delta_e = {\partial J \over \partial e} = {\partial J \over \partial w} {\partial w \over \partial e} = 0
$$

$$
\delta_f = {\partial J \over \partial f} = {\partial J \over \partial w} {\partial w \over \partial f} = 0
$$

#### Mean Pooling

正向公式：

$$w = \frac{1}{4}(a+b+e+f)$$

反向公式（假设Layer-1中的最大值是b）：

$$
{\partial w \over \partial a} = \frac{1}{4}, \quad {\partial w \over \partial b} = \frac{1}{4}
$$

$$
{\partial w \over \partial e} = \frac{1}{4}, \quad {\partial w \over \partial f} = \frac{1}{4}
$$

因为a,b,e,f对w都有贡献，所以偏导数都为1：

$$
\delta_a = {\partial J \over \partial a} = {\partial J \over \partial w} {\partial w \over \partial a} = \frac{1}{4}\delta_w
$$

$$
\delta_b = {\partial J \over \partial b} = {\partial J \over \partial w} {\partial w \over \partial b} = \frac{1}{4}\delta_w
$$

$$
\delta_e = {\partial J \over \partial e} = {\partial J \over \partial w} {\partial w \over \partial e} = \frac{1}{4}\delta_w
$$

$$
\delta_f = {\partial J \over \partial f} = {\partial J \over \partial w} {\partial w \over \partial f} = \frac{1}{4}\delta_w
$$

无论是max pooling还是mean pooling，都没有要学习的参数，所以，在卷积网络的训练中，池化层需要做的只是把误差项向后传递，不需要计算任何梯度。


>>>># 18 卷积神经网络应用

>>### 18.0 经典的卷积神经网络模型
#### LeNet (1998)

图18-1是卷积神经网络的鼻祖LeNet$^{[3]}$的模型结构。

<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/18/Net_LeNet.png" />

图18-1 LeNet模型结构图

LeNet是卷积神经网络的开创者LeCun在1998年提出，用于解决手写数字识别的视觉任务。自那时起，卷积神经网络的最基本的架构就定下来了：卷积层、池化层、全连接层。

1. 输入为单通道32x32灰度图
2. 使用6组5x5的过滤器，每个过滤器里有一个卷积核，stride=1，得到6张28x28的特征图
3. 使用2x2的池化，stride=2，得到6张14x14的特征图
4. 使用16组5x5的过滤器，每个过滤器里有6个卷积核，对应上一层的6个特征图，得到16张10x10的特征图
5. 池化，得到16张5x5的特征图
6. 接全连接层，120个神经元
7. 接全连接层，84个神经元
8. 接全连接层，10个神经元，softmax输出

如今各大深度学习框架中所使用的LeNet都是简化改进过的LeNet-5（5表示具有5个层），和原始的LeNet有些许不同，比如把激活函数改为了现在很常用的ReLu。LeNet-5跟现有的conv->pool->ReLU的套路不同，它使用的方式是conv1->pool->conv2->pool2再接全连接层，但是不变的是，卷积层后紧接池化层的模式依旧不变。

#### AlexNet (2012)

AlexNet$^{[4]}$网络结构在整体上类似于LeNet，都是先卷积然后在全连接。但在细节上有很大不同。AlexNet更为复杂。AlexNet有60 million个参数和65000个神经元，五层卷积，三层全连接网络，最终的输出层是1000通道的Softmax。

AlexNet用两块GPU并行计算，大大提高了训练效率，并且在ILSVRC-2012竞赛中获得了top-5测试的15.3%的error rate，获得第二名的方法error rate是26.2%，差距非常大，足以说明这个网络在当时的影响力。

图18-2是AlexNet的模型结构。

<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/18/Net_AlexNet.png" />

图18-2 AlexNet模型结构图

1. 原始图片是3x224x224三通道彩色图片，分开到上下两块GPU上进行训练
2. 卷积，96核，11x11，stride=4，pad=0，输出96x55x55
3. LRN+池化，3x3，stride=2，pad=0，输出96x27x27
4. 卷积，256核，5x5，stride=1，pad=2，输出256x27x27
5. LRN+池化，3x3，stride=2，输出256x13x13
6. 卷积，384核，3x3，stride=1，pad=1，输出384x13x13
7. 卷积，384核，3x3，stride=1，pad=1，输出384x13x13
8. 卷积，256核，3x3，stride=1，pad=1，输出256x13x13
9. 池化，3x3，stride=2，输出256x6x6
10. 全连接层，4096个神经元，接Dropout和Relu
11. 全连接层，4096个神经元，接Dropout和Relu
12. 全连接层，1000个神经元做分类

AlexNet的特点：

- 比LeNet深和宽的网络
  
  使用了5层卷积和3层全连接，一共8层。特征数在最宽处达到384。

- 数据增强
  
  针对原始图片256x256的数据，做了随机剪裁，得到224x224的图片若干张。

- 使用ReLU做激活函数
- 在全连接层使用DropOut
- 使用LRN
  
  LRN的全称为Local Response Normalizatio，局部响应归一化，是想对线性输出做一个归一化，避免上下越界。发展至今，这个技术已经很少使用了。

#### ZFNet (2013)

ZFNet$^{[5]}$是2013年ImageNet分类任务的冠军，其网络结构没什么改进，只是调了调参，性能较Alex提升了不少。ZF-Net只是将AlexNet第一层卷积核由11变成7，步长由4变为2，第3，4，5卷积层转变为384，384，256。这一年的ImageNet还是比较平静的一届，其冠军ZF-Net的名堂也没其他届的经典网络架构响亮。

图18-3是ZFNet的结构示意图。

<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/18/Net_ZFNet.png" />

图18-3 ZFNet模型结构图

但是ZFNet首次系统化地对卷积神经网络做了可视化的研究，从而找到了AlexNet的缺点并加以改正，提高了网络的能力。总的来说，通过卷积神经网络学习后，我们学习到的特征，是具有辨别性的特征，比如要我们区分人脸和狗头，那么通过卷积神经网络学习后，背景部位的激活度基本很少，我们通过可视化就可以看到我们提取到的特征忽视了背景，而是把关键的信息给提取出来了。

图18-5所示，从Layer 1、Layer 2学习到的特征基本上是颜色、边缘等低层特征。

<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/18/Net_ZFNet_p1.png" />

图18-4 前两层卷积核学到的特征

Layer 3则开始稍微变得复杂，学习到的是纹理特征，比如上面的一些网格纹理，见图18-5。

<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/18/Net_ZFNet_p2.png" />

图18-5 第三层卷积核学到的特征

图18-6所示的Layer 4学习到的则是比较有区别性的特征，比如狗头；Layer 5学习到的则是完整的，具有辨别性关键特征。

<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/18/Net_ZFNet_p3.png" ch="500" />

图18-6 后两层卷积核学到的特征

#### VGGNet (2015)

图18-7为VGG16（16层的VGG）模型结构。

<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/18/Net_Vgg16.png" ch="500" />

图18-7 VGG16模型结构图

VGG比较出名的是VGG-16和VGG-19，最常用的是VGG-16。

VGGNet的卷积层有一个特点：特征图的空间分辨率单调递减，特征图的通道数单调递增，使得输入图像在维度上流畅地转换到分类向量。用通俗的语言讲，就是特征图尺寸单调递减，特征图数量单调递增。从上面的模型图上来看，立体方块的宽和高逐渐减小，但是厚度逐渐增加。

AlexNet的通道数无此规律，VGGNet后续的GoogLeNet和Resnet均遵循此维度变化的规律。

一些其它的特点如下：

1. 选择采用3x3的卷积核是因为3x3是最小的能够捕捉像素8邻域信息的尺寸。
2. 使用1x1的卷积核目的是在不影响输入输出的维度情况下，对输入进行形变，再通过ReLU进行非线性处理，提高决策函数的非线性。
3. 2个3x3卷积堆叠等于1个5x5卷积，3个3x3堆叠等于1个7x7卷积，感受野大小不变，而采用更多层、更小的卷积核可以引入更多非线性（更多的隐藏层，从而带来更多非线性函数），提高决策函数判决力，并且带来更少参数。
4. 每个VGG网络都有3个FC层，5个池化层，1个softmax层。
5. 在FC层中间采用dropout层，防止过拟合。

虽然 VGGNet 减少了卷积层参数，但实际上其参数空间比 AlexNet 大，其中绝大多数的参数都是来自于第一个全连接层，耗费更多计算资源。在随后的 NIN 中发现将这些全连接层替换为全局平均池化，对于性能影响不大，同时显著降低了参数数量。

采用 Pre-trained 方法训练的 VGG model（主要是 D 和 E），相对其他的方法参数空间很大，所以训练一个 VGG 模型通常要花费更长的时间，所幸有公开的 Pre-trained model 让我们很方便的使用。

#### GoogLeNet (2014)

图18-8是GoogLeNet的模型结构图。

<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/18/Net_GoogleNet.png" />

图18-8 GoogLeNet模型结构图

蓝色为卷积运算，红色为池化运算，黄色为softmax分类。

<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/18/Net_GoogleNet_Inception.png" />

图18-9 Inception结构图

图18-9的结构就是Inception，结构里的卷积stride都是1，另外为了保持特征响应图大小一致，都用了零填充。最后每个卷积层后面都立刻接了个ReLU层。在输出前有个叫concatenate的层，直译的意思是“并置”，即把4组不同类型但大小相同的特征响应图一张张并排叠起来，形成新的特征响应图。Inception结构里主要做了两件事：1. 通过3×3的池化、以及1×1、3×3和5×5这三种不同尺度的卷积核，一共4种方式对输入的特征响应图做了特征提取。2. 为了降低计算量。同时让信息通过更少的连接传递以达到更加稀疏的特性，采用1×1卷积核来实现降维。

这里想再详细谈谈1×1卷积核的作用，它究竟是怎么实现降维的。现在运算如下：下面图1是3×3卷积核的卷积，图2是1×1卷积核的卷积过程。对于单通道输入，1×1的卷积确实不能起到降维作用，但对于多通道输入，就不同了。假设你有256个特征输入，256个特征输出，同时假设Inception层只执行3×3的卷积。这意味着总共要进行 256×256×3×3的卷积（589000次乘积累加（MAC）运算）。这可能超出了我们的计算预算，比方说，在Google服务器上花0.5毫秒运行该层。作为替代，我们决定减少需要卷积的特征的数量，比如减少到64（256/4）个。在这种情况下，我们首先进行256到64的1×1卷积，然后在所有Inception的分支上进行64次卷积，接着再使用一个64到256的1×1卷积。

- 256×64×1×1 = 16000
- 64×64×3×3 = 36000
- 64×256×1×1 = 16000

现在的计算量大约是70000(即16000+36000+16000)，相比之前的约600000，几乎减少了10倍。这就通过小卷积核实现了降维。
现在再考虑一个问题：为什么一定要用1×1卷积核，3×3不也可以吗？考虑[50,200,200]的矩阵输入，我们可以使用20个1×1的卷积核进行卷积，得到输出[20,200,200]。有人问，我用20个3×3的卷积核不是也能得到[20,200,200]的矩阵输出吗，为什么就使用1×1的卷积核？我们计算一下卷积参数就知道了，对于1×1的参数总数：20×200×200×（1×1），对于3×3的参数总数：20×200×200×（3×3），可以看出，使用1×1的参数总数仅为3×3的总数的九分之一！所以我们使用的是1×1卷积核。

GoogLeNet网络结构中有3个LOSS单元，这样的网络设计是为了帮助网络的收敛。在中间层加入辅助计算的LOSS单元，目的是计算损失时让低层的特征也有很好的区分能力，从而让网络更好地被训练。在论文中，这两个辅助LOSS单元的计算被乘以0.3，然后和最后的LOSS相加作为最终的损失函数来训练网络。

GoogLeNet还有一个闪光点值得一提，那就是将后面的全连接层全部替换为简单的全局平均pooling，在最后参数会变的更少。而在AlexNet中最后3层的全连接层参数差不多占总参数的90%，使用大网络在宽度和深度允许GoogleNet移除全连接层，但并不会影响到结果的精度，在ImageNet中实现93.3%的精度，而且要比VGG还要快。

#### ResNets (2015)

图18-10是ResNets的模型结构。

<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/18/Net_ResNet.png" />

图18-10 ResNets模型结构图

ResNet称为残差网络。什么是残差呢？

若将输入设为X，将某一有参网络层设为H，那么以X为输入的此层的输出将为H(X)。一般的卷积神经网络网络如Alexnet/VGG等会直接通过训练学习出参数函数H的表达，从而直接学习X -> H(X)。
而残差学习则是致力于使用多个有参网络层来学习输入、输出之间的参差即H(X) - X即学习X -> (H(X) - X) + X。其中X这一部分为直接的identity mapping，而H(X) - X则为有参网络层要学习的输入输出间残差。

图18-11为残差学习这一思想的基本表示。

<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/18/Net_ResNet_flow.png" />

图18-11 残差结构示意图

图18-12展示了两种形态的残差模块，左图是常规残差模块，有两个3×3卷积核组成，但是随着网络进一步加深，这种残差结构在实践中并不是十分有效。针对这问题，右图的“瓶颈残差模块”（bottleneck residual block）可以有更好的效果，它依次由1×1、3×3、1×1这三个卷积层堆积而成，这里的1×1的卷积能够起降维或升维的作用，从而令3×3的卷积可以在相对较低维度的输入上进行，以达到提高计算效率的目的。

<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/18/Net_ResNet_block.png" />

图18-12 两种形态的残差模块

#### DenseNet (2017)

DenseNet$^{[9]}$ 是一种具有密集连接的卷积神经网络。在该网络中，任何两层之间都有直接的连接，也就是说，网络每一层的输入都是前面所有层输出的并集，而该层所学习的特征图也会被直接传给其后面所有层作为输入。下图是 DenseNet 的一个dense block示意图，一个block里面的结构如下，与ResNet中的BottleNeck基本一致：BN-ReLU-Conv(1×1)-BN-ReLU-Conv(3×3) ，而一个DenseNet则由多个这种block组成。每个DenseBlock的之间层称为transition layers，由BN−>Conv(1×1)−>averagePooling(2×2)组成。

图18-13是ResNets的模型结构。

<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/18/Net_DenseNet_1.png" ch="500" />

图18-13 DenseNet模型结构图

DenseNet作为另一种拥有较深层数的卷积神经网络，具有如下优点：

1. 相比ResNet拥有更少的参数数量
2. 旁路加强了特征的重用
3. 网络更易于训练,并具有一定的正则效果
4. 缓解了gradient vanishing和model degradation的问题

>>### 18.1 实现颜色的分类
### 用前馈神经网络解决问题
#### 数据处理

由于输入图片是三通道的彩色图片，我们先把它转换成灰度图，

```Python
class GeometryDataReader(DataReader_2_0):
    def ConvertToGray(self, data):
        (N,C,H,W) = data.shape
        new_data = np.empty((N,H*W))
        if C == 3: # color
            for i in range(N):
                new_data[i] = np.dot(
                    [0.299,0.587,0.114], 
                    data[i].reshape(3,-1)).reshape(1,784)
        elif C == 1: # gray
            new_data[i] = data[i,0].reshape(1,784)
        #end if
        return new_data
```

向量[0.299,0.587,0.114]的作用是，把三通道的彩色图片的RGB值与此向量相乘，得到灰度图，三个因子相加等于1，这样如果原来是[255,255,255]的话，最后的灰度图的值还是255。如果是[255,255,0]的话，最后的结果是：

$$
\begin{aligned}
Y &= 0.299 \cdot R + 0.586 \cdot G + 0.114 \cdot B \\
&= 0.299 \cdot 255 + 0.586 \cdot 255 + 0.114 \cdot 0 \\
&=225.675
\end{aligned}
\tag{1}
$$

也就是说粉色的数值本来是(255,255,0)，变成了单一的值225.675。六种颜色中的每一种都会有不同的值，所以即使是在灰度图中，也会保留部分“彩色”信息，当然会丢失一些信息。这从公式1中很容易看出来，假设$B=0$，不同组合的$R、G$的值有可能得到相同的最终结果，因此会丢失彩色信息。

在转换成灰度图后，立刻用reshape(1,784)把它转变成矢量，该矢量就是每个样本的784维的特征值。

### 1x1卷积

读者可能还记得在GoogLeNet的Inception模块中，有1x1的卷积核。这初看起来是一个非常奇怪的做法，因为1x1的卷积核基本上失去了卷积的作用，并没有建立在同一个通道上的相邻像素之间的相关性。

在本例中，为了识别颜色，我们也使用了1x1的卷积核，并且能够完成颜色分类的任务，这是为什么呢？

我们以三通道的数据举例。

<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/18/OneByOne.png" ch="500" />

图18-19 1x1卷积核的工作原理

假设有一个三通道的1x1的卷积核，其值为[1,2,-1]，则相当于把每个通道的同一位置的像素值乘以卷积核，然后把结果相加，作为输出通道的同一位置的像素值。以左上角的像素点为例：

$$
1 \times 1 + 1 \times 2 + 1 \times (-1)=2
$$

相当于把上图拆开成9个样本，其值为：

```
[1,1,1] # 左上角点
[3,3,0] # 中上点
[0,0,0] # 右上角点
[2,0,0] # 左中点
[0,1,1] # 中点
[4,2,1] # 右中点
[1,1,1] # 左下角点
[2,1,1] # 下中点
[0,0,0] # 右下角点
```

上述值排成一个9行3列的矩阵，然后与一个3行1列的向量$(1,2,-1)^T$相乘，得到9行1列的向量，然后再转换成3x3的矩阵。当然在实际过程中，这个1x1的卷积核的数值是学习出来的，而不是人为指定的。

这样做可以达到两个目的：

1. 跨通道信息整合
2. 降维以减少学习参数

所以1x1的卷积核关注的是不同通道的相同位置的像素之间的相关性，而不是同一通道内的像素的相关性，在本例中，意味着它关心的彩色通道信息，通过不同的卷积核，把彩色通道信息转变成另外一种表达方式，在保留原始信息的同时，还实现了降维。

###  颜色分类可视化解释

如果用人类的视觉神经系统做类比，两个1x1的卷积核可以理解为两只眼睛上的视网膜上的视觉神经细胞，把彩色信息转变成神经电信号传入大脑的过程。最后由全连接层做分类，相当于大脑中的视觉知识体系。

回到神经网络的问题上，只要ReLU的输出结果中仍然含有“颜色”信息（用假彩色图可以证明这一点），并且针对原始图像中的不同的颜色，会生成不同的假彩色图，最后的全连接网络就可以有分辨的能力。

举例来说，从图18-20看，第一行的红色到了第五行变成了黑色，绿色变成了淡绿色，等等，是一一对应的关系。如果红色和绿色都变成了黑色，那么将分辨不出区别来。

>>### 18.2 实现几何图形分类
### 形状分类可视化解释

<img src='../Images/18/shape_cnn_visualization.png'/>

图18-25 可视化解释

参看图18-25，表18-3解释了8个卷积核的作用。

表18-3 8个卷积核的作用

|卷积核序号|作用|直线|三角形|菱形|矩形|圆形|
|:--:|---|:--:|:--:|:--:|:--:|:--:|
|1|左侧边缘|0|1|0|1|1|
|2|大色块区域|0|1|1|1|1|
|3|左上侧边缘|0|1|1|0|1|
|4|45度短边|1|1|1|0|1|
|5|右侧边缘、上横边|0|0|0|1|1|
|6|左上、右上、右下|0|1|1|0|1|
|7|左边框和右下角|0|0|0|1|1|
|8|左上和右下，及背景|0|0|1|0|1|

表18-3中，左侧为卷积核的作用，右侧为某个特征对于5种形状的判别力度，0表示该特征无法找到，1表示可以找到该特征。

1. 比如第一个卷积核，其作用为判断是否有左侧边缘，那么第一行的数据为[0,1,0,1,1]，表示对直线和菱形来说，没有左侧边缘特征，而对于三角形、矩形、圆形来说，有左侧边缘特征。这样的话，就可以根据这个特征把5种形状分为两类：

   - A类有左侧边缘特征：三角形、矩形、圆形
   - B类无左侧边缘特征：直线、菱形

2. 再看第二个卷积核，是判断是否有大色块区域的，只有直线没有该特征，其它4种形状都有。那么看第1个特征的B类种，包括直线、菱形，则第2个特征就可以把直线和菱形分开了。

3. 然后我们只关注A类形状，看第三个卷积核，判断是否有左上侧边缘，对于三角形、矩形、圆形的取值为[1,0,1]，即矩形没有左上侧边缘，这样就可以把矩形从A类中分出来。

4. 对于三角形和圆形，卷积核5、7、8都可以给出不同的值，这就可以把二者分开了。

当然，神经网络可能不是按照我们分析的顺序来判定形状的，这只是其中的一种解释路径，还可以有很多其它种路径的组合，但最终总能够把5种形状分开来。

>>### 18.3 实现几何图形和颜色分类

### 用前馈神经网络解决问题

我们仍然先使用全连接网络来解决这个问题，搭建一个三层的网络如下：

```Python
ef dnn_model():
    num_output = 9
    max_epoch = 50
    batch_size = 16
    learning_rate = 0.01
    params = HyperParameters_4_2(
        learning_rate, max_epoch, batch_size,
        net_type=NetType.MultipleClassifier,
        init_method=InitialMethod.MSRA,
        optimizer_name=OptimizerName.Momentum)

    net = NeuralNet_4_2(params, "color_shape_dnn")
    
    f1 = FcLayer_2_0(784, 128, params)
    net.add_layer(f1, "f1")
    r1 = ActivationLayer(Relu())
    net.add_layer(r1, "relu1")

    f2 = FcLayer_2_0(f1.output_size, 64, params)
    net.add_layer(f2, "f2")
    r2 = ActivationLayer(Relu())
    net.add_layer(r2, "relu2")
    
    f3 = FcLayer_2_0(f2.output_size, num_output, params)
    net.add_layer(f3, "f3")
    s3 = ClassificationLayer(Softmax())
    net.add_layer(s3, "s3")

    return net
```

样本数据为3x28x28的彩色图，所以我们要把它转换成灰度图，然后再展开成1x784的向量，第一层用128个神经元，第二层用64个神经元，输出层用9个神经元接Softmax分类函数。

训练50个epoch后可以得到如下如图18-27所示的训练结果。

<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/18/shape_color_dnn_loss.png" />

图18-27 训练过程中损失函数值和准确度的变化

```
......
epoch=49, total_iteration=15199
loss_train=0.003370, accuracy_train=1.000000
loss_valid=0.510589, accuracy_valid=0.883333
time used: 25.34346342086792
testing...
0.9011111111111111
load parameters
0.8988888888888888
```

在测试集上得到的准确度是89%，这已经超出笔者的预期了，本来猜测准确度会小于80%。有兴趣的读者可以再精调一下这个前馈神经网络网络，看看是否可以得到更高的准确度。

图18-28是部分测试集中的测试样本的预测结果。

<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/18/shape_color_dnn_result.png" ch="500" />

图18-28 测试结果

绝大部分样本预测是正确的，但是第3行第2列的样本，应该是green-rect，被预测成green-circle；最后两行的两个green-tri也被预测错了形状，颜色并没有错。

>>### 18.4 MNIST分类
###  模型搭建

在12.1中，我们用一个三层的神经网络解决MNIST问题，并得到了97.49%的准确率。当时使用的模型如图18-31。

<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/12/nn3.png" ch="500" />

图18-31 前馈神经网络模型解决MNIST问题

这一节中，我们将学习如何使用卷积网络来解决MNIST问题。首先搭建模型如图18-32。

<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/18/mnist_net.png" />

图18-32 卷积神经网络模型解决MNIST问题

表18-5展示了模型中各层的功能和参数。

表18-5 模型中各层的功能和参数

|Layer|参数|输入|输出|参数个数|
|---|---|---|---|---|
|卷积层|8x5x5,s=1|1x28x28|8x24x24|200+8|
|激活层|2x2,s=2, max|8x24x24|8x24x24||
|池化层|Relu|8x24x24|8x12x12||
|卷积层|16x5x5,s=1|8x12x12|16x8x8|400+16|
|激活层|Relu|16x8x8|16x8x8||
|池化层|2x2, s=2, max|16x8x8|16x4x4||
|全连接层|256x32|256|32|8192+32|
|批归一化层||32|32||
|激活层|Relu|32|32||
|全连接层|32x10|32|10|320+10|
|分类层|softmax,10|10|10|

卷积核的大小如何选取呢？大部分卷积神经网络都会用1、3、5、7的方式递增，还要注意在做池化时，应该尽量让输入的矩阵尺寸是偶数，如果不是的话，应该在上一层卷积层加padding，使得卷积的输出结果矩阵的宽和高为偶数。

>>### 18.5 Fashion-MNIST分类
### 提出问题

MNIST手写识别数据集，对卷积神经网络来说已经太简单了，于是科学家们增加了图片的复杂度，用10种物品代替了10个数字，图18-36是它们的部分样本。

<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/18/FashionMnistSample.png" ch="500" />

图18-36 部分样本图展示

每3行是一类样本，按样本类别（从0开始计数）分行显示：

0. T-Shirt，T恤衫（1-3行）
1. Trouser，裤子（4-6行）
2. Pullover，套头衫（7-9行）
3. Dress，连衣裙（10-12行）
4. Coat，外套（13-15行）
5. Sandal，凉鞋（16-18行）
6. Shirt，衬衫（19-21行）
7. Sneaker，运动鞋（22-24行）
8. Bag，包（25-27行）
9. Ankle Boot，短靴（28-30行）

### 用前馈神经网络来解决问题

#### 搭建模型

```Python
def dnn_model():
    num_output = 10
    max_epoch = 10
    batch_size = 128
    learning_rate = 0.1
    params = HyperParameters_4_2(
        learning_rate, max_epoch, batch_size,
        net_type=NetType.MultipleClassifier,
        init_method=InitialMethod.MSRA,
        optimizer_name=OptimizerName.Momentum)

    net = NeuralNet_4_2(params, "fashion_mnist_dnn")
    
    f1 = FcLayer_2_0(784, 128, params)
    net.add_layer(f1, "f1")
    bn1 = BnLayer(f1.output_size)
    net.add_layer(bn1, "bn1")
    r1 = ActivationLayer(Relu())
    net.add_layer(r1, "relu1")

    f2 = FcLayer_2_0(f1.output_size, 64, params)
    net.add_layer(f2, "f2")
    bn2 = BnLayer(f2.output_size)
    net.add_layer(bn2, "bn2")
    r2 = ActivationLayer(Relu())
    net.add_layer(r2, "relu2")
    
    f3 = FcLayer_2_0(f2.output_size, num_output, params)
    net.add_layer(f3, "f3")
    s3 = ClassificationLayer(Softmax())
    net.add_layer(s3, "s3")

    return net
```

#### 训练结果

训练10个epoch后得到如图18-37所示曲线，可以看到网络能力已经接近极限了，再训练下去会出现过拟合现象，准确度也不一定能提高。

<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/18/FashionMnistLoss_dnn.png" />

图18-37 训练过程中损失函数值和准确度的变化

图18-38是在测试集上的预测结果。

<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/18/FashionMnistResult_dnn.png" ch="555" />

图18-38 测试结果

凡是类别名字前面带*号的，表示预测错误，比如第3行第1列，本来应该是第7类“运动鞋”，却被预测成了“凉鞋”。

### 用卷积神经网络来解决问题

#### 搭建模型

```Python
def cnn_model():
    num_output = 10
    max_epoch = 10
    batch_size = 128
    learning_rate = 0.01
    params = HyperParameters_4_2(
        learning_rate, max_epoch, batch_size,
        net_type=NetType.MultipleClassifier,
        init_method=InitialMethod.Xavier,
        optimizer_name=OptimizerName.Momentum)

    net = NeuralNet_4_2(params, "fashion_mnist_conv_test")
    
    c1 = ConvLayer((1,28,28), (32,3,3), (1,0), params)
    net.add_layer(c1, "c1")
    r1 = ActivationLayer(Relu())
    net.add_layer(r1, "relu1")
    p1 = PoolingLayer(c1.output_shape, (2,2), 2, PoolingTypes.MAX)
    net.add_layer(p1, "p1") 

    f3 = FcLayer_2_0(p1.output_size, 128, params)
    net.add_layer(f3, "f3")
    r3 = ActivationLayer(Relu())
    net.add_layer(r3, "relu3")

    f4 = FcLayer_2_0(f3.output_size, 10, params)
    net.add_layer(f4, "f4")
    s4 = ClassificationLayer(Softmax())
    net.add_layer(s4, "s4")

    return net
```

此模型只有一层卷积层，使用了32个卷积核，尺寸为3x3，后接最大池化层，然后两个全连接层。

#### 训练结果

训练10个epoch后得到如图18-39的曲线。

<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/18/FashionMnistLoss_cnn.png" />

图18-39 训练过程中损失函数值和准确度的变化

在测试集上得到91.12%的准确率，在测试集上的前几个样本的预测结果如图18-40所示。

<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/18/FashionMnistResult_cnn.png" ch="555" />

图18-40 测试结果

与前馈神经网络方案相比，这32个样本里只有一个错误，第4行最后一列，把第9类“短靴”预测成了“凉鞋”，因为这个样本中间有一个三角形的黑色块，与凉鞋的镂空设计很像。


>>### 18.6 Cifar-10分类
Cifar 是加拿大政府牵头投资的一个先进科学项目研究所。Hinton、Bengio和他的学生在2004年拿到了 Cifar 投资的少量资金，建立了神经计算和自适应感知项目。这个项目结集了不少计算机科学家、生物学家、电气工程师、神经科学家、物理学家、心理学家，加速推动了 Deep Learning 的进程。从这个阵容来看，DL 已经和 ML 系的数据挖掘分的很远了。Deep Learning 强调的是自适应感知和人工智能，是计算机与神经科学交叉；Data Mining 强调的是高速、大数据、统计数学分析，是计算机和数学的交叉。

Cifar-10 是由 Hinton 的学生 Alex Krizhevsky、Ilya Sutskever 收集的一个用于普适物体识别的数据集。

### 提出问题

图18-41是Cifar-10的样本数据。

<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/18/cifar10_sample.png" ch="500" />

图18-41 Cifar-10样本数据

1. airplane，飞机，6000张
2. automobile，汽车，6000张
3. bird，鸟，6000张
4. cat，猫，6000张
5. deer，鹿，6000张
6. dog，狗，6000张
7. frog，蛙，6000张
8. horse，马，6000张
9. ship，船，6000张
10. truck，卡车，6000张

Cifar-10 由60000张32*32的 RGB 彩色图片构成，共10个分类。50000张训练，10000张测试。分为6个文件，5个训练数据文件，每个文件中包含10000张图片，随机打乱顺序，1个测试数据文件，也是10000张图片。这个数据集最大的特点在于将识别迁移到了普适物体，而且应用于多分类（姊妹数据集Cifar-100达到100类，ILSVRC比赛则是1000类）。

但是，面对彩色数据集，用CPU做训练所花费的时间实在是太长了，所以本节将学习如何使用GPU来训练神经网络。

### 环境搭建

我们将使用Keras$^{[1]}$来训练模型，因为Keras是一个在TensorFlow平台上经过抽象的工具，它的抽象思想与我们在前面学习过的各种Layer的概念完全一致，有利于读者在前面的基础上轻松地继续学习。环境搭建有很多细节，我们在这里不详细介绍，只是把基本步骤列出。

1. 安装Python 3.6（本书中所有案例在Python 3.6上开发测试）
2. 安装CUDA（没有GPU的读者请跳过）
3. 安装cuDNN（没有GPU的读者请跳过）
4. 安装TensorFlow，有GPU硬件的一定要按照GPU版，没有的只能安装CPU版
5. 安装Keras

安装好后用pip list看一下，关键的几个包是：

```
Package              Version
-------------------- ---------
Keras                2.2.5
Keras-Applications   1.0.8
Keras-Preprocessing  1.1.0
matplotlib           3.1.1
numpy                1.17.0
tensorboard          1.13.1
tensorflow-estimator 1.13.0
tensorflow-gpu       1.13.1
```

如果没有GPU，则"tensorflow-gpu"一项会是"tensorflow"。


>>>># 代码实现
#### 17.0
[![DoNkGj.png](https://s3.ax1x.com/2020/12/02/DoNkGj.png)](https://imgchr.com/i/DoNkGj)
[![DoNZMq.png](https://s3.ax1x.com/2020/12/02/DoNZMq.png)](https://imgchr.com/i/DoNZMq)

#### 17.2
[![DoN8zR.png](https://s3.ax1x.com/2020/12/02/DoN8zR.png)](https://imgchr.com/i/DoN8zR)

[![DoNBJH.png](https://s3.ax1x.com/2020/12/02/DoNBJH.png)](https://imgchr.com/i/DoNBJH)

#### 17.3
[![DoUJhQ.png](https://s3.ax1x.com/2020/12/02/DoUJhQ.png)](https://imgchr.com/i/DoUJhQ)
[![DoUtpj.png](https://s3.ax1x.com/2020/12/02/DoUtpj.png)](https://imgchr.com/i/DoUtpj)

#### 17.4
[![DoUwn0.png](https://s3.ax1x.com/2020/12/02/DoUwn0.png)](https://imgchr.com/i/DoUwn0)

#### 17.5
[![DoUg39.png](https://s3.ax1x.com/2020/12/02/DoUg39.png)](https://imgchr.com/i/DoUg39)

#### 18.1
[![DoaVbV.png](https://s3.ax1x.com/2020/12/02/DoaVbV.png)](https://imgchr.com/i/DoaVbV)
[![Doan5F.png](https://s3.ax1x.com/2020/12/03/Doan5F.png)](https://imgchr.com/i/Doan5F)
[![DoalvR.png](https://s3.ax1x.com/2020/12/03/DoalvR.png)](https://imgchr.com/i/DoalvR)

#### 18.4
[![DodRw6.png](https://s3.ax1x.com/2020/12/03/DodRw6.png)](https://imgchr.com/i/DodRw6)
[![DodTld.png](https://s3.ax1x.com/2020/12/03/DodTld.png)](https://imgchr.com/i/DodTld)

#### 18.5
CNN
[![DoBTl4.png](https://s3.ax1x.com/2020/12/03/DoBTl4.png)](https://imgchr.com/i/DoBTl4)
[![DoBHX9.png](https://s3.ax1x.com/2020/12/03/DoBHX9.png)](https://imgchr.com/i/DoBHX9)
DNN
[![DoDn1g.png](https://s3.ax1x.com/2020/12/03/DoDn1g.png)](https://imgchr.com/i/DoDn1g)
[![DoDKXj.png](https://s3.ax1x.com/2020/12/03/DoDKXj.png)](https://imgchr.com/i/DoDKXj)

#### 18.6
[![DoreV1.png](https://s3.ax1x.com/2020/12/03/DoreV1.png)](https://imgchr.com/i/DoreV1)

>>>># 心得体会
通过这张的学习，让我了解到卷积神经网络是深度学习中的一个里程碑式的技术，有了这个技术，才会让计算机有能力理解图片和视频信息，才会有计算机视觉的众多应用。通过代码实现一些实际数据，对前向计算、卷积的反向传播、池化的前向计算与反向传播又了一定的认识。