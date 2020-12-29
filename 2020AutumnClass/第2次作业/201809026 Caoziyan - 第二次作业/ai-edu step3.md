# 六、 多入单出的单层神经网路 - 线性二分类

1.直观理解线性二分类与非线性二分类的区别

|线性二分类|非线性二分类|
|---|---|
|<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/6/linear_binary.png"/>|<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/10/non_linear_binary.png"/>|

2.对率函数Logistic Function，即可以做为激活函数使用，又可以当作二分类函数使用。

- Logistic函数公式

$$Logistic(z) = \frac{1}{1 + e^{-z}}$$

以下记 $a=Logistic(z)$。

- 导数

$$Logistic'(z) = a(1 - a)$$

- 输入值域

$$(-\infty, \infty)$$

- 输出值域

$$(0,1)$$

- 函数图像

<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/8/logistic.png" ch="500" />

3.正向传播

- 矩阵运算

$$
z=x \cdot w + b \tag{1}
$$

- 分类计算

$$
a = Logistic(z)=\frac{1}{1 + e^{-z}} \tag{2}
$$

- 损失函数计算

二分类交叉熵损失函数：

$$
loss(w,b) = -[y \ln a+(1-y) \ln(1-a)] \tag{3}
$$

4.反向传播
#### 求损失函数对 $a$ 的偏导

$$
\frac{\partial loss}{\partial a}=-\left[\frac{y}{a}-\frac{1-y}{1-a}\right]=\frac{a-y}{a(1-a)} \tag{4}
$$

- 求 $a$ 对 $z$ 的偏导

$$
\frac{\partial a}{\partial z}= a(1-a) \tag{5}
$$

- 求误差 $loss$ 对 $z$ 的偏导

$$
\frac{\partial loss}{\partial z}=\frac{\partial loss}{\partial a}\frac{\partial a}{\partial z}=\frac{a-y}{a(1-a)} \cdot a(1-a)=a-y \tag{6}
$$

满足以下条件：

1. 损失函数满足二分类的要求，无论是正例还是反例，都是单调的；
2. 损失函数可导，以便于使用反向传播算法；
3. 计算过程非常简单，一个减法就可以搞定。

5.定义神经网络结构

完成二分类任务的神经元结构

<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/6/BinaryClassifierNN.png" ch="500" />

与上一章的网络结构图的区别：在神经元输出时使用了分类函数，所以输出为 $A$，而不是以往直接输出的 $Z$。

#- 输入层

输入经度 $x_1$ 和纬度 $x_2$ 两个特征：

$$
X=\begin{pmatrix}
x_{1} & x_{2}
\end{pmatrix}
$$

- 权重矩阵

输入是2个特征，输出一个数，则 $W$ 的尺寸就是 $2\times 1$：

$$
W=\begin{pmatrix}
w_{1} \\\\ w_{2}
\end{pmatrix}
$$

$B$ 的尺寸是 $1\times 1$，行数永远是1，列数永远和 $W$ 一样。

$$
B=\begin{pmatrix}
b
\end{pmatrix}
$$

- 输出层

$$
\begin{aligned}    
z &= X \cdot W + B
=\begin{pmatrix}
    x_1 & x_2
\end{pmatrix}
\begin{pmatrix}
    w_1 \\\\ w_2
\end{pmatrix} + b \\\\
&=x_1 \cdot w_1 + x_2 \cdot w_2 + b 
\end{aligned}
\tag{1}
$$
$$a = Logistic(z) \tag{2}$$

- 损失函数

二分类交叉熵损失函数：

$$
loss(W,B) = -[y\ln a+(1-y)\ln(1-a)] \tag{3}
$$

6.线性回归和线性分类的比较

||线性回归|线性分类|
|---|---|---|
|相同点|需要在样本群中找到一条直线|需要在样本群中找到一条直线|
|不同点|用直线来拟合所有样本，使得各个样本到这条直线的距离尽可能最短|用直线来分割所有样本，使得正例样本和负例样本尽可能分布在直线两侧|

7.二分类的代数原理

代数方式：通过一个分类函数计算所有样本点在经过线性变换后的概率值，使得正例样本的概率大于0.5，而负例样本的概率小于0.5。

8.二分类的几何原理

几何方式：让所有正例样本处于直线的一侧，所有负例样本处于直线的另一侧，直线尽可能处于两类样本的中间。

- 二分类函数的几何作用

二分类函数的最终结果是把正例都映射到图中的上半部分的曲线上，而把负类都映射到下半部分的曲线上。

<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/6/sigmoid_binary.png"/>

$Logistic$ 函数把输入的点映射到 $(0,1)$ 区间内实现分类

9.五种逻辑门的结果比较

|逻辑门|分类结果|参数值|
|---|---|---|
|非|<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images\Images\6\LogicNotResult.png" width="300" height="300">|W=-12.468<br/>B=6.031|
|与|<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images\Images\6\LogicAndGateResult.png" width="300" height="300">|W1=11.757<br/>W2=11.757<br/>B=-17.804|
|与非|<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images\Images\6\LogicNandGateResult.png" width="300" height="300">|W1=-11.763<br/>W2=-11.763<br/>B=17.812|
|或|<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images\Images\6\LogicOrGateResult.png" width="300" height="300">|W1=11.743<br/>W2=11.743<br/>B=-11.738|
|或非|<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images\Images\6\LogicNorGateResult.png" width="300" height="300">|W1=-11.738<br/>W2=-11.738<br/>B=5.409|


我们从数值和图形可以得到两个结论：

1. `W1`和`W2`的值基本相同而且符号相同，说明分割线一定是135°斜率
2. 精度越高，则分割线的起点和终点越接近四边的中点0.5的位置

以上两点说明神经网络还是很聪明的，它会尽可能优美而鲁棒地找出那条分割线。
