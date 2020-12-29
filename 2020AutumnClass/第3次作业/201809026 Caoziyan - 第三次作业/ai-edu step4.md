# 八、 激活函数

1.激活函数的基本作用

激活函数的基本性质：

+ 非线性：线性的激活函数和没有激活函数一样；
+ 可导性：做误差反向传播和梯度下降，必须要保证激活函数的可导性；
+ 单调性：单一的输入会得到单一的输出，较大值的输入得到较大值的输出。

何时会用到激活函数

1. 神经网络最后一层不需要激活函数
2. 激活函数只用于连接前后两层神经网络

2.挤压型激活函数（饱和型激活函数）

特点：当输入值域的绝对值较大的时候，其输出在两端是饱和的，都具有S形的函数曲线以及压缩输入值域的作用

(1)Logistic函数（对数几率函数，简称对率函数）

凡是用到“Logistic”词汇的，指的是二分类函数；而用到“Sigmoid”词汇的，指的是本激活函数。

>公式

$$Sigmoid(z) = \frac{1}{1 + e^{-z}} \rightarrow a \tag{1}$$

>导数

$$Sigmoid'(z) = a(1 - a) \tag{2}$$

>值域

- 输入值域：$(-\infty, \infty)$
- 输出值域：$(0,1)$
- 导数值域：$(0,0.25]$

>函数图像

<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/8/sigmoid.png" ch="500" />

>优点

从函数图像来看，Sigmoid函数的作用是将输入压缩到 $(0,1)$ 这个区间范围内，这种输出在0~1之间的函数可以用来模拟一些概率分布的情况。它还是一个连续函数，导数简单易求。  

从数学上来看，Sigmoid函数对中央区的信号增益较大，对两侧区的信号增益小，在信号的特征空间映射上，有很好的效果。 

从神经科学上来看，中央区酷似神经元的兴奋态，两侧区酷似神经元的抑制态，因而在神经网络学习方面，可以将重点特征推向中央区，将非重点特征推向两侧区。

>缺点

指数计算代价大。

反向传播时梯度消失：从梯度图像中可以看到，Sigmoid的梯度在两端都会接近于0，根据链式法则，如果传回的误差是$\delta$，那么梯度传递函数是$\delta \cdot a'$，而$a'$这时接近零，也就是说整体的梯度也接近零。这就出现梯度消失的问题，并且这个问题可能导致网络收敛速度比较慢。

(2)Tanh函数（双曲正切函数）

>公式  
$$Tanh(z) = \frac{e^{z} - e^{-z}}{e^{z} + e^{-z}} = (\frac{2}{1 + e^{-2z}}-1) \rightarrow a \tag{3}$$
即
$$Tanh(z) = 2 \cdot Sigmoid(2z) - 1 \tag{4}$$

>导数公式

$$Tanh'(z) = (1 + a)(1 - a)$$

>值域

- 输入值域：$(-\infty,\infty)$
- 输出值域：$(-1,1)$
- 导数值域：$(0,1)$

>函数图像

<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/8/tanh.png" ch="500" />

>优点

具有Sigmoid的所有优点。

>缺点

exp指数计算代价大。梯度消失问题仍然存在。

比起Sigmoid，Tanh减少了一个缺点，就是他本身是零均值的，也就是说，在传递过程中，输入数据的均值并不会发生改变，这就使他在很多应用中能表现出比Sigmoid优异一些的效果。

3.半线性激活函数（非饱和型激活函数）

（1）ReLU函数 （修正线性单元，线性整流函数，斜坡函数）

>公式

$$ReLU(z) = max(0,z) = \begin{cases} 
  z, & z \geq 0 \\\\ 
  0, & z < 0 
\end{cases}$$

>导数

$$ReLU'(z) = \begin{cases} 1 & z \geq 0 \\\\ 0 & z < 0 \end{cases}$$

>值域

- 输入值域：$(-\infty, \infty)$
- 输出值域：$(0,\infty)$
- 导数值域：$\\{0,1\\}$

<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/8/relu.png"/>

>仿生学原理

相关大脑方面的研究表明生物神经元的信息编码通常是比较分散及稀疏的。通常情况下，大脑中在同一时间大概只有1%~4%的神经元处于活跃状态。使用线性修正以及正则化可以对机器神经网络中神经元的活跃度（即输出为正值）进行调试；相比之下，Sigmoid函数在输入为0时输出为0.5，即已经是半饱和的稳定状态，不够符合实际生物学对模拟神经网络的期望。不过需要指出的是，一般情况下，在一个使用修正线性单元（即线性整流）的神经网络中大概有50%的神经元处于激活态。

>优点

- 反向导数恒等于1，更加有效率的反向传播梯度值，收敛速度快；
- 避免梯度消失问题；
- 计算简单，速度快；
- 活跃度的分散性使得神经网络的整体计算成本下降。

>缺点

无界。

（2）Leaky ReLU函数（带泄露的线性整流函数）

>公式

$$LReLU(z) = \begin{cases} z & z \geq 0 \\\\ \alpha \cdot z & z < 0 \end{cases}$$

>导数

$$LReLU'(z) = \begin{cases} 1 & z \geq 0 \\\\ \alpha & z < 0 \end{cases}$$

>值域

输入值域：$(-\infty, \infty)$

输出值域：$(-\infty,\infty)$

导数值域：$\\{\alpha,1\\}$

>函数图像

<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/8/leakyRelu.png"/>

>优点

继承了ReLU函数的优点。

Leaky ReLU同样有收敛快速和运算复杂度低的优点，而且由于给了$z<0$时一个比较小的梯度$\alpha$,使得$z<0$时依旧可以进行梯度传递和更新，可以在一定程度上避免神经元“死”掉的问题。

（3）Softplus函数

>公式

$$Softplus(z) = \ln (1 + e^z)$$

>导数

$$Softplus'(z) = \frac{e^z}{1 + e^z}$$

>值域

输入值域：$(-\infty, \infty)$

输出值域：$(0,\infty)$

导数值域：$(0,1)$

>函数图像

<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/8/softplus.png"/>

（4）ELU函数

>公式

$$ELU(z) = \begin{cases} z & z \geq 0 \\ \alpha (e^z-1) & z < 0 \end{cases}$$

>导数

$$ELU'(z) = \begin{cases} 1 & z \geq 0 \\ \alpha e^z & z < 0 \end{cases}$$

>值域

输入值域：$(-\infty, \infty)$

输出值域：$(-\alpha,\infty)$

导数值域：$(0,1]$

>函数图像

<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/8/elu.png"/>

# 九、单入单出的双层神经网络 - 非线性回归

1.多项式回归

>一元一次线性模型

$$z = x w + b \tag{1}$$

>多元一次多项式

$$z = x_1 w_1 + x_2 w_2 + ...+ x_m w_m + b \tag{2}$$

这里的多变量，是指样本数据的特征值为多个，上式中的 $x_1,x_2,...,x_m$ 代表了m个特征值。

>一元多次多项式

$$z = x w_1 + x^2 w_2 + ... + x^m w_m + b \tag{3}$$

上式中x是原有的唯一特征值，$x^m$ 是利用 $x$ 的 $m$ 次方作为额外的特征值，这样就把特征值的数量从 $1$ 个变为 $m$ 个。

>多元多次多项式

2.验证与测试

>训练集

Training Set，用于模型训练的数据样本。

>验证集

Validation Set，或者叫做Dev Set，是模型训练过程中单独留出的样本集，它可以用于调整模型的超参数和用于对模型的能力进行初步评估。
  
在神经网络中，验证数据集用于：

- 寻找最优的网络深度
- 或者决定反向传播算法的停止点
- 或者在神经网络中选择隐藏层神经元的数量
- 在普通的机器学习中常用的交叉验证（Cross Validation）就是把训练数据集本身再细分成不同的验证数据集去训练模型。

>测试集

Test Set，用来评估最终模型的泛化能力。但不能作为调参、选择特征等算法相关的选择的依据。

三者之间的关系

<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/9/dataset.png" />

3.双层神经网络实现非线性回归

（1）万能近似定理

万能近似定理(universal approximation theorem) $^{[1]}$，是深度学习最根本的理论依据。它证明了在给定网络具有足够多的隐藏单元的条件下，配备一个线性输出层和一个带有任何“挤压”性质的激活函数（如Sigmoid激活函数）的隐藏层的前馈神经网络，能够以任何想要的误差量近似任何从一个有限维度的空间映射到另一个有限维度空间的Borel可测的函数。

前馈网络的导数也可以以任意好地程度近似函数的导数。

万能近似定理其实说明了理论上神经网络可以近似任何函数。但实践上我们不能保证学习算法一定能学习到目标函数。即使网络可以表示这个函数，学习也可能因为两个不同的原因而失败：

1. 用于训练的优化算法可能找不到用于期望函数的参数值；
2. 训练算法可能由于过拟合而选择了错误的函数。

（2）单入单出的双层神经网络

定义一个两层的神经网络，输入层不算，一个隐藏层，含3个神经元，一个输出层。

<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/9/nn.png" />

输入层就是一个标量x值，如果是成批输入，则是一个矢量或者矩阵，但是特征值数量总为1，因为只有一个横坐标值做为输入。

$$X = (x)$$

>权重矩阵W1/B1

$$
W1=
\begin{pmatrix}
w1_{11} & w1_{12} & w1_{13}
\end{pmatrix}
$$

$$
B1=
\begin{pmatrix}
b1_{1} & b1_{2} & b1_{3} 
\end{pmatrix}
$$

>隐层

我们用3个神经元：

$$
Z1 = \begin{pmatrix}
    z1_1 & z1_2 & z1_3
\end{pmatrix}
$$

$$
A1 = \begin{pmatrix}
    a1_1 & a1_2 & a1_3
\end{pmatrix}
$$

>权重矩阵W2/B2

W2的尺寸是3x1，B2的尺寸是1x1。

$$
W2=
\begin{pmatrix}
w2_{11} \\\\
w2_{21} \\\\
w2_{31}
\end{pmatrix}
$$

$$
B2=
\begin{pmatrix}
b2_{1}
\end{pmatrix}
$$

>输出层

由于我们只想完成一个拟合任务，所以输出层只有一个神经元，尺寸为1x1：

$$
Z2 = 
\begin{pmatrix}
    z2_{1}
\end{pmatrix}
$$

（3）前向计算

前向计算图。

<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/9/forward.png" />

>隐层

- 线性计算

$$
z1_{1} = x \cdot w1_{11} + b1_{1}
$$

$$
z1_{2} = x \cdot w1_{12} + b1_{2}
$$

$$
z1_{3} = x \cdot w1_{13} + b1_{3}
$$

矩阵形式：

$$
\begin{aligned}
Z1 &=x \cdot 
\begin{pmatrix}
    w1_{11} & w1_{12} & w1_{13}
\end{pmatrix}
+
\begin{pmatrix}
    b1_{1} & b1_{2} & b1_{3}
\end{pmatrix}
 \\\\
&= X \cdot W1 + B1  
\end{aligned} \tag{1}
$$

- 激活函数

$$
a1_{1} = Sigmoid(z1_{1})
$$

$$
a1_{2} = Sigmoid(z1_{2})
$$

$$
a1_{3} = Sigmoid(z1_{3})
$$

矩阵形式：

$$
A1 = Sigmoid(Z1) \tag{2}
$$

>输出层

由于我们只想完成一个拟合任务，所以输出层只有一个神经元：

$$
\begin{aligned}
Z2&=a1_{1}w2_{11}+a1_{2}w2_{21}+a1_{3}w2_{31}+b2_{1} \\\\
&= 
\begin{pmatrix}
a1_{1} & a1_{2} & a1_{3}
\end{pmatrix}
\begin{pmatrix}
w2_{11} \\\\ w2_{21} \\\\ w2_{31}
\end{pmatrix}
+b2_1 \\\\
&=A1 \cdot W2+B2
\end{aligned} \tag{3}
$$

>损失函数

均方差损失函数：

$$loss(w,b) = \frac{1}{2} (z2-y)^2 \tag{4}$$

其中，$z2$是预测值，$y$是样本的标签值。

4.曲线拟合

（1）正弦曲线的拟合

>隐层只有一个神经元的情况

令`n_hidden=1`，并指定模型名称为`sin_111`。

训练过程中损失函数值和准确率的变化

<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/9/sin_loss_1n.png" />

一个神经元的拟合效果

<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/9/sin_result_1n.png" ch="500" />

一个神经元的拟合效果

>隐层有两个神经元的情况

两个神经元的训练过程中损失函数值和准确率的变化

<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/9/sin_loss_2n.png"/>

两个神经元的拟合效果

<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/9/sin_result_2n.png"/>

（2）复合函数的拟合

>隐层只有两个神经元的情况

两个神经元的拟合效果

<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/9/complex_result_2n.png" ch="500" />

上图是两个神经元的拟合效果图，拟合情况很不理想，和正弦曲线只用一个神经元的情况类似。观察打印输出的损失值，有波动，久久徘徊在0.003附近不能下降，说明网络能力不够。

>隐层有三个神经元的情况

三个神经元的训练过程中损失函数值和准确率的变化

<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/9/complex_loss_3n.png" />

拟合效果

<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/9/complex_result_3n.png"/>
