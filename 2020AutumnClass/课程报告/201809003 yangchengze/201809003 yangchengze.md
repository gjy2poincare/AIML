# 第一章学习笔记
* * * 
## 神经网络的基本工作原理
> 神经网络的一个重要特性是它能够从环境中学习,并把学习的结果分布存储于网络的突触连接中。  
## 神经元细胞的数学模型
- 输入 input：是外界输入信号，一般是一个训练数据样本的多个属性，   一个神经元可以有多个输入。
- 输出 output：一个神经元只能有一个输出。
- 权重 weights： 是每个输入信号的权重值， 一个神经元的w的数量和输入的数量一致。 
- 偏移 bias：b是临界值。    
- 激活函数 activation：求和之后，神经细胞已经处于兴奋状态了，已经决定要向下一个神经元传递信号了，但是要传递多强烈的信号，要由激活函数来确定。
**回归/拟合 Regression/fitting**
- **分类 Classification**
### 激活函数的作用  
如果不用激活函数，每一层输出都是上层输入的线性函数，无论神经网络有多少层，输出都是输入的线性组合，这种情况就是最原始的感知机。  
如果使用的话，激活函数给神经元引入了非线性因素，使得神经网络可以任意逼近任何非线性函数，这样神经网络就可以应用到众多的非线性模型中。
## 为什么需要深度神经网络与深度学习
通常我们把三层以上的网络称为深度神经网络。两层的神经网络虽然强大，但可能只能完成二维空间上的一些拟合与分类的事情。如果对于图片、语音、文字序列些复杂的事情，就需要更复杂的网络来理解和处理。第一个方式是增加每一层中神经元的数量，但这是线性的，不够有效。另外一个方式是增加层的数量，每一层都处理不同的事情。
1. 卷积神经网络 CNN (Convolutional Neural Networks)

对于图像类的机器学习问题，最有效的就是卷积神经网络。
2. 循环神经网络 RNN (Recurrent Neural Networks)

对于语言类的机器学习问题，最有效的就是循环神经网络。
函数的导数，沿梯度最小方向将误差回传，修正前向计算公式中的各个权重值；
#### 损失函数
##### 1.损失函数的概念

损失函数的作用：
损失函数的作用，就是计算神经网络每次迭代的前向计算结果与真实值的差距，从而指导下一步的训练向正确的方向进行。

损失函数的使用：
1.用随机值初始化前向计算公式的参数；
2.代入样本，计算输出的预测值；
3.用损失函数计算预测值和标签值（真实值）的误差；
4.根据损失函数的导数，沿梯度最小方向将误差回传，修正前向计算公式中的各个权重值；
5.goto 2, 直到损失函数值达到一个满意的值就停止迭代。


##### 2.机器学习常用损失函数
绝对值损失函数
铰链/折页损失函数或最大边界损失函数
对数损失函数，又叫交叉熵损失函数
均方差损失函数
指数损失函数

##### 3.损失函数图像理解
用二维函数图像理解单变量对损失函数的影响
用等高线图理解双变量对损失函数影响

##### 4.神经网络中常用的损失函数
1.均方差函数，主要用于回归
2.交叉熵函数，主要用于分类
二者都是非负函数，极值在底部，用梯度下降法可以求解。

##### 5.均方差函数
工作原理
实际案例

##### 6.损失函数的可视化
损失函数值的3D示意图
损失函数值的2D示意图

##### 7.交叉熵损失函数
交叉熵的由来：
信息量
熵
相对熵(KL散度)
交叉
熵

##### 8.二分类问题交叉熵
##### 9.多分类问题交叉熵
## 梯度下降

### 1 从自然现象中理解梯度下降

在大多数文章中，都以“一个人被困在山上，需要迅速下到谷底”来举例，这个人会“寻找当前所处位置最陡峭的地方向下走”。这个例子中忽略了安全因素，这个人不可能沿着最陡峭的方向走，要考虑坡度。

在自然界中，梯度下降的最好例子，就是泉水下山的过程：

1. 水受重力影响，会在当前位置，沿着最陡峭的方向流动，有时会形成瀑布（梯度下降）；
2. 水流下山的路径不是唯一的，在同一个地点，有可能有多个位置具有同样的陡峭程度，而造成了分流（可以得到多个解）；
3. 遇到坑洼地区，有可能形成湖泊，而终止下山过程（不能得到全局最优解，而是局部最优解）。

### 2 梯度下降的数学理解

梯度下降的数学公式：

$$\theta_{n+1} = \theta_{n} - \eta \cdot \nabla J(\theta) \tag{1}$$

根据公式(1)，迭代公式：

$$x_{n+1} = x_{n} - \eta \cdot \nabla J(x)= x_{n} - \eta \cdot 2x$$

假设终止条件为 $J(x)<0.01$，迭代过程是：
```
x=0.480000, y=0.230400
x=0.192000, y=0.036864
x=0.076800, y=0.005898
x=0.030720, y=0.000944
```

上面的过程如图2-10所示。

<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/2/gd_single_variable.png" ch="500" />
### 3 双变量的梯度下降

假设一个双变量函数：

$$J(x,y) = x^2 + \sin^2(y)$$

我们的目的是找到该函数的最小值，于是计算其微分：

$${\partial{J(x,y)} \over \partial{x}} = 2x$$
$${\partial{J(x,y)} \over \partial{y}} = 2 \sin y \cos y$$

假设初始位置为：

$$(x_0,y_0)=(3,1)$$

假设学习率：

$$\eta = 0.1$$

根据公式(1)，迭代过程是的计算公式：
$$(x_{n+1},y_{n+1}) = (x_n,y_n) - \eta \cdot \nabla J(x,y)$$
$$ = (x_n,y_n) - \eta \cdot (2x,2 \cdot \sin y \cdot \cos y) \tag{1}$$

根据公式(1)，假设终止条件为 $J(x,y)<0.01$，迭代过程如表2-3所示。

表2-3 双变量梯度下降的迭代过程

|迭代次数|x|y|J(x,y)|
|---|---|---|---|
|1|3|1|9.708073|
|2|2.4|0.909070|6.382415|
|...|...|...|...|
|15|0.105553|0.063481|0.015166|
|16|0.084442|0.050819|0.009711|

迭代16次后，$J(x,y)$ 的值为 $0.009711$，满足小于 $0.01$ 的条件，停止迭代。

上面的过程如表2-4所示，由于是双变量，所以需要用三维图来解释。请注意看两张图中间那条隐隐的黑色线，表示梯度下降的过程，从红色的高地一直沿着坡度向下走，直到蓝色的洼地。

表2-4 在三维空间内的梯度下降过程

|观察角度1|观察角度2|
|--|--|
|<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images\Images\2\gd_double_variable.png">|<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images\Images\2\gd_double_variable2.png">|

### 4 学习率η的选择

在公式表达时，学习率被表示为$\eta$。在代码里，我们把学习率定义为`learning_rate`，或者`eta`。针对上面的例子，试验不同的学习率对迭代情况的影响，如表2-5所示。

表2-5 不同学习率对迭代情况的影响

|学习率|迭代路线图|说明|
|---|---|---|
|1.0|<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/2/gd100.png" width="500" height="150"/>|学习率太大，迭代的情况很糟糕，在一条水平线上跳来跳去，永远也不能下降。|
|0.8|<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/2/gd080.png" width="400"/>|学习率大，会有这种左右跳跃的情况发生，这不利于神经网络的训练。|
|0.4|<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/2/gd040.png" width="400"/>|学习率合适，损失值会从单侧下降，4步以后基本接近了理想值。|
|0.1|<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/2/gd010.png" width="400"/>|学习率较小，损失值会从单侧下降，但下降速度非常慢，10步了还没有到达理想状态。|
## 最小二乘法
最小二乘法，也叫做最小平方法（Least Square），它通过最小化误差的平方和寻找数据的最佳函数匹配。利用最小二乘法可以简便地求得未知的数据，并使得这些求得的数据与实际数据之间误差的平方和为最小。最小二乘法还可用于曲线拟合。其他一些优化问题也可通过最小化能量或最小二乘法来表达。
### 1. 数学原理

线性回归试图学得：

$$z_i=w \cdot x_i+b \tag{1}$$

使得：

$$z_i \simeq y_i \tag{2}$$

其中，$x_i$ 是样本特征值，$y_i$ 是样本标签值，$z_i$ 是模型预测值。

如何学得 $w$ 和 $b$ 呢？均方差(MSE - mean squared error)是回归任务中常用的手段：
$$
J = \frac{1}{2m}\sum_{i=1}^m(z_i-y_i)^2 = \frac{1}{2m}\sum_{i=1}^m(y_i-wx_i-b)^2 \tag{3}
$$

$J$ 称为损失函数。实际上就是试图找到一条直线，使所有样本到直线上的残差的平方和最小。

<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/4/mse.png" />

图4-3 均方差函数的评估原理

图4-3中，圆形点是样本点，直线是当前的拟合结果。如左图所示，我们是要计算样本点到直线的垂直距离，需要再根据直线的斜率来求垂足然后再计算距离，这样计算起来很慢；但实际上，在工程上我们通常使用的是右图的方式，即样本点到直线的竖直距离，因为这样计算很方便，用一个减法就可以了。

假设我们计算出初步的结果是虚线所示，这条直线是否合适呢？我们来计算一下图中每个点到这条直线的距离，把这些距离的值都加起来（都是正数，不存在互相抵消的问题）成为误差。

因为上图中的几个点不在一条直线上，所以不能有一条直线能同时穿过它们。所以，我们只能想办法不断改变红色直线的角度和位置，让总体误差最小（永远不可能是 $0$），就意味着整体偏差最小，那么最终的那条直线就是我们要的结果。

如果想让误差的值最小，通过对 $w$ 和 $b$ 求导，再令导数为 $0$（到达最小极值），就是 $w$ 和 $b$ 的最优解。
推导过程如下：

$$
\begin{aligned}
\frac{\partial{J}}{\partial{w}} &=\frac{\partial{(\frac{1}{2m}\sum_{i=1}^m(y_i-wx_i-b)^2)}}{\partial{w}} \\\\
&= \frac{1}{m}\sum_{i=1}^m(y_i-wx_i-b)(-x_i) 
\end{aligned}
\tag{4}
$$

令公式4为 $0$：

$$
\sum_{i=1}^m(y_i-wx_i-b)x_i=0 \tag{5}
$$

$$
\begin{aligned}
\frac{\partial{J}}{\partial{b}} &=\frac{\partial{(\frac{1}{2m}\sum_{i=1}^m(y_i-wx_i-b)^2)}}{\partial{b}} \\\\
&=\frac{1}{m}\sum_{i=1}^m(y_i-wx_i-b)(-1) 
\end{aligned}
\tag{6}
$$

令公式6为 $0$：

$$
\sum_{i=1}^m(y_i-wx_i-b)=0 \tag{7}
$$

由式7得到（假设有 $m$ 个样本）：

$$
\sum_{i=1}^m b = m \cdot b = \sum_{i=1}^m{y_i} - w\sum_{i=1}^m{x_i} \tag{8}
$$

两边除以 $m$：

$$
b = \frac{1}{m}\left(\sum_{i=1}^m{y_i} - w\sum_{i=1}^m{x_i}\right)=\bar y-w \bar x \tag{9}
$$

其中：

$$
\bar y = \frac{1}{m}\sum_{i=1}^m y_i, \bar x=\frac{1}{m}\sum_{i=1}^m x_i \tag{10}
$$

将公式10代入公式5：

$$
\sum_{i=1}^m(y_i-wx_i-\bar y + w \bar x)x_i=0
$$

$$
\sum_{i=1}^m(x_i y_i-wx^2_i-x_i \bar y + w \bar x x_i)=0
$$

$$
\sum_{i=1}^m(x_iy_i-x_i \bar y)-w\sum_{i=1}^m(x^2_i - \bar x x_i) = 0
$$

$$
w = \frac{\sum_{i=1}^m(x_iy_i-x_i \bar y)}{\sum_{i=1}^m(x^2_i - \bar x x_i)} \tag{11}
$$

将公式10代入公式11：

$$
w = \frac{\sum_{i=1}^m (x_i \cdot y_i) - \sum_{i=1}^m x_i \cdot \frac{1}{m} \sum_{i=1}^m y_i}{\sum_{i=1}^m x^2_i - \sum_{i=1}^m x_i \cdot \frac{1}{m}\sum_{i=1}^m x_i} \tag{12}
$$

分子分母都乘以 $m$：

$$
w = \frac{m\sum_{i=1}^m x_i y_i - \sum_{i=1}^m x_i \sum_{i=1}^m y_i}{m\sum_{i=1}^m x^2_i - (\sum_{i=1}^m x_i)^2} \tag{13}
$$

$$
b= \frac{1}{m} \sum_{i=1}^m(y_i-wx_i) \tag{14}
$$

而事实上，式13有很多个变种，大家会在不同的文章里看到不同版本，往往感到困惑，比如下面两个公式也是正确的解：

$$
w = \frac{\sum_{i=1}^m y_i(x_i-\bar x)}{\sum_{i=1}^m x^2_i - (\sum_{i=1}^m x_i)^2/m} \tag{15}
$$

$$
w = \frac{\sum_{i=1}^m x_i(y_i-\bar y)}{\sum_{i=1}^m x^2_i - \bar x \sum_{i=1}^m x_i} \tag{16}
$$

以上两个公式，如果把公式10代入，也应该可以得到和式13相同的答案，只不过需要一些运算技巧。比如，很多人不知道这个神奇的公式：

$$
\begin{aligned}
\sum_{i=1}^m (x_i \bar y) &= \bar y \sum_{i=1}^m x_i =\frac{1}{m}(\sum_{i=1}^m y_i) (\sum_{i=1}^m x_i) \\\\
&=\frac{1}{m}(\sum_{i=1}^m x_i) (\sum_{i=1}^m y_i)= \bar x \sum_{i=1}^m y_i \\\\
&=\sum_{i=1}^m (y_i \bar x) 
\end{aligned}
\tag{17}
$$
## 梯度下降法


有了上一节的最小二乘法做基准，我们这次用梯度下降法求解 $w$ 和 $b$，从而可以比较二者的结果。

### 1. 数学原理

在下面的公式中，我们规定 $x$ 是样本特征值（单特征），$y$ 是样本标签值，$z$ 是预测值，下标 $i$ 表示其中一个样本。

#### 预设函数（Hypothesis Function）

线性函数：

$$z_i = x_i \cdot w + b \tag{1}$$

#### 损失函数（Loss Function）

均方误差：

$$loss_i(w,b) = \frac{1}{2} (z_i-y_i)^2 \tag{2}$$


与最小二乘法比较可以看到，梯度下降法和最小二乘法的模型及损失函数是相同的，都是一个线性模型加均方差损失函数，模型用于拟合，损失函数用于评估效果。

区别在于，最小二乘法从损失函数求导，直接求得数学解析解，而梯度下降以及后面的神经网络，都是利用导数传递误差，再通过迭代方式一步一步（用近似解）逼近真实解。

###  2.梯度计算

#### 计算z的梯度

根据公式2：
$$
\frac{\partial loss}{\partial z_i}=z_i - y_i \tag{3}
$$

#### 计算 $w$ 的梯度

我们用 $loss$ 的值作为误差衡量标准，通过求 $w$ 对它的影响，也就是 $loss$ 对 $w$ 的偏导数，来得到 $w$ 的梯度。由于 $loss$ 是通过公式2->公式1间接地联系到 $w$ 的，所以我们使用链式求导法则，通过单个样本来求导。

根据公式1和公式3：

$$
\frac{\partial{loss}}{\partial{w}} = \frac{\partial{loss}}{\partial{z_i}}\frac{\partial{z_i}}{\partial{w}}=(z_i-y_i)x_i \tag{4}
$$

#### 计算 $b$ 的梯度

$$
\frac{\partial{loss}}{\partial{b}} = \frac{\partial{loss}}{\partial{z_i}}\frac{\partial{z_i}}{\partial{b}}=z_i-y_i \tag{5}
$$
## 神经网络法
### 1 定义神经网络结构

我们是首次尝试建立神经网络，先用一个最简单的单层单点神经元，如图4-4所示。

<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/4/Setup.png" ch="500" />

图4-4 单层单点神经元

下面，我们用这个最简单的线性回归的例子，来说明神经网络中最重要的反向传播和梯度下降的概念、过程以及代码实现。

#### 输入层

此神经元在输入层只接受一个输入特征，经过参数 $w,b$ 的计算后，直接输出结果。这样一个简单的“网络”，只能解决简单的一元线性回归问题，而且由于是线性的，我们不需要定义激活函数，这就大大简化了程序，而且便于大家循序渐进地理解各种知识点。

严格来说输入层在神经网络中并不能称为一个层。

#### 权重 $w,b$

因为是一元线性问题，所以 $w,b$ 都是标量。

#### 输出层

输出层 $1$ 个神经元，线性预测公式是：

$$z_i = x_i \cdot w + b$$

$z$ 是模型的预测输出，$y$ 是实际的样本标签值，下标 $i$ 为样本。

#### 损失函数

因为是线性回归问题，所以损失函数使用均方差函数。

$$loss(w,b) = \frac{1}{2} (z_i-y_i)^2$$

### 2 反向传播

由于我们使用了和上一节中的梯度下降法同样的数学原理，所以反向传播的算法也是一样的，细节请查看4.2.2。

#### 计算 $w$ 的梯度

$$
{\partial{loss} \over \partial{w}} = \frac{\partial{loss}}{\partial{z_i}}\frac{\partial{z_i}}{\partial{w}}=(z_i-y_i)x_i
$$

#### 计算 $b$ 的梯度

$$
\frac{\partial{loss}}{\partial{b}} = \frac{\partial{loss}}{\partial{z_i}}\frac{\partial{z_i}}{\partial{b}}=z_i-y_i
$$
## 梯度下降的三种形式
1 单样本随机梯度下降
2 小批量样本梯度下降
3 全批量样本梯度下降
## 多元线性回归模型解决方法
### 正规方程法
### 神经网络法
与单特征值的线性回归问题类似，多变量（多特征值）的线性回归可以被看做是一种高维空间的线性拟合。以具有两个特征的情况为例，这种线性拟合不再是用直线去拟合点，而是用平面去拟合点。

#### 1 定义神经网络结构

我们定义一个如图5-1所示的一层的神经网络，输入层为2或者更多，反正大于2了就没区别。这个一层的神经网络的特点是：

1. 没有中间层，只有输入项和输出层（输入项不算做一层）；
2. 输出层只有一个神经元；
3. 神经元有一个线性输出，不经过激活函数处理，即在下图中，经过 $\Sigma$ 求和得到 $Z$ 值之后，直接把 $Z$ 值输出。

<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/5/setup.png" ch="500" />

图5-1 多入单出的单层神经元结构

与上一章的神经元相比，这次仅仅是多了一个输入，但却是质的变化，即，一个神经元可以同时接收多个输入，这是神经网络能够处理复杂逻辑的根本。

##### 输入层

单独看第一个样本是这样的：

$$
x_1 =
\begin{pmatrix}
x_{11} & x_{12}
\end{pmatrix} = 
\begin{pmatrix}
10.06 & 60
\end{pmatrix} 
$$

$$
y_1 = \begin{pmatrix} 302.86 \end{pmatrix}
$$

一共有1000个样本，每个样本2个特征值，X就是一个$1000 \times 2$的矩阵：

$$
X = 
\begin{pmatrix} 
x_1 \\\\ x_2 \\\\ \vdots \\\\ x_{1000}
\end{pmatrix} =
\begin{pmatrix} 
x_{1,1} & x_{1,2} \\\\
x_{2,1} & x_{2,2} \\\\
\vdots & \vdots \\\\
x_{1000,1} & x_{1000,2}
\end{pmatrix}
$$

$$
Y =
\begin{pmatrix}
y_1 \\\\ y_2 \\\\ \vdots \\\\ y_{1000}
\end{pmatrix}=
\begin{pmatrix}
302.86 \\\\ 393.04 \\\\ \vdots \\\\ 450.59
\end{pmatrix}
$$


$x_1$ 表示第一个样本，$x_{1,1}$ 表示第一个样本的一个特征值，$y_1$ 是第一个样本的标签值。

##### 权重 $W$ 和 $B$

由于输入层是两个特征，输出层是一个变量，所以 $W$ 的形状是 $2\times 1$，而 $B$ 的形状是 $1\times 1$。

$$
W=
\begin{pmatrix}
w_1 \\\\ w_2
\end{pmatrix}
$$

$$B=(b)$$

$B$ 是个单值，因为输出层只有一个神经元，所以只有一个bias，每个神经元对应一个bias，如果有多个神经元，它们都会有各自的b值。

##### 输出层

由于我们只想完成一个回归（拟合）任务，所以输出层只有一个神经元。由于是线性的，所以没有用激活函数。
$$
\begin{aligned}
Z&=
\begin{pmatrix}
  x_{11} & x_{12}
\end{pmatrix}
\begin{pmatrix}
  w_1 \\\\ w_2
\end{pmatrix}
+(b) \\\\
&=x_{11}w_1+x_{12}w_2+b
\end{aligned}
$$

写成矩阵形式：

$$Z = X\cdot W + B$$

##### 损失函数

因为是线性回归问题，所以损失函数使用均方差函数。

$$loss_i(W,B) = \frac{1}{2} (z_i-y_i)^2 \tag{1}$$

其中，$z_i$ 是样本预测值，$y_i$ 是样本的标签值。

#### 5.2.2 反向传播

##### 单样本多特征计算

与上一章不同，本章中的前向计算是多特征值的公式：

$$\begin{aligned}
z_i &= x_{i1} \cdot w_1 + x_{i2} \cdot w_2 + b \\\\
&=\begin{pmatrix}
  x_{i1} & x_{i2}
\end{pmatrix}
\begin{pmatrix}
  w_1 \\\\
  w_2
\end{pmatrix}+b
\end{aligned} \tag{2}
$$

因为 $x$ 有两个特征值，对应的 $W$ 也有两个权重值。$x_{i1}$ 表示第 $i$ 个样本的第 $1$ 个特征值，所以无论是 $x$ 还是 $W$ 都是一个向量或者矩阵了，那么我们在反向传播方法中的梯度计算公式还有效吗？答案是肯定的，我们来一起做个简单推导。

由于 $W$ 被分成了 $w_1$ 和 $w_2$ 两部分，根据公式1和公式2，我们单独对它们求导：

$$
\frac{\partial loss_i}{\partial w_1}=\frac{\partial loss_i}{\partial z_i}\frac{\partial z_i}{\partial w_1}=(z_i-y_i) \cdot x_{i1} \tag{3}
$$
$$
\frac{\partial loss_i}{\partial w_2}=\frac{\partial loss_i}{\partial z_i}\frac{\partial z_i}{\partial w_2}=(z_i-y_i) \cdot x_{i2} \tag{4}
$$

求损失函数对 $W$ 矩阵的偏导是无法直接求的，所以要变成求各个 $W$ 的分量的偏导。由于 $W$ 的形状是：

$$
W=
\begin{pmatrix}
w_1 \\\\ w_2
\end{pmatrix}
$$

所以求 $loss_i$ 对 $W$ 的偏导，应该这样写：

$$
\begin{aligned}  
\frac{\partial loss_i}{\partial W}&=
\begin{pmatrix}
  \frac{\partial loss_i}{\partial w_1} \\\\
  \frac{\partial loss_i}{\partial w_2}
\end{pmatrix} 
=\begin{pmatrix}
  (z_i-y_i) \cdot x_{i1} \\\\
  (z_i-y_i) \cdot x_{i2}
\end{pmatrix}  \\\\
&=\begin{pmatrix}
  x_{i1} \\\\
  x_{i2}
\end{pmatrix}
(z_i-y_i) 
=\begin{pmatrix}
  x_{i1} & x_{i2}
\end{pmatrix}^{\top}(z_i-y_i) \\\\
&=x_i^{\top}(z_i-y_i)
\end{aligned} \tag{5}
$$

$$
\frac{\partial loss_i}{\partial B}=z_i-y_i \tag{6}
$$

##### 多样本多特征计算

当进行多样本计算时，我们用 $m=3$ 个样本做一个实例化推导：

$$
z_1 = x_{11}w_1+x_{12}w_2+b
$$

$$
z_2= x_{21}w_1+x_{22}w_2+b
$$

$$
z_3 = x_{31}w_1+x_{32}w_2+b
$$

$$
J(W,B) = \frac{1}{2 \times 3}[(z_1-y_1)^2+(z_2-y_2)^2+(z_3-y_3)^2]
$$

$$
\begin{aligned}  
\frac{\partial J}{\partial W}&=
\begin{pmatrix}
  \frac{\partial J}{\partial w_1} \\\\
  \frac{\partial J}{\partial w_2}
\end{pmatrix}
=\begin{pmatrix}
  \frac{\partial J}{\partial z_1}\frac{\partial z_1}{\partial w_1}+\frac{\partial J}{\partial z_2}\frac{\partial z_2}{\partial w_1}+\frac{\partial J}{\partial z_3}\frac{\partial z_3}{\partial w_1} \\\\
  \frac{\partial J}{\partial z_1}\frac{\partial z_1}{\partial w_2}+\frac{\partial J}{\partial z_2}\frac{\partial z_2}{\partial w_2}+\frac{\partial J}{\partial z_3}\frac{\partial z_3}{\partial w_2}  
\end{pmatrix}
\\\\
&=\begin{pmatrix}
  \frac{1}{3}(z_1-y_1)x_{11}+\frac{1}{3}(z_2-y_2)x_{21}+\frac{1}{3}(z_3-y_3)x_{31} \\\\
  \frac{1}{3}(z_1-y_1)x_{12}+\frac{1}{3}(z_2-y_2)x_{22}+\frac{1}{3}(z_3-y_3)x_{32}
\end{pmatrix}
\\\\
&=\frac{1}{3}
\begin{pmatrix}
  x_{11} & x_{21} & x_{31} \\\\
  x_{12} & x_{22} & x_{32}
\end{pmatrix}
\begin{pmatrix}
  z_1-y_1 \\\\
  z_2-y_2 \\\\
  z_3-y_3
\end{pmatrix}
\\\\
&=\frac{1}{3}
\begin{pmatrix}
  x_{11} & x_{12} \\\\
  x_{21} & x_{22} \\\\
  x_{31} & x_{32} 
\end{pmatrix}^{\top}
\begin{pmatrix}
  z_1-y_1 \\\\
  z_2-y_2 \\\\
  z_3-y_3
\end{pmatrix}
\\\\
&=\frac{1}{m}X^{\top}(Z-Y) 
\end{aligned}
\tag{7}
$$
注：3泛化为m。
$$
\frac{\partial J}{\partial B}=\frac{1}{m}(Z-Y) \tag{8}
$$
### 线性二分类原理
#### 二分类的代数原理

代数方式：通过一个分类函数计算所有样本点在经过线性变换后的概率值，使得正例样本的概率大于0.5，而负例样本的概率小于0.5。
#### 二分类函数的几何作用

二分类函数的最终结果是把正例都映射到图6-6中的上半部分的曲线上，而把负类都映射到下半部分的曲线上。

<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/6/sigmoid_binary.png"/>
## 一. 激活函数
### 1 激活函数的基本作用

图8-1是神经网络中的一个神经元，假设该神经元有三个输入，分别为$x_1,x_2,x_3$，那么：

$$z=x_1 w_1 + x_2 w_2 + x_3 w_3 +b \tag{1}$$
$$a = \sigma(z) \tag{2}$$

<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/1/NeuranCell.png" width="500" />

图8-1 激活函数在神经元中的位置

激活函数也就是 $a=\sigma(z)$ 这一步了，他有什么作用呢？

1. 给神经网络增加非线性因素，这个问题在第1章神经网络基本工作原理中已经讲过了；
2. 把公式1的计算结果压缩到 $[0,1]$ 之间，便于后面的计算。

激活函数的基本性质：

+ 非线性：线性的激活函数和没有激活函数一样；
+ 可导性：做误差反向传播和梯度下降，必须要保证激活函数的可导性；
+ 单调性：单一的输入会得到单一的输出，较大值的输入得到较大值的输出。

在物理试验中使用的继电器，是最初的激活函数的原型：当输入电流大于一个阈值时，会产生足够的磁场，从而打开下一级电源通道，如图8-2所示。

<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/8/step.png" width="500" />

图8-2 继电器的阶跃形态

用到神经网络中的概念，用‘1’来代表一个神经元被激活，‘0’代表一个神经元未被激活。

这个Step函数有什么不好的地方呢？主要的一点就是，他的梯度（导数）恒为零（个别点除外)。反向传播公式中，梯度传递用到了链式法则，如果在这样一个连乘的式子其中有一项是零，这样的梯度就会恒为零，是没有办法进行反向传播的。

### 2 何时会用到激活函数

激活函数用在神经网络的层与层之间的连接，神经网络的最后一层不用激活函数。

神经网络不管有多少层，最后的输出层决定了这个神经网络能干什么。在单层神经网络中，我们学习到了表8-1所示的内容。

表8-1 单层的神经网络的参数与功能

|网络|输入|输出|激活函数|分类函数|功能|
|---|---|---|---|---|---|
|单层|单变量|单输出|无|无|线性回归|
|单层|多变量|单输出|无|无|线性回归|
|单层|多变量|单输出|无|二分类函数|二分类|
|单层|多变量|多输出|无|多分类函数|多分类|

从上表可以看到，我们一直没有使用激活函数，而只使用了分类函数。对于多层神经网络也是如此，在最后一层只会用到分类函数来完成二分类或多分类任务，如果是拟合任务，则不需要分类函数。

简言之：

1. 神经网络最后一层不需要激活函数
2. 激活函数只用于连接前后两层神经网络

### 3 Logistic函数(挤压型激活函数)

对数几率函数（Logistic Function，简称对率函数）。

很多文字材料中通常把激活函数和分类函数混淆在一起说，有一个原因是：在二分类任务中最后一层使用的对率函数与在神经网络层与层之间连接的Sigmoid激活函数，是同样的形式。所以它既是激活函数，又是分类函数，是个特例。

对这个函数的叫法比较混乱，在本书中我们约定一下，凡是用到“Logistic”词汇的，指的是二分类函数；而用到“Sigmoid”词汇的，指的是本激活函数。

#### 公式

$$Sigmoid(z) = \frac{1}{1 + e^{-z}} \rightarrow a \tag{1}$$

#### 导数

$$Sigmoid'(z) = a(1 - a) \tag{2}$$
#### 值域

- 输入值域：$(-\infty, \infty)$
- 输出值域：$(0,1)$
- 导数值域：$(0,0.25]$

#### 函数图像

<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/8/sigmoid.png" ch="500" />
### 4 Tanh函数(挤压型激活函数)

TanHyperbolic，即双曲正切函数。

#### 公式  
$$Tanh(z) = \frac{e^{z} - e^{-z}}{e^{z} + e^{-z}} = (\frac{2}{1 + e^{-2z}}-1) \rightarrow a \tag{3}$$
即
$$Tanh(z) = 2 \cdot Sigmoid(2z) - 1 \tag{4}$$

#### 导数公式

$$Tanh'(z) = (1 + a)(1 - a)$$
#### 值域

- 输入值域：$(-\infty,\infty)$
- 输出值域：$(-1,1)$
- 导数值域：$(0,1)$


#### 函数图像

图8-4是双曲正切的函数图像。

<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/8/tanh.png" ch="500" />

图8-4 双曲正切函数图像
#### 代码
``` 
# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for full license information.

import numpy as np
import matplotlib.pyplot as plt

from Activators.Relu import *
from Activators.Elu import *
from Activators.LeakyRelu import *
from Activators.Sigmoid import *
from Activators.Softplus import *
from Activators.Step import *
from Activators.Tanh import *
from Activators.BenIdentity import *

def Draw(start,end,func,lable1,lable2):
    z = np.linspace(start, end, 200)
    a = func.forward(z)
    da, dz = func.backward(z, a, 1)

    p1, = plt.plot(z,a)
    p2, = plt.plot(z,da)
    plt.legend([p1,p2], [lable1, lable2])
    plt.grid()
    plt.xlabel("input : z")
    plt.ylabel("output : a")
    plt.title(lable1)
    plt.show()

if __name__ == '__main__':
    Draw(-7,7,CSigmoid(),"Sigmoid Function","Derivative of Sigmoid")
    Draw(-7,7,CSigmoid(),"Logistic Function","Derivative of Logistic")
    Draw(-5,5,CStep(0.3),"Step Function","Derivative of Step")
```
### 5 ReLU函数(半线性激活函数) 

Rectified Linear Unit，修正线性单元，线性整流函数，斜坡函数。

#### 公式

$$ReLU(z) = max(0,z) = \begin{cases} 
  z, & z \geq 0 \\\\ 
  0, & z < 0 
\end{cases}$$

#### 导数

$$ReLU'(z) = \begin{cases} 1 & z \geq 0 \\\\ 0 & z < 0 \end{cases}$$

#### 值域

- 输入值域：$(-\infty, \infty)$
- 输出值域：$(0,\infty)$
- 导数值域：$\\{0,1\\}$

<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/8/relu.png"/>
### 6 Leaky ReLU函数(半线性激活函数)

LReLU，带泄露的线性整流函数。

#### 公式

$$LReLU(z) = \begin{cases} z & z \geq 0 \\\\ \alpha \cdot z & z < 0 \end{cases}$$

#### 导数

$$LReLU'(z) = \begin{cases} 1 & z \geq 0 \\\\ \alpha & z < 0 \end{cases}$$

#### 值域

输入值域：$(-\infty, \infty)$

输出值域：$(-\infty,\infty)$

导数值域：$\\{\alpha,1\\}$

#### 函数图像

函数图像如图8-7所示。

<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/8/leakyRelu.png"/>

图8-7 LeakyReLU的函数图像
### 7 Softplus函数(半线性激活函数)

#### 公式

$$Softplus(z) = \ln (1 + e^z)$$

#### 导数

$$Softplus'(z) = \frac{e^z}{1 + e^z}$$

#### 

输入值域：$(-\infty, \infty)$

输出值域：$(0,\infty)$

导数值域：$(0,1)$

#### 函数图像

Softplus的函数图像如图8-8所示。

<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/8/softplus.png"/>
### 8 ELU函数(半线性激活函数)

#### 公式

$$ELU(z) = \begin{cases} z & z \geq 0 \\ \alpha (e^z-1) & z < 0 \end{cases}$$

#### 导数

$$ELU'(z) = \begin{cases} 1 & z \geq 0 \\ \alpha e^z & z < 0 \end{cases}$$

#### 值域

输入值域：$(-\infty, \infty)$

输出值域：$(-\alpha,\infty)$

导数值域：$(0,1]$

#### 函数图像

ELU的函数图像如图8-9所示。

<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/8/elu.png"/>

## 二 非线性回归问题
### 1 非线性回归
#### 1 回归模型的评估标准

回归问题主要是求值，评价标准主要是看求得值与实际结果的偏差有多大，所以，回归问题主要以下方法来评价模型。

#### 平均绝对误差

MAE（Mean Abolute Error）。

$$MAE=\frac{1}{m} \sum_{i=1}^m \lvert a_i-y_i \rvert \tag{1}$$

对异常值不如均方差敏感，类似中位数。

#### 绝对平均值率误差

MAPE（Mean Absolute Percentage Error）。

$$MAPE=\frac{100}{m} \sum^m_{i=1} \left\lvert {a_i - y_i \over y_i} \right\rvert \tag{2}$$

#### 和方差

SSE（Sum Squared Error）。

$$SSE=\sum_{i=1}^m (a_i-y_i)^2 \tag{3}$$

得出的值与样本数量有关系，假设有1000个测试样本，得到的值是120；如果只有100个测试样本，得到的值可能是11，我们不能说11就比120要好。

#### 均方差

MSE（Mean Squared Error）。

$$MSE = \frac{1}{m} \sum_{i=1}^m (a_i-y_i)^2 \tag{4}$$

就是实际值减去预测值的平方再求期望，没错，就是线性回归的代价函数。由于MSE计算的是误差的平方，所以它对异常值是非常敏感的，因为一旦出现异常值，MSE指标会变得非常大。MSE越小，证明误差越小。

#### 均方根误差

RMSE（Root Mean Squard Error）。

$$RMSE = \sqrt{\frac{1}{m} \sum_{i=1}^m (a_i-y_i)^2} \tag{5}$$

是均方差开根号的结果，其实质是一样的，只不过对结果有更好的解释。



#### R平方

R-Squared。

上面的几种衡量标准针对不同的模型会有不同的值。比如说预测房价，那么误差单位就是元，比如3000元、11000元等。如果预测身高就可能是0.1、0.2米之类的。也就是说，对于不同的场景，会有不同量纲，因而也会有不同的数值，无法用一句话说得很清楚，必须啰啰嗦嗦带一大堆条件才能表达完整。

我们通常用概率来表达一个准确率，比如89%的准确率。那么线性回归有没有这样的衡量标准呢？答案就是R-Squared。

$$R^2=1-\frac{\sum (a_i - y_i)^2}{\sum(\bar y_i-y_i)^2}=1-\frac{MSE(a,y)}{Var(y)} \tag{6}$$

R平方是多元回归中的回归平方和（分子）占总平方和（分母）的比例，它是度量多元回归方程中拟合程度的一个统计量。R平方值越接近1，表明回归平方和占总平方和的比例越大，回归线与各观测点越接近，回归的拟合程度就越好。

- 如果结果是0，说明模型跟瞎猜差不多；
- 如果结果是1，说明模型无错误；
- 如果结果是0-1之间的数，就是模型的好坏程度；
- 如果结果是负数，说明模型还不如瞎猜。

代码实现：

```Python
def R2(a, y):
    assert (a.shape == y.shape)
    m = a.shape[0]
    var = np.var(y)
    mse = np.sum((a-y)**2)/m
    r2 = 1 - mse / var
    return r2
```
## 三. 验证与测试
### 1 基本概念

#### 训练集

Training Set，用于模型训练的数据样本。

#### 验证集

Validation Set，或者叫做Dev Set，是模型训练过程中单独留出的样本集，它可以用于调整模型的超参数和用于对模型的能力进行初步评估。
  
在神经网络中，验证数据集用于：

- 寻找最优的网络深度
- 或者决定反向传播算法的停止点
- 或者在神经网络中选择隐藏层神经元的数量
- 在普通的机器学习中常用的交叉验证（Cross Validation）就是把训练数据集本身再细分成不同的验证数据集去训练模型。

#### 测试集

Test Set，用来评估最终模型的泛化能力。但不能作为调参、选择特征等算法相关的选择的依据。

三者之间的关系如图9-5所示。

<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/9/dataset.png" />

图9-5 训练集、验证集、测试集的关系

一个形象的比喻：

- 训练集：课本，学生根据课本里的内容来掌握知识。训练集直接参与了模型调参的过程，显然不能用来反映模型真实的能力。即不能直接拿课本上的问题来考试，防止死记硬背课本的学生拥有最好的成绩，即防止过拟合。

- 验证集：作业，通过作业可以知道不同学生学习情况、进步的速度快慢。验证集参与了人工调参（超参数）的过程，也不能用来最终评判一个模型（刷题库的学生不能算是学习好的学生）。

- 测试集：考试，考的题是平常都没有见过，考察学生举一反三的能力。所以要通过最终的考试（测试集）来考察一个学型（模生）真正的能力（期末考试）。

考试题是学生们平时见不到的，也就是说在模型训练时看不到测试集。
## 四 二分类模型
### 1.二分类模型的评估标准
#### 准确率 Accuracy

也可以称之为精度，我们在本书中混用这两个词。

对于二分类问题，假设测试集上一共1000个样本，其中550个正例，450个负例。测试一个模型时，得到的结果是：521个正例样本被判断为正类，435个负例样本被判断为负类，则正确率计算如下：

$$Accuracy=(521+435)/1000=0.956$$

即正确率为95.6%。这种方式对多分类也是有效的，即三类中判别正确的样本数除以总样本数，即为准确率。

但是这种计算方法丢失了很多细节，比如：是正类判断的精度高还是负类判断的精度高呢？因此，我们还有如下一种评估标准。

#### 混淆矩阵

还是用上面的例子，如果具体深入到每个类别上，会分成4部分来评估：

- 正例中被判断为正类的样本数（TP-True Positive）：521
- 正例中被判断为负类的样本数（FN-False Negative）：550-521=29
- 负例中被判断为负类的样本数（TN-True Negative）：435
- 负例中被判断为正类的样本数（FP-False Positive）：450-435=15

可以用图10-3来帮助理解。

<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/10/TPFP.png"/>

图10-3 二分类中四种类别的示意图

- 左侧实心圆点是正类，右侧空心圆是负类；
- 在圆圈中的样本是被模型判断为正类的，圆圈之外的样本是被判断为负类的；
- 左侧圆圈外的点是正类但是误判为负类，右侧圆圈内的点是负类但是误判为正类；
- 左侧圆圈内的点是正类且被正确判别为正类，右侧圆圈外的点是负类且被正确判别为负类。

用表格的方式描述矩阵的话是表10-1的样子。

表10-1 四类样本的矩阵关系

|预测值|被判断为正类|被判断为负类|Total|
|---|---|---|---|
|样本实际为正例|TP-True Positive|FN-False Negative|Actual Positive=TP+FN|
|样本实际为负例|FP-False Positive|TN-True Negative|Actual Negative=FP+TN|
|Total|Predicated Postivie=TP+FP|Predicated Negative=FN+TN|

从混淆矩阵中可以得出以下统计指标：

- 准确率 Accuracy

$$
\begin{aligned}
Accuracy &= \frac{TP+TN}{TP+TN+FP+FN} \\\\
&=\frac{521+435}{521+29+435+15}=0.956
\end{aligned}
$$

这个指标就是上面提到的准确率，越大越好。

- 精确率/查准率 Precision

分子为被判断为正类并且真的是正类的样本数，分母是被判断为正类的样本数。越大越好。

$$
Precision=\frac{TP}{TP+FP}=\frac{521}{521+15}=0.972
$$

- 召回率/查全率 Recall

$$
Recall = \frac{TP}{TP+FN}=\frac{521}{521+29}=0.947
$$

分子为被判断为正类并且真的是正类的样本数，分母是真的正类的样本数。越大越好。

- TPR - True Positive Rate 真正例率

$$
TPR = \frac{TP}{TP + FN}=Recall=0.947
$$

- FPR - False Positive Rate 假正例率

$$
FPR = \frac{FP}{FP+TN}=\frac{15}{15+435}=0.033
$$

分子为被判断为正类的负例样本数，分母为所有负类样本数。越小越好。

- 调和平均值 F1

$$
\begin{aligned}
F1&=\frac{2 \times Precision \times Recall}{recision+Recall}\\\\
&=\frac{2 \times 0.972 \times 0.947}{0.972+0.947}=0.959
\end{aligned}
$$

该值越大越好。

- ROC曲线与AUC

ROC，Receiver Operating Characteristic，接收者操作特征，又称为感受曲线（Sensitivity Curve），是反映敏感性和特异性连续变量的综合指标，曲线上各点反映着相同的感受性，它们都是对同一信号刺激的感受性。
ROC曲线的横坐标是FPR，纵坐标是TPR。

AUC，Area Under Roc，即ROC曲线下面的面积。

在二分类器中，如果使用Logistic函数作为分类函数，可以设置一系列不同的阈值，比如[0.1,0.2,0.3...0.9]，把测试样本输入，从而得到一系列的TP、FP、TN、FN，然后就可以绘制如下曲线，如图10-4。

<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/10/ROC.png"/>

图10-4 ROC曲线图

图中红色的曲线就是ROC曲线，曲线下的面积就是AUC值，取值区间为$[0.5,1.0]$，面积越大越好。

- ROC曲线越靠近左上角，该分类器的性能越好。
- 对角线表示一个随机猜测分类器。
- 若一个学习器的ROC曲线被另一个学习器的曲线完全包住，则可判断后者性能优于前者。
- 若两个学习器的ROC曲线没有包含关系，则可以判断ROC曲线下的面积，即AUC，谁大谁好。

当然在实际应用中，取决于阈值的采样间隔，红色曲线不会这么平滑，由于采样间隔会导致该曲线呈阶梯状。

既然已经这么多标准，为什么还要使用ROC和AUC呢？因为ROC曲线有个很好的特性：当测试集中的正负样本的分布变换的时候，ROC曲线能够保持不变。在实际的数据集中经常会出现样本类不平衡，即正负样本比例差距较大，而且测试数据中的正负样本也可能随着时间变化。

#### Kappa statics 

Kappa值，即内部一致性系数(inter-rater,coefficient of internal consistency)，是作为评价判断的一致性程度的重要指标。取值在0～1之间。

$$
Kappa = \frac{p_o-p_e}{1-p_e}
$$

其中，$p_0$是每一类正确分类的样本数量之和除以总样本数，也就是总体分类精度。$p_e$的定义见以下公式。

- Kappa≥0.75两者一致性较好；
- 0.75>Kappa≥0.4两者一致性一般；
- Kappa<0.4两者一致性较差。 

该系数通常用于多分类情况，如：

||实际类别A|实际类别B|实际类别C|预测总数|
|--|--|--|--|--|
|预测类别A|239|21|16|276|
|预测类别B|16|73|4|93|
|预测类别C|6|9|280|295|
|实际总数|261|103|300|664|


$$
p_o=\frac{239+73+280}{664}=0.8916
$$
$$
p_e=\frac{261 \times 276 + 103 \times 93 + 300 \times 295}{664 \times 664}=0.3883
$$
$$
Kappa = \frac{0.8916-0.3883}{1-0.3883}=0.8228
$$

数据一致性较好，说明分类器性能好。

#### Mean absolute error 和 Root mean squared error 

平均绝对误差和均方根误差，用来衡量分类器预测值和实际结果的差异，越小越好。

#### Relative absolute error 和 Root relative squared error 

相对绝对误差和相对均方根误差，有时绝对误差不能体现误差的真实大小，而相对误差通过体现误差占真值的比重来反映误差大小。
### 2.实现双弧形二分类
#### 代码实现

##### 主过程代码

```Python
if __name__ == '__main__':
    ......
    n_input = dataReader.num_feature
    n_hidden = 2
    n_output = 1
    eta, batch_size, max_epoch = 0.1, 5, 10000
    eps = 0.08

    hp = HyperParameters2(n_input, n_hidden, n_output, eta, max_epoch, batch_size, eps, NetType.BinaryClassifier, InitialMethod.Xavier)
    net = NeuralNet2(hp, "Arc_221")
    net.train(dataReader, 5, True)
    net.ShowTrainingTrace()
```

此处的代码有几个需要强调的细节：

- `n_input = dataReader.num_feature`，值为2，而且必须为2，因为只有两个特征值
- `n_hidden=2`，这是人为设置的隐层神经元数量，可以是大于2的任何整数
- `eps`精度=0.08是后验知识，笔者通过测试得到的停止条件，用于方便案例讲解
- 网络类型是`NetType.BinaryClassifier`，指明是二分类网络

#### 2 运行结果

经过快速的迭代，训练完毕后，会显示损失函数曲线和准确率曲线如图10-15。

<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/10/sin_loss.png" />

图10-15 训练过程中的损失函数值和准确率值的变化

蓝色的线条是小批量训练样本的曲线，波动相对较大，不必理会，因为批量小势必会造成波动。红色曲线是验证集的走势，可以看到二者的走势很理想，经过一小段时间的磨合后，从第200个`epoch`开始，两条曲线都突然找到了突破的方向，然后只用了50个`epoch`，就迅速达到指定精度。

同时在控制台会打印一些信息，最后几行如下：

```
......
epoch=259, total_iteration=18719
loss_train=0.092687, accuracy_train=1.000000
loss_valid=0.074073, accuracy_valid=1.000000
W= [[ 8.88189429  6.09089509]
 [-7.45706681  5.07004428]]
B= [[ 1.99109895 -7.46281087]]
W= [[-9.98653838]
 [11.04185384]]
B= [[3.92199463]]
testing...
1.0
```
一共用了260个`epoch`，达到了指定的loss精度（0.08）时停止迭代。看测试集的情况，准确度1.0，即100%分类正确。
### 3.梯度检查


#### 导数概念回忆

$$
f'(x)=\lim_{h \to 0} \frac{f(x+h)-f(x)}{h} \tag{1}
$$

其含义就是$x$的微小变化$h$（$h$为无限小的值），会导致函数$f(x)$的值有多大变化。在`Python`中可以这样实现：

```Python
def numerical_diff(f, x):
    h = 1e-5
    d = (f(x+h) - f(x))/h
    return d
```

因为计算机的舍入误差的原因，`h`不能太小，比如`1e-10`，会造成计算结果上的误差，所以我们一般用`[1e-4,1e-7]`之间的数值。

但是如果使用上述方法会有一个问题，如图12-4所示。

<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/12/grad_check.png" ch="500" />

图12-4 数值微分方法

红色实线为真实的导数切线，蓝色虚线是上述方法的体现，即从$x$到$x+h$画一条直线，来模拟真实导数。但是可以明显看出红色实线和蓝色虚线的斜率是不等的。因此我们通常用绿色的虚线来模拟真实导数，公式变为：

$$
f'(x) = \lim_{h \to 0} \frac{f(x+h)-f(x-h)}{2h} \tag{2}
$$

公式2被称为双边逼近方法。

用双边逼近形式会比单边逼近形式的误差小100~10000倍左右，可以用泰勒展开来证明。


  
## 深度神经网络框架设计

### 1 功能/模式分析

比较第12章中的三层神经网络的代码，我们可以看到大量的重复之处，比如前向计算中：

```Python
def forward3(X, dict_Param):
    ...
    # layer 1
    Z1 = np.dot(W1,X) + B1
    A1 = Sigmoid(Z1)
    # layer 2
    Z2 = np.dot(W2,A1) + B2
    A2 = Tanh(Z2)
    # layer 3
    Z3 = np.dot(W3,A2) + B3
    A3 = Softmax(Z3)
    ...    
```

1，2，3三层的模式完全一样：矩阵运算+激活/分类函数。

再看看反向传播：

```Python
def backward3(dict_Param,cache,X,Y):
    ...
    # layer 3
    dZ3= A3 - Y
    dW3 = np.dot(dZ3, A2.T)
    dB3 = np.sum(dZ3, axis=1, keepdims=True)
    # layer 2
    dZ2 = np.dot(W3.T, dZ3) * (1-A2*A2) # tanh
    dW2 = np.dot(dZ2, A1.T)
    dB2 = np.sum(dZ2, axis=1, keepdims=True)
    # layer 1
    dZ1 = np.dot(W2.T, dZ2) * A1 * (1-A1)   #sigmoid
    dW1 = np.dot(dZ1, X.T)
    dB1 = np.sum(dZ1, axis=1, keepdims=True)
    ...
```
每一层的模式也非常相近：计算本层的`dZ`，再根据`dZ`计算`dW`和`dB`。

因为三层网络比两层网络多了一层，所以会在初始化、前向、反向、更新参数等四个环节有所不同，但却是有规律的。再加上前面章节中，为了实现一些辅助功能，我们已经写了很多类。所以，现在可以动手搭建一个深度学习的迷你框架了。

### 2 抽象与设计

图14-1是迷你框架的模块化设计，下面对各个模块做功能点上的解释。

<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/14/class.png" />

图14-1 迷你框架设计


##  回归任务功能测试

在第九章中，我们用一个两层的神经网络，验证了万能近似定理。当时是用hard code方式写的，现在我们用迷你框架来搭建一下。

### 1 搭建模型

这个模型很简单，一个双层的神经网络，第一层后面接一个Sigmoid激活函数，第二层直接输出拟合数据，如图14-2所示。

<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/14/ch09_net.png" />

图14-2 完成拟合任务的抽象模型

```Python
def model():
    dataReader = LoadData()
    num_input = 1
    num_hidden1 = 4
    num_output = 1

    max_epoch = 10000
    batch_size = 10
    learning_rate = 0.5

    params = HyperParameters_4_0(
        learning_rate, max_epoch, batch_size,
        net_type=NetType.Fitting,
        init_method=InitialMethod.Xavier,
        stopper=Stopper(StopCondition.StopLoss, 0.001))

    net = NeuralNet_4_0(params, "Level1_CurveFittingNet")
    fc1 = FcLayer_1_0(num_input, num_hidden1, params)
    net.add_layer(fc1, "fc1")
    sigmoid1 = ActivationLayer(Sigmoid())
    net.add_layer(sigmoid1, "sigmoid1")
    fc2 = FcLayer_1_0(num_hidden1, num_output, params)
    net.add_layer(fc2, "fc2")

    net.train(dataReader, checkpoint=100, need_test=True)

    net.ShowLossHistory()
    ShowResult(net, dataReader)
```

超参数说明：

1. 输入层1个神经元，因为只有一个`x`值
2. 隐层4个神经元，对于此问题来说应该是足够了，因为特征很少
3. 输出层1个神经元，因为是拟合任务
4. 学习率=0.5
5. 最大`epoch=10000`轮
6. 批量样本数=10
7. 拟合网络类型
8. Xavier初始化
9. 绝对损失停止条件=0.001

### 2 训练结果

<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/14/ch09_loss.png" />

图14-3 训练过程中损失函数值和准确率的变化

如图14-3所示，损失函数值在一段平缓期过后，开始陡降，这种现象在神经网络的训练中是常见的，最有可能的是当时处于一个梯度变化的平缓地带，算法在艰难地寻找下坡路，然后忽然就找到了。这种情况同时也带来一个弊端：我们会经常遇到缓坡，到底要不要还继续训练？是不是再坚持一会儿就能找到出路呢？抑或是模型能力不够，永远找不到出路呢？这个问题没有准确答案，只能靠试验和经验了。

<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/14/ch09_result.png" />

图14-4 拟合结果

图14-4左侧子图是拟合的情况，绿色点是测试集数据，红色点是神经网路的推理结果，可以看到除了最左侧开始的部分，其它部分都拟合的不错。注意，这里我们不是在讨论过拟合、欠拟合的问题，我们在这个章节的目的就是更好地拟合一条曲线。

图14-4右侧的子图是用下面的代码生成的：

```Python
    y_test_real = net.inference(dr.XTest)
    axes.scatter(y_test_real, y_test_real-dr.YTestRaw, marker='o')
```

以测试集的真实值为横坐标，以真实值和预测值的差为纵坐标。最理想的情况是所有点都在y=0处排成一条横线。从图上看，真实值和预测值二者的差异明显，但是请注意横坐标和纵坐标的间距相差一个数量级，所以差距其实不大。

再看打印输出的最后部分：

```
epoch=4999, total_iteration=449999
loss_train=0.000920, accuracy_train=0.968329
loss_valid=0.000839, accuracy_valid=0.962375
time used: 28.002626419067383
save parameters
total weights abs sum= 45.27530164993504
total weights = 8
little weights = 0
zero weights = 0
testing...
0.9817814550687021
0.9817814550687021
```

由于我们设置了`eps=0.001`，所以在5000多个`epoch`时便达到了要求，训练停止。最后用测试集得到的准确率为98.17%，已经非常不错了。如果训练更多的轮，可以得到更好的结果。
##  二分类任务功能测试

### 1 搭建模型

同样是一个双层神经网络，但是最后一层要接一个Logistic二分类函数来完成二分类任务，如图14-7所示。

<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/14/ch10_net.png" />

图14-7 完成非线性二分类教学案例的抽象模型

```Python

def model(dataReader):
    num_input = 2
    num_hidden = 3
    num_output = 1

    max_epoch = 1000
    batch_size = 5
    learning_rate = 0.1

    params = HyperParameters_4_0(
        learning_rate, max_epoch, batch_size,
        net_type=NetType.BinaryClassifier,
        init_method=InitialMethod.Xavier,
        stopper=Stopper(StopCondition.StopLoss, 0.02))

    net = NeuralNet_4_0(params, "Arc")

    fc1 = FcLayer_1_0(num_input, num_hidden, params)
    net.add_layer(fc1, "fc1")
    sigmoid1 = ActivationLayer(Sigmoid())
    net.add_layer(sigmoid1, "sigmoid1")
    
    fc2 = FcLayer_1_0(num_hidden, num_output, params)
    net.add_layer(fc2, "fc2")
    logistic = ClassificationLayer(Logistic())
    net.add_layer(logistic, "logistic")

    net.train(dataReader, checkpoint=10, need_test=True)
    return net
```

超参数说明：

1. 输入层神经元数为2
2. 隐层的神经元数为3，使用Sigmoid激活函数
3. 由于是二分类任务，所以输出层只有一个神经元，用Logistic做二分类函数
4. 最多训练1000轮
5. 批大小=5
6. 学习率=0.1
7. 绝对误差停止条件=0.02

### 2 运行结果

<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/14/ch10_loss.png" />

图14-8 训练过程中损失函数值和准确率的变化

图14-8是训练记录，再看下面的打印输出结果：

```
......
epoch=419, total_iteration=30239
loss_train=0.010094, accuracy_train=1.000000
loss_valid=0.019141, accuracy_valid=1.000000
time used: 2.149379253387451
testing...
1.0
```
最后的testing...的结果是1.0，表示100%正确，这初步说明mini框架在这个基本case上工作得很好。图14-9所示的分类效果也不错。

<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/14/ch10_result.png" ch="500" />
##  多分类任务 - MNIST手写体识别

### 1 数据读取

MNIST数据本身是图像格式的，我们用`mode="vector"`去读取，转变成矢量格式。

```Python
def LoadData():
    print("reading data...")
    dr = MnistImageDataReader(mode="vector")
    ......
```

### 2 搭建模型

一共4个隐层，都用ReLU激活函数连接，最后的输出层接Softmax分类函数。

<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/14/mnist_net.png" />

图14-18 完成MNIST分类任务的抽象模型

以下是主要的参数设置：

```Python
if __name__ == '__main__':
    dataReader = LoadData()
    num_feature = dataReader.num_feature
    num_example = dataReader.num_example
    num_input = num_feature
    num_hidden1 = 128
    num_hidden2 = 64
    num_hidden3 = 32
    num_hidden4 = 16
    num_output = 10
    max_epoch = 10
    batch_size = 64
    learning_rate = 0.1

    params = HyperParameters_4_0(
        learning_rate, max_epoch, batch_size,
        net_type=NetType.MultipleClassifier,
        init_method=InitialMethod.MSRA,
        stopper=Stopper(StopCondition.StopLoss, 0.12))

    net = NeuralNet_4_0(params, "MNIST")

    fc1 = FcLayer_1_0(num_input, num_hidden1, params)
    net.add_layer(fc1, "fc1")
    r1 = ActivationLayer(Relu())
    net.add_layer(r1, "r1")
    ......
    fc5 = FcLayer_1_0(num_hidden4, num_output, params)
    net.add_layer(fc5, "fc5")
    softmax = ClassificationLayer(Softmax())
    net.add_layer(softmax, "softmax")

    net.train(dataReader, checkpoint=0.05, need_test=True)
    net.ShowLossHistory(xcoord=XCoordinate.Iteration)
```



最后用测试集得到的准确率为96.97%。
##  权重矩阵初始化
### 1 零初始化

即把所有层的`W`值的初始值都设置为0。

$$
W = 0
$$

但是对于多层网络来说，绝对不能用零初始化，否则权重值不能学习到合理的结果。看下面的零值初始化的权重矩阵值打印输出：
```
W1= [[-0.82452497 -0.82452497 -0.82452497]]
B1= [[-0.01143752 -0.01143752 -0.01143752]]
W2= [[-0.68583865]
 [-0.68583865]
 [-0.68583865]]
B2= [[0.68359678]]
```

可以看到`W1`、`B1`、`W2`内部3个单元的值都一样，这是因为初始值都是0，所以梯度均匀回传，导致所有`W`的值都同步更新，没有差别。这样的话，无论多少轮，最终的结果也不会正确。

### 2 标准初始化

标准正态初始化方法保证激活函数的输入均值为0，方差为1。将W按如下公式进行初始化：

$$
W \sim N \begin{bmatrix} 0, 1 \end{bmatrix}
$$

其中的W为权重矩阵，N表示高斯分布，Gaussian Distribution，也叫做正态分布，Normal Distribution，所以有的地方也称这种初始化为Normal初始化。

一般会根据全连接层的输入和输出数量来决定初始化的细节：

$$
W \sim N
\begin{pmatrix} 
0, \frac{1}{\sqrt{n_{in}}}
\end{pmatrix}
$$

$$
W \sim U
\begin{pmatrix} 
-\frac{1}{\sqrt{n_{in}}}, \frac{1}{\sqrt{n_{in}}}
\end{pmatrix}
$$

当目标问题较为简单时，网络深度不大，所以用标准初始化就可以了。但是当使用深度网络时，会遇到如图15-1所示的问题。

<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/15/init_normal_sigmoid.png" ch="500" />

图15-1 标准初始化在Sigmoid激活函数上的表现

图15-1是一个6层的深度网络，使用全连接层+Sigmoid激活函数，图中表示的是各层激活函数的直方图。可以看到各层的激活值严重向两侧[0,1]靠近，从Sigmoid的函数曲线可以知道这些值的导数趋近于0，反向传播时的梯度逐步消失。处于中间地段的值比较少，对参数学习非常不利。

### 3 Xavier初始化方法

基于上述观察，Xavier Glorot等人研究出了下面的Xavier$^{[1]}$初始化方法。

条件：正向传播时，激活值的方差保持不变；反向传播时，关于状态值的梯度的方差保持不变。

$$
W \sim N
\begin{pmatrix}
0, \sqrt{\frac{2}{n_{in} + n_{out}}} 
\end{pmatrix}
$$

$$
W \sim U 
\begin{pmatrix}
 -\sqrt{\frac{6}{n_{in} + n_{out}}}, \sqrt{\frac{6}{n_{in} + n_{out}}} 
\end{pmatrix}
$$

其中的W为权重矩阵，N表示正态分布（Normal Distribution），U表示均匀分布（Uniform Distribution)。下同。

假设激活函数关于0对称，且主要针对于全连接神经网络。适用于tanh和softsign。

即权重矩阵参数应该满足在该区间内的均匀分布。其中的W是权重矩阵，U是Uniform分布，即均匀分布。

论文摘要：神经网络在2006年之前不能很理想地工作，很大原因在于权重矩阵初始化方法上。Sigmoid函数不太适合于深度学习，因为会导致梯度饱和。基于以上原因，我们提出了一种可以快速收敛的参数初始化方法。

Xavier初始化方法比直接用高斯分布进行初始化W的优势所在： 

一般的神经网络在前向传播时神经元输出值的方差会不断增大，而使用Xavier等方法理论上可以保证每层神经元输入输出方差一致。 

图15-2是深度为6层的网络中的表现情况，可以看到，后面几层的激活函数输出值的分布仍然基本符合正态分布，利于神经网络的学习。

<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/15/init_xavier_sigmoid.png" ch="500" />

图15-2 Xavier初始化在Sigmoid激活函数上的表现

表15-1 随机初始化和Xavier初始化的各层激活值与反向传播梯度比较

| |各层的激活值|各层的反向传播梯度|
|---|---|---|
| 随机初始化 |<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images\Images\15\forward_activation1.png"><br/>激活值分布渐渐集中|<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images\Images\15\backward_activation1.png"><br/>反向传播力度逐层衰退|
| Xavier初始化 |<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images\Images\15\forward_activation2.png"><br/>激活值分布均匀|<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images\Images\15\backward_activation2.png"><br/>反向传播力度保持不变|

但是，随着深度学习的发展，人们觉得Sigmoid的反向力度受限，又发明了ReLU激活函数。图15-3显示了Xavier初始化在ReLU激活函数上的表现。

<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/15/init_xavier_relu.png" ch="500" />

图15-3 Xavier初始化在ReLU激活函数上的表现

可以看到，随着层的加深，使用ReLU时激活值逐步向0偏向，同样会导致梯度消失问题。于是He Kaiming等人研究出了MSRA初始化法，又叫做He初始化法。

### 4 MSRA初始化方法

MSRA初始化方法$^{[2]}$，又叫做He方法，因为作者姓何。

条件：正向传播时，状态值的方差保持不变；反向传播时，关于激活值的梯度的方差保持不变。

网络初始化是一件很重要的事情。但是，传统的固定方差的高斯分布初始化，在网络变深的时候使得模型很难收敛。VGG团队是这样处理初始化的问题的：他们首先训练了一个8层的网络，然后用这个网络再去初始化更深的网络。

“Xavier”是一种相对不错的初始化方法，但是，Xavier推导的时候假设激活函数在零点附近是线性的，显然我们目前常用的ReLU和PReLU并不满足这一条件。所以MSRA初始化主要是想解决使用ReLU激活函数后，方差会发生变化，因此初始化权重的方法也应该变化。

只考虑输入个数时，MSRA初始化是一个均值为0，方差为2/n的高斯分布，适合于ReLU激活函数：

$$
W \sim N 
\begin{pmatrix} 
0, \sqrt{\frac{2}{n}} 
\end{pmatrix}
$$

$$
W \sim U 
\begin{pmatrix} 
-\sqrt{\frac{6}{n_{in}}}, \sqrt{\frac{6}{n_{out}}} 
\end{pmatrix}
$$

图15-4中的激活值从0到1的分布，在各层都非常均匀，不会由于层的加深而梯度消失，所以，在使用ReLU时，推荐使用MSRA法初始化。

<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/15/init_msra_relu.png" ch="500" />

图15-4 MSRA初始化在ReLU激活函数上的表现

对于Leaky ReLU：

$$
W \sim N \begin{bmatrix} 0, \sqrt{\frac{2}{(1+\alpha^2) \hat n_i}} \end{bmatrix}
\\\\ \hat n_i = h_i \cdot w_i \cdot d_i
\\\\ h_i: 卷积核高度，w_i: 卷积核宽度，d_i: 卷积核个数
$$
 dW_1$
6. $W_2 = W_1 - v_2 = W_1 - (\alpha v_1 +\eta dW_1) = W_1 - \alpha \cdot \eta \cdot dW_0 - \eta \cdot dW_1$
7. $dW_2 = \nabla J(w)$
8. $v_3=\alpha v_2 + \eta dW_2$
9. $W_3 = W_2 - v_3=W_2-(\alpha v_2 + \eta dW_2) = W_2 - \alpha^2 \eta dW_0 - \alpha \eta dW_1 - \eta dW_2$


根据公式(3)(4)有：

0. $v_0 = 0$
1. $dW_0 = \nabla J(w)$
2. $v_1 = \alpha v_0 - \eta \cdot dW_0 = -\eta \cdot dW_0$
3. $W_1 = W_0 + v_1=W_0 - \eta \cdot dW_0$
4. $dW_1 = \nabla J(w)$
5. $v_2 = \alpha v_1 - \eta dW_1$
6. $W_2 = W_1 + v_2 = W_1 + (\alpha v_1 - \eta dW_1) = W_1 - \alpha \cdot \eta \cdot dW_0 - \eta \cdot dW_1$
7. $dW_2 = \nabla J(w)$
8. $v_3=\alpha v_2 - \eta dW_2$
9. $W_3 = W_2 + v_3=W_2 + (\alpha v_2 - \eta dW_2) = W_2 - \alpha^2 \eta dW_0 - \alpha \eta dW_1-\eta dW_2$

通过手工推导迭代，我们得到两个结论：

1. 可以看到两种方式的第9步结果是相同的，即公式(1)(2)等同于(3)(4)
2. 与普通SGD的算法$W_3 = W_2 - \eta dW_2$相比，动量法不但每次要减去当前梯度，还要减去历史梯度$W_0,W_1$乘以一个不断减弱的因子$\alpha$，因为$\alpha$小于1，所以$\alpha^2$比$\alpha$小，$\alpha^3$比$\alpha^2$小。这种方式的学名叫做指数加权平均。
###  梯度加速算法 NAG

Nesterov Accelerated Gradient，或者叫做Nesterov Momentum。

在小球向下滚动的过程中，我们希望小球能够提前知道在哪些地方坡面会上升，这样在遇到上升坡面之前，小球就开始减速。这方法就是Nesterov Momentum，其在凸优化中有较强的理论保证收敛。并且，在实践中Nesterov Momentum也比单纯的Momentum 的效果好。

#### 输入和参数

- $\eta$ - 全局学习率
- $\alpha$ - 动量参数，缺省取值0.9
- $v$ - 动量，初始值为0
  
#### 算法

---

临时更新：$\hat \theta = \theta_{t-1} - \alpha \cdot v_{t-1}$

前向计算：$f(\hat \theta)$

计算梯度：$g_t = \nabla_{\hat\theta} J(\hat \theta)$

计算速度更新：$v_t = \alpha \cdot v_{t-1} + \eta \cdot g_t$

更新参数：$\theta_t = \theta_{t-1}  - v_t$

---

其核心思想是：注意到 momentum 方法，如果只看 $\alpha \cdot v_{t-1}$ 项，那么当前的θ经过momentum的作用会变成 $\theta - \alpha \cdot v_{t-1}$。既然我们已经知道了下一步的走向，我们不妨先走一步，到达新的位置”展望”未来，然后在新位置上求梯度, 而不是原始的位置。

所以，同Momentum相比，梯度不是根据当前位置θ计算出来的，而是在移动之后的位置$\theta - \alpha \cdot v_{t-1}$计算梯度。理由是，既然已经确定会移动$\theta - \alpha \cdot v_{t-1}$，那不如之前去看移动后的梯度。

图15-7是NAG的前进方向。

<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/15/nag_algorithm.png" ch="500" />

图15-7 梯度加速算法的前进方向

这个改进的目的就是为了提前看到前方的梯度。如果前方的梯度和当前梯度目标一致，那我直接大步迈过去； 如果前方梯度同当前梯度不一致，那我就小心点更新。

#### 实际效果

表15-5 动量法和NAG法的比较

|算法|损失函数和准确率|
|---|---|
|Momentum|<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images\Images\15\op_momentum_ch09_loss_01.png">|
|NAG|<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images\Images\15\op_nag_ch09_loss_01.png">|

表15-9显示，使用动量算法经过2000个epoch迭代结束，NAG算法是加速的动量法，因此只用1400个epoch迭代结束。

NAG 可以使 RNN 在很多任务上有更好的表现。
## 3 自适应学习率算法

### 1 AdaGrad

Adaptive subgradient method.$^{[1]}$

AdaGrad是一个基于梯度的优化算法，它的主要功能是：它对不同的参数调整学习率，具体而言，对低频出现的参数进行大的更新，对高频出现的参数进行小的更新。因此，他很适合于处理稀疏数据。

在这之前，我们对于所有的参数使用相同的学习率进行更新。但 Adagrad 则不然，对不同的训练迭代次数t，AdaGrad 对每个参数都有一个不同的学习率。这里开方、除法和乘法的运算都是按元素运算的。这些按元素运算使得目标函数自变量中每个元素都分别拥有自己的学习率。

#### 输入和参数

- $\eta$ - 全局学习率
- $\epsilon$ - 用于数值稳定的小常数，建议缺省值为`1e-6`
- $r=0$ 初始值
  
#### 算法

---

计算梯度：$g_t = \nabla_\theta J(\theta_{t-1})$

累计平方梯度：$r_t = r_{t-1} + g_t \odot g_t$

计算梯度更新：$\Delta \theta = {\eta \over \epsilon + \sqrt{r_t}} \odot g_t$

更新参数：$\theta_t=\theta_{t-1} - \Delta \theta$

---

从AdaGrad算法中可以看出，随着算法不断迭代，$r$会越来越大，整体的学习率会越来越小。所以，一般来说AdaGrad算法一开始是激励收敛，到了后面就慢慢变成惩罚收敛，速度越来越慢。$r$值的变化如下：

0. $r_0 = 0$
1. $r_1=g_1^2$
2. $r_2=g_1^2+g_2^2$
3. $r_3=g_1^2+g_2^2+g_3^2$

在SGD中，随着梯度的增大，我们的学习步长应该是增大的。但是在AdaGrad中，随着梯度$g$的增大，$r$也在逐渐的增大，且在梯度更新时$r$在分母上，也就是整个学习率是减少的，这是为什么呢？

这是因为随着更新次数的增大，我们希望学习率越来越慢。因为我们认为在学习率的最初阶段，我们距离损失函数最优解还很远，随着更新次数的增加，越来越接近最优解，所以学习率也随之变慢。

但是当某个参数梯度较小时，累积和也会小，那么更新速度就大。
## 过拟合

### 1 拟合程度比较

在深度神经网络中，我们遇到的另外一个挑战，就是网络的泛化问题。所谓泛化，就是模型在测试集上的表现要和训练集上一样好。经常有这样的例子：一个模型在训练集上千锤百炼，能到达99%的准确率，拿到测试集上一试，准确率还不到90%。这说明模型过度拟合了训练数据，而不能反映真实世界的情况。解决过度拟合的手段和过程，就叫做泛化。

神经网络的两大功能：回归和分类。这两类任务，都会出现欠拟合和过拟合现象，如图16-1和16-2所示。

<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/16/fitting.png" />

图16-1 回归任务中的欠拟合、正确的拟合、过拟合

图16-1是回归任务中的三种情况，依次为：欠拟合、正确的拟合、过拟合。

<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/16/classification.png" />

图16-2 分类任务中的欠拟合、正确的拟合、过拟合

图16-2是分类任务中的三种情况，依次为：分类欠妥、正确的分类、分类过度。由于分类可以看作是对分类边界的拟合，所以我们经常也统称其为拟合。

上图中对于“深入敌后”的那颗绿色点样本，正确的做法是把它当作噪音看待，而不要让它对网络产生影响。而对于上例中的欠拟合情况，如果简单的（线性）模型不能很好地完成任务，我们可以考虑使用复杂的（非线性或深度）模型，即加深网络的宽度和深度，提高神经网络的能力。

但是如果网络过于宽和深，就会出现第三张图展示的过拟合的情况。

出现过拟合的原因：

1. 训练集的数量和模型的复杂度不匹配，样本数量级小于模型的参数
2. 训练集和测试集的特征分布不一致
3. 样本噪音大，使得神经网络学习到了噪音，正常样本的行为被抑制
4. 迭代次数过多，过分拟合了训练数据，包括噪音部分和一些非重要特征

既然模型过于复杂，那么我们简化模型不就行了吗？为什么要用复杂度不匹配的模型呢？有两个原因：

1. 因为有的模型以及非常成熟了，比如VGG16，可以不调参而直接用于你自己的数据训练，此时如果你的数据数量不够多，但是又想使用现有模型，就需要给模型加正则项了。
2. 使用相对复杂的模型，可以比较快速地使得网络训练收敛，以节省时间。
##  早停法 Early Stopping



###  1.理论基础

早停法，实际上也是一种正则化的策略，可以理解为在网络训练不断逼近最优解的过程种（实际上这个最优解是过拟合的），在梯度等高线的外围就停止了训练，所以其原理上和L2正则是一样的，区别在于得到解的过程。

我们把图16-21再拿出来讨论一下。

<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/16/regular0.png" />

图16-21 损失函数值的等高线图

图中所示的等高线图，是当前带噪音的样本点所组成梯度图，并不代表测试集数据，所以其中心位置也不代表这个问题的最优解。我们假设红线是最优解，则早停法的目的就是在到达红线附近时停止训练。

### 2. 算法

一般的做法是，在训练的过程中，记录到目前为止最好的validation 准确率，当连续N次Epoch（比如N=10或者更多次）没达到最佳准确率时，则可以认为准确率不再提高了。此时便可以停止迭代了（Early Stopping）。这种策略也称为“No-improvement-in-N”，N即Epoch的次数，可以根据实际情况取，如10、20、30……

算法描述如下：

***

```
初始化
    初始权重均值参数：theta = theta_0
    迭代次数：i = 0
    忍耐次数：patience = N (e.g. N=10)
    忍耐次数计数器：counter = 0
    验证集损失函数值：lastLoss = 10000 (给一个特别大的数值)

while (epoch < maxEpoch) 循环迭代训练过程
    正向计算，反向传播更新theta
    迭代次数加1：i++
    计算验证集损失函数值：newLoss = loss
    if (newLoss < lastLoss) // 新的损失值更小
        忍耐次数计数器归零：counter = 0
        记录当前最佳权重矩阵训练参数：theta_best = theta
        记录当前迭代次数：i_best = i
        更新最新验证集损失函数值：lastLoss = newLoss
    else // 新的损失值大于上一步的损失值
        忍耐次数计数器加1：counter++
        if (counter >= patience) 停止训练！！！
    end if
end while
```

***

此时，`theta_best`和`i_best`就是最佳权重值和迭代次数。




##  丢弃法 Dropout

### 1 基本原理

2012年，Alex、Hinton在其论文《ImageNet Classification with Deep Convolutional Neural Networks》中用到了Dropout算法，用于防止过拟合。

我们假设原来的神经网络是这个结构，最后输出三分类结果，如图16-24所示。

<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/16/dropout_before.png" />

图16-24 输出三分类的神经网络结构图

Dropout可以作为训练深度神经网络的一种正则方法供选择。在每个训练批次中，通过忽略一部分的神经元（让其隐层节点值为0），可以明显地减少过拟合现象。这种方式可以减少隐层节点间的相互作用，高层的神经元需要低层的神经元的输出才能发挥作用，如果高层神经元过分依赖某个低层神经元，就会有过拟合发生。在一次正向/反向的过程中，通过随机丢弃一些神经元，迫使高层神经元和其它的一些低层神经元协同工作，可以有效地防止神经元因为接收到过多的同类型参数而陷入过拟合的状态，来提高泛化程度。

丢弃后的结果如图16-25所示。

<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/16/dropout_after.png" />

图16-25 使用丢弃法的神经网络结构图

其中有叉子的神经元在本次迭代训练中被暂时的封闭了，在下一次迭代训练中，再随机地封闭一些神经元，同一个神经元也许被连续封闭两次，也许一次都没有被封闭，完全随机。封闭多少个神经元是由一个超参来控制的，叫做丢弃率。

### 2 算法与实现

#### 前向计算

正常的隐层计算公式是：

$$
Z = W \cdot X + B \tag{1}
$$

加入随机丢弃步骤后，变成了：

$$
r \sim Bernoulli(p) \tag{2}
$$
$$Y = r \cdot X \tag{3}$$
$$Z = Y \cdot W + B \tag{4}
$$

公式2是得到一个分布概率为p的伯努利分布，伯努利分布在这里可以简单地理解为0-1分布，$p=0.5$时，会以相同概率产生0、1，假设一共10个数，则：
$$
r=[0,0,1,1,0,1,0,1,1,0]
$$
或者
$$
r=[0,1,1,0,0,1,0,1,0,1]
$$
或者其它一些分布。

从公式3，Y将会是X经过r的mask的结果，1的位置保留原x值，0的位置相乘后为0。
### 3 多样本合成法

#### SMOTE

SMOTE,Synthetic Minority Over-sampling Technique$^{[1]}$，通过人工合成新样本来处理样本不平衡问题，提升分类器性能。

类不平衡现象是数据集中各类别数量不近似相等。如果样本类别之间相差很大，会影响分类器的分类效果。假设小样本数据数量极少，仅占总体的1%，所能提取的相应特征也极少，即使小样本被错误地全部识别为大样本，在经验风险最小化策略下的分类器识别准确率仍能达到99%，但在验证环节分类效果不佳。

基于插值的SMOTE方法为小样本类合成新的样本，主要思路为：

1. 定义好特征空间，将每个样本对应到特征空间中的某一点，根据样本不平衡比例确定采样倍率N；
2. 对每一个小样本类样本$(x,y)$，按欧氏距离找K个最近邻样本，从中随机选取一个样本点，假设选择的近邻点为$(x_n,y_n)$。在特征空间中样本点与最近邻样本点的连线段上随机选取一点作为新样本点，满足以下公式:

$$(x_{new},y_{new})=(x,y)+rand(0,1)\times ((x_n-x),(y_n-y))$$

3. 重复选取取样，直到大、小样本数量平衡。

在`python`中，SMOTE算法已经封装到了`imbalanced-learn`库中。

#### SamplePairing

SamplePairing$^{[2]}$方法的处理流程如图16-35所示，从训练集中随机抽取两张图片分别经过基础数据增强操作（如随机翻转等）处理后经像素取平均值的形式叠加合成一个新的样本，标签为原样本标签中的一种。

<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/16/sample_pairing.png" />

图16-35 SamplePairing方法的处理流程

经SamplePairing处理后可使训练集的规模从N扩增到N*N，在CPU上也能完成处理。

训练过程是交替禁用与使用SamplePairing处理操作的结合：

1. 使用传统的数据增强训练网络，不使用SamplePairing 数据增强训练。
2. 在ILSVRC数据集上完成一个epoch或在其他数据集上完成100个epoch后，加入SamplePairing 数据增强训练。
3. 间歇性禁用 SamplePairing。对于 ILSVRC 数据集，为其中的300000 个图像启用SamplePairing，然后在接下来的100000个图像中禁用它。对于其他数据集，在开始的8个epoch中启用，在接下来的2个epoch中禁止。
4. 在训练损失函数和精度稳定后进行微调，禁用SamplePairing。

实验结果表明，因SamplePairing数据增强操作可能引入不同标签的训练样本，导致在各数据集上使用SamplePairing训练的误差明显增加，而在检测误差方面使用SamplePairing训练的验证误差有较大幅度降低。

尽管SamplePairing思路简单，性能上提升效果可观，符合奥卡姆剃刀原理，遗憾的是的可解释性不强，目前尚缺理论支撑。目前仅有图片数据的实验，还需下一步的实验与解读。

#### Mixup

Mixup$^{[3]}$是基于邻域风险最小化（VRM）原则的数据增强方法，使用线性插值得到新样本数据。在邻域风险最小化原则下，根据特征向量线性插值将导致相关目标线性插值的先验知识，可得出简单且与数据无关的mixup公式：

$$
x_n=\lambda x_i + (1-\lambda)x_j \\\\
y_n=\lambda y_i + (1-\lambda)y_j
$$

其中$(x_n，y_n)$是插值生成的新数据，$(x_i,y_i)$和$(x_j，y_j)$是训练集中随机选取的两个数据，λ的取值满足贝塔分布，取值范围介于0到1，超参数α控制特征目标之间的插值强度。

Mixup的实验丰富，实验结果表明可以改进深度学习模型在ImageNet数据集、CIFAR数据集、语音数据集和表格数据集中的泛化误差，降低模型对已损坏标签的记忆，增强模型对对抗样本的鲁棒性和训练对抗生成网络的稳定性。

Mixup处理实现了边界模糊化，提供平滑的预测效果，增强模型在训练数据范围之外的预测能力。随着超参数α增大，实际数据的训练误差就会增加，而泛化误差会减少。说明Mixup隐式地控制着模型的复杂性。随着模型容量与超参数的增加，训练误差随之降低。

尽管有着可观的效果改进，但mixup在偏差—方差平衡方面尚未有较好的解释。在其他类型的有监督学习、无监督、半监督和强化学习中，Mixup还有很大的发展空间。
#### 手写数字识别
手写数字识别，需要用到监督式学习方法。
监督式学习的基本方法，核心是x*w=y的公式。从模型训练，模型评估再到模型预测。

监督式学习的核心思想就是用已知的数据xy训练出模型w，然后再用w作出预测和判断。
我们运用x*y=w来理解相关问题：
在机器翻译问题中左边的x是英文，右边的y是中文，
在语音识别问题中左边的x是语音，右边的y是文本
![yang](./photo/2.jpg)
手写数字识别，可以采用图像识别的方法，左边的x是手写之后的图像，右边的y是对应的数字。接下来是重点，如果机器学习的各种内容是一栋大厦，那这一点就 是奠基石。对于图像语音文字这些信息计算机都是用数值来进行表示的，机器学习让计算机具备智能，实际上是训练出数值模型w对于新的输入x，可以通过与w进行数值运算得到y从而进行预测判断的。
手写数字的识别，是依靠一个训练的数据集，该数据集经过许多次的训练，使机器学会了对于数据集中相似的识别，这样，我们通过使用该数据集，就可以进行手写数字的识别了。
下图为手写数字识别的实验结果：

![yang](./photo/1.png)
### 学习心得
这学期所学的《人工智能概论》让我深刻的体会到了人类无穷的智慧从中我学到了不少的知识。 
人工智能英文缩写为AI。它是研究、开发用于模拟、延伸和扩展人的智能的理论、方法、技术及应用系统的一门新的技术科学。 人工智能是计算机科学的一个分支。它企图了解智能的实质并生产出一种新的能以人类智能相似的方式作出反应的智能机器。
通过学习我了解到了人工智能所用到的一些更深层次的算法，例如神经法，最小二乘法，梯度下降法等等。。。。 通过这些方法，我更加深刻的了解了人工智能所蕴含的科学依据以及道理，并使我对人工智能产生浓厚的兴趣。
人工智能的定义可以分为两部分，即“人工”和“智能”。“人工”比较好理解，争议性也不大。有时我们会要考虑什么是人力所能及制造的，或者人自身的智能程度有没有高到可以创造人工智能的地步，等等。但总的来说，“人工系统”就是通常意义下的人工系统。
人工智能目前在计算机领域内，得到了愈加广泛的重视。并在机器人，经济政治决策，控制系统，仿真系统中得到应用。如今，人工智能研究出现了新的高潮，这一方面是因为在人工智能理论方面有了新的进展，另一方面也是因为计算机硬件突飞猛进的发展。随着计算机速度的不断提高、存储容量的不断扩大、价格的不断降低以及网络技
术的不断发展，许多原来无法完成的工作现在已经能够实现。
人工智能的学习，让我明白了人工智能始终处于计算机发展的最前沿。高级计算机语言、计算机界面及文字处理器的存在或多或少都得归功于人工智能的研究。人工智能研究带来的理论和洞察力指引了计算技术发展的未来方向。现有的人工智能产品相对于即将到来的人工智能应用可以说微不足道，但是它们预示着人工智能的未来。将来我们会对人工智有能更高层次的需人工智能也会继续影响我们的工作、学习和生活，我们也要支持人工智能的发展。
 