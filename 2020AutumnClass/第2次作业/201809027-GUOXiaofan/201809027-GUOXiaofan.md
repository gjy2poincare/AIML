# STEP $2$ & STEP $3$预习 ##
智能181 郭小凡 201809027
## 4 线性回归 ##
1. 单层的神经网络：即一个神经元，可完成一些线性工作。
2. 单变量线性回归：当神经元只接收一个输入时。
3. 多变量线性回归：当神经元接收多个变量输入时。

## 4.0 单变量线性回归问题 ##
### 一元线性回归模型 ###
**回归分析**是一种数学模型。当因变量和自变量为线性关系时，它是一种特殊的线性模型。
最简单的情形是**一元线性回归**，由大体上有线性关系的一个自变量和一个因变量组成。
一元线性回归数学模型为：
$$Y=a+bX+\varepsilon $$
其中：
1. $X$ ：自变量。
2. $Y$ ：因变量。
3. $\varepsilon$ ：随机误差。
4. $a$ ， $b$ ：参数。在线性回归模型中$a,b$需要通过算法学习得到。
   
### 线性回归模型有关概念 ###
- 通常假定随机误差 $\varepsilon$ 的均值为 $0$，方差为$σ^2$（$σ^2>0$，$σ^2$ 与 $X$ 的值无关）
- 若进一步假定随机误差遵从正态分布，就叫做正态线性模型
- 一般地，若有 $k$ 个自变量和 $1$ 个因变量（即公式1中的 $Y$），则因变量的值分为两部分：一部分由自变量影响，即表示为它的函数，函数形式已知且含有未知参数；另一部分由其他的未考虑因素和随机性影响，即随机误差
- 当函数为参数未知的线性函数时，称为线性回归分析模型
- 当函数为参数未知的非线性函数时，称为非线性回归分析模型
- 当自变量个数大于 $1$ 时称为多元回归
- 当因变量个数大于 $1$ 时称为多重回归

## 4.1 最小二乘法 ##
最小二乘法数学模型为：
$$z_i=w \cdot x_i+b $$
使得：
$$z_i \simeq y_i $$
求 $w$ & $b$ ：均方差(MSE - mean squared error)
$$J = \frac{1}{2m}\sum_{i=1}^m(z_i-y_i)^2 = \frac{1}{2m}\sum_{i=1}^m(y_i-wx_i-b)^2 $$
可得：
$$w = \frac{m\sum_{i=1}^m x_i y_i - \sum_{i=1}^m x_i \sum_{i=1}^m y_i}{m\sum_{i=1}^m x^2_i - (\sum_{i=1}^m x_i)^2} $$
$$b= \frac{1}{m} \sum_{i=1}^m(y_i-wx_i) $$
其中：
1. $x_i$ :样本特征值。
2. $y_i$ 是样本标签值。
3. $z_i$ 是模型预测值。
4. $J$：损失函数。

## 4.2 梯度下降法 ##
梯度下降法可用来求解 $w$ 和 $b$。
对于预设线性函数：
$$z_i = x_i \cdot w + b $$
有误差函数均方误差：
$$loss_i(w,b) = \frac{1}{2} (z_i-y_i)^2 $$
其中：
1. $x$ ：样本特征值（单特征）。
2. $y$ ：样本标签值。
3. $z$ ：预测值。
4. 下标 $i$ ：其中一个样本。
#### 计算 $w$ 的梯度 ####

$$\frac{\partial{loss}}{\partial{w}} = \frac{\partial{loss}}{\partial{z_i}}\frac{\partial{z_i}}{\partial{w}}=(z_i-y_i)x_i $$

#### 计算 $b$ 的梯度 ####

$$\frac{\partial{loss}}{\partial{b}} = \frac{\partial{loss}}{\partial{z_i}}\frac{\partial{z_i}}{\partial{b}}=z_i-y_i $$

## 4.3 神经网络法 ##
**神经网络线性拟合原理**：
1. 初始化权重值
2. 根据权重值放出一个解
3. 根据均方差函数求误差
4. 误差反向传播给线性计算部分以调整权重值
5. 判断是否满足终止条件；若不满足则跳回2

### 神经网络结构 ###

<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/4/Setup.png" ch="500" />

                图4.3.1 单层单点神经元

#### 输入层 ####
此神经元在输入层只接受一个输入特征，经过参数 $w,b$ 的计算后，直接输出结果。
严格来说输入层在神经网络中并不能称为一个层。
#### 权重 $w,b$ ####
在一元线性问题中 $w,b$ 均为标量。
#### 输出层 ####
输出层 $1$ 个神经元，线性预测公式是：

$$z_i = x_i \cdot w + b$$

$z$ 是模型的预测输出，$y$ 是实际的样本标签值，下标 $i$ 为样本。

#### 损失函数
在线性回归问题中损失函数使用均方差函数，即：
$$loss(w,b) = \frac{1}{2} (z_i-y_i)^2$$

### 反向传播 ###
反向传播算法也可用来求解 $w$ 和 $b$。
#### 计算 $w$ 的梯度 ####
$$
{\partial{loss} \over \partial{w}} = \frac{\partial{loss}}{\partial{z_i}}\frac{\partial{z_i}}{\partial{w}}=(z_i-y_i)x_i
$$
#### 计算 $b$ 的梯度 ####
$$
\frac{\partial{loss}}{\partial{b}} = \frac{\partial{loss}}{\partial{z_i}}\frac{\partial{z_i}}{\partial{b}}=z_i-y_i
$$

## 4.4 多样本计算 ##
单样本计算存在缺点：
1. 前后相邻的样本可能会对反向传播产生相反的作用而互相抵消。
2. 在样本数据量大时，逐个计算花费时间长。

### 前向计算 ###
对于数学模型：
$$
Z = X \cdot w + b 
$$

把它展开成i行，（每行代表一个样本,此处以i=3为例）的形式：

$$
X=\begin{pmatrix}
    x_1 \\\\ 
    x_2 \\\\ 
    x_3
\end{pmatrix}
$$

$$
Z= 
\begin{pmatrix}
    x_1 \\\\ 
    x_2 \\\\ 
    x_3
\end{pmatrix} \cdot w + b 
=\begin{pmatrix}
    x_1 \cdot w + b \\\\ 
    x_2 \cdot w + b \\\\ 
    x_3 \cdot w + b
\end{pmatrix}
=\begin{pmatrix}
    z_1 \\\\ 
    z_2 \\\\ 
    z_3
\end{pmatrix} 
$$
其中：
1. $X$ 是样本组成的矩阵
2. $x_i$ ：第 $i$ 个样本
3. $Z$ ：计算结果矩阵
4. $w$ & $b$ ：均为标量。

### 损失函数 ###
损失函数为：
$$J(w,b) = \frac{1}{2m}\sum_{i=1}^{m}(z_i - y_i)^2$$
其中：
1. $z$ ：每一次迭代的预测输出。
2. $y$ ：样本标签数据。
3. $m$ ：参与计算样本个数。
   
### $w$ 的梯度 ###
$J$ 对 $w$ 的偏导为：

$$
\begin{aligned}
\frac{\partial{J}}{\partial{w}}&=\frac{\partial{J}}{\partial{z_1}}\frac{\partial{z_1}}{\partial{w}}+\frac{\partial{J}}{\partial{z_2}}\frac{\partial{z_2}}{\partial{w}}+\frac{\partial{J}}{\partial{z_3}}\frac{\partial{z_3}}{\partial{w}} \\\\
&=\frac{1}{3}[(z_1-y_1)x_1+(z_2-y_2)x_2+(z_3-y_3)x_3] \\\\
&=\frac{1}{3}
\begin{pmatrix}
    x_1 & x_2 & x_3
\end{pmatrix}
\begin{pmatrix}
    z_1-y_1 \\\\
    z_2-y_2 \\\\
    z_3-y_3 
\end{pmatrix} \\\\
&=\frac{1}{m} \sum^m_{i=1} (z_i-y_i)x_i \\\\ 
&=\frac{1}{m} X^{\top} \cdot (Z-Y) \\\\ 
\end{aligned} 
$$
其中：
$$X = 
\begin{pmatrix}
    x_1 \\\\ 
    x_2 \\\\ 
    x_3
\end{pmatrix}, X^{\top} =
\begin{pmatrix}
    x_1 & x_2 & x_3
\end{pmatrix}
$$

### $b$ 的梯度 ###

$$
\begin{aligned}    
\frac{\partial{J}}{\partial{b}}&=\frac{\partial{J}}{\partial{z_1}}\frac{\partial{z_1}}{\partial{b}}+\frac{\partial{J}}{\partial{z_2}}\frac{\partial{z_2}}{\partial{b}}+\frac{\partial{J}}{\partial{z_3}}\frac{\partial{z_3}}{\partial{b}} \\\\
&=\frac{1}{3}[(z_1-y_1)+(z_2-y_2)+(z_3-y_3)] \\\\
&=\frac{1}{m} \sum^m_{i=1} (z_i-y_i) \\\\ 
&=\frac{1}{m}(Z-Y)
\end{aligned} 
$$

## 4.5 梯度下降的三种形式 ##
### 单样本随机梯度下降 ### 
单样本随机梯度下降SGD(Stochastic Gradient Descent)样本访问示意图：
 <img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/4/SingleSample-example.png" />

                图4.5.1 单样本访问方式

#### 特点 ####
- 训练样本：每次使用一个样本数据进行一次训练，更新一次梯度;重复以上过程。
- 优点：训练开始时损失值下降很快，随机性大，找到最优解的可能性大。
- 缺点：受单个样本的影响最大，损失函数值波动大，到后期徘徊不前，在最优解附近震荡;不能并行计算。

### 小批量样本梯度下降 ###
小批量样本梯度下降(Mini-Batch Gradient Descent)样本访问示意图：
<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/4/MiniBatch-example.png" />

                图4.5.2 小批量样本访问方式


#### 特点 ####
  - 训练样本：选择一小部分样本进行训练，更新一次梯度，然后再选取另外一小部分样本进行训练，再更新一次梯度。
  - 优点：不受单样本噪声影响，训练速度较快。
  - 缺点：batch size的数值选择很关键，会影响训练结果。

#### 小批量的大小通常决定因素 ####
- 更大的批量会计算更精确的梯度，但回报小于线性。
- 极小批量通常难以充分利用多核架构。这决定了最小批量的数值，低于这个值的小批量处理不会减少计算时间。
- 若批量处理中的所有样本可以并行地处理，那么内存消耗和批量大小成正比。对于多硬件设施，这是批量大小的限制因素。
- 某些硬件上使用特定大小的数组时，运行时间会更少。

**在实际工程中，我们通常使用小批量梯度下降形式。**

### 全批量样本梯度下降 ###
全批量样本梯度下降(Full Batch Gradient Descent)样本访问示意图如下：
<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/4/FullBatch-example.png" />

             图4.5.3  全批量样本访问方式


#### 特点 ####
  - 训练样本：每次使用全部数据集进行一次训练，更新一次梯度，重复以上过程。
  - 优点：受单个样本的影响最小，一次计算全体样本速度快，损失函数值没有波动，到达最优点平稳。方便并行计算。
  - 缺点：数据量较大时不能实现（内存限制），训练过程变慢。初始值不同，可能导致获得局部最优解，并非全局最优解。

### 三种方式比较 ###

||单样本|小批量|全批量|
|---|---|---|---|
|梯度下降过程图解|<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/4/SingleSample-Trace.png"/>|<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/4/MiniBatch-Trace.png"/>|<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/4/FullBatch-Trace.png"/>|
|批大小|1|10|100|
|学习率|0.1|0.3|0.5|
|迭代次数|304|110|60|
|epoch|3|10|60|
|结果|w=2.003, b=2.990|w=2.006, b=2.997|w=1.993, b=2.998|
从结果来看，三种方式的结果都接近于 $w=2,b=3$ 的原始解。

## 5.0 多变量线性回归问题 ##
典型的多元线性回归，即包括两个或两个以上自变量的回归。函数模型如下：

$$y=a_0+a_1x_1+a_2x_2+\dots+a_kx_k$$
为了保证回归模型具有优良的解释能力和预测效果，应首先注意**自变量的选择**，其准则是：

1. 自变量对因变量必须有显著的影响，并呈密切的线性相关；
2. 自变量与因变量之间的线性相关必须是真实的，而不是形式上的；
3. 自变量之间应具有一定的互斥性，即自变量之间的相关程度不应高于自变量与因变量之因的相关程度；
4. 自变量应具有完整的统计数据，其预测值容易确定。

## 5.1 正规方程解法 ##
多元线性回归问题可以用正规方程来解决。这种解法可以解决下面这个公式描述的问题：

$$y=a_0+a_1x_1+a_2x_2+\dots+a_kx_k $$


## 5.2 神经网络解法 ##
多变量（多特征值）的线性回归可以被看做是一种高维空间的线性拟合。
### 神经网络结构定义 ###

定义一个如图所示的一层的神经网络：

<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/5/setup.png" ch="500" />

                图5.2.1 多入单出的单层神经元结构

该神经网络的特点为：
1. 没有中间层，只有输入项和输出层（输入项不算为一层）。
2. 输出层只有一个神经元。
3. 神经元有一个线性输出，不经过激活函数处理；即在下图中，经过 $\Sigma$ 求和得到 $Z$ 值之后，直接把 $Z$ 值输出。

#### 输入层 ####
对 $i$ 个样本，每个样本 $2$ 个特征值，X就是一个 $i \times 2$ 的矩阵：

$$
X = 
\begin{pmatrix} 
x_1 \\\\ x_2 \\\\ \vdots \\\\ x_{i}
\end{pmatrix} =
\begin{pmatrix} 
x_{1,1} & x_{1,2} \\\\
x_{2,1} & x_{2,2} \\\\
\vdots & \vdots \\\\
x_{i,1} & x_{i,2}
\end{pmatrix}
$$

$$
Y =
\begin{pmatrix}
y_1 \\\\ y_2 \\\\ \vdots \\\\ y_{i}
\end{pmatrix}=
\begin{pmatrix}
302.86 \\\\ 393.04 \\\\ \vdots \\\\ 450.59
\end{pmatrix}
$$
其中：
1. $x_1$ :第一个样本。
2. $x_{1,1}$ ：第一个样本的一个特征值。
3. $y_1$ ：第一个样本的标签值。

#### 权重 $W$ 和 $B$ ####

由于输入层是两个特征，输出层是一个变量，所以 $W$ 的形状是 $2\times 1$，而 $B$ 的形状是 $1\times 1$。

$$
W=
\begin{pmatrix}
w_1 \\\\ w_2
\end{pmatrix}
$$

$$B=(b)$$

$B$ 是个单值。
若有多个神经元，它们都会有各自的 $b$ 值。

#### 输出层 ####

由于目标是完成一个回归（拟合）任务，故输出层只有一个神经元。由于是线性的，所以没有用激活函数。
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

#### 损失函数 ####

因为是线性回归问题，所以损失函数使用均方差函数。

$$loss_i(W,B) = \frac{1}{2} (z_i-y_i)^2 $$

其中:
1. $z_i$ ：样本预测值。
2. $y_i$ ：样本的标签值。

### 反向传播 ###

#### 单样本多特征计算

前向计算（多特征值的公式）：

$$\begin{aligned}
z_i &= x_{i1} \cdot w_1 + x_{i2} \cdot w_2 + b \\\\
&=\begin{pmatrix}
  x_{i1} & x_{i2}
\end{pmatrix}
\begin{pmatrix}
  w_1 \\\\
  w_2
\end{pmatrix}+b
\end{aligned}
$$
其中：
1. $x_{i1}$ ：第 $i$ 个样本的第 $1$ 个特征值。
**因为 $x$ 有两个特征值，对应的 $W$ 也有两个权重值。**

## 5.3 样本特征数据标准化 ##
数据标准化（Normalization），又成为数据归一化。
### 标准化目的 ###
理论层面上，神经网络是以样本在事件中的统计分布概率为基础进行训练和预测的，所以它对样本数据的要求比较苛刻。
具体说明如下：

1. 样本的各个特征的取值要符合概率分布，即 $[0,1]$。
2. 样本的度量单位要相同。
3. 神经网络假设所有的输入输出数据都是标准差为1，均值为0，包括权重值的初始化，激活函数的选择，以及优化算法的设计。

4. 数值问题

    标准化可以避免一些不必要的数值问题。因为激活函数sigmoid/tanh的非线性区间大约在 $[-1.7，1.7]$。意味着要使神经元有效，线性计算输出的值的数量级应该在1（1.7所在的数量级）左右。这时如果输入较大，就意味着权值必须较小，一个较大，一个较小，两者相乘，就引起数值问题了。
    
5. 梯度更新
    
    若果输出层的数量级很大，会引起损失函数的数量级很大，这样做反向传播时的梯度也就很大，这时会给梯度的更新带来数值问题。
    
6. 学习率
   
    如果梯度非常大，学习率就必须非常小，因此，学习率（学习率初始值）的选择需要参考输入的范围，不如直接将数据标准化，这样学习率就不必再根据数据范围作调整。

### 标准化的常用方法 ###

- Min-Max标准化（离差标准化），将数据映射到 $[0,1]$ 区间

$$x_{new}=\frac{x-x_{min}}{x_{max} - x_{min}} $$

- 平均值标准化，将数据映射到[-1,1]区间
   
$$x_{new} = \frac{x - \bar{x}}{x_{max} - x_{min}} $$

- 对数转换
$$x_{new}=\ln(x_i) $$

- 反正切转换
$$x_{new}=\frac{2}{\pi}\arctan(x_i) $$

- Z-Score法

把每个特征值中的所有数据，变成平均值为0，标准差为1的数据，最后为正态分布。Z-Score规范化（标准差标准化 / 零均值标准化，其中std是标准差）：

$$x_{new} = \frac{x_i - \bar{x}}{std} $$

- 中心化，平均值为0，无标准差要求
  
$$x_{new} = x_i - \bar{x} $$

- 比例法，要求数据全是正值

$$
x_{new} = \frac{x_k}{\sum_{i=1}^m{x_i}} $$

## 6 线性分类 ##
神经网络的一个重要功能就是分类。
## 6.1 线性二分类 ##
回归问题可以分为两类：**线性回归**和**逻辑回归**。
1. 线性回归使用一条直线拟合样本数据；
2. 逻辑回归的目标是“拟合”0或1两个数值，而不是具体连续数值，也称为广义线性模型。
### 逻辑回归 ###
逻辑回归（Logistic Regression）：回归给出的结果是事件成功或失败的概率。其自变量既可以是连续的，也可以是分类的。
当因变量的类型属于二值（1/0，真/假，是/否）变量时，我们就应该使用逻辑回归。

## 6.1 二分类函数 ##
此函数对线性和非线性二分类都适用。
### 对率函数 ###
对率函数(Logistic Function)，即可以做为激活函数使用，又可以当作二分类函数使用。在二分类任务中，称其为Logistic函数；而在作为激活函数时，成为Sigmoid函数。
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

                   图6.1.1 Logistic函数图像

- 使用方式
此函数实际上是一个概率计算——它把 $(-\infty, \infty)$ 之间的任何数字都压缩到 $(0,1)$ 之间，返回一个概率值，这个概率值接近 $1$ 时，认为是正例，否则认为是负例。

### 正向传播 ###
#### 矩阵运算
$$
z=x \cdot w + b 
$$
#### 分类计算
$$
a = Logistic(z)=\frac{1}{1 + e^{-z}} 
$$
#### 损失函数计算
二分类交叉熵损失函数：
$$
loss(w,b) = -[y \ln a+(1-y) \ln(1-a)] 
$$

### 反向传播 ###
#### 求损失函数对 $a$ 的偏导
$$
\frac{\partial loss}{\partial a}=-\left[\frac{y}{a}-\frac{1-y}{1-a}\right]=\frac{a-y}{a(1-a)} 
$$
#### 求 $a$ 对 $z$ 的偏导
$$
\frac{\partial a}{\partial z}= a(1-a) 
$$
#### 求误差 $loss$ 对 $z$ 的偏导

$$
\frac{\partial loss}{\partial z}=\frac{\partial loss}{\partial a}\frac{\partial a}{\partial z}=\frac{a-y}{a(1-a)} \cdot a(1-a)=a-y 
$$

## 6.2 用神经网络实现先行二分类 ##
### 神经网络结构定义 ###
神经网络结构图如下：
<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/6/BinaryClassifierNN.png" ch="500" />

              图6.2.1 完成二分类任务的神经元结构

#### 输入层
输入经度 $x_1$ 和纬度 $x_2$ 两个特征：
$$
X=\begin{pmatrix}
x_{1} & x_{2}
\end{pmatrix}
$$

#### 权重矩阵
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

#### 输出层

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
$$
$$a = Logistic(z) $$

#### 损失函数
二分类交叉熵损失函数：
$$
loss(W,B) = -[y\ln a+(1-y)\ln(1-a)] 
$$

### 反向传播 ###
对$W$ 为一个2行1列的向量求偏导时，要对向量求导，即：

$$
\frac{\partial loss}{\partial w}=
\begin{pmatrix}
    \frac{\partial loss}{\partial w_1} \\\\ 
    \frac{\partial loss}{\partial w_2}
\end{pmatrix}
$$
$$
=\begin{pmatrix}
 \frac{\partial loss}{\partial z}\frac{\partial z}{\partial w_1} \\\\
 \frac{\partial loss}{\partial z}\frac{\partial z}{\partial w_2}   
\end{pmatrix}
=\begin{pmatrix}
    (a-y)x_1 \\\\
    (a-y)x_2
\end{pmatrix}
$$
$$
=(x_1 \ x_2)^{\top} (a-y)
$$

## 6.3 线性二分类原理 ##
### 线性分类 VS 线性回归

||线性回归|线性分类|
|---|---|---|
|相同点|需要在样本群中找到一条直线|需要在样本群中找到一条直线|
|不同点|用直线来拟合所有样本，使得各个样本到这条直线的距离尽可能最短|用直线来分割所有样本，使得正例样本和负例样本尽可能分布在直线两侧|

## 6.5 实现逻辑与或非门 ##
单层神经网络，又叫做感知机，它可以轻松实现逻辑与、或、非门。由于逻辑与、或门，需要有两个变量输入，而逻辑非门只有一个变量输入。
### 实现逻辑非门 ###
逻辑非问题的样本数据：

|样本序号|样本值$x$|标签值$y$|
|:---:|:---:|:---:|
|1|0|1|
|2|1|0|

逻辑非门神经元模型如下：
<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/6/LogicNot2.png" width="500" />
 
           图6.5.1 逻辑非门的神经元实现

### 实现逻辑与或门 ###
#### 神经元模型

神经元模型如下：

<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/6/BinaryClassifierNN.png" ch="500" />

          图6.5.2 逻辑与或门的神经元实现

#### 训练样本

每个类型的逻辑门都只有4个训练样本。四种逻辑门的样本和标签数据：
|样本|$x_1$|$x_2$|逻辑与$y$|逻辑与非$y$|逻辑或$y$|逻辑或非$y$|
|:---:|:--:|:--:|:--:|:--:|:--:|:--:|
|1|0|0|0|1|0|1|
|2|0|1|0|1|1|0|
|3|1|0|0|1|1|0|
|4|1|1|1|0|1|0|

### 五种逻辑门比较 ###

|逻辑门|分类结果|参数值|
|---|---|---|
|非|<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images\Images\6\LogicNotResult.png" width="300" height="300">|W=-12.468<br/>B=6.031|
|与|<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images\Images\6\LogicAndGateResult.png" width="300" height="300">|W1=11.757<br/>W2=11.757<br/>B=-17.804|
|与非|<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images\Images\6\LogicNandGateResult.png" width="300" height="300">|W1=-11.763<br/>W2=-11.763<br/>B=17.812|
|或|<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images\Images\6\LogicOrGateResult.png" width="300" height="300">|W1=11.743<br/>W2=11.743<br/>B=-11.738|
|或非|<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images\Images\6\LogicNorGateResult.png" width="300" height="300">|W1=-11.738<br/>W2=-11.738<br/>B=5.409|

**结论**：
1. `W1`和`W2`的值基本相同而且符号相同，说明分割线一定是135°斜率
2. 精度越高，则分割线的起点和终点越接近四边的中点0.5的位置

## 7 多入多出的单层神经网路 ##
### 7.0 线性多分类问题 ###
#### 线性多分类和非线性多分类的区别 ####
线性多分类与非线性多分类示意图：

<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/7/linear_vs_nonlinear.png" />

           图7.0.1 线性多分类与分线性多分类示意图

左侧为线性多分类，右侧为非线性多分类。它们的区别在于不同类别的样本点之间是否可以用一条直线来互相分割。对神经网络来说，线性多分类可以使用单层结构来解决，而分线性多分类需要使用双层结构。

### 7.1 多分类函数定义 ###
#### 引入Softmax ####

Softmax加了个"soft"来模拟max的行为，但同时又保留了相对大小的信息。

$$
a_j = \frac{e^{z_j}}{\sum\limits_{i=1}^m e^{z_i}}=\frac{e^{z_j}}{e^{z_1}+e^{z_2}+\dots+e^{z_m}}
$$

其中:

- $z_j$ ：对第 $j$ 项的分类原始值，即矩阵运算的结果
- $z_i$ ：参与分类计算的每个类别的原始值
- $m$ ：总分类数
- $a_j$ ：对第 $j$ 项的计算结果

### 正向传播 ###

#### 矩阵运算 ####

$$
z=x \cdot w + b 
$$

#### 分类计算

$$
a_j = \frac{e^{z_j}}{\sum\limits_{i=1}^m e^{z_i}}=\frac{e^{z_j}}{e^{z_1}+e^{z_2}+\dots+e^{z_m}} 
$$

#### 损失函数计算

计算单样本时，m是分类数：
$$
loss(w,b)=-\sum_{i=1}^m y_i \ln a_i $$

计算多样本时，m是分类数，n是样本数：
$$J(w,b) =- \sum_{j=1}^n \sum_{i=1}^m y_{ij} \log a_{ij} $$

如图所示：

<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/7/Loss-A-Z.jpg" ch="500" />

图7.1.1 Softmax在神经网络结构中的示意图

### 线性多分类实现 ###
多入多出单层神经网络结构图：
<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/7/MultipleClassifierNN.png" ch="500" />

             图7.2.1 多入多出单层神经网络


#### 输入层

输入经度 $x_1$ 和纬度 $x_2$ 两个特征：

$$
x=\begin{pmatrix}
x_1 & x_2
\end{pmatrix}
$$

#### 权重矩阵

$W$权重矩阵的尺寸，可以从前往后看，比如：输入层是2个特征，输出层是3个神经元，则$W$的尺寸就是 $2\times 3$。

$$
W=\begin{pmatrix}
w_{11} & w_{12} & w_{13}\\\\
w_{21} & w_{22} & w_{23} 
\end{pmatrix}
$$

$B$的尺寸是1x3，列数永远和神经元的数量一样，行数永远是1。

$$
B=\begin{pmatrix}
b_1 & b_2 & b_3 
\end{pmatrix}
$$

#### 输出层

输出层三个神经元，再加上一个Softmax计算，最后有$A1,A2,A3$三个输出，写作：

$$
Z = \begin{pmatrix}z_1 & z_2 & z_3 \end{pmatrix}
$$
$$
A = \begin{pmatrix}a_1 & a_2 & a_3 \end{pmatrix}
$$

其中，$Z=X \cdot W+B，A = Softmax(Z)$

## 7.3 线性多分类原理 ##
### 多分类过程 ###

1. 线性计算

$$z_1 = x_1 w_{11} + x_2 w_{21} + b_1 $$
$$z_2 = x_1 w_{12} + x_2 w_{22} + b_2 $$
$$z_3 = x_1 w_{13} + x_2 w_{23} + b_3 $$

2. 分类计算

$$
a_1=\frac{e^{z_1}}{\sum_i e^{z_i}}=\frac{e^{z_1}}{e^{z_1}+e^{z_2}+e^{z_3}}  $$
$$
a_2=\frac{e^{z_2}}{\sum_i e^{z_i}}=\frac{e^{z_2}}{e^{z_1}+e^{z_2}+e^{z_3}}  $$
$$
a_3=\frac{e^{z_3}}{\sum_i e^{z_i}}=\frac{e^{z_3}}{e^{z_1}+e^{z_2}+e^{z_3}}  $$

3. 损失函数计算

单样本时，$n$表示类别数，$j$表示类别序号：

$$
\begin{aligned}
loss(w,b)&=-(y_1 \ln a_1 + y_2 \ln a_2 + y_3 \ln a_3) \\\\
&=-\sum_{j=1}^{n} y_j \ln a_j 
\end{aligned}
$$

批量样本时，$m$ 表示样本数，$i$ 表示样本序号：

$$
\begin{aligned}
J(w,b) &=- \sum_{i=1}^m (y_{i1} \ln a_{i1} + y_{i2} \ln a_{i2} + y_{i3} \ln a_{i3}) \\\\
&=- \sum_{i=1}^m \sum_{j=1}^n y_{ij} \ln a_{ij}
\end{aligned}
 $$

## 总结 ##
在对$step 2$ 和 $step 3$的预习过程中，主要侧重对概念的初步了解和对公式的熟悉。很多涉及道德高深概念等等甚至很难看懂，故没有进行整理。
整理的缺失部分即完全未理解的部分，有待课堂上的学习。