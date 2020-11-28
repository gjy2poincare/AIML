# 第三步  线性分类

## 摘要

分类问题在很多资料中都称之为逻辑回归，Logistic Regression，其原因是使用了线性回归中的线性模型，加上一个Logistic二分类函数，共同构造了一个分类器。我们在本书中统称之为分类。

神经网络的一个重要功能就是分类。本部分中，我们从最简单的线性二分类开始学习，包括其原理，实现，训练过程，推理过程等等，并且以可视化的方式来帮助大家更好地理解这些过程。

在本部分中，我们将利用学到的二分类知识，实现逻辑与门、与非门，或门，或非门。


然后我们进入线性多分类的学习。多分类时，可以一对一、一对多、多对多

Softmax函数是多分类问题的分类函数，通过对它的分析，我们学习多分类的原理、实现、以及可视化结果，从而理解神经网络的工作方式。

# 第6章 多入单出的单层神经网路 - 线性二分类

## 6.0 线性二分类问题

### 6.01 逻辑回归模型
我们本章要学习的路径是：回归问题->逻辑回归问题->线性逻辑回归即分类问题->线性二分类问题。

表6-2示意说明了线性二分类和非线性二分类的区别。

表6-2 直观理解线性二分类与非线性二分类的区别

|线性二分类|非线性二分类|
|---|---|
|<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/6/linear_binary.png"/>|<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/10/non_linear_binary.png"/>|

我们先学习如何解决线性二分类为标题，在此基础上可以扩展为非线性二分类问题。

## 6.1 二分类函数

此函数对线性和非线性二分类都适用。

### 6.1.1 二分类函数

对率函数Logistic Function，即可以做为激活函数使用，又可以当作二分类函数使用。而在很多不太正规的文字材料中，把这两个概念混用了，比如下面这个说法：“我们在最后使用Sigmoid激活函数来做二分类”，这是不恰当的。在本书中，我们会根据不同的任务区分激活函数和分类函数这两个概念，在二分类任务中，叫做Logistic函数，而在作为激活函数时，叫做Sigmoid函数。

- Logistic函数公式

$$Logistic(z) = \frac{1}{1 + e^{-z}}$$

以下记 $a=Logistic(z)$。

- 导数

$$Logistic'(z) = a(1 - a)$$

具体求导过程可以参考8.1节。

- 输入值域

$$(-\infty, \infty)$$

- 输出值域

$$(0,1)$$

- 函数图像（图6-2）

<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/8/logistic.png" ch="500" />

图6-2 Logistic函数图像

- 使用方式

此函数实际上是一个概率计算，它把 $(-\infty, \infty)$ 之间的任何数字都压缩到 $(0,1)$ 之间，返回一个概率值，这个概率值接近 $1$ 时，认为是正例，否则认为是负例。

训练时，一个样本 $x$ 在经过神经网络的最后一层的矩阵运算结果作为输入 $z$，经过Logistic计算后，输出一个 $(0,1)$ 之间的预测值。我们假设这个样本的标签值为 $0$ 属于负类，如果其预测值越接近 $0$，就越接近标签值，那么误差越小，反向传播的力度就越小。

推理时，我们预先设定一个阈值比如 $0.5$，则当推理结果大于 $0.5$ 时，认为是正类；小于 $0.5$ 时认为是负类；等于 $0.5$ 时，根据情况自己定义。阈值也不一定就是 $0.5$，也可以是 $0.65$ 等等，阈值越大，准确率越高，召回率越低；阈值越小则相反，准确度越低，召回率越高。


### 6.1.2 正向传播

#### 矩阵运算

$$
z=x \cdot w + b \tag{1}
$$

#### 分类计算

$$
a = Logistic(z)=\frac{1}{1 + e^{-z}} \tag{2}
$$

#### 损失函数计算

二分类交叉熵损失函数：

$$
loss(w,b) = -[y \ln a+(1-y) \ln(1-a)] \tag{3}
$$

### 6.1.3 反向传播

#### 求损失函数对 $a$ 的偏导

$$
\frac{\partial loss}{\partial a}=-\left[\frac{y}{a}-\frac{1-y}{1-a}\right]=\frac{a-y}{a(1-a)} \tag{4}
$$

#### 求 $a$ 对 $z$ 的偏导

$$
\frac{\partial a}{\partial z}= a(1-a) \tag{5}
$$

#### 求误差 $loss$ 对 $z$ 的偏导

使用链式法则链接公式4和公式5：

$$
\frac{\partial loss}{\partial z}=\frac{\partial loss}{\partial a}\frac{\partial a}{\partial z}=\frac{a-y}{a(1-a)} \cdot a(1-a)=a-y \tag{6}
$$

我们惊奇地发现，使用交叉熵函数求导得到的分母，与Logistic分类函数求导后的结果，正好可以抵消，最后只剩下了 $a-y$ 这一项。真的有这么巧合的事吗？实际上这是依靠科学家们的聪明才智寻找出了这种匹配关系，以满足以下条件：

1. 损失函数满足二分类的要求，无论是正例还是反例，都是单调的；
2. 损失函数可导，以便于使用反向传播算法；
3. 计算过程非常简单，一个减法就可以搞定。

#### 多样本情况

我们用三个样本做实例化推导：

$$Z=
\begin{pmatrix}
  z_1 \\\\ z_2 \\\\ z_3
\end{pmatrix},
A=Logistic\begin{pmatrix}
  z_1 \\\\ z_2 \\\\ z_3
\end{pmatrix}=
\begin{pmatrix}
  a_1 \\\\ a_2 \\\\ a_3
\end{pmatrix}
$$

$$
\begin{aligned}
J(w,b)=&-[y_1 \ln a_1+(1-y_1)\ln(1-a_1)]  \\\\
&-[y_2 \ln a_2+(1-y_2)\ln(1-a_2)]  \\\\
&-[y_3 \ln a_3+(1-y_3)\ln(1-a_3)]  \\\\
\end{aligned}
$$
代入公式6结果：
$$ 
\frac{\partial J(w,b)}{\partial Z}=
\begin{pmatrix}
  \frac{\partial J(w,b)}{\partial z_1} \\\\
  \frac{\partial J(w,b)}{\partial z_2} \\\\
  \frac{\partial J(w,b)}{\partial z_3}
\end{pmatrix}
=\begin{pmatrix}
  a_1-y_1 \\\\
  a_2-y_2 \\\\
  a_3-y_3 
\end{pmatrix}=A-Y
$$

所以，用矩阵运算时可以简化为矩阵相减的形式：$A-Y$。

### 6.1.4 对数几率的来历

经过调整 $w$ 和 $b$ 的值，把所有正例的样本都归纳到大于 $0.5$ 的范围内，所有负例都小于 $0.5$。但是如果只说大于或者小于，无法做准确的量化计算，所以用一个对率函数来模拟。

泛化一下，如果正面的概率是 $a$，则反面的概率就是 $1-a$，则几率等于：

$$odds = \frac{a}{1-a} \tag{9}$$

上式中，如果 $a$ 是把样本 $x$ 的预测为正例的可能性，那么 $1-a$ 就是其负例的可能性，$\frac{a}{1-a}$就是正负例的比值，称为几率(odds)，它反映了 $x$作为正例的相对可能性，而对几率取对数就叫做对数几率(log odds, logit)。

假设概率如表6-3。

表6-3 概率到对数几率的对照表

|概率$a$|0|0.1|0.2|0.3|0.4|0.5|0.6|0.7|0.8|0.9|1|
|--|--|--|--|--|--|--|--|--|--|--|--|
|反概率$1-a$|1|0.9|0.8|0.7|0.6|0.5|0.4|0.3|0.2|0.1|0|
|几率 $odds$ |0|0.11|0.25|0.43|0.67|1|1.5|2.33|4|9|$\infty$|
|对数几率 $\ln(odds)$|N/A|-2.19|-1.38|-0.84|-0.4|0|0.4|0.84|1.38|2.19|N/A|

可以看到几率的值不是线性的，不利于分析问题，所以在表中第4行对几率取对数，可以得到一组成线性关系的值，并可以用直线方程来表示，即：

$$
\ln(odds) = \ln \frac{a}{1-a}= xw + b \tag{10}
$$

对公式10两边取自然指数：

$$
\frac{a}{1-a}=e^{xw+b} \tag{11}
$$

$$
a=\frac{1}{1+e^{-(xw+b)}}
$$

令$z=xw+b$：

$$
a=\frac{1}{1+e^{-z}} \tag{12}
$$

公式12就是公式2！对数几率的函数形式可以认为是这样得到的。

以上推导过程，实际上就是用线性回归模型的预测结果来逼近样本分类的对数几率。这就是为什么它叫做逻辑回归(logistic regression)，但其实是分类学习的方法。这种方法的优点如下：

- 把线性回归的成功经验引入到分类问题中，相当于对“分界线”的预测进行建模，而“分界线”在二维空间上是一条直线，这就不需要考虑具体样本的分布（比如在二维空间上的坐标位置），避免了假设分布不准确所带来的问题；
- 不仅预测出类别（0/1），而且得到了近似的概率值（比如0.31或0.86），这对许多需要利用概率辅助决策的任务很有用；
- 对率函数是任意阶可导的凸函数，有很好的数学性，许多数值优化算法都可以直接用于求取最优解。


## 6.2 用神经网络实现线性二分类

用神经网络在两组不同标签的样本之间画一条明显的分界线。这条分界线可以是直线，也可以是曲线。这就是二分类问题。如果只画一条分界线的话，无论是直线还是曲线，我们可以用一支假想的笔（即一个神经元），就可以达到目的，也就是说笔的走向，完全依赖于这一个神经元根据输入信号的判断。

再看楚汉城池示意图，在两个颜色区域之间似乎存在一条分割的直线，即线性可分的。

1. 从视觉上判断是线性可分的，所以我们使用单层神经网络即可；
2. 输入特征是经度和纬度，所以我们在输入层设置两个输入单元。其中$x_1=$经度，$x_2=$纬度；
3. 最后输出的是一个二分类结果，分别是楚汉地盘，可以看成非0即1的二分类问题，所以我们只用一个输出单元就可以了。

### 6.2.1 定义神经网络结构

根据前面的猜测，看来我们只需要一个二入一出的神经元就可以搞定。这个网络只有输入层和输出层，由于输入层不算在内，所以是一层网络，见图6-3。

<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/6/BinaryClassifierNN.png" ch="500" />

图6-3 完成二分类任务的神经元结构

与上一章的网络结构图的区别是，这次我们在神经元输出时使用了分类函数，所以输出为 $A$，而不是以往直接输出的 $Z$。

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
\tag{1}
$$
$$a = Logistic(z) \tag{2}$$

#### 损失函数

二分类交叉熵损失函数：

$$
loss(W,B) = -[y\ln a+(1-y)\ln(1-a)] \tag{3}
$$

### 6.2.2 反向传播

我们在6.1节已经推导了 $loss$ 对 $z$ 的偏导数，结论为 $A-Y$。接下来，我们求 $loss$ 对 $W$ 的导数。本例中，$W$ 的形式是一个2行1列的向量，所以求 $W$ 的偏导时，要对向量求导：

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
=(x_1 \ x_2)^{\top} (a-y) \tag{4}
$$

上式中$x_1,x_2$是一个样本的两个特征值。如果是多样本的话，公式4将会变成其矩阵形式，以3个样本为例：

$$
\frac{\partial J(W,B)}{\partial W}=
\begin{pmatrix}
    x_{11} & x_{12} \\\\
    x_{21} & x_{22} \\\\
    x_{31} & x_{32} 
\end{pmatrix}^{\top}
\begin{pmatrix}
    a_1-y_1 \\\\
    a_2-y_2 \\\\
    a_3-y_3 
\end{pmatrix}
=X^{\top}(A-Y) \tag{5}
$$

### 6.2.3 代码实现

我们对第5章的`HelperClass5`中，把一些已经写好的类copy过来，然后稍加改动，就可以满足我们的需要了。

由于以前我们的神经网络只会做线性回归，现在多了一个做分类的技能，所以我们加一个枚举类型，可以让调用者通过指定参数来控制神经网络的功能。

```Python
class NetType(Enum):
    Fitting = 1,
    BinaryClassifier = 2,
    MultipleClassifier = 3,
```

然后在超参类里把这个新参数加在初始化函数里：

```Python
class HyperParameters(object):
    def __init__(self, eta=0.1, max_epoch=1000, batch_size=5, eps=0.1, net_type=NetType.Fitting):
        self.eta = eta
        self.max_epoch = max_epoch
        self.batch_size = batch_size
        self.eps = eps
        self.net_type = net_type
```
再增加一个`Logistic`分类函数：

```Python
class Logistic(object):
    def forward(self, z):
        a = 1.0 / (1.0 + np.exp(-z))
        return a
```

以前只有均方差函数，现在我们增加了交叉熵函数，所以新建一个类便于管理：

```Python
class LossFunction(object):
    def __init__(self, net_type):
        self.net_type = net_type
    # end def

    def MSE(self, A, Y, count):
        ...

    # for binary classifier
    def CE2(self, A, Y, count):
        ...
```
上面的类是通过初始化时的网络类型来决定何时调用均方差函数(MSE)，何时调用交叉熵函数(CE2)的。

下面修改一下`NeuralNet`类的前向计算函数，通过判断当前的网络类型，来决定是否要在线性变换后再调用`Logistic`分类函数：

```Python
class NeuralNet(object):
    def __init__(self, params, input_size, output_size):
        self.params = params
        self.W = np.zeros((input_size, output_size))
        self.B = np.zeros((1, output_size))

    def __forwardBatch(self, batch_x):
        Z = np.dot(batch_x, self.W) + self.B
        if self.params.net_type == NetType.BinaryClassifier:
            A = Sigmoid().forward(Z)
            return A
        else:
            return Z
```

最后是主过程：

```Python
if __name__ == '__main__':
    ......
    params = HyperParameters(eta=0.1, max_epoch=100, batch_size=10, eps=1e-3, net_type=NetType.BinaryClassifier)
    ......
```

与以往不同的是，我们设定了超参中的网络类型是`BinaryClassifier`。

### 6.2.4 运行结果

图6-4所示的损失函数值记录很平稳地下降，说明网络收敛了。

<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/6/binary_loss.png" ch="500" />

图6-4 训练过程中损失函数值的变化

最后几行的打印输出：

```
......
99 19 0.20742586902509108
W= [[-7.66469954]
 [ 3.15772116]]
B= [[2.19442993]]
A= [[0.65791301]
 [0.30556477]
 [0.53019727]]
```

打印出来的`W`,`B`的值对我们来说是几个很神秘的数字，下一节再解释。`A`值是返回的预测结果：

1. 经纬度相对值为(0.58,0.92)时，概率为0.65，属于汉；
2. 经纬度相对值为(0.62,0.55)时，概率为0.30，属于楚；
3. 经纬度相对值为(0.39,0.29)时，概率为0.53，属于汉。

分类的方式是，可以指定当 $A>0.5$ 时是正例，$A\leq 0.5$ 时就是反例。有时候正例反例的比例不一样或者有特殊要求时，也可以用不是 $0.5$ 的数来当阈值。

## 6.3 线性二分类原理

### 6.3.1 线性分类和线性回归的异同

此原理对线性和非线性二分类都适用。

表6-4 线性回归和线性分类的比较

||线性回归|线性分类|
|---|---|---|
|相同点|需要在样本群中找到一条直线|需要在样本群中找到一条直线|
|不同点|用直线来拟合所有样本，使得各个样本到这条直线的距离尽可能最短|用直线来分割所有样本，使得正例样本和负例样本尽可能分布在直线两侧|

可以看到线性回归中的目标--“距离最短”，还是很容易理解的，但是线性分类的目标--“分布在两侧”，用数学方式如何描述呢？我们可以有代数和几何两种方式来描述。

### 6.3.2 二分类的代数原理

代数方式：通过一个分类函数计算所有样本点在经过线性变换后的概率值，使得正例样本的概率大于0.5，而负例样本的概率小于0.5。

#### 基本公式回顾

下面我们以单样本双特征值为例来说明神经网络的二分类过程，这是用代数方式来解释其工作原理。

1. 正向计算

$$
z = x_1 w_1+ x_2 w_2 + b  \tag{1}
$$

2. 分类计算

$$
a={1 \over 1 + e^{-z}} \tag{2}
$$

3. 损失函数计算

$$
loss = -[y \ln (a)+(1-y) \ln (1-a)] \tag{3}
$$

用图6-5举例来说明计算过程。

<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/6/sample.png" ch="500" />

图6-5 不正确的分类线试图分类红绿两色样本点

平面上有三个点，分成两类，绿色方块为正类，红色三角为负类。各个点的坐标为：$A(2,4)，B(2,1)，C(3,3)$。

#### 分类线为 $L_1$ 时

假设神经网络第一次使用 $L_1$ 做为分类线，此时：$w_1=-1,w_2=2,b=-2$，我们来计算一下三个点的情况。

$A$点：

$$
z_A = (-1)\times 2 + 2 \times 4 -2 = 4 > 0 \tag{正确}
$$

$B$点：

$$
z_B = (-1)\times 2 + 2 \times 1 -2 = -2 < 0 \tag{正确}
$$

$C$点：

$$
z_C = (-1)\times 3 + 2 \times 3 -2 = 1 > 0 \tag{错误}
$$

从6.1节中我们知道当 $z>0$ 时，$Logistic(z) > 0.5$ 为正例，反之为负例，所以我们只需要看三个点的 $z$ 值是大于0还是小于0就可以了，不用再计算 $Logistic$ 函数值。

其中，$A,B$点处于正确的分类区，而 $C$ 点处于错误的分类区。此时 $C$ 点的损失函数值为（注意 $C$ 的标签值 $y=0$）：

$$
a_C = Logistic(z_C) = 0.731
$$

$$
loss_Z = -(0 \cdot \ln(0.731) + 1 \cdot \ln(1-0.731))=1.313
$$

读者可能对 $1.313$ 这个值没有什么概念，是大还是小呢？我们不妨计算一下分类正确的 $A,B$ 点的坐标：

$$
loss_A = 0.018, \quad loss_B = 0.112
$$

可见，对于分类正确的 $A,B$ 点来说，其损失函数值比 $C$ 点要小很多，所以 $C$ 点的反向传播的力度就大。对比总结如表6-5。

表6-5 对比三个点在各个环节的计算值

|点|坐标值|$z$ 值|$a$ 值|$y$ 值|$loss$ 值|分类情况|
|---|---|---|---|---|---|---|
|A|(2,4)|4|0.982|1|0.018|正确|
|B|(2,1)|-2|0.119|0|0.112|正确|
|C|(3,3)|1|0.731|0|1.313|错误|

- 在正例情况 $y=1$ 时，$a$ 如果越靠近 $1$，表明分类越正确，此时损失值会越小。点 $A$ 就是这种情况：$a=0.982$，距离 $1$ 不远；$loss$ 值 $0.018$，很小；
- 在负例情况 $y=0$ 时，$a$ 如果越靠近 $0$，表明分类越正确，此时损失值会越小。点 $B$ 就是这种情况：$a=0.119$，距离 $0$ 不远；$loss$ 值 $0.112$，不算很大；
- 点 $C$ 是分类错误的情况，$a=0.731$，本应小于 $0.5$，实际上距离 $0$ 远，距离 $1$ 反而近，它的 $loss=1.313$，与其它两个点的相对值来看非常大，这样误差就大，反向传播的力度也大。

#### 分类线为 $L_2$ 时

我们假设经过反向传播后，神经网络把直线的位置调整到 $L_2$，以 $L_2$ 做为分类线，即 $w_1=-1,w_2=1,b=-1$，则三个点的 $z$ 值都会是符合其分类的：

$$
z_A = (-1)\times 2 + 1 \times 4 -1 = 1 > 0 \tag{正确}
$$

$$
z_B = (-1)\times 2 + 1 \times 1 -1 = -2 < 0 \tag{正确}
$$

$$
z_C = (-1)\times 3 + 1 \times 3 -1 = -1 < 0 \tag{正确}
$$

这里可能会产生一个疑问：既然用 $z$ 值是否大于0这个条件就可以判断出分类是否正确，那么二分类理论中为什么还要用 $Logistic$ 函数做一次分类呢？

原因是这样的：只有 $z$ 值的话，我们只能知道是大于0还是小于0，并不能有效地进行反向传播，也就是说我们无法告诉神经网络反向传播的误差的力度有多大。比如 $z=5$ 和 $z=-1$ 相比，难度意味着前者的力度是后者的5倍吗？

而有了 $Logistic$ 分类计算后，得到的值是一个 $(0,1)$ 之间的概率，比如：当 $z=5$ 时，$Logistic(5) = 0.993$；当 $z=-1$ 时，$Logistic(-1)=0.269$。这两个数值的含义是这两个样本在分类区内的概率，前者概率为 $99.3%$，偏向正例，后者概率为 $26.9%$，偏向负例。然后再计算损失函数，就可以得到神经网络可以理解的反向传播误差，比如上面曾经计算过的 $loss_A,loss_B,loss_C$。

### 6.3.3 二分类的几何原理

几何方式：让所有正例样本处于直线的一侧，所有负例样本处于直线的另一侧，直线尽可能处于两类样本的中间。

#### 二分类函数的几何作用

二分类函数的最终结果是把正例都映射到图6-6中的上半部分的曲线上，而把负类都映射到下半部分的曲线上。

<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/6/sigmoid_binary.png"/>

图6-6 $Logistic$ 函数把输入的点映射到 $(0,1)$ 区间内实现分类

我们用正例来举例：

$$a = Logistic(z) = \frac{1}{1 + e^{-z}} > 0.5$$

做公式变形，两边取自然对数，可以得到：

$$z > 0$$

即：
$$
z = x_1 \cdot w_1 + x_2 \cdot w_2 + b > 0
$$

对上式做一下变形，把$x_2$放在左侧，其他项放在右侧（假设$w_2>0$，则不等号方向不变）：
$$
x_2 > - \frac{w_1}{w_2}x_1 - \frac{b}{w_2} \tag{5}
$$

简化一下两个系数，令$w'=-w1/w2,b'=-b/w2$：

$$
x_2 > w' \cdot x_1 + b' \tag{6}
$$

公式6用几何方式解释，就是：有一条直线，方程为 $z = w' \cdot x_1+b'$，所有的正例样本都处于这条直线的上方；同理可得所有的负例样本都处于这条直线的下方。

<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/6/linear_binary_analysis.png" ch="500" />

图6-7 用直线分开的两类样本

我们再观察一下分类正确的图，如图6-7所示。假设绿色方块为正类：标签值 $y=1$，红色三角形为负类：标签值 $y=0$。从几何关系上理解，如果我们有一条直线，其公式为：$z = w' \cdot x_1+b'$，如图中的虚线 $L_1$ 所示，则所有正类的样本的 $x_2$ 都大于 $z$，而所有的负类样本的 $x_2$ 都小于 $z$，那么这条直线就是我们需要的分割线。

这就说明神经网络的工作原理和我们在二维平面上的直观感觉是相同的，即神经网络的工作就是找到这么一条合适的直线，尽量让所有正例样本都处于直线上方时，负例样本处于直线的下方。其实这与线性回归中找到一条直线穿过所有样本点的过程有异曲同工之处。

我们还有一个额外的收获，即：

$$w' = - w1 / w2 \tag{7}$$

$$b' = -b/w2 \tag{8}$$

我们可以使用神经网络计算出 $w_1,w_2,b$ 三个值以后，转换成 $w',b'$，以便在二维平面上画出分割线，来直观地判断神经网络训练结果的正确性。


## 6.4 二分类结果可视化

### 6.4.1 可视化的重要性
在实际的工程实践中，一般我们会把样本分成训练集、验证集、测试集，用测试集来测试训练结果的正确性。

### 6.4.2 权重值的含义

在6.2节中的训练结果如下，这几个数字如何解读呢？

```
W= [[-7.66469954]
 [ 3.15772116]]
B= [[2.19442993]]
A= [[0.65791301]
 [0.30556477]
 [0.53019727]]
``````

在6.1节中我们学习了线性二分类的原理，如果我们能够根据训练结果，在图上画出一条直线来分割正例和负例两个区域，是不是就很直观了呢？

$$
z = x_{1} \cdot w_1 + x_{2} \cdot w_2 + b \tag{1}
$$
$$
a=Logistic(z) \tag{2}
$$

对公式2来说，当 $a>0.5$ 时，属于正例（属于汉），当 $a<0.5$ 时，属于负例（属于楚）。那么 $a=0.5$ 时，就是楚汉边界啦！事实上，我们有
$$a = 0.5 \Leftrightarrow z=0$$
$$z = x_{1} \cdot w_1 + x_{2} \cdot w_2 + b = 0$$

把 $x_2$ 留在等式左侧，其它的挪到右侧去，就可以得到一条直线的方程了：

$$x_{2} \cdot w_2 = -x_{1} \cdot w_1 - b$$
$$x_2 = -\frac{w_1}{w_2}x_1 - \frac{b}{w_2} \tag{3}$$

好了，这就是标准直线方程 $y=ax+b$ 的形式了。这个公式等同于二分类原理中的公式7，8。

### 6.4.3 代码实现

用`Python`代码实现公式3如下：

```Python
def draw_split_line(net):
    b12 = -net.B[0,0]/net.W[1,0]
    w12 = -net.W[0,0]/net.W[1,0]
    print(w12,b12)
    x = np.linspace(0,1,10)
    y = w12 * x + b12
    plt.plot(x,y)
    plt.axis([-0.1,1.1,-0.1,1.1])
    plt.show()
```
上面代码中的`w12`,`b12`就是根据公式3计算得来的，只不过我们对 $W$ 的定义是$(w_1, w_2)$，而`Python`是"zero-based"，所以：
$w_1 = W[0,0],w_2 = W[0,1],b = B[0,0]$。

同时需要展示样本数据，以便判断分割线和样本数据的吻合程度：

```Python
def draw_source_data(net, dataReader):
    fig = plt.figure(figsize=(6.5,6.5))
    X,Y = dataReader.GetWholeTrainSamples()
    for i in range(200):
        if Y[i,0] == 1:
            plt.scatter(X[i,0], X[i,1], marker='x', c='g')
        else:
            plt.scatter(X[i,0], X[i,1], marker='o', c='r')
        #end if
    #end for
```

最后还可以显示一下三个预测点的位置，看看是否正确：

```Python
def draw_predicate_data(net):
    x = np.array([0.58,0.92,0.62,0.55,0.39,0.29]).reshape(3,2)
    a = net.inference(x)
    print("A=", a)
    for i in range(3):
        if a[i,0] > 0.5:
            plt.scatter(x[i,0], x[i,1], marker='^', c='g', s=100)
        else:
            plt.scatter(x[i,0], x[i,1], marker='^', c='r', s=100)
        #end if
    #end for
```
主程序：

```Python
# 主程序
if __name__ == '__main__':
    ......
    # show result
    draw_source_data(net, reader)
    draw_predicate_data(net)
    draw_split_line(net)
    plt.show()
```

### 6.4.4 运行结果

图6-8为二分类结果。

<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/6/binary_result.png" ch="500" />

图6-8 稍有欠缺的二分类结果

虽然蓝色的分割线大体分开了楚汉两国，但是细心的读者会发现在上下两端，还是绿点在分割线右侧，而红点在分割线左侧的情况。这说明我们的神经网络的训练精度不够。所以，稍微改一下超参，再训练一遍：

```Python
params = HyperParameters(eta=0.1, max_epoch=10000, batch_size=10, eps=1e-3, net_type=NetType.BinaryClassifier)
```
把`max_epoch`从`100`改成了`10000`，再跑一次。

<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/6/binary_loss_10k.png" ch="500" />

图6-9 训练过程中损失函数值的变化

从图6-9的曲线看，损失函数值一直在下降，说明网络还在继续收敛。再看图6-10的直线位置，已经比较完美地分开了红色和绿色区域。

<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/6/binary_result_10k.png" ch="500" />

图6-10 比较完美的二分类的结果

三个三角点是求解问题的三个坐标，其中第三个三角点处于分割线附近，用肉眼不是很容易分得出来，看打印输出：

```
W= [[-42.62417571]
 [ 21.36558218]]
B= [[10.5773054]]
A= [[0.99597669]
 [0.01632475]
 [0.53740392]]
w12= 1.994992477013739
b12= -0.49506282174794675
```

前两个点的概率分别是0.995和0.016，可以明确地区分正例负例，第三个点是0.537，大于0.5，可以算作正例。

在`matplot`的绘图控件中，我们也可以放大局部观察，可以图6-11的细节。

<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/6/binary_result_10k_zoom.png" ch="500" />

图6-11 放大后的细节，绿色点确实在直线左侧，分类正确

第三个点位于左侧正例区域。

好了，我们已经自信满满地找到了解释神经网络工作的原理，有数值计算验证，有公式推导，有图形显示，至少可以自圆其说了。但实际情况是不是这样呢？有没有更高深的原理还没有接触到呢？暂且留下这个问题，留在以后的章节去继续学习。


## 6.5 实现逻辑与或非门

单层神经网络，又叫做感知机，它可以轻松实现逻辑与、或、非门。由于逻辑与、或门，需要有两个变量输入，而逻辑非门只有一个变量输入。但是它们共同的特点是输入为0或1，可以看作是正负两个类别。

所以，在学习了二分类知识后，我们可以用分类的思想来实现下列5个逻辑门：

- 与门 AND
- 与非门 NAND
- 或门 OR
- 或非门 NOR
- 非门 NOT
 
以逻辑AND为例，图6-12中的4个点分别代表4个样本数据，蓝色圆点表示负例（$y=0$），红色三角表示正例（$y=1$）。

<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/6/LogicAndGateData.png" ch="500" />

图6-12 可以解决逻辑与问题的多条分割线

如果用分类思想的话，根据前面学到的知识，应该在红色点和蓝色点之间划出一条分割线来，可以正好把正例和负例完全分开。图中的三条直线，都可以是这个问题的解。

### 6.5.1 实现逻辑非门

很多阅读材料上会这样介绍：模型 $y=wx+b$，令$w=-1,b=1$，则：

- 当 $x=0$ 时，$y = -1 \times 0 + 1 = 1$
- 当 $x=1$ 时，$y = -1 \times 1 + 1 = 0$

于是有如图6-13所示的神经元结构。

<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/6/LogicNot.png" width="400"/>

图6-13 不正确的逻辑非门的神经元实现

但是，这变成了一个拟合问题，而不是分类问题。比如，令$x=0.5$，代入公式中有：

$$
y=wx+b = -1 \times 0.5 + 1 = 0.5
$$

即，当 $x=0.5$ 时，$y=0.5$，且其结果 $x$ 和 $y$ 的值并没有丝毫“非”的意思。所以，应该定义如图6-14所示的神经元来解决问题，而其样本数据也很简单，如表6-6所示，一共只有两行数据。

<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/6/LogicNot2.png" width="500" />

图6-14 正确的逻辑非门的神经元实现

表6-6 逻辑非问题的样本数据

|样本序号|样本值$x$|标签值$y$|
|:---:|:---:|:---:|
|1|0|1|
|2|1|0|

建立样本数据的代码如下：

```Python
    def Read_Logic_NOT_Data(self):
        X = np.array([0,1]).reshape(2,1)
        Y = np.array([1,0]).reshape(2,1)
        self.XTrain = self.XRaw = X
        self.YTrain = self.YRaw = Y
        self.num_train = self.XRaw.shape[0]
```

在主程序中，令：
```Python
num_input = 1
num_output = 1
```
执行训练过程，最终得到图6-16所示的分类结果和下面的打印输出结果。
```
......
2514 1 0.0020001369266925305
2515 1 0.0019993382569061806
W= [[-12.46886021]]
B= [[6.03109791]]
[[0.99760291]
 [0.00159743]]
```

<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/6/LogicNotResult.png" width="400" />

图6-15 逻辑非门的分类结果

从图6-15中，可以理解神经网络在左右两类样本点之间画了一条直线，来分开两类样本，该直线的方程就是打印输出中的W和B值所代表的直线：

$$
y = -12.468x + 6.031
$$

结果显示这不是一条垂直于 $x$ 轴的直线，而是稍微有些“歪”。这体现了神经网络的能力的局限性，它只是“模拟”出一个结果来，而不能精确地得到完美的数学公式。这个问题的精确的数学公式是一条垂直线，相当于$w=\infty$，这不可能训练得出来。

### 6.5.2 实现逻辑与或门

#### 神经元模型

依然使用第6.2节中的神经元模型，如图6-16。

<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/6/BinaryClassifierNN.png" ch="500" />

图6-16 逻辑与或门的神经元实现

因为输入特征值只有两个，输出一个二分类，所以模型和前一节的一样。

#### 训练样本

每个类型的逻辑门都只有4个训练样本，如表6-7所示。

表6-7 四种逻辑门的样本和标签数据

|样本|$x_1$|$x_2$|逻辑与$y$|逻辑与非$y$|逻辑或$y$|逻辑或非$y$|
|:---:|:--:|:--:|:--:|:--:|:--:|:--:|
|1|0|0|0|1|0|1|
|2|0|1|0|1|1|0|
|3|1|0|0|1|1|0|
|4|1|1|1|0|1|0|

#### 读取数据
  
```Python
class LogicDataReader(SimpleDataReader):
    def Read_Logic_AND_Data(self):
        X = np.array([0,0,0,1,1,0,1,1]).reshape(4,2)
        Y = np.array([0,0,0,1]).reshape(4,1)
        self.XTrain = self.XRaw = X
        self.YTrain = self.YRaw = Y
        self.num_train = self.XRaw.shape[0]

    def Read_Logic_NAND_Data(self):
        ......

    def Read_Logic_OR_Data(self):
        ......

    def Read_Logic_NOR_Data(self):        
        ......
```

以逻辑AND为例，我们从`SimpleDataReader`派生出自己的类`LogicDataReader`，并加入特定的数据读取方法`Read_Logic_AND_Data()`，其它几个逻辑门的方法类似，在此只列出方法名称。

#### 测试函数

```Python
def Test(net, reader):
    X,Y = reader.GetWholeTrainSamples()
    A = net.inference(X)
    print(A)
    diff = np.abs(A-Y)
    result = np.where(diff < 1e-2, True, False)
    if result.sum() == 4:
        return True
    else:
        return False
```

我们知道了神经网络只能给出近似解，但是这个“近似”能到什么程度，是需要我们在训练时自己指定的。相应地，我们要有测试手段，比如当输入为 $(1，1)$ 时，AND的结果是$1$，但是神经网络只能给出一个 $0.721$ 的概率值，这是不满足精度要求的，必须让4个样本的误差都小于`1e-2`。

#### 训练函数

```Python
def train(reader, title):
    ...
    params = HyperParameters(eta=0.5, max_epoch=10000, batch_size=1, eps=2e-3, net_type=NetType.BinaryClassifier)
    num_input = 2
    num_output = 1
    net = NeuralNet(params, num_input, num_output)
    net.train(reader, checkpoint=1)
    # test
    print(Test(net, reader))
    ......
```
在超参中指定了最多10000次的`epoch`，0.5的学习率，停止条件为损失函数值低至`2e-3`时。在训练结束后，要先调用测试函数，需要返回`True`才能算满足要求，然后用图形显示分类结果。

#### 运行结果

逻辑AND的运行结果的打印输出如下：

```
......
epoch=4236
4236 3 0.0019998012999365928
W= [[11.75750515]
 [11.75780362]]
B= [[-17.80473354]]
[[9.96700157e-01]
 [2.35953140e-03]
 [1.85140939e-08]
 [2.35882891e-03]]
True
```
迭代了4236次，达到精度$loss<1e-2$。当输入$(1,1)、(1,0)、(0,1)、(0,0)$四种组合时，输出全都满足精度要求。

### 6.5.3 结果比较

把5组数据放入表6-8中做一个比较。

表6-8 五种逻辑门的结果比较

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


## 6.6 用双曲正切函数做二分类函数


### 6.6.1 提出问题

在二分类问题中，一般都使用对率函数（Logistic Function，常被称为Sigmoid Function）作为分类函数，并配合二分类交叉熵损失函数：

$$a_i=Logisitc(z_i) = \frac{1}{1 + e^{-z_i}} \tag{1}$$

$$loss_i(w,b)=-[y_i \ln a_i + (1-y_i) \ln (1-a_i)] \tag{2}$$

还有一个与对率函数长得非常像的函数，即双曲正切函数（Tanh Function），公式如下：

$$Tanh(z) = \frac{e^{z} - e^{-z}}{e^{z} + e^{-z}} = \frac{2}{1 + e^{-2z}} - 1 \tag{3}$$

比较一下二者的函数图像，如表6-9。

表6-9 对率函数和双曲正切函数的比较

|对率函数|双曲正切函数|
|---|---|
|<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/6/logistic_seperator.png">|<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/6/tanh_seperator.png">|
|正负类分界线：$a=0.5$|正负类分界线：$a=0$|

对于对率函数来说，一般使用 $0.5$ 作为正负类的分界线，那我们会自然地想到，对于双曲正切函数来说，可以使用 $0$ 作为正负类的分界线。

所谓分界线，其实只是人们理解神经网络做二分类的一种方式，对于神经网络来说，其实并没有分界线这个概念，它要做的是通过线性变换，尽量把正例向上方推，把负例向下方推。

### 6.6.2 修改前向计算和反向传播函数

现在我们就开始动手实践，第一步显然是要用双曲正切来代替对率函数，修改正向计算的同时，不要忘记修改反向传播的代码。

#### 增加双曲正切分类函数

```Python
def Tanh(z):
    a = 2.0 / (1.0 + np.exp(-2*z)) - 1.0
    return a
```

#### 修改前向计算方法

软件开发中一个重要的原则是开放封闭原则：对“增加”开放，对“修改”封闭。主要是防止改出bug来。为了不修改已有的`NeuralNet`类的代码，我们派生出一个子类来，在子类中增加新的方法来覆盖父类的方法，对于其它代码来说，仍然使用父类的逻辑；对于本例来说，使用子类的逻辑。

```Python
class TanhNeuralNet(NeuralNet):
    def forwardBatch(self, batch_x):
        Z = np.dot(batch_x, self.W) + self.B
        if self.params.net_type == NetType.BinaryClassifier:
            A = Sigmoid().forward(Z)
            return A
        elif self.params.net_type == NetType.BinaryTanh:
            A = Tanh().forward(Z)
            return A
        else:
            return Z

```

子类的新方法Overwrite掉以前的前向计算方法，通过判断网络类型`NetType`参数值来调用Tanh函数。相应地，要在网络类型中增加一个枚举值：`BinaryTanh`，意为用Tanh做二分类。

```Python
class NetType(Enum):
    Fitting = 1,
    BinaryClassifier = 2,
    MultipleClassifier = 3,
    BinaryTanh = 4,
```

#### 修改反向传播方法

正向计算很容易改，反向传播需要自己推导公式！

对公式2的交叉熵函数求导，为了方便起见，我们使用单样本方式书写求导过程：

$$
\frac{\partial{loss_i}}{\partial{a_i}}= \frac{y_i-a_i}{a_i(1-a_i)} \tag{4}
$$

通常是用损失函数对Logistic函数求导，但现在我们需要用Tanh函数做分类函数，所以改成对公式3的Tanh函数求导：

$$
\frac{\partial{a_i}}{\partial{z_i}}=(1-a_i)(1+a_i) \tag{5}
$$

用链式法则结合公式4，5：

$$
\begin{aligned}    
\frac{\partial loss_i}{\partial z_i}&=\frac{\partial loss_i}{\partial a_i} \frac{\partial a_i}{\partial z_i} \\\\
&= \frac{y_i-a_i}{a_i(1-a_i)} (1+a_i)(1-a_i) \\\\
&= \frac{(a_i-y_i)(1+a_i)}{a_i}
\end{aligned} \tag{6}
$$

反向传播代码的实现，同样是在`TanhNeuralNet`子类中，写一个新的`backwardBatch`方法来覆盖父类的方法：

```Python
class TanhNeuralNet(NeuralNet):
    def backwardBatch(self, batch_x, batch_y, batch_a):
        m = batch_x.shape[0]
        dZ = (batch_a - batch_y) * (1 + batch_a) / batch_a
        dB = dZ.sum(axis=0, keepdims=True)/m
        dW = np.dot(batch_x.T, dZ)/m
        return dW, dB
```

这个实现利用了公式6的结果。再仔细地推导一遍公式，确认无误后，我们可以试着运行：

```
epoch=0
Level4_TanhAsBinaryClassifier.py:29: RuntimeWarning: divide by zero encountered in true_divide
  dZ = (batch_a - batch_y) * (1 + batch_a) / batch_a
Level4_TanhAsBinaryClassifier.py:29: RuntimeWarning: invalid value encountered in true_divide
  dZ = (batch_a - batch_y) * (1 + batch_a) / batch_a
0 1 nan
0 3 nan
0 5 nan
......
```

不出意外，出错了！看第一个错误应该是除数为0，即`batch_a`值为0。为什么在使用对率函数时没有出过这样的异常呢？原因有二：

1. 用对率函数，输出值域为 $(0,1)$，所以a值永远会大于0，不可能为0。而Tanh函数的输出值域是 $(-1,1)$，有可能是0；
2. 以前的误差项 dZ = batch_a - batch_y，并没有除法项。


### 6.6.3 修改损失函数

交叉熵函数原公式为：

$$loss_i=-[y_i \ln a_i + (1-y_i) \ln (1-a_i)]$$

改成：

$$loss_i=-[(1+y_i) \ln (1+a_i) + (1-y_i) \ln (1-a_i)] \tag{7}$$

对公式7求导：

$$
\frac{\partial loss}{\partial a_i} = \frac{2(a_i-y_i)}{(1+a_i)(1-a_i)} \tag{8}
$$


结合公式5的Tanh的导数：

$$
\begin{aligned}
\frac{\partial loss_i}{\partial z_i}&=\frac{\partial loss_i}{\partial a_i}\frac{\partial a_i}{\partial z_i} \\\\
&=\frac{2(a_i-y_i)}{(1+a_i)(1-a_i)} (1+a_i)(1-a_i) \\\\
&=2(a_i-y_i) 
\end{aligned}
\tag{9}
$$

好，我们成功地把分母消除了！现在我们需要同时修改损失函数和反向传播函数。

#### 增加新的损失函数

```Python
class LossFunction(object):
    def CE2_tanh(self, A, Y, count):
        p = (1-Y) * np.log(1-A) + (1+Y) * np.log(1+A)
        LOSS = np.sum(-p)
        loss = LOSS / count
        return loss
```
在原`LossFunction`类中，新增加了一个叫做`CE2_tanh`的损失函数，完全按照公式7实现。

#### 修改反向传播方法

```Python
class NeuralNet(object):
    def backwardBatch(self, batch_x, batch_y, batch_a):
        m = batch_x.shape[0]
        # setp 1 - use original cross-entropy function
#        dZ = (batch_a - batch_y) * (1 + batch_a) / batch_a
        # step 2 - modify cross-entropy function
        dZ = 2 * (batch_a - batch_y)
        ......
```
注意我们注释掉了step1的代码，利用公式9的结果，代替为step2的代码。

第二次运行，结果只运行了一轮就停止了。看打印信息和损失函数值，损失函数居然是个负数！

```
epoch=0
0 1 -0.1882585728753378
W= [[0.04680528]
 [0.10793676]]
B= [[0.16576018]]
A= [[0.28416676]
 [0.24881074]
 [0.21204905]]
w12= -0.4336361115243373
b12= -1.5357156668786782
```
如果忽略损失函数为负数而强行继续训练的话，可以看到损失函数图6-17。

<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/6/tanh_loss_2.png" ch="500" />

图6-17 训练过程中损失函数值的变化

从表面上看损失值不断下降，好像是收敛了，但是全程都是负数。从交叉熵损失函数的原始定义来看，其值本身应该是永远大于0的。难道是因为改了损失函数形式从而得到了负值吗？让我们再来比较一下公式2和公式7：

$$loss_i=-[y_i \ln(a_i)+(1-y_i) \ln (1-a_i)] \tag{2}$$

由于使用对率函数计算输出a，所以a的值永远在 $(0,1)$ 之间，那么1-a也在 $(0,1)$ 之间，所以 $\ln(a_i)$ 和 $\ln(1-a_i)$ 都是负数。而y的值是0或1，两项相加也是负数。最前面有个负号，所以最终loss的结果是个正数。

改成1+a后：

$$loss_i=-[(1+y_i) \ln (1+a_i) + (1-y_i) \ln (1-a_i)] \tag{7}$$

Tanh函数输出值 $a$ 为 $(-1,1)$，这样$1+a \in (0,2)$，$1-a \in (0,2)$，当处于(1,2)区间时，$ln(1+a)$ 和 $ln(1-a)$的值大于0，最终导致loss为负数。如果仍然想用交叉熵函数，必须符合其原始设计思想，让 $1+a$ 和 $1-a$ 都在 $(0,1)$ 值域内！

### 6.6.4 再次修改损失函数代码

既然 $1+a$ 和 $1-a$ 都在 $(0,2)$ 区间内，我们把它们都除以2，就可以变成 $(0,1)$ 区间了。

$$loss_i=-[(1+y_i) \ln (\frac{1+a_i}{2})+(1-y_i) \ln (\frac{1-a_i}{2})] \tag{9}$$

虽然分母有个2，但是对导数公式没有影响，最后的结果仍然是公式8的形式：

$$\frac{\partial loss_i}{\partial z_i} =2(a_i-y_i) \tag{8}$$

```Python
class LossFunction(object):
    def CE2_tanh(self, A, Y, count):
        #p = (1-Y) * np.log(1-A) + (1+Y) * np.log(1+A)
        p = (1-Y) * np.log((1-A)/2) + (1+Y) * np.log((1+A)/2)
        ......
```

注意我们注释掉了上一次的代码，增加了分母为2的代码，完全按照公式9实现。

第三次运行，终于能跑起来了，得到图6-18、图6-19所示的结果。

<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/6/tanh_loss_3.png">

图6-18 训练过程中损失函数值的变化

<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/6/tanh_result_3.png">

图6-19 有偏差的分类效果图

这次的损失函数值曲线非常好，值域正确并且收敛了。可是看图6-19的分类结果，为什么分界线整体向右偏移了呢？

这个偏移让我们想起了本节最开始logistic function和Tanh function的比较图，Tanh的输出值域在 $(-1,1)$ 之间，而logistic的输出值域在 $(0,1)$ 之间，相当于把 $(0,1)$ 拉伸为 $(-1,1)$。这与分类结果的偏移有没有关系呢？

### 6.6.5 修改样本数据标签值

仔细观察原始数据，其标签值是非0即1的，表示正负类，这符合对率函数的输出值域。而Tanh要求正负类的标签是-1和1，所以我们要把标签值改一下。

在`SimpleDataReader`类上派生出子类`SimpleDataReader_tanh`，并增加一个`ToZeroOne()`方法，目的是把原来的[0/1]标签变成[-1/1]标签。

```Python
class SimpleDataReader_tanh(SimpleDataReader):
    def ToZeroOne(self):
        Y = np.zeros((self.num_train, 1))
        for i in range(self.num_train):
            if self.YTrain[i,0] == 0:     # 第一类的标签设为0
                Y[i,0] = -1
            elif self.YTrain[i,0] == 1:   # 第二类的标签设为1
                Y[i,0] = 1
        ......
```

同时不要忘记把预测函数里的0.5变成0，因为Tanh函数的正负类分界线是0，而不是0.5。

```Python
def draw_predicate_data(net):
    ......
    for i in range(3):
        # if a[i,0] > 0.5:  # logistic function
        if a[i,0] > 0:      # tanh function
            ......
```

最后别忘记在主程序里调用修改标签值的方法：

```Python
if __name__ == '__main__':
    ......
    reader.ToZeroOne()  # change lable value from 0/1 to -1/1
    # net
    params = HyperParameters(eta=0.1, max_epoch=100, batch_size=10, eps=1e-3, net_type=NetType.BinaryTanh)
    ......
    net = TanhNeuralNet(params, num_input, num_output)
    ......
```

第四次运行！......Perfect！无论是打印输出还是最终的可视化结果图6-20都很完美。

<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/6/tanh_result_4.png" ch="500" />

图6-20 完美的分类效果图

最后我们对比一下两个分类函数以及与它们对应的交叉熵函数的图像，如表6-10。

表6-10 对比使用不同分类函数的交叉熵函数的不同 

|分类函数|交叉熵函数|
|---|---|
|<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/6/logistic_seperator.png">|<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/3/crossentropy2.png"/>|
|输出值域a在(0,1)之间，分界线为a=0.5，标签值为y=0/1|y=0为负例，y=1为正例，输入值域a在(0,1)之间，符合对率函数的输出值域|
|<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/6/tanh_seperator.png">|<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/6/modified_crossentropy.png"/>|
|输出值域a在(-1,1)之间，分界线为a=0，标签值为y=-1/1|y=-1为负例，y=1为正例，输入值域a在(-1,1)之间，符合双曲正切函数的输出值域|

可以形象地总结出，当使用Tanh函数后，相当于把Logistic的输出值域范围拉伸到2倍，下边界从0变成-1；而对应的交叉熵函数，是把输入值域的范围拉伸到2倍，左边界从0变成-1，完全与分类函数匹配。


# 第7章 多入多出的单层神经网路 - 线性多分类

## 7.0 线性多分类问题

### 7.01 多分类学习策略

#### 线性多分类和非线性多分类的区别

图7-2显示了线性多分类和非线性多分类的区别。

<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/7/linear_vs_nonlinear.png" />

图7-2 直观理解线性多分类与分线性多分类的区别

左侧为线性多分类，右侧为非线性多分类。它们的区别在于不同类别的样本点之间是否可以用一条直线来互相分割。对神经网络来说，线性多分类可以使用单层结构来解决，而分线性多分类需要使用双层结构。

#### 二分类与多分类的关系

在传统的机器学习中，有些二分类算法可以直接推广到多分类，但是在更多的时候，我们会基于一些基本策略，利用二分类学习器来解决多分类问题。

多分类问题一共有三种解法：

1. 一对一方式
   
每次先只保留两个类别的数据，训练一个分类器。如果一共有 $N$ 个类别，则需要训练 $C^2_N$ 个分类器。以 $N=3$ 时举例，需要训练 $A|B，B|C，A|C$ 三个分类器。

<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/7/one_vs_one.png" />

图7-3 一对一方式

如图7-3最左侧所示，这个二分类器只关心蓝色和绿色样本的分类，而不管红色样本的情况，也就是说在训练时，只把蓝色和绿色样本输入网络。
   
推理时，$(A|B)$ 分类器告诉你是A类时，需要到 $(A|C)$ 分类器再试一下，如果也是A类，则就是A类。如果 $(A|C)$ 告诉你是C类，则基本是C类了，不可能是B类，不信的话可以到 $(B|C)$ 分类器再去测试一下。

2. 一对多方式
   
如图7-4，处理一个类别时，暂时把其它所有类别看作是一类，这样对于三分类问题，可以得到三个分类器。

<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/7/one_vs_multiple.png" />

图7-4 一对多方式

如最左图，这种情况是在训练时，把红色样本当作一类，把蓝色和绿色样本混在一起当作另外一类。

推理时，同时调用三个分类器，再把三种结果组合起来，就是真实的结果。比如，第一个分类器告诉你是“红类”，那么它确实就是红类；如果告诉你是非红类，则需要看第二个分类器的结果，绿类或者非绿类；依此类推。

3. 多对多方式

假设有4个类别ABCD，我们可以把AB算作一类，CD算作一类，训练一个分类器1；再把AC算作一类，BD算作一类，训练一个分类器2。
    
推理时，第1个分类器告诉你是AB类，第二个分类器告诉你是BD类，则做“与”操作，就是B类。

#### 多分类与多标签

多分类学习中，虽然有多个类别，但是每个样本只属于一个类别。

有一种情况也很常见，比如一幅图中，既有蓝天白云，又有花草树木，那么这张图片可以有两种标注方法：

- 标注为“风景”，而不是“人物”，属于风景图片，这叫做分类
- 被同时标注为“蓝天”、“白云”、“花草”、“树木”等多个标签，这样的任务不叫作多分类学习，而是“多标签”学习，multi-label learning。我们此处不涉及这类问题。
  

## 7.1 多分类函数

此函数对线性多分类和非线性多分类都适用。

先回忆一下二分类问题，在线性计算后，使用了Logistic函数计算样本的概率值，从而把样本分成了正负两类。那么对于多分类问题，应该使用什么方法来计算样本属于各个类别的概率值呢？又是如何作用到反向传播过程中的呢？我们这一节主要研究这个问题。

### 7.1.1 多分类函数定义 - Softmax

#### 如何得到多分类问题的分类结果概率值？

Logistic函数可以得到诸如0.8、0.3这样的二分类概率值，前者接近1，后者接近0。那么多分类问题如何得到类似的概率值呢？

我们依然假设对于一个样本的分类值是用这个线性公式得到的：

$$
z = x \cdot w + b
$$

但是，我们要求 $z$ 不是一个标量，而是一个向量。如果是三分类问题，我们就要求 $z$ 是一个三维的向量，向量中的每个单元的元素值代表该样本分别属于三个分类的值，这样不就可以了吗？

具体的说，假设$x$是一个 (1x2) 的向量，把w设计成一个(2x3)的向量，b设计成(1x3)的向量，则z就是一个(1x3)的向量。我们假设z的计算结果是$[3,1,-3]$，这三个值分别代表了样本x在三个分类中的数值，下面我们把它转换成概率值。

#### 取max值

z值是 $[3,1,-3]$，如果取max操作会变成 $[1,0,0]$，这符合我们的分类需要，即三者相加为1，并且认为该样本属于第一类。但是有两个不足：

1. 分类结果是 $[1,0,0]$，只保留的非0即1的信息，没有各元素之间相差多少的信息，可以理解是“Hard Max”；
2. max操作本身不可导，无法用在反向传播中。

#### 引入Softmax

Softmax加了个"soft"来模拟max的行为，但同时又保留了相对大小的信息。

$$
a_j = \frac{e^{z_j}}{\sum\limits_{i=1}^m e^{z_i}}=\frac{e^{z_j}}{e^{z_1}+e^{z_2}+\dots+e^{z_m}}
$$

上式中:

- $z_j$ 是对第 $j$ 项的分类原始值，即矩阵运算的结果
- $z_i$ 是参与分类计算的每个类别的原始值
- $m$ 是总分类数
- $a_j$ 是对第 $j$ 项的计算结果

假设 $j=1,m=3$，上式为：
  
$$a_1=\frac{e^{z_1}}{e^{z_1}+e^{z_2}+e^{z_3}}$$

用图7-5来形象地说明这个过程。

<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/7/softmax.png" />

图7-5 Softmax工作过程

当输入的数据$[z_1,z_2,z_3]$是$[3,1,-3]$时，按照图示过程进行计算，可以得出输出的概率分布是$[0.879,0.119,0.002]$。

对比MAX运算和Softmax的不同，如表7-2所示。

表7-2 MAX运算和Softmax的不同

|输入原始值|MAX计算|Softmax计算|
|:---:|:---:|:---:|
|$[3, 1, -3]$|$[1, 0, 0]$|$[0.879, 0.119, 0.002]$|

也就是说，在（至少）有三个类别时，通过使用Softmax公式计算它们的输出，比较相对大小后，得出该样本属于第一类，因为第一类的值为0.879，在三者中最大。注意这是对一个样本的计算得出的数值，而不是三个样本，亦即Softmax给出了某个样本分别属于三个类别的概率。

它有两个特点：

1. 三个类别的概率相加为1
2. 每个类别的概率都大于0

#### Softmax的反向传播工作原理

我们仍假设网络输出的预测数据是 $z=[3,1,-3]$，而标签值是 $y=[1,0,0]$。在做反向传播时，根据前面的经验，我们会用 $z-y$，得到：

$$z-y=[2,1,-3]$$

在使用Softmax之后，我们得到的值是 $a=[0.879,0.119,0.002]$，用 $a-y$：

$$a-y=[-0.121, 0.119, 0.002]$$

再来分析这个信息：

- 第一项-0.121是奖励给该类别0.121，因为它做对了，但是可以让这个概率值更大，最好是1
- 第二项0.119是惩罚，因为它试图给第二类0.119的概率，所以需要这个概率值更小，最好是0
- 第三项0.002是惩罚，因为它试图给第三类0.002的概率，所以需要这个概率值更小，最好是0

这个信息是完全正确的，可以用于反向传播。Softmax先做了归一化，把输出值归一到[0,1]之间，这样就可以与标签值的0或1去比较，并且知道惩罚或奖励的幅度。

从继承关系的角度来说，Softmax函数可以视作Logistic函数扩展，比如一个二分类问题：

$$
a_1 = \frac{e^{z_1}}{e^{z_1} + e^{z_2}} = \frac{1}{1 + e^{z_2 - z_1}}
$$

是不是和Logistic函数形式非常像？其实Logistic函数也是给出了当前样本的一个概率值，只不过是依靠偏近0或偏近1来判断属于正类还是负类。

### 7.1.2 正向传播

#### 矩阵运算

$$
z=x \cdot w + b \tag{1}
$$

#### 分类计算

$$
a_j = \frac{e^{z_j}}{\sum\limits_{i=1}^m e^{z_i}}=\frac{e^{z_j}}{e^{z_1}+e^{z_2}+\dots+e^{z_m}} \tag{2}
$$

#### 损失函数计算

计算单样本时，m是分类数：
$$
loss(w,b)=-\sum_{i=1}^m y_i \ln a_i \tag{3}
$$

计算多样本时，m是分类数，n是样本数：
$$J(w,b) =- \sum_{j=1}^n \sum_{i=1}^m y_{ij} \log a_{ij} \tag{4}$$

如图7-6示意。

<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/7/Loss-A-Z.jpg" ch="500" />

图7-6 Softmax在神经网络结构中的示意图

### 7.1.3 反向传播

#### 实例化推导

我们先用实例化的方式来做反向传播公式的推导，然后再扩展到一般性上。假设有三个类别，则：

$$
z_1 = x \cdot w+ b_1 \tag{5}
$$
$$
z_2 = x \cdot w + b_2 \tag{6}
$$
$$
z_3 = x \cdot w + b_3 \tag{7}
$$
$$
a_1=\frac{e^{z_1}}{\sum_i e^{z_i}}=\frac{e^{z_1}}{e^{z_1}+e^{z_2}+e^{z_3}}  \tag{8}
$$
$$
a_2=\frac{e^{z_2}}{\sum_i e^{z_i}}=\frac{e^{z_2}}{e^{z_1}+e^{z_2}+e^{z_3}}  \tag{9}
$$
$$
a_3=\frac{e^{z_3}}{\sum_i e^{z_i}}=\frac{e^{z_3}}{e^{z_1}+e^{z_2}+e^{z_3}}  \tag{10}
$$

为了方便书写，我们令：

$$
E ={e^{z_1}+e^{z_2}+e^{z_3}}
$$

$$
loss(w,b)=-(y_1 \ln a_1 + y_2 \ln a_2 + y_3 \ln a_3)  \tag{11}
$$

$$
\frac{\partial{loss}}{\partial{z_1}}= \frac{\partial{loss}}{\partial{a_1}}\frac{\partial{a_1}}{\partial{z_1}} + \frac{\partial{loss}}{\partial{a_2}}\frac{\partial{a_2}}{\partial{z_1}} + \frac{\partial{loss}}{\partial{a_3}}\frac{\partial{a_3}}{\partial{z_1}}  \tag{12}
$$

依次求解公式12中的各项：

$$
\frac{\partial loss}{\partial a_1}=- \frac{y_1}{a_1} \tag{13}
$$
$$
\frac{\partial loss}{\partial a_2}=- \frac{y_2}{a_2} \tag{14}
$$
$$
\frac{\partial loss}{\partial a_3}=- \frac{y_3}{a_3} \tag{15}
$$

$$
\begin{aligned}
\frac{\partial a_1}{\partial z_1}&=(\frac{\partial e^{z_1}}{\partial z_1} E -\frac{\partial E}{\partial z_1}e^{z_1})/E^2 \\\\
&=\frac{e^{z_1}E - e^{z_1}e^{z_1}}{E^2}=a_1(1-a_1)  
\end{aligned}
\tag{16}
$$

$$
\begin{aligned}
\frac{\partial a_2}{\partial z_1}&=(\frac{\partial e^{z_2}}{\partial z_1} E -\frac{\partial E}{\partial z_1}e^{z_2})/E^2 \\\\
&=\frac{0 - e^{z_1}e^{z_2}}{E^2}=-a_1 a_2 
\end{aligned}
\tag{17}
$$

$$
\begin{aligned}
\frac{\partial a_3}{\partial z_1}&=(\frac{\partial e^{z_3}}{\partial z_1} E -\frac{\partial E}{\partial z_1}e^{z_3})/E^2 \\\\
&=\frac{0 - e^{z_1}e^{z_3}}{E^2}=-a_1 a_3  
\end{aligned}
\tag{18}
$$

把公式13~18组合到12中：

$$
\begin{aligned}    
\frac{\partial loss}{\partial z_1}&=-\frac{y_1}{a_1}a_1(1-a_1)+\frac{y_2}{a_2}a_1a_2+\frac{y_3}{a_3}a_1a_3 \\\\
&=-y_1+y_1a_1+y_2a_1+y_3a_1 \\\\
&=-y_1+a_1(y_1+y_2+y_3) \\\\
&=a_1-y_1 
\end{aligned}
\tag{19}
$$

不失一般性，由公式19可得：
$$
\frac{\partial loss}{\partial z_i}=a_i-y_i \tag{20}
$$

#### 一般性推导

1. Softmax函数自身的求导

由于Softmax涉及到求和，所以有两种情况：

- 求输出项 $a_1$ 对输入项 $z_1$ 的导数，此时：$j=1, i=1, i=j$，可以扩展到 $i,j$ 为任意相等值
- 求输出项 $a_2$ 或 $a_3$ 对输入项 $z_1$ 的导数，此时：$j$ 为 $2$ 或 $3$, $i=1,i \neq j$，可以扩展到 $i,j$ 为任意不等值

Softmax函数的分子：因为是计算 $a_j$，所以分子是 $e^{z_j}$。

Softmax函数的分母：
$$
\sum\limits_{i=1}^m e^{z_i} = e^{z_1} + \dots + e^{z_j} + \dots +e^{z_m} \Rightarrow E
$$

- $i=j$时（比如输出分类值 $a_1$ 对 $z_1$ 的求导），求 $a_j$ 对 $z_i$ 的导数，此时分子上的 $e^{z_j}$ 要参与求导。参考基本数学导数公式33：

$$
\begin{aligned}
\frac{\partial{a_j}}{\partial{z_i}} &= \frac{\partial{}}{\partial{z_i}}(e^{z_j}/E) \\\\
&= \frac{\partial{}}{\partial{z_j}}(e^{z_j}/E) \quad (因为z_i==z_i)\\\\
&=\frac{e^{z_j}E-e^{z_j}e^{z_j}}{E^2} 
=\frac{e^{z_j}}{E} - \frac{(e^{z_j})^2}{E^2} \\\\
&= a_j-a^2_j=a_j(1-a_j)  \\\\
\end{aligned}
\tag{21}
$$

- $i \neq j$时（比如输出分类值 $a_1$ 对 $z_2$ 的求导，$j=1,i=2$），$a_j$对$z_i$的导数，分子上的 $z_j$ 与 $i$ 没有关系，求导为0，分母的求和项中$e^{z_i}$要参与求导。同样是公式33，因为分子$e^{z_j}$对$e^{z_i}$求导的结果是0：

$$
\frac{\partial{a_j}}{\partial{z_i}}=\frac{-(E)'e^{z_j}}{E^2}
$$
求和公式对$e^{z_i}$的导数$(E)'$，除了$e^{z_i}$项外，其它都是0：
$$
(E)' = (e^{z_1} + \dots + e^{z_i} + \dots +e^{z_m})'=e^{z_i}
$$
所以：
$$
\begin{aligned}
\frac{\partial{a_j}}{\partial{z_i}}&=\frac{-(E)'e^{z_j}}{(E)^2}=-\frac{e^{z_j}e^{z_i}}{{(E)^2}} \\\\
&=-\frac{e^{z_j}}{{E}}\frac{e^{z_j}}{{E}}=-a_{i}a_{j} 
\end{aligned}
\tag{22}
$$

2. 结合损失函数的整体反向传播公式

看上图，我们要求Loss值对Z1的偏导数。和以前的Logistic函数不同，那个函数是一个z对应一个a，所以反向关系也是一对一。而在这里，a1的计算是有z1,z2,z3参与的，a2的计算也是有z1,z2,z3参与的，即所有a的计算都与前一层的z有关，所以考虑反向时也会比较复杂。

先从Loss的公式看，$loss=-(y_1lna_1+y_2lna_2+y_3lna_3)$，a1肯定与z1有关，那么a2,a3是否与z1有关呢？

再从Softmax函数的形式来看：

无论是a1，a2，a3，都是与z1相关的，而不是一对一的关系，所以，想求Loss对Z1的偏导，必须把Loss->A1->Z1， Loss->A2->Z1，Loss->A3->Z1，这三条路的结果加起来。于是有了如下公式：

$$
\begin{aligned}    
\frac{\partial{loss}}{\partial{z_i}} &= \frac{\partial{loss}}{\partial{a_1}}\frac{\partial{a_1}}{\partial{z_i}} + \frac{\partial{loss}}{\partial{a_2}}\frac{\partial{a_2}}{\partial{z_i}} + \frac{\partial{loss}}{\partial{a_3}}\frac{\partial{a_3}}{\partial{z_i}} \\\\
&=\sum_j \frac{\partial{loss}}{\partial{a_j}}\frac{\partial{a_j}}{\partial{z_i}}
\end{aligned}
$$

当上式中$i=1,j=3$，就完全符合我们的假设了，而且不失普遍性。

前面说过了，因为Softmax涉及到各项求和，A的分类结果和Y的标签值分类是否一致，所以需要分情况讨论：

$$
\frac{\partial{a_j}}{\partial{z_i}} = \begin{cases} a_j(1-a_j), & i = j \\\\ -a_ia_j, & i \neq j \end{cases}
$$

因此，$\frac{\partial{loss}}{\partial{z_i}}$应该是 $i=j$ 和 $i \neq j$两种情况的和：

- $i = j$ 时，loss通过 $a_1$ 对 $z_1$ 求导（或者是通过 $a_2$ 对 $z_2$ 求导）：

$$
\begin{aligned}
\frac{\partial{loss}}{\partial{z_i}} &= \frac{\partial{loss}}{\partial{a_j}}\frac{\partial{a_j}}{\partial{z_i}}=-\frac{y_j}{a_j}a_j(1-a_j) \\\\
&=y_j(a_j-1)=y_i(a_i-1) 
\end{aligned}
\tag{23}
$$

- $i \neq j$，loss通过 $a_2+a_3$ 对 $z_1$ 求导：

$$
\begin{aligned}    
\frac{\partial{loss}}{\partial{z_i}} &= \frac{\partial{loss}}{\partial{a_j}}\frac{\partial{a_j}}{\partial{z_i}}=\sum_j^m(-\frac{y_j}{a_j})(-a_ja_i) \\\\
&=\sum_j^m(y_ja_i)=a_i\sum_{j \neq i}{y_j} 
\end{aligned}
\tag{24}
$$

把两种情况加起来：

$$
\begin{aligned}    
\frac{\partial{loss}}{\partial{z_i}} &= y_i(a_i-1)+a_i\sum_{j \neq i}y_j \\\\
&=-y_i+a_iy_i+a_i\sum_{j \neq i}y_j \\\\
&=-y_i+a_i(y_i+\sum_{j \neq i}y_j) \\\\
&=-y_i + a_i*1 \\\\
&=a_i-y_i 
\end{aligned}
\tag{25}$$

因为$y_j$取值$[1,0,0]$或者$[0,1,0]$或者$[0,0,1]$，这三者加起来，就是$[1,1,1]$，在矩阵乘法运算里乘以$[1,1,1]$相当于什么都不做，就等于原值。

我们惊奇地发现，最后的反向计算过程就是：$a_i-y_i$，假设当前样本的$a_i=[0.879, 0.119, 0.002]$，而$y_i=[0, 1, 0]$，则：
$$a_i - y_i = [0.879, 0.119, 0.002]-[0,1,0]=[0.879,-0.881,0.002]$$

其含义是，样本预测第一类，但实际是第二类，所以给第一类0.879的惩罚值，给第二类0.881的奖励，给第三类0.002的惩罚，并反向传播给神经网络。

后面对 $z=wx+b$ 的求导，与二分类一样，不再赘述。

### 7.1.4 代码实现

第一种，直截了当按照公式写：
```Python
def Softmax1(x):
    e_x = np.exp(x)
    v = np.exp(x) / np.sum(e_x)
    return v
```
这个可能会发生的问题是，当x很大时，`np.exp(x)`很容易溢出，因为是指数运算。所以，有了下面这种改进的代码：

```Python
def Softmax2(Z):
    shift_Z = Z - np.max(Z)
    exp_Z = np.exp(shift_Z)
    A = exp_Z / np.sum(exp_Z)
    return A
```
测试一下：
```Python
Z = np.array([3,0,-3])
print(Softmax1(Z))
print(Softmax2(Z))
```
两个实现方式的结果一致：
```
[0.95033021 0.04731416 0.00235563]
[0.95033021 0.04731416 0.00235563]
```

为什么一样呢？从代码上看差好多啊！我们来证明一下：

假设有3个值a，b，c，并且a在三个数中最大，则b所占的Softmax比重应该这样写：

$$P(b)=\frac{e^b}{e^a+e^b+e^c}$$

如果减去最大值变成了a-a，b-a，c-a，则b'所占的Softmax比重应该这样写：

$$
\begin{aligned}
P(b') &= \frac{e^{b-a}}{e^{a-a}+e^{b-a}+e^{c-a}} \\
&=\frac{e^b/e^a}{e^a/e^a+e^b/e^a+e^c/e^a} \\
&= \frac{e^b}{e^a+e^b+e^c}
\end{aligned}
$$
所以：
$$
P(b) == P(b')
$$

`Softmax2`的写法对一个一维的向量或者数组是没问题的，如果遇到Z是个$M \times N$维(M,N>1)的矩阵的话，就有问题了，因为`np.sum(exp_Z)`这个函数，会把$M\times N$矩阵里的所有元素加在一起，得到一个标量值，而不是相关列元素加在一起。

所以应该这么写：

```Python
class Softmax(object):
    def forward(self, z):
        shift_z = z - np.max(z, axis=1, keepdims=True)
        exp_z = np.exp(shift_z)
        a = exp_z / np.sum(exp_z, axis=1, keepdims=True)
        return a

```

`axis=1`这个参数非常重要，因为如果输入Z是单样本的预测值话，如果是分三类，则应该是个 $3\times 1$ 的数组，如果：

- $z = [3,1,-3]$
- $a = [0.879,0.119,0.002]$

但是，如果是批量训练，假设每次用两个样本，则：
```
if __name__ == '__main__':
    z = np.array([[3,1,-3],[1,-3,3]]).reshape(2,3)
    a = Softmax().forward(z)
    print(a)
```
结果：
```
[[0.87887824 0.11894324 0.00217852]
 [0.11894324 0.00217852 0.87887824]]
```
其中，a是包含两个样本的softmax结果，每个数组里面的三个数字相加为1。

如果`s = np.sum(exp_z)`，不指定`axis=1`参数，则：
```
[[0.43943912 0.05947162 0.00108926]
 [0.05947162 0.00108926 0.43943912]]
```
A虽然仍然包含两个样本，但是变成了两个样本所有的6个元素相加为1，这不是softmax的本意，softmax只计算一个样本（一行）中的数据。


## 7.2 线性多分类的神经网络实现  

### 7.2.1 定义神经网络结构

从图7-1来看，似乎在三个颜色区间之间有两个比较明显的分界线，而且是直线，即线性可分的。我们如何通过神经网络精确地找到这两条分界线呢？

- 从视觉上判断是线性可分的，所以我们使用单层神经网络即可
- 输入特征有两个，分别为经度、纬度
- 最后输出的是三个分类，分别是魏蜀吴，所以输出层有三个神经元

如果有三个以上的分类同时存在，我们需要对每一类别分配一个神经元，这个神经元的作用是根据前端输入的各种数据，先做线性处理（$Z=WX+B$），然后做一次非线性处理，计算每个样本在每个类别中的预测概率，再和标签中的类别比较，看看预测是否准确，如果准确，则奖励这个预测，给与正反馈；如果不准确，则惩罚这个预测，给与负反馈。两类反馈都反向传播到神经网络系统中去调整参数。

这个网络只有输入层和输出层，由于输入层不算在内，所以是一层网络，如图7-7所示。

<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/7/MultipleClassifierNN.png" ch="500" />

图7-7 多入多出单层神经网络

与前面的单层网络不同的是，图7-7最右侧的输出层还多出来一个Softmax分类函数，这是多分类任务中的标准配置，可以看作是输出层的激活函数，并不单独成为一层，与二分类中的Logistic函数一样。

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

### 7.2.2 样本数据

使用`SimpleDataReader`类读取数据后，观察一下数据的基本属性：

```Python
reader.XRaw.shape
(140, 2)
reader.XRaw.min()
0.058152279749505986
reader.XRaw.max()
9.925126526921046

reader.YRaw.shape
(140, 1)
reader.YRaw.min()
1.0
reader.YRaw.max()
3.0
```

- 训练数据X，140个记录，两个特征，最小值0.058，最大值9.925
- 标签数据Y，140个记录，一个分类值，取值范围是$\\{1,2,3\\}$

#### 样本标签数据

一般来说，在标记样本时，我们会用$1,2,3$这样的标记来指明是哪一类。所以样本数据是这个样子的：
$$
Y = 
\begin{pmatrix}
y_1 \\\\ y_2 \\\\ \vdots \\\\ y_{140}
\end{pmatrix}=
\begin{pmatrix}3 \\\\ 2 \\\\ \vdots \\\\ 1\end{pmatrix}
$$

在有Softmax的多分类计算时，我们用下面这种等价的方式，俗称OneHot。
$$
Y = 
\begin{pmatrix}
y_1 \\\\ y_2 \\\\ \vdots \\\\ y_{140}
\end{pmatrix}=
\begin{pmatrix}
0 & 0 & 1 \\\\
0 & 1 & 0 \\\\
... & ... & ... \\\\
1 & 0 & 0
\end{pmatrix}
$$

OneHot的意思，在这一列数据中，只有一个1，其它都是0。1所在的列数就是这个样本的分类类别。标签数据对应到每个样本数据上，列对齐，只有$(1,0,0),(0,1,0),(0,0,1)$三种可能，分别表示第一类、第二类和第三类。

在`SimpleDataReader`中实现`ToOneHot()`方法，把原始标签转变成One-Hot编码：

```Python
class SimpleDataReader(object):
    def ToOneHot(self, num_category, base=0):
        ......
```

### 7.2.3 代码实现

绝大部分代码可以从上一章的`HelperClass`中拷贝出来，但是需要我们为了本章的特殊需要稍加修改。

#### 添加分类函数

在`Activators.py`中，增加Softmax的实现，并添加单元测试。

```Python
class Softmax(object):
    def forward(self, z):
        ......
```

#### 前向计算

前向计算需要增加分类函数调用：

```Python
class NeuralNet(object):
    def forwardBatch(self, batch_x):
        Z = np.dot(batch_x, self.W) + self.B
        if self.params.net_type == NetType.BinaryClassifier:
            A = Logistic().forward(Z)
            return A
        elif self.params.net_type == NetType.MultipleClassifier:
            A = Softmax().forward(Z)
            return A
        else:
            return Z
```

#### 反向传播

在多分类函数一节详细介绍了反向传播的推导过程，推导的结果很令人惊喜，就是一个简单的减法，与前面学习的拟合、二分类的算法结果都一样。

```Python
class NeuralNet(object):
    def backwardBatch(self, batch_x, batch_y, batch_a):
        ......
```

#### 计算损失函数值

损失函数不再是均方差和二分类交叉熵了，而是交叉熵函数对于多分类的形式，并且添加条件分支来判断只在网络类型为多分类时调用此损失函数。

```Python
class LossFunction(object):
    # fcFunc: feed forward calculation
    def CheckLoss(self, A, Y):
        m = Y.shape[0]
        if self.net_type == NetType.Fitting:
            loss = self.MSE(A, Y, m)
        elif self.net_type == NetType.BinaryClassifier:
            loss = self.CE2(A, Y, m)
        elif self.net_type == NetType.MultipleClassifier:
            loss = self.CE3(A, Y, m)
        #end if
        return loss
    # end def

    # for multiple classifier
    def CE3(self, A, Y, count):
        ......
```

#### 推理函数

```Python
def inference(net, reader):
    xt_raw = np.array([5,1,7,6,5,6,2,7]).reshape(4,2)
    xt = reader.NormalizePredicateData(xt_raw)
    output = net.inference(xt)
    r = np.argmax(output, axis=1)+1
    print("output=", output)
    print("r=", r)
```
注意在推理之前，先做了归一化，因为原始数据是在[0,10]范围的。

函数`np.argmax`的作用是比较`output`里面的几个数据的值，返回最大的那个数据的行数或者列数，0-based。比如output=(1.02,-3,2.2)时，会返回2，因为2.2最大，所以我们再加1，把返回值变成1,2,3的其中一个。

`np.argmax`函数的参数`axis=1`，是因为有4个样本参与预测，所以需要在第二维上区分开来，分别计算每个样本的argmax值。

#### 主程序

```Python
if __name__ == '__main__':
    num_category = 3
    ......
    num_input = 2
    params = HyperParameters(num_input, num_category, eta=0.1, max_epoch=100, batch_size=10, eps=1e-3, net_type=NetType.MultipleClassifier)
    ......
```

### 7.2.4 运行结果

#### 损失函数历史记录

从图7-8所示的趋势上来看，损失函数值还有进一步下降的可能，以提高模型精度。有兴趣的读者可以多训练几轮，看看效果。

<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/7/loss.png" ch="500" />

图7-8 训练过程中损失函数值的变化

下面是打印输出的最后几行：

```
......
epoch=99
99 13 0.25497053433985734
W= [[-1.43234109 -3.57409342  5.00643451]
 [ 4.47791288 -2.88936887 -1.58854401]]
B= [[-1.81896724  3.66606162 -1.84709438]]
output= [[0.01801124 0.73435241 0.24763634]
 [0.24709055 0.15438074 0.59852871]
 [0.38304995 0.37347646 0.24347359]
 [0.51360269 0.46266935 0.02372795]]
r= [2 3 1 1]
```
注意，r是分类预测结果，对于每个测试样本的结果，是按行看的，即第一行是第一个测试样本的分类结果。

1. 经纬度相对值为 $(5,1)$ 时，概率0.734最大，属于2，蜀国
2. 经纬度相对值为 $(7,6)$ 时，概率0.598最大，属于3，吴国
3. 经纬度相对值为 $(5,6)$ 时，概率0.383最大，属于1，魏国
4. 经纬度相对值为 $(2,7)$ 时，概率0.513最大，属于1，魏国


## 7.3 线性多分类原理

此原理对线性多分类和非线性多分类都适用。

### 7.3.1 多分类过程

我们在此以具有两个特征值的三分类举例。可以扩展到更多的分类或任意特征值，比如在ImageNet的图像分类任务中，最后一层全连接层输出给分类器的特征值有成千上万个，分类有1000个。

1. 线性计算

$$z_1 = x_1 w_{11} + x_2 w_{21} + b_1 \tag{1}$$
$$z_2 = x_1 w_{12} + x_2 w_{22} + b_2 \tag{2}$$
$$z_3 = x_1 w_{13} + x_2 w_{23} + b_3 \tag{3}$$

2. 分类计算

$$
a_1=\frac{e^{z_1}}{\sum_i e^{z_i}}=\frac{e^{z_1}}{e^{z_1}+e^{z_2}+e^{z_3}}  \tag{4}
$$
$$
a_2=\frac{e^{z_2}}{\sum_i e^{z_i}}=\frac{e^{z_2}}{e^{z_1}+e^{z_2}+e^{z_3}}  \tag{5}
$$
$$
a_3=\frac{e^{z_3}}{\sum_i e^{z_i}}=\frac{e^{z_3}}{e^{z_1}+e^{z_2}+e^{z_3}}  \tag{6}
$$

3. 损失函数计算

单样本时，$n$表示类别数，$j$表示类别序号：

$$
\begin{aligned}
loss(w,b)&=-(y_1 \ln a_1 + y_2 \ln a_2 + y_3 \ln a_3) \\\\
&=-\sum_{j=1}^{n} y_j \ln a_j 
\end{aligned}
\tag{7}
$$

批量样本时，$m$ 表示样本数，$i$ 表示样本序号：

$$
\begin{aligned}
J(w,b) &=- \sum_{i=1}^m (y_{i1} \ln a_{i1} + y_{i2} \ln a_{i2} + y_{i3} \ln a_{i3}) \\\\
&=- \sum_{i=1}^m \sum_{j=1}^n y_{ij} \ln a_{ij}
\end{aligned}
 \tag{8}
$$

损失函数计算在交叉熵函数一节有详细介绍。

### 7.3.2 数值计算举例

假设对预测一个样本的计算得到的 $z$ 值为：

$$z=[z_1,z_2,z_3]=[3,1,-3]$$

则按公式4、5、6进行计算，可以得出Softmax的概率分布是：

$$a=[a_1,a_2,a_3]=[0.879,0.119,0.002]$$

#### 如果标签值表明此样本为第一类

即：

$$y=[1,0,0]$$

则损失函数为：

$$
loss_1=-(1 \times \ln 0.879 + 0 \times \ln 0.119 + 0 \times \ln 0.002)=0.123
$$

反向传播误差矩阵为：

$$a-y=[-0.121,0.119,0.002]$$

因为$a_1=0.879$，为三者最大，分类正确，所以$a-y$的三个值都不大。

#### 如果标签值表明此样本为第二类

即：

$$y=[0,1,0]$$

则损失函数为：

$$
loss_2=-(0 \times \ln 0.879 + 1 \times \ln 0.119 + 0 \times \ln 0.002)=2.128
$$

可以看到由于分类错误，$loss_2$的值比$loss_1$的值大很多。

反向传播误差矩阵为：

$$a-y=[0.879,0.881,0.002]$$

本来是第二类，误判为第一类，所以前两个元素的值很大，反向传播的力度就大。

### 7.3.3 多分类的几何原理

在前面的二分类原理中，很容易理解为我们用一条直线分开两个部分。对于多分类问题，是否可以沿用二分类原理中的几何解释呢？答案是肯定的，只不过需要单独判定每一个类别。

假设一共有三类样本，蓝色为1，红色为2，绿色为3，那么Softmax的形式应该是：

$$
a_j = \frac{e^{z_j}}{\sum\limits_{i=1}^3 e^{z_i}}=\frac{e^{z_j}}{e^{z_1}+e^{z_2}+^{z_3}}
$$

#### 当样本属于第一类时

把蓝色点与其它颜色的点分开。

如果判定一个点属于第一类，则 $a_1$ 概率值一定会比 $a_2,a_3$ 大，表示为公式：

$$a_1 > a_2 且 a_1 > a_3 \tag{9}$$

由于Softmax的特殊形式，分母都一样，所以只比较分子就行了。而分子是一个自然指数，输出值域大于零且单调递增，所以只比较指数就可以了，因此，公式9等同于下式：

$$z_1 > z_2, z_1 > z_3 \tag{10}$$

把公式1、2、3引入到10：

$$x_1 w_{11}  + x_2 w_{21} + b_1  > x_1 w_{12} + x_2 w_{22} + b_2 \tag{11}$$
$$x_1 w_{11}  + x_2 w_{21} + b_1  > x_1 w_{13} + x_2 w_{23} + b_3 \tag{12}$$

变形：

$$(w_{21} - w_{22})x_2 > (w_{12} - w_{11})x_1 + (b_2 - b_1) \tag{13}$$

$$(w_{21} - w_{23})x_2 > (w_{13} - w_{11})x_1 + (b_3 - b_1) \tag{14}$$

我们先假设：

$$w_{21} > w_{22}, w_{21}> w_{23} \tag{15}$$

所以公式13、14左侧的系数都大于零，两边同时除以系数：

$$x_2 > \frac{w_{12} - w_{11}}{w_{21} - w_{22}}x_1 + \frac{b_2 - b_1}{w_{21} - w_{22}} \tag{16}$$

$$x_2 > \frac{w_{13} - w_{11}}{w_{21} - w_{23}} x_1 + \frac{b_3 - b_1}{w_{21} - w_{23}} \tag{17}$$

简化，即把公式16、17的分数形式的因子分别简化成一个变量$W$和$B$，并令$y = x_2,x = x_1$，利于理解：

$$y > W_{12} \cdot x + B_{12} \tag{18}$$

$$y > W_{13} \cdot x + B_{13} \tag{19}$$

此时y代表了第一类的蓝色点。

<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/7/z1.png" ch="500" />

图7-9 如何把蓝色样本与其它两色的样本分开

借用二分类中的概念，公式18的几何含义是：有一条直线可以分开第一类（蓝色点）和第二类（红色点），使得所有蓝色点都在直线的上方，所有的红色点都在直线的下方。于是我们可以画出图7-9中的那条绿色直线。

而公式19的几何含义是：有一条直线可以分开第一类（蓝色点）和第三类（绿色点），使得所有蓝色点都在直线的上方，所有的绿色点都在直线的下方。于是我们可以画出图7-9中的那条红色直线。

也就是说在图中画两条直线，所有蓝点都同时在红线和绿线这两条直线的上方。

#### 当样本属于第二类时

即如何把红色点与其它两色点分开。

$$z_2 > z_1,z_2 > z_3 \tag{20}$$

同理可得

$$y < W_{12} \cdot x + B_{12} \tag{21}$$

$$y > W_{23} \cdot x + B_{23} \tag{22}$$

<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/7/z2.png" ch="500" />

图7-10 如何把红色样本与其它两色的样本分开


此时$y$代表了第二类的红色点。

公式21和公式18几何含义相同，不等号相反，代表了图7-10中绿色直线的分割作用，即红色点在绿色直线下方。

公式22的几何含义是，有一条蓝色直线可以分开第二类（红色点）和第三类（绿色点），使得所有红色点都在直线的上方，所有的绿色点都在直线的下方。

#### 当样本属于第三类时

即如何把绿色点与其它两色点分开。

$$z_3 > z_1,z_3 > z_2 \tag{22}$$

最后可得：

$$y < W_{13} \cdot x + B_{13} \tag{23}$$

$$y < W_{23} \cdot x + B_{23} \tag{24}$$

此时$y$代表了第三类的绿色点。

<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/7/z3.png" ch="500" />

图7-11 如何把绿色样本与其它两色的样本分开

公式23与公式19不等号相反，几何含义相同，代表了图7-11中红色直线的分割作用，绿色点在红色直线下方。

公式24与公式22不等号相反，几何含义相同，代表了图7-11中蓝色直线的分割作用，绿色点在蓝色直线下方。

#### 综合效果

把三张图综合在一起，应该是图7-12的样子。

<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/7/z123.png" ch="500" />

图7-12 三条线分开三种样本


## 7.4 多分类结果可视化

神经网络到底是一对一方式，还是一对多方式呢？从Softmax公式，好像是一对多方式，因为只取一个最大值，那么理想中的一对多方式应该是图7-13所示的样子。

<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/7/OneVsOthers.png" ch="500" />

图7-13 理想中的一对多方式的分割线

实际上是什么样子的，我们来看下面的具体分析。

### 7.4.1 显示原始数据图

与二分类时同样的问题，如何直观地理解多分类的结果？三分类要复杂一些，我们先把原始数据显示出来。

```Python
def ShowData(X,Y):
    for i in range(X.shape[0]):
        if Y[i,0] == 1:
            plt.plot(X[i,0], X[i,1], '.', c='r')
        elif Y[i,0] == 2:
            plt.plot(X[i,0], X[i,1], 'x', c='g')
        elif Y[i,0] == 3:
            plt.plot(X[i,0], X[i,1], '^', c='b')
        # end if
    # end for
    plt.show()
```

会画出图7-1来。

### 7.4.2 显示分类结果分割线图

下面的数据是神经网络训练出的权重和偏移值的结果：

```
......
epoch=98
98 1385 0.25640040547970516
epoch=99
99 1399 0.2549651316913006
W= [[-1.43299777 -3.57488388  5.00788165]
 [ 4.47527075 -2.88799216 -1.58727859]]
B= [[-1.821679    3.66752583 -1.84584683]]
......
```

其实在7.2节中讲解多分类原理的时候，我们已经解释了其几何理解，那些公式的推导就可以用于指导我们画出多分类的分割线来。先把几个有用的结论拿过来。

从7.2节中的公式16，把不等号变成等号，即$z_1=z_2$，则代表了那条绿色的分割线，用于分割第一类和第二类的：

$$x_2 = \frac{w_{12} - w_{11}}{w_{21} - w_{22}}x_1 + \frac{b_2 - b_1}{w_{21} - w_{22}} \tag{1}$$

$$即：y = W_{12} \cdot x + B_{12}$$

由于`Python`数组是从0开始的，所以公式1中的所有下标都减去1，写成代码：

```Python
b12 = (net.B[0,1] - net.B[0,0])/(net.W[1,0] - net.W[1,1])
w12 = (net.W[0,1] - net.W[0,0])/(net.W[1,0] - net.W[1,1])
```

从7.2节中的公式17，把不等号变成等号，即$z_1=z_3$，则代表了那条红色的分割线，用于分割第一类和第三类的：

$$x_2 = \frac{w_{13} - w_{11}}{w_{21} - w_{23}} x_1 + \frac{b_3 - b_1}{w_{21} - w_{23}} \tag{2}$$
$$即：y = W_{13} \cdot x + B_{13}$$

写成代码：

```Python
b13 = (net.B[0,0] - net.B[0,2])/(net.W[1,2] - net.W[1,0])
w13 = (net.W[0,0] - net.W[0,2])/(net.W[1,2] - net.W[1,0])
```

从7.2节中的公式24，把不等号变成等号，即$z_2=z_3$，则代表了那条蓝色的分割线，用于分割第二类和第三类的：

$$x_2 = \frac{w_{13} - w_{12}}{w_{22} - w_{23}} x_1 + \frac{b_3 - b_2}{w_{22} - w_{23}} \tag{3}$$
$$即：y = W_{23} \cdot x + B_{23}$$

写成代码：

```Python
b23 = (net.B[0,2] - net.B[0,1])/(net.W[1,1] - net.W[1,2])
w23 = (net.W[0,2] - net.W[0,1])/(net.W[1,1] - net.W[1,2])
```

完整代码请看ch07 Level2的python文件。

改一下主函数，增加对以上两个函数`ShowData()`和`ShowResult()`的调用，最后可以看到图7-14所示的分类结果图，注意，这个结果图和我们在7.2中分析的一样，只是蓝线斜率不同。

<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/7/result.png" ch="500" />

图7-14 神经网络绘出的分类结果图

图7-14中的四个三角形的大点是需要我们预测的四个坐标值，其中三个点的分类都比较明确，只有那个蓝色的点看不清在边界那一侧，可以通过在实际的运行结果图上放大局部来观察。

### 7.4.3 理解神经网络的分类方式

做为实际结果，图7-14与我们猜想的图7-13完全不同：

- 蓝色线是2|3的边界，不考虑第1类
- 绿色线是1|2的边界，不考虑第3类
- 红色线是1|3的边界，不考虑第2类

我们只看蓝色的第1类，当要区分1|2和1|3时，神经网络实际是用了两条直线（绿色和红色）同时作为边界。那么它是一对一方式还是一对多方式呢？

图7-14的分割线是我们令$z_1=z_2, z_2=z_3, z_3=z_1$三个等式得到的，但实际上神经网络的工作方式不是这样的，它不会单独比较两类，而是会同时比较三类，这个从Softmax会同时输出三个概率值就可以理解。比如，当我们想得到第一类的分割线时，需要同时满足两个条件：

$$z_1=z_2，且：z_1=z_3 \tag{4}$$

即，同时，找到第一类和第三类的边界。

这就意味着公式4其实是一个线性分段函数，而不是两条直线，即图7-15中红色射线和绿色射线所组成的函数。

<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/7/multiple_result_true.png" ch="500" />

图7-15 分段线性的分割作用

同理，用于分开红色点和其它两类的分割线是蓝色射线和绿色射线，用于分开绿色点和其它两类的分割线是红色射线和蓝色射线。

训练一对多分类器时，是把蓝色样本当作一类，把红色和绿色样本混在一起当作另外一类。训练一对一分类器时，是把绿色样本扔掉，只考虑蓝色样本和红色样本。而我们在此并没有这样做，三类样本是同时参与训练的。所以我们只能说神经网络从结果上看，是一种一对多的方式，至于它的实质，我们在后面的非线性分类时再进一步探讨。

