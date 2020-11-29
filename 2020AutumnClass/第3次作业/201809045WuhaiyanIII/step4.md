# 第四步  非线性回归

## 摘要

从这一步开始，我们进入双层神经网络的学习，并解决非线性问题。

在两层神经网络之间，必须有激活函数连接，从而加入非线性因素，提高神经网络的能力。所以，我们先从激活函数学起，一类是挤压型的激活函数，常用于简单网络的学习；另一类是半线性的激活函数，常用于深度网络的学习。

接下来我们将验证著名的万能近似定理，建立一个双层的神经网络，来拟合一个比较复杂的函数。

在上面的双层神经网络中，已经出现了很多的超参，都会影响到神经网络的训练结果。所以在完成了基本的拟合任务之后，我们将尝试着调试这些参数，得到又快又好的训练效果，从而得到超参调试的第一手经验。

# 第8章 激活函数

## 8.0 激活函数概论
### 8.0.1 激活函数的基本作用

图8-1是神经网络中的一个神经元，假设该神经元有三个输入，分别为$x_1,x_2,x_3$，那么：

$$z=x_1 w_1 + x_2 w_2 + x_3 w_3 +b \tag{1}$$
$$a = \sigma(z) \tag{2}$$

<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/1/NeuranCell.png" width="500" />

图8-1 激活函数在神经元中的位置

激活函数的基本性质：

+ 非线性：线性的激活函数和没有激活函数一样；
+ 可导性：做误差反向传播和梯度下降，必须要保证激活函数的可导性；
+ 单调性：单一的输入会得到单一的输出，较大值的输入得到较大值的输出。

在物理试验中使用的继电器，是最初的激活函数的原型：当输入电流大于一个阈值时，会产生足够的磁场，从而打开下一级电源通道，如图8-2所示。

<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/8/step.png" width="500" />

图8-2 继电器的阶跃形态

用到神经网络中的概念，用‘1’来代表一个神经元被激活，‘0’代表一个神经元未被激活。

这个Step函数的梯度（导数）恒为零（个别点除外)。反向传播公式中，梯度传递用到了链式法则，如果在这样一个连乘的式子其中有一项是零，这样的梯度就会恒为零，是没有办法进行反向传播的。

### 8.0.2 何时会用到激活函数

激活函数用在神经网络的层与层之间的连接，神经网络的最后一层不用激活函数。

神经网络不管有多少层，最后的输出层决定了这个神经网络能干什么。

对于多层神经网络也是如此，在最后一层只会用到分类函数来完成二分类或多分类任务，如果是拟合任务，则不需要分类函数。

简言之：

1. 神经网络最后一层不需要激活函数
2. 激活函数只用于连接前后两层神经网络

当不需要指定具体的激活函数形式时，会使用 $\sigma()$ 符号来代表激活函数运算。


## 8.1 挤压型激活函数

这一类函数的特点是，当输入值域的绝对值较大的时候，其输出在两端是饱和的，都具有S形的函数曲线以及压缩输入值域的作用，所以叫挤压型激活函数，又可以叫饱和型激活函数。

在英文中，通常用Sigmoid来表示，原意是S型的曲线，在数学中是指一类具有压缩作用的S型的函数，在神经网络中，有两个常用的Sigmoid函数，一个是Logistic函数，另一个是Tanh函数。

### 8.1.1 Logistic函数

对数几率函数（Logistic Function，简称对率函数）。

在二分类任务中最后一层使用的对率函数与在神经网络层与层之间连接的Sigmoid激活函数，是同样的形式。所以它既是激活函数，又是分类函数，是个特例。

凡是用到“Logistic”词汇的，指的是二分类函数；而用到“Sigmoid”词汇的，指的是本激活函数。

#### 公式

$$Sigmoid(z) = \frac{1}{1 + e^{-z}} \rightarrow a \tag{1}$$

#### 导数

$$Sigmoid'(z) = a(1 - a) \tag{2}$$

注意，如果是矩阵运算的话，需要在公式2中使用$\odot$符号表示按元素的矩阵相乘：$a\odot (1-a)$，后面不再强调。

推导过程如下：

令：$u=1,v=1+e^{-z}$ 则：

$$
\begin{aligned}
Sigmoid'(z)&= (\frac{u}{v})'=\frac{u'v-v'u}{v^2} \\\\
&=\frac{0-(1+e^{-z})'}{(1+e^{-z})^2}=\frac{e^{-z}}{(1+e^{-z})^2} \\\\
&=\frac{1+e^{-z}-1}{(1+e^{-z})^2}=\frac{1}{1+e^{-z}}-(\frac{1}{1+e^{-z}})^2 \\\\
&=a-a^2=a(1-a)
\end{aligned}
$$

#### 值域

- 输入值域：$(-\infty, \infty)$
- 输出值域：$(0,1)$
- 导数值域：$(0,0.25]$

#### 函数图像

<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/8/sigmoid.png" ch="500" />

图8-3 Sigmoid函数图像

#### 优点

从函数图像来看，Sigmoid函数的作用是将输入压缩到 $(0,1)$ 这个区间范围内，这种输出在0~1之间的函数可以用来模拟一些概率分布的情况。它还是一个连续函数，导数简单易求。  

从数学上来看，Sigmoid函数对中央区的信号增益较大，对两侧区的信号增益小，在信号的特征空间映射上，有很好的效果。 

从神经科学上来看，中央区酷似神经元的兴奋态，两侧区酷似神经元的抑制态，因而在神经网络学习方面，可以将重点特征推向中央区，
将非重点特征推向两侧区。

Sigmoid函数在这里就起到了如何把一个数值转化成一个通俗意义上的“把握”的表示。z坐标值越大，经过Sigmoid函数之后的结果就越接近1，把握就越大。

#### 缺点

指数计算代价大。

### 8.1.2 Tanh函数

TanHyperbolic，即双曲正切函数。

#### 公式  
$$Tanh(z) = \frac{e^{z} - e^{-z}}{e^{z} + e^{-z}} = (\frac{2}{1 + e^{-2z}}-1) \rightarrow a \tag{3}$$
即
$$Tanh(z) = 2 \cdot Sigmoid(2z) - 1 \tag{4}$$

#### 导数公式

$$Tanh'(z) = (1 + a)(1 - a)$$

利用基本导数公式23，令：$u={e^{z}-e^{-z}}，v=e^{z}+e^{-z}$ 则有：

$$
\begin{aligned}
Tanh'(z)&=\frac{u'v-v'u}{v^2} \\\\
&=\frac{(e^{z}-e^{-z})'(e^{z}+e^{-z})-(e^{z}+e^{-z})'(e^{z}-e^{-z})}{(e^{z}+e^{-z})^2} \\\\
&=\frac{(e^{z}+e^{-z})(e^{z}+e^{-z})-(e^{z}-e^{-z})(e^{z}-e^ {-z})}{(e^{z}+e^{-z})^2} \\\\
&=\frac{(e^{z}+e^{-z})^2-(e^{z}-e^{-z})^2}{(e^{z}+e^{-z})^2} \\\\
&=1-(\frac{(e^{z}-e^{-z}}{e^{z}+e^{-z}})^2=1-a^2
\end{aligned}
$$

#### 值域

- 输入值域：$(-\infty,\infty)$
- 输出值域：$(-1,1)$
- 导数值域：$(0,1)$


#### 函数图像

图8-4是双曲正切的函数图像。

<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/8/tanh.png" ch="500" />

图8-4 双曲正切函数图像

#### 优点

具有Sigmoid的所有优点。

无论从理论公式还是函数图像，这个函数都是一个和Sigmoid非常相像的激活函数，他们的性质也确实如此。但是比起Sigmoid，Tanh减少了一个缺点，就是他本身是零均值的，也就是说，在传递过程中，输入数据的均值并不会发生改变，这就使他在很多应用中能表现出比Sigmoid优异一些的效果。

#### 缺点

exp指数计算代价大。梯度消失问题仍然存在。

### 8.1.3 其它函数

图8-5展示了其它S型函数，除了$Tanh(x)$以外，其它的基本不怎么使用，目的是告诉大家这类函数有很多，但是常用的只有Sigmoid和Tanh两个。

<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/8/others.png" />

图8-5 其它S型函数

再强调一下本书中的约定：

1. Sigmoid，指的是对数几率函数用于激活函数时的称呼；
2. Logistic，指的是对数几率函数用于二分类函数时的称呼；
3. Tanh，指的是双曲正切函数用于激活函数时的称呼。


## 8.2 半线性激活函数

又可以叫非饱和型激活函数。

### 8.2.1 ReLU函数 

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

图8-6 线性整流函数ReLU

#### 仿生学原理

相关大脑方面的研究表明生物神经元的信息编码通常是比较分散及稀疏的。通常情况下，大脑中在同一时间大概只有1%~4%的神经元处于活跃状态。使用线性修正以及正则化可以对机器神经网络中神经元的活跃度（即输出为正值）进行调试；相比之下，Sigmoid函数在输入为0时输出为0.5，即已经是半饱和的稳定状态，不够符合实际生物学对模拟神经网络的期望。不过需要指出的是，一般情况下，在一个使用修正线性单元（即线性整流）的神经网络中大概有50%的神经元处于激活态。

#### 优点

- 反向导数恒等于1，更加有效率的反向传播梯度值，收敛速度快；
- 避免梯度消失问题；
- 计算简单，速度快；
- 活跃度的分散性使得神经网络的整体计算成本下降。

#### 缺点

无界。

梯度很大的时候可能导致的神经元“死”掉。

### 8.2.2 Leaky ReLU函数

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

#### 优点

继承了ReLU函数的优点。

Leaky ReLU同样有收敛快速和运算复杂度低的优点，而且由于给了$z<0$时一个比较小的梯度$\alpha$,使得$z<0$时依旧可以进行梯度传递和更新，可以在一定程度上避免神经元“死”掉的问题。

### 8.2.3 Softplus函数

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

图8-8 Softplus的函数图像

### 8.2.4 ELU函数

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

图8-9 ELU的函数图像


# 第9章 单入单出的双层神经网络 - 非线性回归

## 9.0 非线性回归问题

### 9.0.1 提出问题一

<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/9/sin_data.png" ch="500" />

图9-1 成正弦曲线分布的样本点

问题：如何使用神经网络拟合一条有很强规律的曲线，比如正弦曲线？

### 9.0.2 提出问题二

如何使用神经网络方法来拟合这条曲线？

<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/9/Sample.png"/>

图9-2 复杂曲线样本可视化

在本章，我们将会学习另外一个定理：前馈神经网络的通用近似定理。

上面这条“蛇形”曲线，实际上是由下面这个公式添加噪音后生成的：

$$y=0.4x^2 + 0.3x\sin(15x) + 0.01\cos(50x)-0.3$$

我们特意把数据限制在[0,1]之间，避免做归一化的麻烦。

以上问题可以叫做非线性回归，即自变量X和因变量Y之间不是线性关系。常用的传统的处理方法有线性迭代法、分段回归法、迭代最小二乘法等。在神经网络中，解决这类问题的思路非常简单，就是使用带有一个隐层的两层神经网络。

#### 9.0.3 回归模型的评估标准

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

我们通常用概率来表达一个准确率，比如89%的准确率。那么线性回归的衡量标准就是R-Squared。

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


## 9.1 用多项式回归法拟合正弦曲线

### 9.1.1 多项式回归的概念

多项式回归有几种形式：

#### 一元一次线性模型

$$z = x w + b \tag{1}$$

#### 多元一次多项式

多变量的线性回归其模型为：

$$z = x_1 w_1 + x_2 w_2 + ...+ x_m w_m + b \tag{2}$$

这里的多变量，是指样本数据的特征值为多个，上式中的 $x_1,x_2,...,x_m$ 代表了m个特征值。

#### 一元多次多项式

对于只有一个特征值的问题，就是把特征值的高次方作为另外的特征值，加入到回归分析中，用公式描述：

$$z = x w_1 + x^2 w_2 + ... + x^m w_m + b \tag{3}$$

上式中x是原有的唯一特征值，$x^m$ 是利用 $x$ 的 $m$ 次方作为额外的特征值，这样就把特征值的数量从 $1$ 个变为 $m$ 个。

换一种表达形式，令：$x_1 = x,x_2=x^2,\ldots,x_m=x^m$，则：

$$z = x_1 w_1 + x_2 w_2 + ... + x_m w_m + b \tag{4}$$

可以看到公式4和上面的公式2是一样的，所以解决方案也一样。

#### 多元多次多项式

<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/9/polynomial_10_pic.png" />

图9-3 对有噪音的正弦曲线的拟合

一堆散点，看上去像是一条带有很大噪音的正弦曲线，从左上到右下，分别是1次多项式、2次多项式......10次多项式，其中：

- 第4、5、6、7图是比较理想的拟合
- 第1、2、3图欠拟合，多项式的次数不够高
- 第8、9、10图，多项式次数过高，过拟合了

在做多项式拟合之前，所有的特征值都会先做归一化，然后再获得x的平方值，三次方值等等。在归一化之后，x的值变成了[0,1]之间，那么x的平方值会比x值要小，x的三次方值会比x的平方值要小。假设$x=0.5，x^2=0.25，x^3=0.125$，所以次数越高，权重值会越大，特征值与权重值的乘积才会是一个不太小的数，以此来弥补特征值小的问题。

### 9.1.2 用二次多项式拟合

鉴于以上的认知，我们要考虑使用几次的多项式来拟合正弦曲线。在没有什么经验的情况下，可以先试一下二次多项式，即：

$$z = x w_1 + x^2 w_2 + b \tag{5}$$

#### 数据增强

在`ch08.train.npz`中，读出来的`XTrain`数组，只包含1列x的原始值，根据公式5，我们需要再增加一列x的平方值，所以代码如下：

```Python
file_name = "../../data/ch08.train.npz"
class DataReaderEx(SimpleDataReader):
    def Add(self):
        X = self.XTrain[:,]**2
        self.XTrain = np.hstack((self.XTrain, X))
```

从`SimpleDataReader`类中派生出子类`DataReaderEx`，然后添加`Add()`方法，先计算`XTrain`第一列的平方值，放入矩阵X中，然后再把X合并到`XTrain`右侧，这样`XTrain`就变成了两列，第一列是x的原始值，第二列是x的平方值。

#### 主程序

在主程序中，先加载数据，做数据增强，然后建立一个net，参数`num_input=2`，对应着`XTrain`中的两列数据，相当于两个特征值，

```Python
if __name__ == '__main__':
    dataReader = DataReaderEx(file_name)
    dataReader.ReadData()
    dataReader.Add()
    # net
    num_input = 2
    num_output = 1
    params = HyperParameters(num_input, num_output, eta=0.2, max_epoch=10000, batch_size=10, eps=0.005, net_type=NetType.Fitting)
    net = NeuralNet(params)
    net.train(dataReader, checkpoint=10)
    ShowResult(net, dataReader, params.toString())
```

#### 运行结果

表9-4 二次多项式训练过程与结果

|损失函数值|拟合结果|
|---|---|
|<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/9/sin_loss_2p.png">|<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/9/sin_result_2p.png">|

从表9-4的损失函数曲线上看，没有任何损失值下降的趋势；再看拟合情况，只拟合成了一条直线。这说明二次多项式不能满足要求。以下是最后几行的打印输出：

```
......
9989 49 0.09410913779071385
9999 49 0.09628814270449357
W= [[-1.72915813]
 [-0.16961507]]
B= [[0.98611283]]
```

对此结论持有怀疑的读者，可以尝试着修改主程序中的各种超参数，比如降低学习率、增加循环次数等，来验证一下这个结论。

### 9.1.3 用三次多项式拟合

三次多项式的公式：

$$z = x w_1 + x^2 w_2 + x^3 w_3 + b \tag{6}$$

在二次多项式的基础上，把训练数据的再增加一列x的三次方，作为一个新的特征。以下为数据增强代码：

```Python
class DataReaderEx(SimpleDataReader):
    def Add(self):
        X = self.XTrain[:,]**2
        self.XTrain = np.hstack((self.XTrain, X))
        X = self.XTrain[:,0:1]**3
        self.XTrain = np.hstack((self.XTrain, X))
```

同时不要忘记修改主过程参数中的`num_input`值：

```Python
    num_input = 3
```

再次运行，得到表9-5所示的结果。

表9-5 三次多项式训练过程与结果

|损失函数值|拟合结果|
|---|---|
|<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/9/sin_loss_3p.png">|<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/9/sin_result_3p.png">|

表9-5中左侧图显示损失函数值下降得很平稳，说明网络训练效果还不错。拟合的结果也很令人满意，虽然红色线没有严丝合缝地落在蓝色样本点内，但是这完全是因为训练的次数不够多。

以下为打印输出：

```
......
2369 49 0.0050611643902918856
2379 49 0.004949680631526745
W= [[ 10.49907256]
 [-31.06694195]
 [ 20.73039288]]
B= [[-0.07999603]]
```

可以观察到达到0.005的损失值，这个神经网络迭代了2379个`epoch`。而在二次多项式的试验中，用了10000次的迭代也没有达到要求。

### 9.1.4 用四次多项式拟合
第一步依然是增加x的4次方作为特征值：

```Python
        X = self.XTrain[:,0:1]**4
        self.XTrain = np.hstack((self.XTrain, X))
```

第二步设置超参num_input=4，然后训练，得到表9-6的结果。

表9-6 四次多项式训练过程与结果

|损失函数值|拟合结果|
|---|---|
|<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/9/sin_loss_4p.png">|<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/9/sin_result_4p.png">|


```
......
8279 49 0.00500000873141068
8289 49 0.0049964143635271635
W= [[  8.78717   ]
 [-20.55757649]
 [  1.28964911]
 [ 10.88610303]]
B= [[-0.04688634]]
```

### 9.1.5 结果比较

表9-7 不同项数的多项式拟合结果比较

|多项式次数|迭代数|损失函数值|
|:---:|---:|---:|
|2|10000|0.095|
|3|2380|0.005|
|4|8290|0.005|

从表9-7的结果比较中可以得到以下结论：

1. 二次多项式的损失值在下降了一定程度后，一直处于平缓期，不再下降，说明网络能力到了一定的限制，直到10000次迭代也没有达到目的；
2. 损失值达到0.005时，四项式迭代了8290次，比三次多项式的2380次要多很多，说明四次多项式多出的一个特征值，没有给我们带来什么好处，反而是增加了网络训练的复杂度。

由此可以知道，多项式次数并不是越高越好，对不同的问题，有特定的限制，需要在实践中摸索，并无理论指导。


## 9.2 用多项式回归法拟合复合函数曲线

还记得我们在本章最开始提出的两个问题吗？在上一节中我们解决了问题一，学习了用多项式拟合正弦曲线，在本节中，我们尝试着用多项式解决问题二，拟合复杂的函数曲线。

<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/9/Sample.png" ch="500" />

图9-4 样本数据可视化

再把图9-4所示的这条“眼镜蛇形”曲线拿出来观察一下，不但有正弦式的波浪，还有线性的爬升，转折处也不是很平滑，所以难度很大。从正弦曲线的拟合经验来看，三次多项式以下肯定无法解决，所以我们可以从四次多项式开始试验。

### 9.2.1 用四次多项式拟合

代码与正弦函数拟合方法区别不大，不再赘述，我们本次主要说明解决问题的思路。

超参的设置情况：

```Python
    num_input = 4
    num_output = 1    
    params = HyperParameters(num_input, num_output, eta=0.2, max_epoch=10000, batch_size=10, eps=1e-3, net_type=NetType.Fitting)
```
最开始设置`max_epoch=10000`，运行结果如表9-8所示。

表9-8 四次多项式1万次迭代的训练结果

|损失函数历史|曲线拟合结果|
|---|---|
|<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/9/complex_loss_4_10k.png">|<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/9/complex_result_4_10k.png">|

可以看到损失函数值还有下降的空间，拟合情况很糟糕。以下是打印输出结果：

```
......
9899 99 0.004994434937236122
9999 99 0.0049819495247358375
W= [[-0.70780292]
 [ 5.01194857]
 [-9.6191971 ]
 [ 6.07517269]]
B= [[-0.27837814]]
```

所以我们增加`max_epoch`到100000再试一次。

表9-9 四次多项式10万次迭代的训练结果

|损失函数历史|曲线拟合结果|
|---|---|
|<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/9/complex_loss_4_100k.png">|<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/9/complex_result_4_100k.png">|

从表9-9中的左图看，损失函数值到了一定程度后就不再下降了，说明网络能力有限。再看下面打印输出的具体数值，在0.005左右是一个极限。

```
......
99899 99 0.004685711600240152
99999 99 0.005299305272730845
W= [[ -2.18904889]
 [ 11.42075916]
 [-19.41933987]
 [ 10.88980241]]
B= [[-0.21280055]]
```

### 9.2.2 用六次多项式拟合

接下来跳过5次多项式，直接用6次多项式来拟合。这次不需要把`max_epoch`设置得很大，可以先试试50000个`epoch`。

表9-10 六次多项式5万次迭代的训练结果

|损失函数历史|曲线拟合结果|
|---|---|
|<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/9/complex_loss_6_50k.png">|<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/9/complex_result_6_50k.png">|

打印输出：

```
999 99 0.005154576065966749
1999 99 0.004889156300531125
......
48999 99 0.0047460241904710935
49999 99 0.004669517756696059
W= [[-1.46506264]
 [ 6.60491296]
 [-6.53643709]
 [-4.29857685]
 [ 7.32734744]
 [-0.85129652]]
B= [[-0.21745171]]
```

从表9-10的损失函数历史图看，损失值下降得比较理想，但是实际看打印输出时，损失值最开始几轮就已经是0.0047了，到了最后一轮，是0.0046，并不理想，说明网络能力还是不够。因此在这个级别上，不用再花时间继续试验了，应该还需要提高多项式次数。

### 9.2.3 用八次多项式拟合

再跳过7次多项式，直接使用8次多项式。先把`max_epoch`设置为50000试验一下。

表9-11 八项式5万次迭代的训练结果

|损失函数历史|曲线拟合结果|
|---|---|
|<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/9/complex_loss_8_50k.png">|<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/9/complex_result_8_50k.png">|

表9-11中损失函数值下降的趋势非常可喜，似乎还没有遇到什么瓶颈，仍有下降的空间，并且拟合的效果也已经初步显现出来了。

再看下面的打印输出，损失函数值已经可以突破0.004的下限了。

```
......
49499 99 0.004086918553033752
49999 99 0.0037740488283595657
W= [[ -2.44771419]
 [  9.47854206]
 [ -3.75300184]
 [-14.39723202]
 [ -1.10074631]
 [ 15.09613263]
 [ 13.37017924]
 [-15.64867322]]
B= [[-0.16513259]]
```

根据以上情况，可以认为8次多项式很有可能得到比较理想的解，所以我们需要增加`max_epoch`数值，让网络得到充分的训练。好，设置`max_epoch=1000000`试一下！没错，是一百万次！开始运行后，大家就可以去做些别的事情，一两个小时之后再回来看结果。

表9-12 八项式100万次迭代的训练结果

|损失函数历史|曲线拟合结果|
|---|---|
|<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/9/complex_loss_8_1M.png">|<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/9/complex_result_8_1M.png">|

从表9-12的结果来看，损失函数值还有下降的空间和可能性，已经到了0.0016的水平（从后面的章节中可以知道，0.001的水平可以得到比较好的拟合效果），拟合效果也已经初步呈现出来了，所有转折的地方都可以复现，只是精度不够，相信更多的训练次数可以达到更好的效果。

```
......
998999 99 0.0015935143877633367
999999 99 0.0016124984420510522
W= [[  2.75832935]
 [-30.05663986]
 [ 99.68833781]
 [-85.95142109]
 [-71.42918867]
 [ 63.88516377]
 [104.44561608]
 [-82.7452897 ]]
B= [[-0.31611388]]
```

分析打印出的`W`权重值，x的原始特征值的权重值比后面的权重值小了一到两个数量级，这与归一化后x的高次幂的数值很小有关系。

至此，我们可以得出结论，多项式回归确实可以解决复杂曲线拟合问题，但是代价有些高，我们训练了一百万次，才得到初步满意的结果。


## 9.3 验证与测试

### 9.3.1 基本概念

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

### 9.3.2 交叉验证

#### 传统的机器学习

在传统的机器学习中，我们经常用交叉验证的方法，比如把数据分成10份，$V_1\sim V_{10}$，其中 $V_1 \sim V_9$ 用来训练，$V_{10}$ 用来验证。然后用 $V_2\sim V_{10}$ 做训练，$V_1$ 做验证……如此我们可以做10次训练和验证，大大增加了模型的可靠性。

这样的话，验证集也可以做训练，训练集数据也可以做验证，当样本很少时，这个方法很有用。

#### 神经网络/深度学习

那么深度学习中的用法是什么呢？

比如在神经网络中，训练时到底迭代多少次停止呢？或者我们设置学习率为多少何时呢？或者用几个中间层，以及每个中间层用几个神经元呢？如何正则化？这些都是超参数设置，都可以用验证集来解决。

但对于实际应用中的问题，我们可以用验证集来验证一下准确率，假设只有90%的准确率，可能是局部最优解。这样我们可以继续迭代，寻找全局最优解。

举个例子：一个BP神经网络，我们无法确定隐层的神经元数目，因为没有理论支持。此时可以按图9-6的示意图这样做。

<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/9/CrossValidation.png" ch="500" />

图9-6 交叉训练的数据配置方式

1. 随机将训练数据分成K等份（通常建议 $K=10$），得到$D_0,D_1,D_9$；
2. 对于一个模型M，选择 $D_9$ 为验证集，其它为训练集，训练若干轮，用 $D_9$ 验证，得到误差 $E$。再训练，再用 $D_9$ 测试，如此N次。对N次的误差做平均，得到平均误差；
3. 换一个不同参数的模型的组合，比如神经元数量，或者网络层数，激活函数，重复2，但是这次用 $D_8$ 去得到平均误差；
4. 重复步骤2，一共验证10组组合；
5. 最后选择具有最小平均误差的模型结构，用所有的 $D_0 \sim D_9$ 再次训练，成为最终模型，不用再验证；
6. 用测试集测试。

### 9.3.3 留出法 Hold out

使用交叉验证的方法虽然比较保险，但是非常耗时，尤其是在大数据量时，训练出一个模型都要很长时间，没有可能去训练出10个模型再去比较。

在深度学习中，有另外一种方法使用验证集，称为留出法。亦即从训练数据中保留出验证样本集，主要用于解决过拟合情况，这部分数据不用于训练。如果训练数据的准确度持续增长，但是验证数据的准确度保持不变或者反而下降，说明神经网络亦即过拟合了，此时需要停止训练，用测试集做最终测试。

所以，训练步骤的伪代码如下：

```
for each epoch
    shuffle
    for each iteraion
        获得当前小批量数据
        前向计算
        反向传播
        更新梯度
        if is checkpoint
            用当前小批量数据计算训练集的loss值和accuracy值并记录
            计算验证集的loss值和accuracy值并记录
            如果loss值不再下降，停止训练
            如果accuracy值满足要求，停止训练
        end if
    end for
end for
```

从本章开始，我们将使用新的`DataReader`类来管理训练/测试数据，与前面的`SimpleDataReader`类相比，这个类有以下几个不同之处：

- 要求既有训练集，也有测试集
- 提供`GenerateValidationSet()`方法，可以从训练集中产生验证集

以上两个条件保证了我们在以后的训练中，可以使用本节中所描述的留出法，来监控整个训练过程。

关于三者的比例关系，在传统的机器学习中，三者可以是6:2:2。在深度学习中，一般要求样本数据量很大，所以可以给训练集更多的数据，比如8:1:1。

如果有些数据集已经给了你训练集和测试集，那就不关心其比例问题了，只需要从训练集中留出10%左右的验证集就可以了。

### 9.3.4 代码实现

定义DataReader类如下：

```Python
class DataReader(object):
    def __init__(self, train_file, test_file):
        self.train_file_name = train_file
        self.test_file_name = test_file
        self.num_train = 0        # num of training examples
        self.num_test = 0         # num of test examples
        self.num_validation = 0   # num of validation examples
        self.num_feature = 0      # num of features
        self.num_category = 0     # num of categories
        self.XTrain = None        # training feature set
        self.YTrain = None        # training label set
        self.XTest = None         # test feature set
        self.YTest = None         # test label set
        self.XTrainRaw = None     # training feature set before normalization
        self.YTrainRaw = None     # training label set before normalization
        self.XTestRaw = None      # test feature set before normalization
        self.YTestRaw = None      # test label set before normalization
        self.XVld = None          # validation feature set
        self.YVld = None          # validation lable set
```

命名规则：

1. 以`num_`开头的表示一个整数，后面跟着数据集的各种属性的名称，如训练集（`num_train`）、测试集（`num_test`）、验证集（`num_validation`）、特征值数量（`num_feature`）、分类数量（`num_category`）；
2. `X`表示样本特征值数据，`Y`表示样本标签值数据；
3. `Raw`表示没有经过归一化的原始数据。

#### 得到训练集和测试集

一般的数据集都有训练集和测试集，如果没有，需要从一个单一数据集中，随机抽取出一小部分作为测试集，剩下的一大部分作为训练集，一旦测试集确定后，就不要再更改。然后在训练过程中，从训练集中再抽取一小部分作为验证集。

#### 读取数据

```Python
    def ReadData(self):
        train_file = Path(self.train_file_name)
        if train_file.exists():
            ...

        test_file = Path(self.test_file_name)
        if test_file.exists():
            ...
```

在读入原始数据后，数据存放在`XTrainRaw`、`YTrainRaw`、`XTestRaw`、`YTestRaw`中。由于有些数据不需要做归一化处理，所以，在读入数据集后，令：`XTrain=XTrainRaw`、`YTrain=YTrainRaw`、`XTest=XTestRaw`、`YTest=YTestRaw`，如此一来，就可以直接使用`XTrain`、`YTrain`、`XTest`、`YTest`做训练和测试了，避免不做归一化时上述4个变量为空。

#### 特征值归一化

```Python
    def NormalizeX(self):
        x_merge = np.vstack((self.XTrainRaw, self.XTestRaw))
        x_merge_norm = self.__NormalizeX(x_merge)
        train_count = self.XTrainRaw.shape[0]
        self.XTrain = x_merge_norm[0:train_count,:]
        self.XTest = x_merge_norm[train_count:,:]
```

如果需要归一化处理，则`XTrainRaw` -> `XTrain`、`YTrainRaw` -> `YTrain`、`XTestRaw` -> `XTest`、`YTestRaw` -> `YTest`。注意需要把`Train`、`Test`同时归一化，如上面代码中，先把`XTrainRaw`和`XTestRaw`合并，一起做归一化，然后再拆开，这样可以保证二者的值域相同。

比如，假设`XTrainRaw`中的特征值只包含1、2、3三种值，在对其归一化时，1、2、3会变成0、0.5、1；而`XTestRaw`中的特征值只包含2、3、4三种值，在对其归一化时，2、3、4会变成0、0.5、1。这就造成了0、0.5、1这三个值的含义在不同数据集中不一样。

把二者merge后，就包含了1、2、3、4四种值，再做归一化，会变成0、0.333、0.666、1，在训练和测试时，就会使用相同的归一化值。

#### 标签值归一化

根据不同的网络类型，标签值的归一化方法也不一样。

```Python
    def NormalizeY(self, nettype, base=0):
        if nettype == NetType.Fitting:
            ...
        elif nettype == NetType.BinaryClassifier:
            ...
        elif nettype == NetType.MultipleClassifier:
            ...
```

- 如果是`Fitting`任务，即线性回归、非线性回归，对标签值使用普通的归一化方法，把所有的值映射到[0,1]之间
- 如果是`BinaryClassifier`，即二分类任务，把标签值变成0或者1。`base`参数是指原始数据中负类的标签值。比如，原始数据的两个类别标签值是1、2，则`base=1`，把1、2变成0、1
- 如果是`MultipleClassifier`，即多分类任务，把标签值变成One-Hot编码。

#### 生成验证集

```Python
    def GenerateValidationSet(self, k = 10):
        self.num_validation = (int)(self.num_train / k)
        self.num_train = self.num_train - self.num_validation
        # validation set
        self.XVld = self.XTrain[0:self.num_validation]
        self.YVld = self.YTrain[0:self.num_validation]
        # train set
        self.XTrain = self.XTrain[self.num_validation:]
        self.YTrain = self.YTrain[self.num_validation:]
```

验证集是从归一化好的训练集中抽取出来的。上述代码假设`XTrain`已经做过归一化，并且样本是无序的。如果样本是有序的，则需要先打乱。

#### 获得批量样本
```Python
    def GetBatchTrainSamples(self, batch_size, iteration):
        start = iteration * batch_size
        end = start + batch_size
        batch_X = self.XTrain[start:end,:]
        batch_Y = self.YTrain[start:end,:]
        return batch_X, batch_Y
```
训练时一般采样Mini-batch梯度下降法，所以要指定批大小`batch_size`和当前批次`iteration`，就可以从已经打乱过的样本中获得当前批次的数据，在一个epoch中根据iteration的递增调用此函数。

#### 样本打乱
```Python
    def Shuffle(self):
        seed = np.random.randint(0,100)
        np.random.seed(seed)
        XP = np.random.permutation(self.XTrain)
        np.random.seed(seed)
        YP = np.random.permutation(self.YTrain)
        self.XTrain = XP
        self.YTrain = YP
```

样本打乱操作只涉及到训练集，在每个epoch开始时调用此方法。打乱时，要注意特征值X和标签值Y是分开存放的，所以要使用相同的`seed`来打乱，保证打乱顺序后的特征值和标签值还是一一对应的。


## 9.4 双层神经网络实现非线性回归

### 9.4.1 万能近似定理

万能近似定理(universal approximation theorem) $^{[1]}$，是深度学习最根本的理论依据。它证明了在给定网络具有足够多的隐藏单元的条件下，配备一个线性输出层和一个带有任何“挤压”性质的激活函数（如Sigmoid激活函数）的隐藏层的前馈神经网络，能够以任何想要的误差量近似任何从一个有限维度的空间映射到另一个有限维度空间的Borel可测的函数。

万能近似定理其实说明了理论上神经网络可以近似任何函数。但实践上我们不能保证学习算法一定能学习到目标函数。即使网络可以表示这个函数，学习也可能因为两个不同的原因而失败：

1. 用于训练的优化算法可能找不到用于期望函数的参数值；
2. 训练算法可能由于过拟合而选择了错误的函数。


### 9.4.2 定义神经网络结构

本节的目的是要用神经网络完成图9-1和图9-2中的曲线拟合。

根据万能近似定理的要求，我们定义一个两层的神经网络，输入层不算，一个隐藏层，含3个神经元，一个输出层。图9-7显示了此次用到的神经网络结构。

<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/9/nn.png" />

图9-7 单入单出的双层神经网络

输入层只有一个特征值，我们不需要在隐层放很多的神经元，先用3个神经元试验一下。如果不够的话再增加，神经元数量是由超参控制的。

#### 输入层

输入层就是一个标量x值，如果是成批输入，则是一个矢量或者矩阵，但是特征值数量总为1，因为只有一个横坐标值做为输入。

$$X = (x)$$

#### 权重矩阵W1/B1

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

#### 隐层

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


#### 权重矩阵W2/B2

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

#### 输出层

由于我们只想完成一个拟合任务，所以输出层只有一个神经元，尺寸为1x1：

$$
Z2 = 
\begin{pmatrix}
    z2_{1}
\end{pmatrix}
$$

### 9.4.3 前向计算

根据图9-7的网络结构，我们可以得到如图9-8的前向计算图。

<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/9/forward.png" />

图9-8 前向计算图

#### 隐层

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

#### 输出层

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

#### 损失函数

均方差损失函数：

$$loss(w,b) = \frac{1}{2} (z2-y)^2 \tag{4}$$

其中，$z2$是预测值，$y$是样本的标签值。

### 9.4.4 反向传播

我们比较一下本章的神经网络和第5章的神经网络的区别，看表9-13。

表9-13 本章中的神经网络与第5章的神经网络的对比

|第5章的神经网络|本章的神经网络|
|---|---|
|<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images\Images\5\setup.png"/>|<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/9/nn.png"/>|

我们需要推导一下反向传播的各个过程。看一下计算图，然后用链式求导法则反推。

#### 求损失函数对输出层的反向误差

根据公式4：

$$
\frac{\partial loss}{\partial z2} = z2 - y \rightarrow dZ2 \tag{5}
$$

#### 求W2的梯度

根据公式3和W2的矩阵形状，把标量对矩阵的求导分解到矩阵中的每一元素：

$$
\begin{aligned}
\frac{\partial loss}{\partial W2} &= 
\begin{pmatrix}
    \frac{\partial loss}{\partial z2}\frac{\partial z2}{\partial w2_{11}} \\\\
    \frac{\partial loss}{\partial z2}\frac{\partial z2}{\partial w2_{21}} \\\\
    \frac{\partial loss}{\partial z2}\frac{\partial z2}{\partial w2_{31}}
\end{pmatrix}
\begin{pmatrix}
    dZ2 \cdot a1_{1} \\\\
    dZ2 \cdot a1_{2} \\\\
    dZ2 \cdot a1_{3}
\end{pmatrix} \\\\
&=\begin{pmatrix}
    a1_{1} \\\\ a1_{2} \\\\ a1_{3}
\end{pmatrix} \cdot dZ2
=A1^{\top} \cdot dZ2 \rightarrow dW2
\end{aligned} \tag{6}
$$

#### 求B2的梯度

$$
\frac{\partial loss}{\partial B2}=dZ2 \rightarrow dB2 \tag{7}
$$

与第5章相比，除了把X换成A以外，其它的都一样。对于输出层来说，A就是它的输入，也就相当于是X。

#### 求损失函数对隐层的反向误差

下面的内容是双层神经网络独有的内容，也是深度神经网络的基础，请大家仔细阅读体会。我们先看看正向计算和反向计算图，即图9-9。

<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/9/backward.png" />

图9-9 正向计算和反向传播路径图

图9-9中：

- 蓝色矩形表示数值或矩阵；
- 蓝色圆形表示计算单元；
- 蓝色的箭头表示正向计算过程；
- 红色的箭头表示反向计算过程。

如果想计算W1和B1的反向误差，必须先得到Z1的反向误差，再向上追溯，可以看到Z1->A1->Z2->Loss这条线，Z1->A1是一个激活函数的运算，比较特殊，所以我们先看Loss->Z->A1如何解决。

根据公式3和A1矩阵的形状：

$$
\begin{aligned}
\frac{\partial loss}{\partial A1}&=
\begin{pmatrix}
    \frac{\partial loss}{\partial Z2}\frac{\partial Z2}{\partial a1_{11}}
    &
    \frac{\partial loss}{\partial Z2}\frac{\partial Z2}{\partial a1_{12}}
    &
    \frac{\partial loss}{\partial Z2}\frac{\partial Z2}{\partial a1_{13}}
\end{pmatrix} \\\\
&=
\begin{pmatrix}
dZ2 \cdot w2_{11} & dZ2 \cdot w2_{12} & dZ2 \cdot w2_{13}
\end{pmatrix} \\\\
&=dZ2 \cdot
\begin{pmatrix}
    w2_{11} & w2_{21} & w2_{31}
\end{pmatrix} \\\\
&=dZ2 \cdot
\begin{pmatrix}
    w2_{11} \\\\ w2_{21} \\\\ w2_{31}
\end{pmatrix}^{\top}=dZ2 \cdot W2^{\top}
\end{aligned} \tag{8}
$$

现在来看激活函数的误差传播问题，由于公式2在计算时，并没有改变矩阵的形状，相当于做了一个矩阵内逐元素的计算，所以它的导数也应该是逐元素的计算，不改变误差矩阵的形状。根据Sigmoid激活函数的导数公式，有：

$$
\frac{\partial A1}{\partial Z1}= Sigmoid'(A1) = A1 \odot (1-A1) \tag{9}
$$

所以最后到达Z1的误差矩阵是：

$$
\begin{aligned}
\frac{\partial loss}{\partial Z1}&=\frac{\partial loss}{\partial A1}\frac{\partial A1}{\partial Z1} \\\\
&=dZ2 \cdot W2^T \odot Sigmoid'(A1) \rightarrow dZ1
\end{aligned} \tag{10}
$$

有了dZ1后，再向前求W1和B1的误差，就和第5章中一样了，我们直接列在下面：

$$
dW1=X^T \cdot dZ1 \tag{11}
$$

$$
dB1=dZ1 \tag{12}
$$

### 9.4.5 代码实现

主要讲解神经网络`NeuralNet2`类的代码，其它的类都是辅助类。

#### 前向计算

```Python
class NeuralNet2(object):
    def forward(self, batch_x):
        # layer 1
        self.Z1 = np.dot(batch_x, self.wb1.W) + self.wb1.B
        self.A1 = Sigmoid().forward(self.Z1)
        # layer 2
        self.Z2 = np.dot(self.A1, self.wb2.W) + self.wb2.B
        if self.hp.net_type == NetType.BinaryClassifier:
            self.A2 = Logistic().forward(self.Z2)
        elif self.hp.net_type == NetType.MultipleClassifier:
            self.A2 = Softmax().forward(self.Z2)
        else:   # NetType.Fitting
            self.A2 = self.Z2
        #end if
        self.output = self.A2
```        
在`Layer2`中考虑了多种网络类型，在此我们暂时只关心`NetType.Fitting`类型。

#### 反向传播
```Python
class NeuralNet2(object):
    def backward(self, batch_x, batch_y, batch_a):
        # 批量下降，需要除以样本数量，否则会造成梯度爆炸
        m = batch_x.shape[0]
        # 第二层的梯度输入 公式5
        dZ2 = self.A2 - batch_y
        # 第二层的权重和偏移 公式6
        self.wb2.dW = np.dot(self.A1.T, dZ2)/m 
        # 公式7 对于多样本计算，需要在横轴上做sum，得到平均值
        self.wb2.dB = np.sum(dZ2, axis=0, keepdims=True)/m 
        # 第一层的梯度输入 公式8
        d1 = np.dot(dZ2, self.wb2.W.T) 
        # 第一层的dZ 公式10
        dZ1,_ = Sigmoid().backward(None, self.A1, d1)
        # 第一层的权重和偏移 公式11
        self.wb1.dW = np.dot(batch_x.T, dZ1)/m
        # 公式12 对于多样本计算，需要在横轴上做sum，得到平均值
        self.wb1.dB = np.sum(dZ1, axis=0, keepdims=True)/m 
```
反向传播部分的代码完全按照公式推导的结果实现。

#### 保存和加载权重矩阵数据

在训练结束后，或者每个epoch结束后，都可以选择保存训练好的权重矩阵值，避免每次使用时重复训练浪费时间。

而在初始化完毕神经网络后，可以立刻加载历史权重矩阵数据（前提是本次的神经网络设置与保存时的一致），这样可以在历史数据的基础上继续训练，不会丢失以前的进度。

```Python
    def SaveResult(self):
        self.wb1.SaveResultValue(self.subfolder, "wb1")
        self.wb2.SaveResultValue(self.subfolder, "wb2")

    def LoadResult(self):
        self.wb1.LoadResultValue(self.subfolder, "wb1")
        self.wb2.LoadResultValue(self.subfolder, "wb2")
```

#### 辅助类

- `Activators` - 激活函数类，包括Sigmoid/Tanh/Relu等激活函数的实现，以及Losistic/Softmax分类函数的实现
- `DataReader` - 数据操作类，读取、归一化、验证集生成、获得指定类型批量数据
- `HyperParameters2` - 超参类，各层的神经元数量、学习率、批大小、网络类型、初始化方法等

```Python
class HyperParameters2(object):
    def __init__(self, n_input, n_hidden, n_output, 
                 eta=0.1, max_epoch=10000, batch_size=5, eps = 0.1, 
                 net_type = NetType.Fitting,
                 init_method = InitialMethod.Xavier):
```

- `LossFunction` - 损失函数类，包含三种损失函数的代码实现
- `NeuralNet2` - 神经网络类，初始化、正向、反向、更新、训练、验证、测试等一系列方法
- `TrainingTrace` - 训练记录类，记录训练过程中的损失函数值、验证精度
- `WeightsBias` - 权重矩阵类，初始化、加载数据、保存数据

### 代码位置

ch09, HelperClass2

- 双层神经网络解决方案的基本代码都在`HelperClass2`子目录下

### 参考文献

[1] Hornik, K., Stinchcombe, M., and White, H. (1989). Multilayer feedforward networks are uni-versal approximators. Neural Networks, 2, 359–366. 171


## 9.5 曲线拟合

在上一节我们已经写好了神经网络的核心模块及其辅助功能，现在我们先来做一下正弦曲线的拟合，然后再试验复合函数的曲线拟合。

### 9.5.1 正弦曲线的拟合

#### 隐层只有一个神经元的情况

令`n_hidden=1`，并指定模型名称为`sin_111`，训练过程见图9-10。图9-11为拟合效果图。

<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/9/sin_loss_1n.png" />

图9-10 训练过程中损失函数值和准确率的变化

<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/9/sin_result_1n.png" ch="500" />

图9-11 一个神经元的拟合效果

从图9-10可以看到，损失值到0.04附近就很难下降了。图9-11中，可以看到只有中间线性部分拟合了，两端的曲线部分没有拟合。

```
......
epoch=4999, total_iteration=224999
loss_train=0.015787, accuracy_train=0.943360
loss_valid=0.038609, accuracy_valid=0.821760
testing...
0.8575700023301912
```

打印输出最后的测试集精度值为85.7%，不是很理想。所以隐层1个神经元是基本不能工作的，这只比单层神经网络的线性拟合强一些，距离目标还差很远。

#### 隐层有两个神经元的情况

```Python
if __name__ == '__main__':
    ......
    n_input, n_hidden, n_output = 1, 2, 1
    eta, batch_size, max_epoch = 0.05, 10, 5000
    eps = 0.001
    hp = HyperParameters2(n_input, n_hidden, n_output, eta, max_epoch, batch_size, eps, NetType.Fitting, InitialMethod.Xavier)
    net = NeuralNet2(hp, "sin_121")
    #net.LoadResult()
    net.train(dataReader, 50, True)
    ......
```

初始化神经网络类的参数有两个，第一个是超参组合`hp`，第二个是指定模型专有名称，以便把结果保存在名称对应的子目录中。保存训练结果的代码在训练结束后自动调用，但是如果想加载历史训练结果，需要在主过程中手动调用，比如上面代码中注释的那一行：`net.LoadResult()`。这样的话，如果下次再训练，就可以在以前的基础上继续训练，不必从头开始。

注意在主过程代码中，我们指定了n_hidden=2，意为隐层神经元数量为2。

#### 运行结果

图9-12为损失函数曲线和验证集精度曲线，都比较正常。而2个神经元的网络损失值可以达到0.004，少一个数量级。验证集精度到82%左右，而2个神经元的网络可以达到97%。图9-13为拟合效果图。

<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/9/sin_loss_2n.png"/>

图9-12 两个神经元的训练过程中损失函数值和准确率的变化

<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/9/sin_result_2n.png"/>

图9-13 两个神经元的拟合效果

再看下面的打印输出结果，最后测试集的精度为98.8%。如果需要精度更高的话，可以增加迭代次数。

```
......
epoch=4999, total_iteration=224999
loss_train=0.007681, accuracy_train=0.971567
loss_valid=0.004366, accuracy_valid=0.979845
testing...
0.9881468747638157
```

### 9.5.2 复合函数的拟合

基本过程与正弦曲线相似，区别是这个例子要复杂不少，所以首先需要耐心，增大`max_epoch`的数值，多迭代几次。其次需要精心调参，找到最佳参数组合。

#### 隐层只有两个神经元的情况

<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/9/complex_result_2n.png" ch="500" />

图9-14 两个神经元的拟合效果

图9-14是两个神经元的拟合效果图，拟合情况很不理想，和正弦曲线只用一个神经元的情况类似。观察打印输出的损失值，有波动，久久徘徊在0.003附近不能下降，说明网络能力不够。

```
epoch=99999, total_iteration=8999999
loss_train=0.000751, accuracy_train=0.968484
loss_valid=0.003200, accuracy_valid=0.795622
testing...
0.8641114405898856
```

#### 隐层有三个神经元的情况

```Python
if __name__ == '__main__':
    ......
    n_input, n_hidden, n_output = 1, 3, 1
    eta, batch_size, max_epoch = 0.5, 10, 10000
    eps = 0.001
    hp = HyperParameters2(n_input, n_hidden, n_output, eta, max_epoch, batch_size, eps, NetType.Fitting, InitialMethod.Xavier)
    net = NeuralNet2(hp, "model_131")
    ......
```

#### 运行结果

图9-15为损失函数曲线和验证集精度曲线，都比较正常。图9-16是拟合效果。

<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/9/complex_loss_3n.png" />

图9-15 三个神经元的训练过程中损失函数值和准确率的变化

<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/9/complex_result_3n.png"/>

图9-16 三个神经元的拟合效果

再看下面的打印输出结果，最后测试集的精度为97.6%，已经令人比较满意了。如果需要精度更高的话，可以增加迭代次数。

```
......
epoch=4199, total_iteration=377999
loss_train=0.001152, accuracy_train=0.963756
loss_valid=0.000863, accuracy_valid=0.944908
testing...
0.9765910104463337
```

以下就是笔者找到的最佳组合：

- 隐层3个神经元
- 学习率=0.5
- 批量=10

### 9.5.3 广义的回归/拟合

我们用简单的例子讲解了神经网络的功能，但是此功能完全可以用于多变量的复杂非线性回归。

简言之，只要是数值拟合问题，确定不能用线性回归的话，都可以用非线性回归来尝试解决。


## 9.6 非线性回归的工作原理

### 9.6.1 多项式为何能拟合曲线

当我们使用双层神经网络时，在隐层只放置了三个神经元，就轻松解决了复合函数拟合的问题，效率高出十几倍，复杂度却降低了几倍。那么含有隐层的神经网络究竟是如何完成这个任务的呢？

我们以正弦曲线拟合为例来说明这个问题，首先看一下多项式回归方法的示意图，如图9-17。

<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/9/polynomial_concept.png"/>

图9-17 多项式回归方法的特征值输入

单层神经网络的多项式回归法，需要$x,x^2,x^3$三个特征值，组成如下公式来得到拟合结果：

$$
z = x \cdot w_1 + x^2 \cdot w_2 + x^3 \cdot w_3 + b \tag{1}
$$

我们可以回忆一下第5章学习的多变量线性回归问题，公式1实际上是把一维的x的特征信息，增加到了三维，然后再使用多变量线性回归来解决问题的。本来一维的特征只能得到线性的结果，但是三维的特征就可以得到非线性的结果，这就是多项式拟合的原理。

我们用具体的数值计算方式来理解一下其工作过程。

```Python
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    x = np.linspace(0,1,10)
    w = 1.1
    b = 0.2
    y = x * w + b
    p1, = plt.plot(x,y, marker='.')

    x2 = x*x
    w2 = -0.5
    y2 = x * w + x2 * w2 + b
    p2, = plt.plot(x, y2, marker='s')

    x3 = x*x*x
    w3 = 2.3
    y3 = x * w + x2 * w2 + x3 * w3 + b
    p3, = plt.plot(x, y3, marker='x')

    plt.grid()
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("linear and non-linear")
    plt.legend([p1,p2,p3], ["x","x*x","x*x*x"])
    plt.show()
```

上述代码完成了如下任务：

1. 定义 $[0,1]$ 之间的等距的10个点
2. 使用 $w=1.1,b=0.2$ 计算一个线性回归数值`y`
3. 使用 $w=1.1,w2=-0.5,b=0.2$ 计算一个二项式回归数值`y2`
4. 使用 $w=1.1,w2=-0.5,w3=2.3,b=0.2$ 计算一个三项式回归数值`y3`
5. 绘制出三条曲线，如图9-18所示。

<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/9/concept1.png" ch="500" />

图9-18 线性到非线性的变换

可以清楚地看到：

- 蓝色直线是线性回归数值序列；
- 红色曲线是二项式回归数值序列；
- 绿色曲线是三项式回归数值序列。

也就是说，我们只使用了同一个x序列的原始值，却可以得到三种不同数值序列，这就是多项式拟合的原理。当多项式次数很高、数据样本充裕、训练足够多的时候，甚至可以拟合出非单调的曲线。

### 9.6.2 神经网络的非线性拟合工作原理

我们以正弦曲线的例子来讲解神经网络非线性回归的工作过程和原理。

表9-14

|单层多项式回归|双层神经网络|
|---|---|
|<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/9/polynomial_concept.png">|<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/9/neuralnet_concept.png">|

比较一下表9-14中的两张图，左侧为单特征多项式拟合的示意图，右侧为双层神经网络的示意图。

左图中，通过人为的方式，给Z的输入增加了$x^2和x^3$项。

右图中，通过线性变换的方式，把x变成了两部分：$z_{11}/a_{11}，z_{12}/a_{12}$，然后再通过一次线性变换把两者组合成为Z，这种方式和多项式回归非常类似：

1. 隐层把x拆成不同的特征，根据问题复杂度决定神经元数量，神经元的数量相当于特征值的数量；
2. 隐层通过激活函数做一次非线性变换；
3. 输出层使用多变量线性回归，把隐层的输出当作输入特征值，再做一次线性变换，得出拟合结果。

与多项式回归不同的是，不需要指定变换参数，而是从训练中学习到参数，这样的话权重值不会大得离谱。

下面讲述具体的工作步骤。

#### 第一步 把X拆成两个线性序列z1和z2

假设原始值x有21个点，样本数据如表9-15所示。

表9-15

|id|0|1|2|...|19|20|21|
|--|--|--|--|--|--|--|--|
|x|0.|0.05|0.1|...|0.9|0.95|1.|

通过以下线性变换，被分成了两个线性序列，得到表9-16所示的隐层值：

$$
z1 = x \cdot w_{11} + b_{11} \tag{2}
$$
$$
z2 = x \cdot w_{12} + b_{12} \tag{3}
$$

其中：

- $w_{11} = -2.673$
- $b_{11} = 1.303$
- $w_{12} = -9.036$
- $b_{12} = 4.507$

表9-16 隐层线性变化结果

||0|1|2|...|19|20|21|
|--|--|--|--|--|--|--|--|
|z1|1.303|1.169|1.035|...|-1.102|-1.236|-1.369|
|z2|4.507|4.055|3.603|...|-3.625|-4.077|-4.528|

三个线性序列如图9-19所示，黑色点是原始数据序列，红色和绿色点是拆分后的两个序列。

<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/9/nn_concept_x_z1_z2.png" ch="500" />

图9-19 从原始数据序列拆分成的两个数据序列

这个运算相当于把特征值分解成两个部分。

#### 第二步 计算z1的激活函数值a1

表9-17和图9-20分别展示了隐层对第一个特征值的计算结果数值和示意图。

表9-17 第一个特征值及其激活函数结果数值

||0|1|2|...|19|20|21|
|--|--|--|--|--|--|--|--|
|z1|1.303|1.169|1.035|...|-1.102|-1.236|-1.369|
|a1|0.786|0.763|0.738|...|0.249|0.225|0.203|

第二行的a1值等于第1行的z1值的sigmoid函数值：

$$a1 = {1 \over 1+e^{-z1}} \tag{4}$$

<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/9/nn_concept_x_z1_a1.png" ch="500" />

图9-20 第一个特征值及其激活函数结果可视化

z1还是一条直线，但是经过激活函数后的a1已经不是一条直线了。上面这张图由于z1的跨度大，所以a1的曲线程度不容易看出来。

#### 第三步 计算z2的激活函数值a2

表9-18和图9-21分别展示了隐层对第二个特征值的计算结果数值和示意图。

表9-18 第二个特征值及其激活函数结果数值

||0|1|2|...|19|20|21|
|--|--|--|--|--|--|--|--|
|z2|4.507|4.055|3.603|...|-3.625|-4.077|-4.528|
|a2|0.989|0.983|0.973|...|0.026|0.017|0.011|

$$a2 = {1 \over 1+e^{-z2}} \tag{5}$$

<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/9/nn_concept_x_z2_a2.png" ch="500" />

图9-21 第二个特征值及其激活函数结果可视化

z2还是一条直线，但是经过激活函数后的a2已经明显看出是一条曲线了。

#### 第四步 计算Z值

表9-19和图9-22分别展示了输出层对两个特征值的计算结果数值和示意图。

表9-19 输出层的计算结果数值

||0|1|2|...|19|20|21|
|--|--|--|--|--|--|--|--|
|a1|0.786|0.763|0.738|...|0.249|0.225|0.203|
|a2|0.989|0.983|0.973|...|0.026|0.017|0.011|
|z|0.202|0.383|0.561|...|-0.580|-0.409|-0.235|

$$z = a1 \cdot w_{11} + a2 \cdot w_{21} + b \tag{6}$$

其中：

- $w_{11}=-9.374$
- $w_{21}=6.039$
- $b=1.599$
  
<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/9/nn_concept_a1_a2_z.png" ch="500" />

图9-22 输出层的计算结果可视化

也就是说，相同x值的红点a1和绿点a2，经过公式6计算后得到蓝点z，而所有的蓝点就拟合出一条正弦曲线。

### 9.6.3 比较多项式回归和双层神经网络解法

表9-20列出了多项式回归和神经网络的比较结果，可以看到神经网络处于绝对的优势地位。

表9-20 多项式回归和神经网络的比较

||多项式回归|双层神经网络|
|---|---|---|
|特征提取方式|特征值的高次方|线性变换拆分|
|特征值数量级|高几倍的数量级|数量级与原特征值相同|
|训练效率|低，需要迭代次数多|高，比前者少好几个数量级|



## 9.7 超参数优化的初步认识

超参数优化（Hyperparameter Optimization）主要存在两方面的困难：

1. 超参数优化是一个组合优化问题，无法像一般参数那样通过梯度下降方法来优化，也没有一种通用有效的优化方法。
2. 评估一组超参数配置（Conﬁguration）的时间代价非常高，从而导致一些优化方法（比如演化算法）在超参数优化中难以应用。

对于超参数的设置，比较简单的方法有人工搜索、网格搜索和随机搜索。 

### 9.7.1 可调的参数

我们使用表9-21所示的参数做第一次的训练。

表9-21 参数配置

|参数|缺省值|是否可调|注释|
|---|---|---|---|
|输入层神经元数|1|No|
|隐层神经元数|4|Yes|影响迭代次数|
|输出层神经元数|1|No|
|学习率|0.1|Yes|影响迭代次数|
|批样本量|10|Yes|影响迭代次数|
|最大epoch|10000|Yes|影响终止条件,建议不改动|
|损失门限值|0.001|Yes|影响终止条件,建议不改动|
|损失函数|MSE|No|
|权重矩阵初始化方法|Xavier|Yes|参看15.1|

表9-21中的参数，最终可以调节的其实只有三个：

- 隐层神经元数
- 学习率
- 批样本量

另外还有一个权重矩阵初始化方法需要特别注意，我们在后面的章节中讲解。

另外两个要提一下的参数，第一个是最大epoch数，根据不同的模型和案例会有所不同，在本例中10000次足以承载所有的超参组合了；另外一个是损失门限值，它是一个先验数值，也就是说笔者通过试验，事先知道了当`eps=0.001`时，会训练出精度可接受的模型来，但是在实践中没有这种先验知识，只能摸着石头过河，因为在训练一个特定模型之前，谁也不能假设它能到达的精度值是多少，损失函数值的下限也是通过多次试验，通过历史记录的趋势来估算出来的。

如果读者不了解神经网络中的基本原理，那么所谓“调参”就是碰运气了。今天咱们可以试着改变几个参数，来看看训练结果，以此来增加对神经网络中各种参数的了解。

#### 避免权重矩阵初始化的影响

权重矩阵中的参数，是神经网络要学习的参数，所以不能称作超参数。

权重矩阵初始化是神经网络训练非常重要的环节之一，不同的初始化方法，甚至是相同的方法但不同的随机值，都会给结果带来或多或少的影响。

在后面的几组比较中，都是用Xavier方法初始化的。在两次参数完全相同的试验中，即使两次都使用Xavier初始化，因为权重矩阵参数的差异，也会得到不同的结果。为了避免这个随机性，我们在代码`WeightsBias.py`中使用了一个小技巧，调用下面这个函数：

```Python
class WeightsBias(object):
    def InitializeWeights(self, folder, create_new):
        self.folder = folder
        if create_new:
            self.__CreateNew()
        else:
            self.__LoadExistingParameters()
        # end if

    def __CreateNew(self):
        self.W, self.B = WeightsBias.InitialParameters(self.num_input, self.num_output, self.init_method)
        self.__SaveInitialValue()
        
    def __LoadExistingParameters(self):
        file_name = str.format("{0}\\{1}.npz", self.folder, self.initial_value_filename)
        w_file = Path(file_name)
        if w_file.exists():
            self.__LoadInitialValue()
        else:
            self.__CreateNew()
```

第一次调用`InitializeWeights()`时，会得到一个随机初始化矩阵。以后再次调用时，如果设置`create_new=False`，只要隐层神经元数量不变并且初始化方法不变，就会用第一次的初始化结果，否则后面的各种参数调整的结果就没有可比性了。

至于为什么使用Xavier方法初始化，将在15.1中讲解。

### 9.7.2 手动调整参数

手动调整超参数，我们必须了解超参数、训练误差、泛化误差和计算资源（内存和运行时间）之间的关系。手动调整超参数的主要目标是调整模型的有效容量以匹配任务的复杂性。有效容量受限于3个因素：

- 模型的表示容量；
- 学习算法与代价函数的匹配程度；
- 代价函数和训练过程正则化模型的程度。

表9-22比较了几个超参数的作用。具有更多网络层、每层有更多隐藏单元的模型具有较高的表示能力，能够表示更复杂的函数。学习率是最重要的超参数。如果你只有一个超参数调整的机会，那就调整学习率。

表9-22 各种超参数的作用

|超参数|目标|作用|副作用|
|---|---|---|---|
|学习率|调至最优|低的学习率会导致收敛慢，高的学习率会导致错失最佳解|容易忽略其它参数的调整|
|隐层神经元数量|增加|增加数量会增加模型的表示能力|参数增多、训练时间增长|
|批大小|有限范围内尽量大|大批量的数据可以保持训练平稳，缩短训练时间|可能会收敛速度慢|

通常的做法是，按经验设置好隐层神经元数量和批大小，并使之相对固定，然后调整学习率。

### 9.7.3 网格搜索

当有3个或更少的超参数时，常见的超参数搜索方法是网格搜索（grid search）。对于每个超参数，选择一个较小的有限值集去试验。然后，这些超参数的笛卡儿乘积（所有的排列组合）得到若干组超参数，网格搜索使用每组超参数训练模型。挑选验证集误差最小的超参数作为最好的超参数组合。

用学习率和隐层神经元数量来举例，横向为学习率，取值 $[0.1,0.3,0.5,0.7]$；纵向为隐层神经元数量，取值 $[2,4,8,12]$，在每个组合上测试验证集的精度。我们假设其中最佳的组合精度达到0.97，学习率为0.5，神经元数为8，那么这个组合就是我们需要的模型超参，可以拿到测试集上去做最终测试了。

表9-23数据为假设的结果值，用于说明如何选择最终参数值。

表9-23 各种组合下的准确率

||eta=0.1|eta=0.3|eta=0.5|eta=0.7|
|---|---|---|---|---|
|ne=2|0.63|0.68|0.71|0.73|
|ne=4|0.86|0.89|0.91|0.3|
|ne=8|0.92|0.94|0.97|0.95|
|ne=12|0.69|0.84|0.88|0.87|

针对我们这个曲线拟合问题，规模较小，模型简单，所以可以用上表列出的数据做搜索。对于大规模模型问题，学习率的取值集合可以是 $\\{0.1,0.01,0.001,0.0001,0.00001\\}$，隐层单元数集合可以是 $\\{50,100,200,500,1000,2000\\}$，亦即在对数尺度上搜索，确定范围后，可以做进一步的小颗粒步长的搜索。

网格搜索带来的一个明显问题是，计算代价会随着超参数数量呈指数级增长。如果有m个超参数，每个最多取n个值，那么训练和估计所需的试验数将是$O(n^m)$。我们可以并行地进行实验，并且并行要求十分宽松（进行不同搜索的机器之间几乎没有必要进行通信）。令人遗憾的是，由于网格搜索指数级增长计算代价，即使是并行，我们也无法提供令人满意的搜索规模。

下面我们做一下具体的试验。

#### 学习率的调整

我们固定其它参数，即隐层神经元`ne=4`、`batch_size=10`不变，改变学习率，来试验网络训练情况。为了节省时间，不做无限轮次的训练，而是设置`eps=0.001`为最低精度要求，一旦到达，就停止训练。

表9-24和图9-23展示了四种学习率值的不同结果。

表9-24 四种学习率值的比较

|学习率|迭代次数|说明|
|----|----|----|
|0.1|10000|学习率小，收敛最慢，没有在规定的次数内达到精度要求|
|0.3|10000|学习率增大，收敛慢，没有在规定的次数内达到精度要求|
|0.5|8200|学习率增大，在8200次左右达到精度要求|
|0.7|3500|学习率进一步增大，在3500次达到精度|

<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/9/eta.png" ch="500" />

图9-23 四种学习率值造成的损失函数值的变化

#### 批大小的调整

我们固定其它参数，即隐层神经元`ne=4`、`eta=0.5`不变，调整批大小，来试验网络训练情况，设置`eps=0.001`为精度要求。

表9-25和图9-24展示了四种批大小值的不同结果。

表9-25 四种批大小数值的比较

|批大小|迭代次数|说明|
|----|----|----|
|5|2500|批数据量小到1，收敛最快|
|10|8200|批数据量增大，收敛变慢|
|15|10000|批数据量进一步增大，收敛变慢|
|20|10000|批数据量太大，反而会降低收敛速度|

<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/9/batchsize.png" ch="500" />

图9-24 四种批大小值造成的损失函数值的变化

合适的批样本量会带来较快的收敛，前提是我们固定了学习率。如果想用较大的批数据，底层数据库计算的速度较快，但是需要同时调整学习率，才会相应地提高收敛速度。

#### 隐层神经元数量的调整

我们固定其它参数，即`batch_size=10`、`eta=0.5`不变，调整隐层神经元的数量，来试验网络训练情况，设置`eps=0.001`为精度要求。

表9-26和图9-25展示了四种神经元数值的不同结果。

表9-26 四种隐层神经元数量值的比较

|隐层神经元数量|迭代次数|说明|
|---|---|---|
|2|10000|神经元数量少，拟合能力低|
|4|8000|神经元数量增加会有帮助|
|6|5500|神经元数量进一步增加，收敛更快|
|8|3500|再多一些神经元，还会继续提供收敛速度|

<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/9/neuron_number.png" ch="500" />

图9-25 四种隐层神经元数量值造成的损失函数值的变化

### 9.7.4 随机搜索

随机搜索（Bergstra and Bengio，2012），是一个替代网格搜索的方法，并且编程简单，使用更方便，能更快地收敛到超参数的良好取值。

随机搜索过程如下：

首先，我们为每个超参 数定义一个边缘分布，例如，Bernoulli分布或范畴分布（分别对应着二元超参数或离散超参数），或者对数尺度上的均匀分布（对应着正实 值超参数）。例如，其中，$U(a,b)$ 表示区间$(a,b)$ 上均匀采样的样本。类似地，`log_number_of_hidden_units`可以从 $U(\ln(50),\ln(2000))$ 上采样。

与网格搜索不同，我们不需要离散化超参数的值。这允许我们在一个更大的集合上进行搜索，而不产生额外的计算代价。实际上，当有几个超参数对性能度量没有显著影响时，随机搜索相比于网格搜索指数级地高效。

Bergstra and Bengio（2012）进行了详细的研究并发现相比于网格搜索，随机搜索能够更快地减小验证集误差（就每个模型运行的试验数而 言）。

与网格搜索一样，我们通常会重复运行不同 版本的随机搜索，以基于前一次运行的结果改进下一次搜索。

随机搜索能比网格搜索更快地找到良好超参数的原因是，没有浪费的实验，不像网格搜索有时会对一个超参数的两个不同值（给定其他超参 数值不变）给出相同结果。在网格搜索中，其他超参数将在这两次实验中拥有相同的值，而在随机搜索中，它们通常会具有不同的值。因此，如果这两个值的变化所对应的验证集误差没有明显区别的话，网格搜索没有必要重复两个等价的实验，而随机搜索仍然会对其他超参数进行两次独立的探索。



