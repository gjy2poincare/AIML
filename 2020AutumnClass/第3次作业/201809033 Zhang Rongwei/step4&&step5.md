# 第8章 激活函数
### 8.0.1 激活函数的基本作用

图8-1是神经网络中的一个神经元，假设该神经元有三个输入，分别为$x_1,x_2,x_3$，那么：

$$z=x_1 w_1 + x_2 w_2 + x_3 w_3 +b \tag{1}$$
$$a = \sigma(z) \tag{2}$$

<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/1/NeuranCell.png" width="500" />

图8-1 激活函数在神经元中的位置

激活函数也就是 $a=\sigma(z)$ 这一步

1. 给神经网络增加非线性因素。
2. 把公式1的计算结果压缩到 $[0,1]$ 之间，便于后面的计算。

激活函数的基本性质：

+ 非线性：线性的激活函数和没有激活函数一样；
+ 可导性：做误差反向传播和梯度下降，必须要保证激活函数的可导性；
+ 单调性：单一的输入会得到单一的输出，较大值的输入得到较大值的输出。

在物理试验中使用的继电器，是最初的激活函数的原型：当输入电流大于一个阈值时，会产生足够的磁场，从而打开下一级电源通道，如图8-2所示。

<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/8/step.png" width="500" />

图8-2 继电器的阶跃形态

用到神经网络中的概念，用‘1’来代表一个神经元被激活，‘0’代表一个神经元未被激活。

### 8.0.2 何时会用到激活函数

激活函数用在神经网络的层与层之间的连接，神经网络的最后一层不用激活函数。

表8-1 单层的神经网络的参数与功能

|网络|输入|输出|激活函数|分类函数|功能|
|---|---|---|---|---|---|
|单层|单变量|单输出|无|无|线性回归|
|单层|多变量|单输出|无|无|线性回归|
|单层|多变量|单输出|无|二分类函数|二分类|
|单层|多变量|多输出|无|多分类函数|多分类|

从上表可以看到，一直没有使用激活函数，而只使用了分类函数。对于多层神经网络也是如此，在最后一层只会用到分类函数来完成二分类或多分类任务，如果是拟合任务，则不需要分类函数。

简言之：

1. 神经网络最后一层不需要激活函数
2. 激活函数只用于连接前后两层神经网络

# 8.1 挤压型激活函数

当输入值域的绝对值较大的时候，其输出在两端是饱和的，都具有S形的函数曲线以及压缩输入值域的作用，所以叫挤压型激活函数，又可以叫饱和型激活函数。

在英文中，通常用Sigmoid来表示，在神经网络中，有两个常用的Sigmoid函数，一个是Logistic函数，另一个是Tanh函数。

### 8.1.1 Logistic函数

对数几率函数（Logistic Function，简称对率函数）。

#### 公式

$$Sigmoid(z) = \frac{1}{1 + e^{-z}} \rightarrow a \tag{1}$$

#### 导数

$$Sigmoid'(z) = a(1 - a) \tag{2}$$


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

#### 缺点

指数计算代价大。

反向传播时梯度消失：从梯度图像中可以看到，Sigmoid的梯度在两端都会接近于0，根据链式法则，如果传回的误差是$\delta$，那么梯度传递函数是$\delta \cdot a'$，而$a'$这时接近零，也就是说整体的梯度也接近零。这就出现梯度消失的问题，并且这个问题可能导致网络收敛速度比较慢。


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

# 8.2 半线性激活函数

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

例如：要做房价预测，每平方是万元，我们预测结果也是万元，那么MSE差值的平方单位应该是千万级别的。假设我们的模型预测结果与真实值相差1000元，则用MSE的计算结果是1000,000，这个值没有单位，如何描述这个差距？于是就求个平方根就好了，这样误差可以是标签值是同一个数量级的，在描述模型的时候就说，我们模型的误差是多少元。

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

## 9.1 用多项式回归法拟合正弦曲线

#### 一元一次线性模型


$$z = x w + b \tag{1}$$

#### 多元一次多项式

多变量的线性回归，其模型为：

$$z = x_1 w_1 + x_2 w_2 + ...+ x_m w_m + b \tag{2}$$

这里的多变量，是指样本数据的特征值为多个，上式中的 $x_1,x_2,...,x_m$ 代表了m个特征值。

#### 一元多次多项式
用公式描述：

$$z = x w_1 + x^2 w_2 + ... + x^m w_m + b \tag{3}$$

上式中x是原有的唯一特征值，$x^m$ 是利用 $x$ 的 $m$ 次方作为额外的特征值，这样就把特征值的数量从 $1$ 个变为 $m$ 个。

换一种表达形式，令：$x_1 = x,x_2=x^2,\ldots,x_m=x^m$，则：

$$z = x_1 w_1 + x_2 w_2 + ... + x_m w_m + b \tag{4}$$

#### 多元多次多项式

多变量的非线性回归，其参数与特征组合繁复，但最终都可以归结为公式2和公式4的形式。

<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/9/polynomial_10_pic.png" />

图9-3 对有噪音的正弦曲线的拟合

一堆散点，看上去像是一条带有很大噪音的正弦曲线，从左上到右下，分别是1次多项式、2次多项式......10次多项式，其中：

- 第4、5、6、7图是比较理想的拟合
- 第1、2、3图欠拟合，多项式的次数不够高
- 第8、9、10图，多项式次数过高，过拟合了

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

        ### 主程序

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

表9-5中左侧图显示损失函数值下降得很平稳，说明网络训练效果还不错。拟合的结果也很令人满意，虽然红色线没有严丝合缝地落在蓝色样本点内，但是这完全是因为训练的次数不够多，有兴趣的读者可以修改超参后做进一步的试验。

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
<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/9/Sample.png" ch="500" />

图9-4 样本数据可视化

再把图9-4所示的这条“眼镜蛇形”曲线拿出来观察一下，不但有正弦式的波浪，还有线性的爬升，转折处也不是很平滑，所以难度很大。从正弦曲线的拟合经验来看，三次多项式以下肯定无法解决，所以可以从四次多项式开始试验。

### 9.2.1 用四次多项式拟合

表9-8 四次多项式1万次迭代的训练结果

|损失函数历史|曲线拟合结果|
|---|---|
|<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/9/complex_loss_4_10k.png">|<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/9/complex_result_4_10k.png">|

可以看到损失函数值还有下降的空间，拟合情况很糟糕。


表9-9 四次多项式10万次迭代的训练结果

|损失函数历史|曲线拟合结果|
|---|---|
|<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/9/complex_loss_4_100k.png">|<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/9/complex_result_4_100k.png">|

从表9-9中的左图看，损失函数值到了一定程度后就不再下降了，说明网络能力有限。

### 9.2.2 用六次多项式拟合

接下来跳过5次多项式，直接用6次多项式来拟合。这次不需要把`max_epoch`设置得很大，可以先试试50000个`epoch`。

表9-10 六次多项式5万次迭代的训练结果

|损失函数历史|曲线拟合结果|
|---|---|
|<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/9/complex_loss_6_50k.png">|<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/9/complex_result_6_50k.png">|
### 9.2.3 用八次多项式拟合

再跳过7次多项式，直接使用8次多项式。先把`max_epoch`设置为50000试验一下。

表9-11 八项式5万次迭代的训练结果

|损失函数历史|曲线拟合结果|
|---|---|
|<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/9/complex_loss_8_50k.png">|<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/9/complex_result_8_50k.png">|

表9-12 八项式100万次迭代的训练结果

|损失函数历史|曲线拟合结果|
|---|---|
|<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/9/complex_loss_8_1M.png">|<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/9/complex_result_8_1M.png">|

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

### 9.3.2 交叉验证

#### 传统的机器学习

当样本很少时，这个方法很有用。
#### 神经网络/深度学习


举个例子：一个BP神经网络，我们无法确定隐层的神经元数目，因为没有理论支持。此时可以按图9-6的示意图这样做。

<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/9/CrossValidation.png" ch="500" />

图9-6 交叉训练的数据配置方式

### 9.3.3 留出法 Hold out

亦即从训练数据中保留出验证样本集，主要用于解决过拟合情况，这部分数据不用于训练。如果训练数据的准确度持续增长，但是验证数据的准确度保持不变或者反而下降，说明神经网络亦即过拟合了，此时需要停止训练，用测试集做最终测试。

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

## 9.4 双层神经网络实现非线性回归

### 9.4.1 万能近似定理

万能近似定理(universal approximation theorem) $^{[1]}$，是深度学习最根本的理论依据。它证明了在给定网络具有足够多的隐藏单元的条件下，配备一个线性输出层和一个带有任何“挤压”性质的激活函数（如Sigmoid激活函数）的隐藏层的前馈神经网络，能够以任何想要的误差量近似任何从一个有限维度的空间映射到另一个有限维度空间的Borel可测的函数。### 9.4.2 定义神经网络结构



根据万能近似定理的要求，我们定义一个两层的神经网络，输入层不算，一个隐藏层，含3个神经元，一个输出层。图9-7显示了此次用到的神经网络结构。

<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/9/nn.png" />

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
### 9.4.4 反向传播

我们比较一下本章的神经网络和第5章的神经网络的区别，看表9-13。

表9-13 本章中的神经网络与第5章的神经网络的对比

|第5章的神经网络|本章的神经网络|
|---|---|
|<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images\Images\5\setup.png"/>|<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/9/nn.png"/>|

## 9.5 曲线拟合

### 9.5.1 正弦曲线的拟合

#### 隐层只有一个神经元的情况

令`n_hidden=1`，并指定模型名称为`sin_111`，训练过程见图9-10。图9-11为拟合效果图。

<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/9/sin_loss_1n.png" />

图9-10 训练过程中损失函数值和准确率的变化

<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/9/sin_result_1n.png" ch="500" />

图9-11 一个神经元的拟合效果

从图9-10可以看到，损失值到0.04附近就很难下降了。图9-11中，可以看到只有中间线性部分拟合了，两端的曲线部分没有拟合。

### 隐层有两个神经元的情况

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
#### 运行结果

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

#### 隐层只有两个神经元的情况

<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/9/complex_result_2n.png" ch="500" />

图9-14 两个神经元的拟合效果

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

### 9.5.3 广义的回归/拟合

只要是数值拟合问题，确定不能用线性回归的话，都可以用非线性回归来尝试解决。

## 9.6 非线性回归的工作原理

多项式回归方法的示意图，如图9-17。

<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/9/polynomial_concept.png"/>

图9-17 多项式回归方法的特征值输入

单层神经网络的多项式回归法，需要$x,x^2,x^3$三个特征值，组成如下公式来得到拟合结果：

$$
z = x \cdot w_1 + x^2 \cdot w_2 + x^3 \cdot w_3 + b \tag{1}
$$

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

## 9.6.2 神经网络的非线性拟合工作原理

以正弦曲线的例子来讲解神经网络非线性回归的工作过程和原理。

表9-14

|单层多项式回归|双层神经网络|
|---|---|
|<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/9/polynomial_concept.png">|<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/9/neuralnet_concept.png">|
### 9.6.3 比较多项式回归和双层神经网络解法
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

### 9.7.2 手动调整参数

有效容量受限于3个因素：

- 模型的表示容量；
- 学习算法与代价函数的匹配程度；
- 代价函数和训练过程正则化模型的程度。

表9-22 各种超参数的作用

|超参数|目标|作用|副作用|
|---|---|---|---|
|学习率|调至最优|低的学习率会导致收敛慢，高的学习率会导致错失最佳解|容易忽略其它参数的调整|
|隐层神经元数量|增加|增加数量会增加模型的表示能力|参数增多、训练时间增长|
|批大小|有限范围内尽量大|大批量的数据可以保持训练平稳，缩短训练时间|可能会收敛速度慢|
### 9.7.3 网格搜索

表9-23 各种组合下的准确率

||eta=0.1|eta=0.3|eta=0.5|eta=0.7|
|---|---|---|---|---|
|ne=2|0.63|0.68|0.71|0.73|
|ne=4|0.86|0.89|0.91|0.3|
|ne=8|0.92|0.94|0.97|0.95|
|ne=12|0.69|0.84|0.88|0.87|

#### 学习率的调整
表9-24 四种学习率值的比较

|学习率|迭代次数|说明|
|----|----|----|
|0.1|10000|学习率小，收敛最慢，没有在规定的次数内达到精度要求|
|0.3|10000|学习率增大，收敛慢，没有在规定的次数内达到精度要求|
|0.5|8200|学习率增大，在8200次左右达到精度要求|
|0.7|3500|学习率进一步增大，在3500次达到精度|

<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/9/eta.png" ch="500" />

图9-23 四种学习率值造成的损失函数值的变化
较大的学习率可以带来很快的收敛速度，但是有两点：

- 但并不是对所有问题都这样，有的问题可能需要0.001或者更小的学习率
- 学习率大时，开始时收敛快，但是到了后来有可能会错失最佳解
#### 批大小的调整
固定其它参数，即隐层神经元`ne=4`、`eta=0.5`不变，调整批大小，来试验网络训练情况，设置`eps=0.001`为精度要求。

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

固定其它参数，即`batch_size=10`、`eta=0.5`不变，调整隐层神经元的数量，来试验网络训练情况，设置`eps=0.001`为精度要求。

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

# 第10章 多入单出的双层神经网络 - 非线性二分类

## 10.0 非线性二分类问题

### 10.0.1 提出问题一：异或问题
<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/10/xor_source_data.png" ch="500" />

图10-1 异或问题的样本数据
从图10-1看，两类样本点（红色叉子和蓝色圆点）交叉分布在[0,1]空间的四个角上，用一条直线无法分割开两类样本。

### 10.0.2 提出问题二：双弧形问题
<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/10/sin_data_source.png" ch="500" />

图10-2 呈弧线分布的两类样本数据

平面上有两类样本数据，都成弧形分布，由于弧度的存在，使得我们无法使用一根直线来分开红蓝两种样本点。

#### 准确率 Accuracy

也可以称之为精度

对于二分类问题，假设测试集上一共1000个样本，其中550个正例，450个负例。测试一个模型时，得到的结果是：521个正例样本被判断为正类，435个负例样本被判断为负类，则正确率计算如下：

$$Accuracy=(521+435)/1000=0.956$$
即正确率为95.6%。这种方式对多分类也是有效的，即三类中判别正确的样本数除以总样本数，即为准确率。

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

<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/10/ROC.png"/>

图10-4 ROC曲线图

图中红色的曲线就是ROC曲线，曲线下的面积就是AUC值，取值区间为$[0.5,1.0]$，面积越大越好。

- ROC曲线越靠近左上角，该分类器的性能越好。
- 对角线表示一个随机猜测分类器。
- 若一个学习器的ROC曲线被另一个学习器的曲线完全包住，则可判断后者性能优于前者。
- 若两个学习器的ROC曲线没有包含关系，则可以判断ROC曲线下的面积，即AUC，谁大谁好。
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
## 10.1 为什么必须用双层神经网络

### 10.1.1 分类

我们先回忆一下各种分类的含义：

- 从复杂程度上分，有线性/非线性之分；
- 从样本类别上分，有二分类/多分类之分。
表10-2 各种分类的组合关系

||二分类|多分类|
|---|---|---|
|线性|<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/6/linear_binary.png"/>|<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/6/linear_multiple.png"/>|
|非线性|<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/10/non_linear_binary.png"/>|<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/10/non_linear_multiple.png"/>|

在第三步中我们学习过线性分类，如果用于此处的话，我们可能会得到表10-3所示的绿色分割线。

表10-3 线性分类结果

|XOR问题|弧形问题|
|---|---|
|<img src='../Images/10/xor_data_line.png'/>|<img src='../Images/10/sin_data_line.png'/>|
|图中两根直线中的任何一根，都不可能把蓝色点分到一侧，同时红色点在另一侧|对于线性技术来说，它已经尽力了，使得两类样本尽可能地分布在直线的两侧|

## 10.2 非线性二分类实现

### 10.2.1 定义神经网络结构

首先定义可以完成非线性二分类的神经网络结构图，如图10-6所示。

<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/10/xor_nn.png" />

图10-6 非线性二分类神经网络结构图

- 输入层两个特征值$x_1,x_2$
  $$
  X=\begin{pmatrix}
    x_1 & x_2
  \end{pmatrix}
  $$
- 隐层$2\times 2$的权重矩阵$W1$
$$
  W1=\begin{pmatrix}
    w1_{11} & w1_{12} \\\\
    w1_{21} & w1_{22} 
  \end{pmatrix}
$$
- 隐层$1\times 2$的偏移矩阵$B1$

$$
  B1=\begin{pmatrix}
    b1_{1} & b1_{2}
  \end{pmatrix}
$$

- 隐层由两个神经元构成
$$
Z1=\begin{pmatrix}
  z1_{1} & z1_{2}
\end{pmatrix}
$$
$$
A1=\begin{pmatrix}
  a1_{1} & a1_{2}
\end{pmatrix}
$$
- 输出层$2\times 1$的权重矩阵$W2$
$$
  W2=\begin{pmatrix}
    w2_{11} \\\\
    w2_{21}  
  \end{pmatrix}
$$

- 输出层$1\times 1$的偏移矩阵$B2$

$$
  B2=\begin{pmatrix}
    b2_{1}
  \end{pmatrix}
$$

- 输出层有一个神经元使用Logistic函数进行分类
$$
  Z2=\begin{pmatrix}
    z2_{1}
  \end{pmatrix}
$$
$$
  A2=\begin{pmatrix}
    a2_{1}
  \end{pmatrix}
$$

对于一般的用于二分类的双层神经网络可以是图10-7的样子。

<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/10/binary_classifier.png" width="600" ch="500" />

图10-7 通用的二分类神经网络结构图

输入特征值可以有很多，隐层单元也可以有很多，输出单元只有一个，且后面要接Logistic分类函数和二分类交叉熵损失函数。
### 10.2.2 前向计算

根据网络结构，我们有了前向计算过程图10-8。

<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/10/binary_forward.png" />

图10-8 前向计算过程

#### 第一层

- 线性计算

$$
z1_{1} = x_{1} w1_{11} + x_{2} w1_{21} + b1_{1}
$$
$$
z1_{2} = x_{1} w1_{12} + x_{2} w1_{22} + b1_{2}
$$
$$
Z1 = X \cdot W1 + B1
$$

- 激活函数

$$
a1_{1} = Sigmoid(z1_{1})
$$
$$
a1_{2} = Sigmoid(z1_{2})
$$
$$
A1=\begin{pmatrix}
  a1_{1} & a1_{2}
\end{pmatrix}=Sigmoid(Z1)
$$

#### 第二层

- 线性计算

$$
z2_1 = a1_{1} w2_{11} + a1_{2} w2_{21} + b2_{1}
$$
$$
Z2 = A1 \cdot W2 + B2
$$

- 分类函数

$$a2_1 = Logistic(z2_1)$$
$$A2 = Logistic(Z2)$$

#### 损失函数

我们把异或问题归类成二分类问题，所以使用二分类交叉熵损失函数：

$$
loss = -Y \ln A2 + (1-Y) \ln (1-A2) \tag{12}
$$
## 10.3 实现逻辑异或门

### 10.3.1 代码实现
#### 准备数据
```Python
class XOR_DataReader(DataReader):
    def ReadData(self):
        self.XTrainRaw = np.array([0,0,0,1,1,0,1,1]).reshape(4,2)
        self.YTrainRaw = np.array([0,1,1,0]).reshape(4,1)
        self.XTrain = self.XTrainRaw
        self.YTrain = self.YTrainRaw
        self.num_category = 1
        self.num_train = self.XTrainRaw.shape[0]
        self.num_feature = self.XTrainRaw.shape[1]
        self.XTestRaw = self.XTrainRaw
        self.YTestRaw = self.YTrainRaw
        self.XTest = self.XTestRaw
        self.YTest = self.YTestRaw
        self.num_test = self.num_train

    def GenerateValidationSet(self, k = 10):
        self.XVld = self.XTrain
        self.YVld = self.YTrain
```

#### 测试函数
```Python
def Test(dataReader, net):
    print("testing...")
    X,Y = dataReader.GetTestSet()
    A = net.inference(X)
    diff = np.abs(A-Y)
    result = np.where(diff < 1e-2, True, False)
    if result.sum() == dataReader.num_test:
        return True
    else:
        return False
```

#### 主过程代码

```Python
if __name__ == '__main__':
    ......
    n_input = dataReader.num_feature
    n_hidden = 2
    n_output = 1
    eta, batch_size, max_epoch = 0.1, 1, 10000
    eps = 0.005
    hp = HyperParameters2(n_input, n_hidden, n_output, eta, max_epoch, batch_size, eps, NetType.BinaryClassifier, InitialMethod.Xavier)
    net = NeuralNet2(hp, "Xor_221")
    net.train(dataReader, 100, True)
    ......
```
### 10.3.2 运行结果

经过快速的迭代后，会显示训练过程如图10-10所示。

<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/10/xor_loss.png" />

图10-10 训练过程中的损失函数值和准确率值的变化

可以看到二者的走势很理想。
表10-7 异或计算值与神经网络推理值的比较

|x1|x2|XOR|Inference|diff|
|---|---|---|---|---|
|0|0|0|0.0041|0.0041|
|0|1|1|0.9945|0.0055|
|1|0|1|0.9945|0.0055|
|1|1|0|0.0047|0.0047|

## 10.4 逻辑异或门的工作原理

表10-8 XOR问题的推理过程

|||
|---|---|
|<img src='../Images/10/xor_source_data.png'/>|<img src='../Images/10/xor_z1.png'/>|
|原始样本|Z1是第一层网络线性计算结果|
|<img src='../Images/10/xor_a1.png'/>|<img src='../Images/10/xor_z2_a2.png'/>|
|A1是Z1的激活函数计算结果|Z2是第二层线性计算结果，A2是二分类结果|
### 10.4.2 更直观的可视化结果

#### 3D图
表10-10 平面拟合的可视化结果

|正向|侧向|
|---|---|
|<img src='../Images/5/level3_result_1.png'/>|<img src='../Images/5/level3_result_2.png'/>|
表10-11 异或分类结果可视化

|斜侧视角|顶视角|
|---|---|
|<img src='../Images/10/xor_result_3D_1.png'/>|<img src='../Images/10/xor_result_3D_2.png'/>|
#### 2.5D图
```Python
def ShowResultContour(net, dr):
    DrawSamplePoints(dr.XTrain[:,0], dr.XTrain[:,1], dr.YTrain, "classification result", "x1", "x2", show=False)
    X,Y,Z = Prepare3DData(net, 50)
    plt.contourf(X, Y, Z, cmap=plt.cm.Spectral)
    plt.show()
```

在二维平面上，可以通过plt.contourf()函数画出着色的等高线图，Z作为等高线高度，可以得到图10-13。

<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/10/xor_result_25d.png" ch="500" />

图10-13 分类结果的等高线图
## 10.5 实现双弧形二分类
### 10.5.1 代码实现

#### 主过程代码

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

### 10.5.2 运行结果

经过快速的迭代，训练完毕后，会显示损失函数曲线和准确率曲线如图10-15。

<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/10/sin_loss.png" />

图10-15 训练过程中的损失函数值和准确率值的变化

## 10.6 双弧形二分类的工作原理
#### 几个辅助的函数
- `DrawSamplePoints(x1, x2, y, title, xlabel, ylabel, show=True)`
  
  画样本点，把正例绘制成红色的`x`，把负例绘制成蓝色的点。输入的`x1`和`x2`组成横纵坐标，`y`是正负例的标签值。

- `Prepare3DData(net, count)

  准备3D数据，把平面划分成`count` * `count`的网格，并形成矩阵。如果传入的`net`不是None的话，会使用`net.inference()`做一次推理，以便得到和平面上的网格相对应的输出值。

- `DrawGrid(Z, count)`

  绘制网格。这个网格不一定是正方形的，有可能会由于矩阵的平移缩放而扭曲，目的是观察神经网络对空间的变换。

- `ShowSourceData(dataReader)`

  显示原始训练样本数据。

- `ShowTransformation(net, dr, epoch)`

  绘制经过神经网络第一层的线性计算即激活函数计算后，空间变换的结果。神经网络的第二层就是在第一层的空间变换的结果之上来完成分类任务的。

- `ShowResult2D(net, dr, epoch)`

  在二维平面上显示分类结果，实际是用等高线方式显示2.5D的分类结果。

#### 训练函数

```Python
def train(dataReader, max_epoch):
    n_input = dataReader.num_feature
    n_hidden = 2
    n_output = 1
    eta, batch_size = 0.1, 5
    eps = 0.01

    hp = HyperParameters2(n_input, n_hidden, n_output, eta, max_epoch, batch_size, eps, NetType.BinaryClassifier, InitialMethod.Xavier)
    net = NeuralNet2(hp, "Arc_221_epoch")
    
    net.train(dataReader, 5, True)
    
    ShowTransformation(net, dataReader, max_epoch)
    ShowResult2D(net, dataReader, max_epoch)
```
- `n_input=2`，输入的特征值数量
- `n_hidden=2`，隐层的神经元数
- `n_output=1`，输出为二分类
- `eta=0.1`，学习率
- `batch_size=5`，批量样本数为5
- `eps=0.01`，停止条件
- `NetType.BinaryClassifier`，二分类网络
- `InitialMethod.Xavier`，初始化方法为Xavier

#### 主过程

```Python
if __name__ == '__main__':
    dataReader = DataReader(train_data_name, test_data_name)
    dataReader.ReadData()
    dataReader.NormalizeX()
    dataReader.Shuffle()
    dataReader.GenerateValidationSet()

    ShowSourceData(dataReader)
    plt.show()

    train(dataReader, 20)
    train(dataReader, 50)
    train(dataReader, 100)
    train(dataReader, 150)
    train(dataReader, 200)
    train(dataReader, 600)
```
读取数据后，以此用20、50、100、150、200、600个`epoch`来做为训练停止条件，以便观察中间状态，笔者经过试验事先知道了600次迭代一定可以达到满意的效果。而上述`epoch`的取值，是通过观察损失函数的下降曲线来确定的。

### 10.6.2 运行结果

运行后，首先会显示一张原始样本的位置如图10-16，以便确定训练样本是否正确，并得到基本的样本分布概念。

<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/10/sin_data_source.png" ch="500" />

图10-16 双弧形的样本数据

随着每一个`train()`函数的调用，会在每一次训练结束后依次显示以下图片：

- 第一层神经网络的线性变换结果
- 第一层神经网络的激活函数结果
- 第二层神经网络的分类结果

表10-15 训练过程可视化

|迭代|线性变换|激活结果|分类结果|
|---|---|---|---|
|20次|<img src='../Images/10/sin_z1_20.png'/>|<img src='../Images/10/sin_a1_20.png'/>|<img src='../Images/10/sin_a2_20.png'/>|
|100次|<img src='../Images/10/sin_z1_100.png'/>|<img src='../Images/10/sin_a1_100.png'/>|<img src='../Images/10/sin_a2_100.png'/>|
|200次|<img src='../Images/10/sin_z1_200.png'/>|<img src='../Images/10/sin_a1_200.png'/>|<img src='../Images/10/sin_a2_200.png'/>|
|600次|<img src='../Images/10/sin_z1_600.png'/>|<img src='../Images/10/sin_a1_600.png'/>|<img src='../Images/10/sin_a2_600.png'/>|

分析表10-15中各列图片的变化

# 第11章 多入多出的双层神经网络 - 非线性多分类
<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/11/data.png" ch="500" />

图11-1 可视化样本数据

一共有3个类别：

1. 蓝色方点
2. 红色叉点
3. 绿色圆点

样本组成了一个貌似铜钱的形状，我们就把这个问题叫做“铜钱孔形分类”问题
### 11.0.2 多分类模型的评估标准

我们以三分类问题举例，假设每类有100个样本，一共300个样本，最后的分类结果如表11-2所示。

表11-2 多分类结果的混淆矩阵

|样本所属类别|分到类1|分到类2|分到类3|各类样本总数|精(准)确率|
|---|---|---|---|---|---|
|类1|90|4|6|100|90%|
|类2|9|84|5|100|84%|
|类3|1|4|95|100|95%|
|总数|101|93|106|300|89.67%|

- 第1类样本，被错分到2类4个，错分到3类6个，正确90个；
- 第2类样本，被错分到1类9个，错分到3类5个，正确84个；
- 第3类样本，被错分到1类1个，错分到2类4个，正确95个。
 
总体的准确率是89.67%。三类的精确率是90%、84%、95%。

## 11.1 非线性多分类

### 11.1.1 定义神经网络结构

先设计出能完成非线性多分类的网络结构，如图11-2所示。

<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/11/nn.png" />

图11-2 非线性多分类的神经网络结构图

- 输入层两个特征值$x_1, x_2$
$$
x=
\begin{pmatrix}
    x_1 & x_2
\end{pmatrix}
$$
- 隐层$2\times 3$的权重矩阵$W1$
$$
W1=
\begin{pmatrix}
    w1_{11} & w1_{12} & w1_{13} \\\\
    w1_{21} & w1_{22} & w1_{23}
\end{pmatrix}
$$

- 隐层$1\times 3$的偏移矩阵$B1$

$$
B1=\begin{pmatrix}
    b1_1 & b1_2 & b1_3 
\end{pmatrix}
$$

- 隐层由3个神经元构成
- 输出层$3\times 3$的权重矩阵$W2$
$$
W2=\begin{pmatrix}
    w2_{11} & w2_{12} & w2_{13} \\\\
    w2_{21} & w2_{22} & w2_{23} \\\\
    w2_{31} & w2_{32} & w2_{33} 
\end{pmatrix}
$$

- 输出层$1\times 1$的偏移矩阵$B2$

$$
B2=\begin{pmatrix}
    b2_1 & b2_2 & b2_3 
  \end{pmatrix}
$$

- 输出层有3个神经元使用Softmax函数进行分类

### 11.1.4 代码实现

绝大部分代码都在`HelperClass2`目录中的基本类实现，这里只有主过程：

```Python
if __name__ == '__main__':
    ......
    n_input = dataReader.num_feature
    n_hidden = 3
    n_output = dataReader.num_category
    eta, batch_size, max_epoch = 0.1, 10, 5000
    eps = 0.1
    hp = HyperParameters2(n_input, n_hidden, n_output, eta, max_epoch, batch_size, eps, NetType.MultipleClassifier, InitialMethod.Xavier)
    # create net and train
    net = NeuralNet2(hp, "Bank_233")
    net.train(dataReader, 100, True)
    net.ShowTrainingTrace()
    # show result
    ......
```

过程描述：

1. 读取数据文件
2. 显示原始数据样本分布图
3. 其它数据操作：归一化、打乱顺序、建立验证集
4. 设置超参
5. 建立神经网络开始训练
6. 显示训练结果

### 11.1.5 运行结果

训练过程如图11-5所示。

<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/11/loss.png" />

图11-5 训练过程中的损失函数值和准确率值的变化

迭代了5000次，没有到达损失函数小于0.1的条件。

分类结果如图11-6所示。

<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/11/result.png" ch="500" />

图11-6 分类效果图

因为没达到精度要求，所以分类效果一般。从分类结果图上看，外圈圆形差不多拟合住了，但是内圈的方形还差很多。

# 11.2 非线性多分类的工作原理

### 11.2.1 隐层神经元数量的影响

下面列出了隐层神经元为2、4、8、16、32、64的情况，设定好最大epoch数为10000，以比较它们各自能达到的最大精度值的区别。每个配置只测试一轮，所以测出来的数据有一定的随机性。

表11-3展示了隐层神经元数与分类结果的关系。

表11-3 神经元数与网络能力及分类结果的关系

|神经元数|损失函数|分类结果|
|---|---|---|
|2|<img src='../Images/11/loss_n2.png'/>|<img src='../Images/11/result_n2.png'/>|
||测试集准确度0.618，耗时49秒，损失函数值0.795。类似这种曲线的情况，损失函数值降不下去，准确度值升不上去，主要原因是网络能力不够。|没有完成分类任务|
|4|<img src='../Images/11/loss_n4.png'/>|<img src='../Images/11/result_n4.png'/>|
||测试准确度0.954，耗时51秒，损失函数值0.132。虽然可以基本完成分类任务，网络能力仍然不够。|基本完成，但是边缘不够清晰|
|8|<img src='../Images/11/loss_n8.png'/>|<img src='../Images/11/result_n8.png'/>|
||测试准确度0.97，耗时52秒，损失函数值0.105。可以先试试在后期衰减学习率，如果再训练5000轮没有改善的话，可以考虑增加网络能力。|基本完成，但是边缘不够清晰|
|16|<img src='../Images/11/loss_n16.png'/>|<img src='../Images/11/result_n16.png'/>|
||测试准确度0.978，耗时53秒，损失函数值0.094。同上，可以同时试着使用优化算法，看看是否能收敛更快。|较好地完成了分类任务|
|32|<img src='../Images/11/loss_n32.png'/>|<img src='../Images/11/result_n32.png'/>|
||测试准确度0.974，耗时53秒，损失函数值0.085。网络能力够了，从损失值下降趋势和准确度值上升趋势来看，可能需要更多的迭代次数。|较好地完成了分类任务|
|64|<img src='../Images/11/loss_n64.png'/>|<img src='../Images/11/result_n64.png'/>|
||测试准确度0.972，耗时64秒，损失函数值0.075。网络能力足够。|较好地完成了分类任务|

### 11.2.2 三维空间内的变换过程
``Python
def Show3D(net, dr):
    ......
```

上述代码首先使用测试集（500个样本点）在已经训练好的网络上做一次推理，得到了隐层的计算结果，然后分别用`net.Z1`和`net.A1`的三列数据做为XYZ坐标值，绘制三维点图，列在表11-4中做比较。

表11-4 工作原理可视化

||正视角|侧视角|
|---|---|---|
|z1|<img src='../Images/11/bank_z1_1.png'/>|<img src='../Images/11/bank_z1_2.png'/>|
||通过线性变换得到在三维空间中的线性平面|从侧面看的线性平面|
|a1|<img src='../Images/11/bank_a1_1.png'/>|<img src='../Images/11/bank_a1_2.png'/>|
||通过激活函数的非线性变化，被空间挤压成三角形|从侧面看三种颜色分成了三层|
#### 3D分类结果图
表11-5 分类效果图

|||
|---|---|
|<img src='../Images/11/multiple_3d_c1_1.png'/>|<img src='../Images/11/multiple_3d_c2_1.png'/>|
|红色：类别1样本区域|红色：类别2样本区域|
|<img src='../Images/11/multiple_3d_c3_1.png'/>|<img src='../Images/11/multiple_3d_c1_c2_1.png'/>|
|红色：类别3样本区域|红色：类别1，青色：类别2，紫色：类别3|
## 11.3 分类样本不平衡问题

### 11.3.1 什么是样本不平衡

英文名叫做Imbalanced Data。

在一般的分类学习方法中都有一个假设，就是不同类别的训练样本的数量相对平衡。

### 11.3.2 如何解决样本不平衡问题

#### 平衡数据集
一些经验法则：

- 考虑对大类下的样本（超过1万、十万甚至更多）进行欠采样，即删除部分样本；
- 考虑对小类下的样本（不足1万甚至更少）进行过采样，即添加部分样本的副本；
- 考虑尝试随机采样与非随机采样两种采样方法；
- 考虑对各类别尝试不同的采样比例，比一定是1:1，有时候1:1反而不好，因为与现实情况相差甚远；
- 考虑同时使用过采样（over-sampling）与欠采样（under-sampling）。
# 第12章 多入多出的三层神经网络 - 深度非线性多分类

## 12.0 多变量非线性多分类

#### 图片数据归一化
要处理的对象是图片，需要把整张图片看作一个样本，因此使用下面这段代码做数据归一化方法：

```Python
    def __NormalizeData(self, XRawData):
        X_NEW = np.zeros(XRawData.shape)
        x_max = np.max(XRawData)
        x_min = np.min(XRawData)
        X_NEW = (XRawData - x_min)/(x_max-x_min)
        return X_NEW
```
## 12.1 三层神经网络的实现

### 12.1.1 定义神经网络

为了完成MNIST分类，我们需要设计一个三层神经网络结构，如图12-2所示。

<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/12/nn3.png" ch="500" />

图12-2 三层神经网络结构

#### 输入层

共计$28\times 28=784$个特征值：

$$
X=\begin{pmatrix}
    x_1 & x_2 & \cdots & x_{784}
  \end{pmatrix}
$$

#### 隐层1

- 权重矩阵$W1$形状为$784\times 64$

$$
W1=\begin{pmatrix}
    w1_{1,1} & w1_{1,2} & \cdots & w1_{1,64} \\\\
    \vdots & \vdots & \cdots & \vdots \\\\
    w1_{784,1} & w1_{784,2} & \cdots & w1_{784,64} 
  \end{pmatrix}
$$

- 偏移矩阵$B1$的形状为$1\times 64$

$$
B1=\begin{pmatrix}
    b1_{1} & b1_{2} & \cdots & b1_{64}
  \end{pmatrix}
$$

- 隐层1由64个神经元构成，其结果为$1\times 64$的矩阵

$$
Z1=\begin{pmatrix}
    z1_{1} & z1_{2} & \cdots & z1_{64}
  \end{pmatrix}
$$
$$
A1=\begin{pmatrix}
    a1_{1} & a1_{2} & \cdots & a1_{64}
  \end{pmatrix}
$$

#### 隐层2

- 权重矩阵$w2$形状为$64\times 16$

$$
W2=\begin{pmatrix}
    w2_{1,1} & w2_{1,2} & \cdots & w2_{1,16} \\\\
    \vdots & \vdots & \cdots & \vdots \\\\
    w2_{64,1} & w2_{64,2} & \cdots & w2_{64,16} 
  \end{pmatrix}
$$

- 偏移矩阵#B2#的形状是$1\times 16$

$$
B2=\begin{pmatrix}
    b2_{1} & b2_{2} & \cdots & b2_{16}
  \end{pmatrix}
$$

- 隐层2由16个神经元构成

$$
Z2=\begin{pmatrix}
    z2_{1} & z2_{2} & \cdots & z2_{16}
  \end{pmatrix}
$$
$$
A2=\begin{pmatrix}
    a2_{1} & a2_{2} & \cdots & a2_{16}
  \end{pmatrix}
$$

#### 输出层

- 权重矩阵$W3$的形状为$16\times 10$

$$
W3=\begin{pmatrix}
    w3_{1,1} & w3_{1,2} & \cdots & w3_{1,10} \\\\
    \vdots & \vdots & \cdots & \vdots \\\\
    w3_{16,1} & w3_{16,2} & \cdots & w3_{16,10} 
  \end{pmatrix}
$$

- 输出层的偏移矩阵$B3$的形状是$1\times 10$

$$
B3=\begin{pmatrix}
    b3_{1}& b3_{2} & \cdots & b3_{10}
  \end{pmatrix}
$$

- 输出层有10个神经元使用Softmax函数进行分类

$$
Z3=\begin{pmatrix}
    z3_{1} & z3_{2} & \cdots & z3_{10}
  \end{pmatrix}
$$
$$
A3=\begin{pmatrix}
    a3_{1} & a3_{2} & \cdots & a3_{10}
  \end{pmatrix}
$$

### 12.1.2 前向计算

我们都是用大写符号的矩阵形式的公式来描述，在每个矩阵符号的右上角是其形状。

#### 隐层1

$$Z1 = X \cdot W1 + B1 \tag{1}$$

$$A1 = Sigmoid(Z1) \tag{2}$$

#### 隐层2

$$Z2 = A1 \cdot W2 + B2 \tag{3}$$

$$A2 = Tanh(Z2) \tag{4}$$

#### 输出层

$$Z3 = A2 \cdot W3  + B3 \tag{5}$$

$$A3 = Softmax(Z3) \tag{6}$$
### 12.1.4 代码实现

在`HelperClass3` / `NeuralNet3.py`中，下面主要列出与两层网络不同的代码。

#### 初始化

```Python
class NeuralNet3(object):
    def __init__(self, hp, model_name):
        ...
        self.wb1 = WeightsBias(self.hp.num_input, self.hp.num_hidden1, self.hp.init_method, self.hp.eta)
        self.wb1.InitializeWeights(self.subfolder, False)
        self.wb2 = WeightsBias(self.hp.num_hidden1, self.hp.num_hidden2, self.hp.init_method, self.hp.eta)
        self.wb2.InitializeWeights(self.subfolder, False)
        self.wb3 = WeightsBias(self.hp.num_hidden2, self.hp.num_output, self.hp.init_method, self.hp.eta)
        self.wb3.InitializeWeights(self.subfolder, False)
```
初始化部分需要构造三组`WeightsBias`对象，请注意各组的输入输出数量，决定了矩阵的形状。

#### 前向计算

```Python
    def forward(self, batch_x):
        # 公式1
        self.Z1 = np.dot(batch_x, self.wb1.W) + self.wb1.B
        # 公式2
        self.A1 = Sigmoid().forward(self.Z1)
        # 公式3
        self.Z2 = np.dot(self.A1, self.wb2.W) + self.wb2.B
        # 公式4
        self.A2 = Tanh().forward(self.Z2)
        # 公式5
        self.Z3 = np.dot(self.A2, self.wb3.W) + self.wb3.B
        # 公式6
        if self.hp.net_type == NetType.BinaryClassifier:
            self.A3 = Logistic().forward(self.Z3)
        elif self.hp.net_type == NetType.MultipleClassifier:
            self.A3 = Softmax().forward(self.Z3)
        else:   # NetType.Fitting
            self.A3 = self.Z3
        #end if
        self.output = self.A3
```
前向计算部分增加了一层，并且使用`Tanh()`作为激活函数。

- 反向传播
```Python
    def backward(self, batch_x, batch_y, batch_a):
        # 批量下降，需要除以样本数量，否则会造成梯度爆炸
        m = batch_x.shape[0]

        # 第三层的梯度输入 公式7
        dZ3 = self.A3 - batch_y
        # 公式8
        self.wb3.dW = np.dot(self.A2.T, dZ3)/m
        # 公式9
        self.wb3.dB = np.sum(dZ3, axis=0, keepdims=True)/m 

        # 第二层的梯度输入 公式10
        dA2 = np.dot(dZ3, self.wb3.W.T)
        # 公式11
        dZ2,_ = Tanh().backward(None, self.A2, dA2)
        # 公式12
        self.wb2.dW = np.dot(self.A1.T, dZ2)/m 
        # 公式13
        self.wb2.dB = np.sum(dZ2, axis=0, keepdims=True)/m 

        # 第一层的梯度输入 公式8
        dA1 = np.dot(dZ2, self.wb2.W.T) 
        # 第一层的dZ 公式10
        dZ1,_ = Sigmoid().backward(None, self.A1, dA1)
        # 第一层的权重和偏移 公式11
        self.wb1.dW = np.dot(batch_x.T, dZ1)/m
        self.wb1.dB = np.sum(dZ1, axis=0, keepdims=True)/m 

    def update(self):
        self.wb1.Update()
        self.wb2.Update()
        self.wb3.Update()
```
反向传播也相应地增加了一层，注意要用对应的`Tanh()`的反向公式。梯度更新时也是三组权重值同时更新。

- 主过程

```Python
if __name__ == '__main__':
    ......
    n_input = dataReader.num_feature
    n_hidden1 = 64
    n_hidden2 = 16
    n_output = dataReader.num_category
    eta = 0.2
    eps = 0.01
    batch_size = 128
    max_epoch = 40

    hp = HyperParameters3(n_input, n_hidden1, n_hidden2, n_output, eta, max_epoch, batch_size, eps, NetType.MultipleClassifier, InitialMethod.Xavier)
    net = NeuralNet3(hp, "MNIST_64_16")
    net.train(dataReader, 0.5, True)
    net.ShowTrainingTrace(xline="iteration")
```
超参配置：第一隐层64个神经元，第二隐层16个神经元，学习率0.2，批大小128，Xavier初始化，最大训练40个epoch。

### 12.1.5 运行结果

损失函数值和准确度值变化曲线如图12-3。

<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/12/loss.png" />

图12-3 训练过程中损失函数和准确度的变化

## 12.2 梯度检查

### 12.2.1 为何要做梯度检查？
为了确认代码中反向传播计算的梯度是否正确，可以采用梯度检验（gradient check）的方法。通过计算数值梯度，得到梯度的近似值，然后和反向传播得到的梯度进行比较，若两者相差很小的话则证明反向传播的代码是正确无误的。

### 12.2.2 数值微分

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

#### 泰勒公式

$$f(x)=\frac{f(x_0)}{0!} + \frac{f'(x_0)}{1!}(x-x_0)+\frac{f''(x_0)}{2!}(x-x_0)^2 + ...+\frac{f^{(n)}(x_0)}{n!}(x-x_0)^n+R_n(x) \tag{3}$$

其中,$f^{(n)}(x)$表示$f(x)$的$n$阶导数，等号后的多项式称为函数$f(x)$在$x_0$处的泰勒展开式，剩余的$R_n(x)$是泰勒公式的余项，是$(x-x_0)^n$的高阶无穷小。 

利用泰勒展开公式，令$x=\theta + h, x_0=\theta$，我们可以得到：

$$f(\theta + h)=f(\theta) + f'(\theta)h + O(h^2) \tag{4}$$

#### 单边逼近误差

如果用单边逼近，把公式4两边除以$h$后变形：

$$f'(\theta) + O(h)=\frac{f(\theta+h)-f(\theta)}{h} \tag{5}$$

公式5已经和公式1的定义非常接近了，只是左侧多出来的第二项，就是逼近的误差，是个$O(h)$级别的误差项。

#### 双边逼近误差

如果用双边逼近，我们用三阶泰勒展开：

令$x=\theta + h, x_0=\theta$，我们可以得到：

$$f(\theta + h)=f(\theta) + f'(\theta)h + f''(\theta)h^2 + O(h^3) \tag{6}$$

再令$x=\theta - h, x_0=\theta$我们可以得到：

$$f(\theta - h)=f(\theta) - f'(\theta)h + f''(\theta)h^2 - O(h^3) \tag{7}$$

公式6减去公式7，有：

$$f(\theta + h) - f(\theta - h)=2f'(\theta)h + 2O(h^3) \tag{8}$$

两边除以$2h$：

$$f'(\theta) + O(h^2)={f(\theta + h) - f(\theta - h) \over 2h} \tag{9}$$

公式9中，左侧多出来的第二项，就是双边逼近的误差，是个$O(h^2)$级别的误差项，比公式5中的误差项小很多数量级。
### 12.2.3 算法实现

在神经网络中，我们假设使用多分类的交叉熵函数，则其形式为：

$$J(w,b) =- \sum_{i=1}^m \sum_{j=1}^n y_{ij} \ln a_{ij}$$

m是样本数，n是分类数。
## 12.3 学习率与批大小

在梯度下降公式中：

$$
w_{t+1} = w_t - \frac{\eta}{m} \sum_i^m \nabla J(w,b) \tag{1}
$$

其中，$\eta$是学习率，m是批大小。所以，学习率与批大小是对梯度下降影响最大的两个因子。
表12-1 鞍点和驻点

|鞍点|驻点|
|---|---|
|<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images\Images\12\saddle_point.png" width="640">|<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images\Images\9\sgd_loss_8.png">|
### 12.3.2 初始学习率的选择

我们前面一直使用固定的学习率，比如0.1或者0.05，而没有采用0.5、0.8这样高的学习率。这是因为在接近极小点时，损失函数的梯度也会变小，使用小的学习率时，不会担心步子太大越过极小点。

保证SGD收敛的充分条件是：

$$\sum_{k=1}^\infty \eta_k = \infty \tag{2}$$

且： 

$$\sum_{k=1}^\infty \eta^2_k < \infty \tag{3}$$ 

图12-5是不同的学习率的选择对训练结果的影响。

<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/12/learning_rate.png" ch="500" />

图12-5 学习率对训练的影响

- 黄色：学习率太大，loss值增高，网络发散
- 红色：学习率可以使网络收敛，但值较大，开始时loss值下降很快，但到达极值点附近时，在最优解附近来回跳跃
- 绿色：正确的学习率设置
- 蓝色：学习率值太小，loss值下降速度慢，训练次数长，收敛慢

### 12.3.3 学习率的后期修正

用12.1的MNIST的例子，固定批大小为128时，我们分别使用学习率为0.2，0.3，0.5，0.8来比较一下学习曲线。

<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/12/acc_bs_128.png" ch="500" />

图12-9 不同学习率对应的迭代次数与准确度值的

####心得
  通过这学期的学习，我对人工智能有了一定的感性认识，个人觉得人工智能是一门极富挑战性的科学，从事这项工作的人必须懂得计算机知识，心理学和哲学。人工智能是包括十分广泛的科学，它由不同的领域组成，如机器学习，计算机视觉等等。总的来说，人工智能研究的一个主要目标是使机器能够胜任一些通常需要人类智能才能完成的复杂工作。逻辑性很强，很多数学公式需要反复推敲才能理解。