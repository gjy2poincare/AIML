# 线性分类
神经网络的一个重要功能就是分类，现实世界中的分类任务复杂多样，但万变不离其宗，我们都可以用同一种模式的神经网络来处理。
## 多入单出的单层神经网络- 线性二分类
### 逻辑回归模型

回归问题可以分为两类：线性回归和逻辑回归。
逻辑回归（Logistic Regression），回归给出的结果是事件成功或失败的概率。当因变量的类型属于二值（1/0，真/假，是/否）变量时，我们就应该使用逻辑回归。

线性回归使用一条直线拟合样本数据，而逻辑回归的目标是“拟合”0或1两个数值，而不是具体连续数值，所以称为广义线性模型。逻辑回归又称Logistic回归分析，常用于数据挖掘，疾病自动诊断，经济预测等领域。
表6-1示意说明了线性二分类和非线性二分类的区别。

表6-1 直观理解线性二分类与非线性二分类的区别

|线性二分类|非线性二分类|
|---|---|
|<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/6/linear_binary.png"/>|<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/10/non_linear_binary.png"/>|

## 二分类函数
此函数对线性和非线性二分类都适用。

###  二分类函数

对率函数Logistic Function，即可以做为激活函数使用，又可以当作二分类函数使用。我们会根据不同的任务区分激活函数和分类函数这两个概念，在二分类任务中，叫做Logistic函数，而在作为激活函数时，叫做Sigmoid函数。

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

### 正向传播
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

用线性回归模型的预测结果来逼近样本分类的对数几率。这就是为什么它叫做逻辑回归(logistic regression)，但其实是分类学习的方法。这种方法的优点如下：

- 把线性回归的成功经验引入到分类问题中，相当于对“分界线”的预测进行建模，而“分界线”在二维空间上是一条直线，这就不需要考虑具体样本的分布（比如在二维空间上的坐标位置），避免了假设分布不准确所带来的问题；
- 不仅预测出类别（0/1），而且得到了近似的概率值（比如0.31或0.86），这对许多需要利用概率辅助决策的任务很有用；
- 对率函数是任意阶可导的凸函数，有很好的数学性，许多数值优化算法都可以直接用于求取最优解。

## 线性二分类实现
### 代码实现
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

###  运行结果

图6-3所示的损失函数值记录很平稳地下降，说明网络收敛了。

<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/6/binary_loss.png" ch="500" />

图6-3 训练过程中损失函数值的变化

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
## 线性二分类的工作原理
表6-4 线性回归和线性分类的比较

||线性回归|线性分类|
|---|---|---|
|相同点|需要在样本群中找到一条直线|需要在样本群中找到一条直线|
|不同点|用直线来拟合所有样本，使得各个样本到这条直线的距离尽可能最短|用直线来分割所有样本，使得正例样本和负例样本尽可能分布在直线两侧|

### 二分类的代数原理
代数方式：通过一个分类函数计算所有样本点在经过线性变换后的概率值，使得正例样本的概率大于0.5，而负例样本的概率小于0.5。

### 二分类的几何原理
几何方式：让所有正例样本处于直线的一侧，所有负例样本处于直线的另一侧，直线尽可能处于两类样本的中间。

#### 二分类函数的几何作用
二分类函数的最终结果是把正例都映射到图6-5中的上半部分的曲线上，而把负类都映射到下半部分的曲线上。

<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/6/sigmoid_binary.png"/>

图6-5 $Logistic$ 函数把输入的点映射到 $(0,1)$ 区间内实现分类
## 线性二分类结果可视化
神经网络的可视化，说简单也很简单，说难也很难，关键是对框架系统的理解，对运行机制和工作原理的理解，掌握了这些，可视化就会使一件轻而易举且令人愉快的事情。
#### 代码实现

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

#### 运行结果

图6-7为二分类结果。

<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/6/binary_result.png" ch="500" />

图6-7 稍有欠缺的二分类结果
后面就是我们修改代码使二分类的实现更加完美。

## 实现逻辑与或非门
单层神经网络，又叫做感知机，它可以轻松实现逻辑与、或、非门。由于逻辑与、或门，需要有两个变量输入，而逻辑非门只有一个变量输入。但是它们共同的特点是输入为0或1，可以看作是正负两个类别。

所以，在学习了二分类知识后，我们可以用分类的思想来实现下列5个逻辑门：

- 与门 AND
- 与非门 NAND
- 或门 OR
- 或非门 NOR
- 非门 NOT
 
以逻辑AND为例，图6-8中的4个点分别代表4个样本数据，蓝色圆点表示负例（$y=0$），红色三角表示正例（$y=1$）。

<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/6/LogicAndGateData.png" ch="500" />

图6-8 可以解决逻辑与问题的多条分割线
### 实现逻辑非门
<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/6/LogicNot2.png" width="500" />

图6-9 正确的逻辑非门的神经元实现

表6-10 逻辑非问题的样本数据

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
执行训练过程，最终得到图6-11所示的分类结果和下面的打印输出结果。
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

图6-11 逻辑非门的分类结果

### 实现逻辑与或门
#### 神经元模型
<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/6/BinaryClassifierNN.png" ch="500" />

图6-12 逻辑与或门的神经元实现
### 用双曲正切函数分类
双曲正切函数（Tanh Function），公式如下：

$$Tanh(z) = \frac{e^{z} - e^{-z}}{e^{z} + e^{-z}} = \frac{2}{1 + e^{-2z}} - 1 \tag{3}$$
比较一下二者的函数图像，如表6-13。

表6-13 对率函数和双曲正切函数的比较

|对率函数|双曲正切函数|
|---|---|
|<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/6/logistic_seperator.png">|<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/6/tanh_seperator.png">|
|正负类分界线：$a=0.5$|正负类分界线：$a=0$|

对于对率函数来说，一般使用 $0.5$ 作为正负类的分界线，那我们会自然地想到，对于双曲正切函数来说，可以使用 $0$ 作为正负类的分界线。

所谓分界线，其实只是人们理解神经网络做二分类的一种方式，对于神经网络来说，其实并没有分界线这个概念，它要做的是通过线性变换，尽量把正例向上方推，把负例向下方推。

## 多入多出单层神经网络- 线性多分类
#### 线性多分类和非线性多分类的区别

图6.14显示了线性多分类和非线性多分类的区别。

<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/7/linear_vs_nonlinear.png" />

图6-14 直观理解线性多分类与分线性多分类的区别
左侧为线性多分类，右侧为非线性多分类。它们的区别在于不同类别的样本点之间是否可以用一条直线来互相分割。对神经网络来说，线性多分类可以使用单层结构来解决，而分线性多分类需要使用双层结构。

#### 二分类与多分类的关系
多分类问题一共有三种解法：

1. 一对一方式
   
每次先只保留两个类别的数据，训练一个分类器。如果一共有 $N$ 个类别，则需要训练 $C^2_N$ 个分类器。以 $N=3$ 时举例，需要训练 $A|B，B|C，A|C$ 三个分类器。

<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/7/one_vs_one.png" />

图6-15 一对一方式

如图6-15最左侧所示，这个二分类器只关心蓝色和绿色样本的分类，而不管红色样本的情况，也就是说在训练时，只把蓝色和绿色样本输入网络。
   
推理时，$(A|B)$ 分类器告诉你是A类时，需要到 $(A|C)$ 分类器再试一下，如果也是A类，则就是A类。如果 $(A|C)$ 告诉你是C类，则基本是C类了，不可能是B类，不信的话可以到 $(B|C)$ 分类器再去测试一下。

2. 一对多方式
   
如图6-16，处理一个类别时，暂时把其它所有类别看作是一类，这样对于三分类问题，可以得到三个分类器。

<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/7/one_vs_multiple.png" />

图6-16 一对多方式

如最左图，这种情况是在训练时，把红色样本当作一类，把蓝色和绿色样本混在一起当作另外一类。

推理时，同时调用三个分类器，再把三种结果组合起来，就是真实的结果。比如，第一个分类器告诉你是“红类”，那么它确实就是红类；如果告诉你是非红类，则需要看第二个分类器的结果，绿类或者非绿类；依此类推。

3. 多对多方式

假设有4个类别ABCD，我们可以把AB算作一类，CD算作一类，训练一个分类器1；再把AC算作一类，BD算作一类，训练一个分类器2。
    
推理时，第1个分类器告诉你是AB类，第二个分类器告诉你是BD类，则做“与”操作，就是B类。
## 多分类函数
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
### 正向传播

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
如图6-17示意。

<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/7/Loss-A-Z.jpg" ch="500" />

图6-17 Softmax在神经网络结构中的示意图

## 线性多分类实现

这个网络只有输入层和输出层，由于输入层不算在内，所以是一层网络，如图6-18所示。

<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/7/MultipleClassifierNN.png" ch="500" />

图6-18 多入多出单层神经网络
## 线性多分类的工作原理
### 多分类过程
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
### 多分类的几何原理
样本共分为三类，每一类之间略有不同。
### 综合效果

把三张图综合在一起，应该是下图的样子。

<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/7/z123.png" ch="500" />

图 三条线分开三种样本
## 线性多分类结果可视化
神经网络到底是一对一方式，还是一对多方式呢？从Softmax公式，好像是一对多方式，因为只取一个最大值，那么理想中的一对多方式应该是图6-19所示的样子。

<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/7/OneVsOthers.png" ch="500" />

图6-19 理想中的一对多方式的分割线

## 总结
在本部分中，我们从最简单的线性二分类开始学习，包括其原理，实现，训练过程，推理过程等等，并且以可视化的方式来帮助大家更好地理解这些过程。并且，我们将利用学到的二分类知识，实现逻辑与门、与非门，或门，或非门。
