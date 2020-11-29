# 线性回归
单层的神经网络，其实就是一个神经元，可以完成一些线性的工作，比如拟合一条直线，这用一个神经元就可以实现。当这个神经元只接收一个输入时，就是单变量线性回归，可以在二维平面上用可视化方法理解。当接收多个变量输入时，叫做多变量线性回归，此时可视化方法理解就比较困难了，通常我们会用变量两两组对的方式来表现。

当变量多于一个时，两个变量的量纲和数值有可能差别很大，这种情况下，我们通常需要对样本特征数据做归一化，然后把数据喂给神经网络进行训练，否则会出现“消化不良”的情况。

## 一、单入单出的单层神经网络
### 单变量线性回归

利用已有值，预测未知值。如果只有一个自变量和一个因变量，可以用简单直接的方法来解决问题。但是，当有多个自变量时，这种直接的办法可能就会失效了。假设有三个自变量，很有可能不能够在样本中找到和这三个自变量的组合非常接近的数据，此时我们就应该借助更系统的方法。
#### 一元线性回归
回归分析是一种数学模型。当因变量和自变量为线性关系时，它是一种特殊的线性模型。

最简单的情形是一元线性回归，由大体上有线性关系的一个自变量和一个因变量组成，模型是：

$$Y=a+bX+\varepsilon \tag{1}$$

$X$ 是自变量，$Y$ 是因变量，$\varepsilon$ 是随机误差，$a$ 和 $b$ 是参数，在线性回归模型中，$a,b$ 是我们要通过算法学习出来的。

对于线性回归模型，有如下一些概念需要了解：

- 通常假定随机误差 $\varepsilon$ 的均值为 $0$，方差为$σ^2$（$σ^2>0$，$σ^2$ 与 $X$ 的值无关）
- 若进一步假定随机误差遵从正态分布，就叫做正态线性模型
- 一般地，若有 $k$ 个自变量和 $1$ 个因变量（即公式1中的 $Y$），则因变量的值分为两部分：一部分由自变量影响，即表示为它的函数，函数形式已知且含有未知参数；另一部分由其他的未考虑因素和随机性影响，即随机误差
- 当函数为参数未知的线性函数时，称为线性回归分析模型
- 当函数为参数未知的非线性函数时，称为非线性回归分析模型
- 当自变量个数大于 $1$ 时称为多元回归
- 当因变量个数大于 $1$ 时称为多重回归

解决办法： 1.最小二乘法 2.梯度下降法 3.简单的神经网络法 4.更通用的神经网络算法

## 二、最小二乘法
>代码实现
#### 计算 $w$ 值

```Python
# 根据公式15
def method1(X,Y,m):
    x_mean = X.mean()
    p = sum(Y*(X-x_mean))
    q = sum(X*X) - sum(X)*sum(X)/m
    w = p/q
    return w

# 根据公式16
def method2(X,Y,m):
    x_mean = X.mean()
    y_mean = Y.mean()
    p = sum(X*(Y-y_mean))
    q = sum(X*X) - x_mean*sum(X)
    w = p/q
    return w

# 根据公式13
def method3(X,Y,m):
    p = m*sum(X*Y) - sum(X)*sum(Y)
    q = m*sum(X*X) - sum(X)*sum(X)
    w = p/q
    return w
```

由于有函数库的帮助，我们不需要手动计算`sum()`, `mean()`这样的基本函数。

#### 计算 $b$ 值

```Python
# 根据公式14
def calculate_b_1(X,Y,w,m):
    b = sum(Y-w*X)/m
    return b

# 根据公式9
def calculate_b_2(X,Y,w):
    b = Y.mean() - w * X.mean()
    return b
```
>运算结果

```
w1=2.056827, b1=2.965434
w2=2.056827, b2=2.965434
w3=2.056827, b3=2.965434
```
## 三、梯度下降法

#### 损失函数（Loss Function）

均方误差：

$$loss_i(w,b) = \frac{1}{2} (z_i-y_i)^2 \tag{2}$$


与最小二乘法比较可以看到，梯度下降法和最小二乘法的模型及损失函数是相同的，都是一个线性模型加均方差损失函数，模型用于拟合，损失函数用于评估效果。

区别在于，最小二乘法从损失函数求导，直接求得数学解析解，而梯度下降以及后面的神经网络，都是利用导数传递误差，再通过迭代方式一步一步（用近似解）逼近真实解。
>代码实现
```Python
if __name__ == '__main__':

    reader = SimpleDataReader()
    reader.ReadData()
    X,Y = reader.GetWholeTrainSamples()

    eta = 0.1
    w, b = 0.0, 0.0
    for i in range(reader.num_train):
        # get x and y value for one sample
        xi = X[i]
        yi = Y[i]
        # 公式1
        zi = xi * w + b
        # 公式3
        dz = zi - yi
        # 公式4
        dw = dz * xi
        # 公式5
        db = dz
        # update w,b
        w = w - eta * dw
        b = b - eta * db

    print("w=", w)    
    print("b=", b)
```
>运算结果
```
w= [1.71629006]
b= [3.19684087]
```
## 四、神经网络法

在梯度下降法中，我们简单讲述了一下神经网络做线性拟合的原理，即：

1. 初始化权重值
2. 根据权重值放出一个解
3. 根据均方差函数求误差
4. 误差反向传播给线性计算部分以调整权重值
5. 是否满足终止条件？不满足的话跳回2

### 定义神经网络结构

建立最简单的单层单点神经元：
<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/4/Setup.png" ch="500" />
然后我们用这个最简单的线性回归的例子，来说明神经网络中最重要的反向传播和梯度下降的概念、过程以及代码实现。
### 代码实现

其实神经网络法和梯度下降法在本质上是一样的，只不过神经网络法使用一个崭新的编程模型，即以神经元为中心的代码结构设计，这样便于以后的功能扩充。

在`Python`中可以使用面向对象的技术，通过创建一个类来描述神经网络的属性和行为，下面我们将会创建一个叫做`NeuralNet`的`class`，然后通过逐步向此类中添加方法，来实现神经网络的训练和推理过程。
#### 定义类

```Python
class NeuralNet(object):
    def __init__(self, eta):
        self.eta = eta
        self.w = 0
        self.b = 0
```
`NeuralNet`类从`object`类派生，并具有初始化函数，其参数是`eta`，也就是学习率，需要调用者指定。另外两个成员变量是`w`和`b`，初始化为`0`。

#### 前向计算

```Python
    def __forward(self, x):
        z = x * self.w + self.b
        return z
```
这是一个私有方法，所以前面有两个下划线，只在`NeuralNet`类中被调用，不对外公开。

#### 反向传播

下面的代码是通过梯度下降法中的公式推导而得的，也设计成私有方法：

```Python
    def __backward(self, x,y,z):
        dz = z - y
        db = dz
        dw = x * dz
        return dw, db
```
`dz`是中间变量，避免重复计算。`dz`又可以写成`delta_Z`，是当前层神经网络的反向误差输入。

#### 梯度更新

```Python
    def __update(self, dw, db):
        self.w = self.w - self.eta * dw
        self.b = self.b - self.eta * db
```

每次更新好新的`w`和`b`的值以后，直接存储在成员变量中，方便下次迭代时直接使用，不需要在全局范围当作参数内传来传去的。

#### 训练过程

只训练一轮的算法是：

***

`for` 循环，直到所有样本数据使用完毕：

1. 读取一个样本数据
2. 前向计算
3. 反向传播
4. 更新梯度

***

```Python
    def train(self, dataReader):
        for i in range(dataReader.num_train):
            # get x and y value for one sample
            x,y = dataReader.GetSingleTrainSample(i)
            # get z from x,y
            z = self.__forward(x)
            # calculate gradient of w and b
            dw, db = self.__backward(x, y, z)
            # update w,b
            self.__update(dw, db)
        # end for
```

#### 推理预测

```Python
    def inference(self, x):
        return self.__forward(x)
```

推理过程，实际上就是一个前向计算过程，我们把它单独拿出来，方便对外接口的设计，所以这个方法被设计成了公开的方法。

#### 主程序

```Python
if __name__ == '__main__':
    # read data
    sdr = SimpleDataReader()
    sdr.ReadData()
    # create net
    eta = 0.1
    net = NeuralNet(eta)
    net.train(sdr)
    # result
    print("w=%f,b=%f" %(net.w, net.b))
    # predication
    result = net.inference(0.346)
    print("result=", result)
    ShowResult(net, sdr)
```

### 运行结果可视化

打印输出结果：

```
w=1.716290,b=3.196841
result= [3.79067723]
```
上述代码是对这个问题的分析过程也是我们学习计算与Python相关类的问题的步骤。

## 五、多样本单特征值计算
在前面的代码中，我们一直使用单样本计算来实现神经网络的训练过程，但是单样本计算有一些缺点：

1. 前后两个相邻的样本很有可能会对反向传播产生相反的作用而互相抵消。假设样本1造成了误差为 $0.5$，$w$ 的梯度计算结果是 $0.1$；紧接着样本2造成的误差为 $-0.5$，$w$ 的梯度计算结果是 $-0.1$，那么前后两次更新 $w$ 就会产生互相抵消的作用。
2. 在样本数据量大时，逐个计算会花费很长的时间。由于我们在本例中样本量不大（200个样本），所以计算速度很快，觉察不到这一点。在实际的工程实践中，动辄10万甚至100万的数据量，轮询一次要花费很长的时间。

如果使用多样本计算，就要涉及到矩阵运算了，而所有的深度学习框架，都对矩阵运算做了优化，会大幅提升运算速度。

## 六、梯度下降的三种形式
我们先从最简单的梯度下降的三种形式说起。
在下面的说明中，我们使用如下假设，以便简化问题易于理解：

1. 使用可以解决本章的问题的线性回归模型，即 $z=x \cdot w+b$；
2. 样本特征值数量为1，即 $x,w,b$ 都是标量；
3. 使用均方差损失函数。

计算 $w$ 的梯度：

$$
\frac{\partial{loss}}{\partial{w}} = \frac{\partial{loss}}{\partial{z_i}}\frac{\partial{z_i}}{\partial{w}}=(z_i-y_i)x_i
$$

计算 $b$ 的梯度：

$$
\frac{\partial{loss}}{\partial{b}} = \frac{\partial{loss}}{\partial{z_i}}\frac{\partial{z_i}}{\partial{b}}=z_i-y_i
$$

### 单样本随机梯度下降

GD(Stochastic Gradient Descent)

样本访问示意图如图所示。
  
<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/4/SingleSample-example.png" />
图 单样本访问方式

#### 特点
 
  - 训练样本：每次使用一个样本数据进行一次训练，更新一次梯度，重复以上过程。
  - 优点：训练开始时损失值下降很快，随机性大，找到最优解的可能性大。
  - 缺点：受单个样本的影响最大，损失函数值波动大，到后期徘徊不前，在最优解附近震荡。不能并行计算。
  
单样本方式的训练情况

|损失函数值|梯度下降过程|
|---|---|
|<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/4/SingleSample-Loss.png"/>|<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/4/SingleSample-Trace.png"/>|
左图，由于我们使用了限定的停止条件，即当损失函数值小于等于 $0.02$ 时停止训练，所以，单样本方式迭代了300次后达到了精度要求。

右图是 $w$ 和 $b$ 共同构成的损失函数等高线图。梯度下降时，开始收敛较快，稍微有些弯曲地向中央地带靠近。到后期波动较大，找不到准确的前进方向，曲折地达到中心附近。
### 小批量样本梯度下降
本访问示意图如图所示。

<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/4/MiniBatch-example.png" />

图 小批量样本访问方式

#### 特点
 - 训练样本：选择一小部分样本进行训练，更新一次梯度，然后再选取另外一小部分样本进行训练，再更新一次梯度。
  - 优点：不受单样本噪声影响，训练速度较快。
  - 缺点：batch size的数值选择很关键，会影响训练结果。
  
小批量样本方式的训练情况

|损失函数值|梯度下降过程|
|---|---|
|<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/4/MiniBatch-Loss.png"/>|<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/4/MiniBatch-Trace.png"/>|
右图，梯度下降时，在接近中心时有小波动。图太小看不清楚，可以用matplot工具放大局部来观察。和单样本方式比较，在中心区的波动已经缓解了很多。

在实际工程中，我们通常使用小批量梯度下降形式。

### 全批量样本梯度下降
Full Batch Gradient Descent

样本访问示意图如图所示。

<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/4/FullBatch-example.png" />

图 全批量样本访问方式

#### 特点
  - 训练样本：每次使用全部数据集进行一次训练，更新一次梯度，重复以上过程。
  - 优点：受单个样本的影响最小，一次计算全体样本速度快，损失函数值没有波动，到达最优点平稳。方便并行计算。
  - 缺点：数据量较大时不能实现（内存限制），训练过程变慢。初始值不同，可能导致获得局部最优解，并非全局最优解。
  
全批量样本方式的训练情况

|损失函数值|梯度下降过程|
|---|---|
|<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/4/FullBatch-Loss.png"/>|<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/4/FullBatch-Trace.png"/>|
右图，梯度下降时，在整个过程中只拐了一个弯儿，就直接到达了中心点。

### 三种方式的比较
三种方式的比较

||单样本|小批量|全批量|
|---|---|---|---|
|梯度下降过程图解|<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/4/SingleSample-Trace.png"/>|<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/4/MiniBatch-Trace.png"/>|<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/4/FullBatch-Trace.png"/>|
|批大小|1|10|100|
|学习率|0.1|0.3|0.5|
|迭代次数|304|110|60|
|epoch|3|10|60|
|结果|w=2.003, b=2.990|w=2.006, b=2.997|w=1.993, b=2.998|
表中比较了三种方式的结果，从结果看，都接近于 $w=2,b=3$ 的原始解。最后的可视化结果图如图4-10，可以看到直线已经处于样本点比较中间的位置。

## 多入单出的单层神经网络-多变量线性回归

#### 提出问题
对于一般的应用问题，建立多元线性回归模型时，为了保证回归模型具有优良的解释能力和预测效果，应首先注意自变量的选择，其准则是：

1. 自变量对因变量必须有显著的影响，并呈密切的线性相关；
2. 自变量与因变量之间的线性相关必须是真实的，而不是形式上的；
3. 自变量之间应具有一定的互斥性，即自变量之间的相关程度不应高于自变量与因变量之因的相关程度；
4. 自变量应具有完整的统计数据，其预测值容易确定。

#### 解决方案
两种方法的比较

|方法|正规方程|梯度下降|
|---|-----|-----|
|原理|几次矩阵运算|多次迭代|
|特殊要求|$X^{\top}X$ 的逆矩阵存在|需要确定学习率|
|复杂度|$O(n^3)$|$O(n^2)$|
|适用样本数|$m \lt 10000$|$m \ge 10000$|

### 一、正规方程法（Normal Equations）
对于多元线性回归，可以用正规方程来解决，也就是得到一个数学上的解析解。它可以解决下面这个公式描述的问题：

$$y=a_0+a_1x_1+a_2x_2+\dots+a_kx_k \tag{1}$$

#### 代码实现
我们把表5-1的样本数据带入方程内。根据公式(5)，我们应该建立如下的 $X,Y$ 矩阵：

$$
X = \begin{pmatrix} 
1 & 10.06 & 60 \\\\
1 & 15.47 & 74 \\\\
1 & 18.66 & 46 \\\\
1 & 5.20 & 77 \\\\
\vdots & \vdots & \vdots \\\\
\end{pmatrix} \tag{17}
$$

$$
Y= \begin{pmatrix}
302.86 \\\\
393.04 \\\\
270.67 \\\\
450.59 \\\\
\vdots \\\\
\end{pmatrix}  \tag{18}
$$

根据公式(10)：

$$W = (X^{\top} X)^{-1}{X^{\top} Y} \tag{19}$$

1. $X$ 是 $1000\times 3$ 的矩阵，$X$ 的转置是 $3\times 1000$，$X^{\top}X$ 生成 $3\times 3$ 的矩阵；
2. $(X^{\top}X)^{-1}$ 也是 $3\times 3$ 的矩阵；
3. 再乘以 $X^{\top}$，即 $3\times 3$ 的矩阵乘以 $3\times 1000$ 的矩阵，变成 $3\times 1000$ 的矩阵；
4. 再乘以 $Y$，$Y$是 $1000\times 1$ 的矩阵，所以最后变成 $3\times 1$ 的矩阵，成为 $W$ 的解，其中包括一个偏移值 $b$ 和两个权重值 $w$，3个值在一个向量里。

```Python
if __name__ == '__main__':
    reader = SimpleDataReader()
    reader.ReadData()
    X,Y = reader.GetWholeTrainSamples()
    num_example = X.shape[0]
    one = np.ones((num_example,1))
    x = np.column_stack((one, (X[0:num_example,:])))
    a = np.dot(x.T, x)
    # need to convert to matrix, because np.linalg.inv only works on matrix instead of array
    b = np.asmatrix(a)
    c = np.linalg.inv(b)
    d = np.dot(c, x.T)
    e = np.dot(d, Y)
    #print(e)
    b=e[0,0]
    w1=e[1,0]
    w2=e[2,0]
    print("w1=", w1)
    print("w2=", w2)
    print("b=", b)
    # inference
    z = w1 * 15 + w2 * 93 + b
    print("z=",z)
```
- 运行结果

```
w1= -2.0184092853092226
w2= 5.055333475112755
b= 46.235258613837644
z= 486.1051325196855
```
 我们得到了两个权重值和一个偏移值,至此,我们得到了解析解。我们可以用这个做为标准答案，去验证我们的神经网络的训练结果。

### 二、神经网络法
单特征值的线性回归问题类似，多变量（多特征值）的线性回归可以被看做是一种高维空间的线性拟合。以具有两个特征的情况为例，这种线性拟合不再是用直线去拟合点，而是用平面去拟合点。

#### 定义神经网络结构
我们定义一个如图所示的一层的神经网络，输入层为2或者更多，反正大于2了就没区别。这个一层的神经网络的特点是：

1. 没有中间层，只有输入项和输出层（输入项不算做一层）；
2. 输出层只有一个神经元；
3. 神经元有一个线性输出，不经过激活函数处理，即在下图中，经过 $\Sigma$ 求和得到 $Z$ 值之后，直接把 $Z$ 值输出。

<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/5/setup.png" ch="500" />
图 多入单出的单层神经元结构

与上一章的神经元相比，这次仅仅是多了一个输入，但却是质的变化，即，一个神经元可以同时接收多个输入，这是神经网络能够处理复杂逻辑的根本。

###  反向传播

#### 单样本多特征计算

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
#### 多样本多特征计算
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

### 三、样本特征数据标准化
数据标准化又可以叫做数据归一化。

#### 为什么要做标准化

理论层面上，神经网络是以样本在事件中的统计分布概率为基础进行训练和预测的，所以它对样本数据的要求比较苛刻。具体说明如下：

1. 样本的各个特征的取值要符合概率分布，即 $[0,1]$。
2. 样本的度量单位要相同。我们并没有办法去比较1米和1公斤的区别，但是，如果我们知道了1米在整个样本中的大小比例，以及1公斤在整个样本中的大小比例，比如一个处于0.2的比例位置，另一个处于0.3的比例位置，就可以说这个样本的1米比1公斤要小。
3. 神经网络假设所有的输入输出数据都是标准差为1，均值为0，包括权重值的初始化，激活函数的选择，以及优化算法的设计。

4. 数值问题

    标准化可以避免一些不必要的数值问题。因为激活函数sigmoid/tanh的非线性区间大约在 $[-1.7，1.7]$。意味着要使神经元有效，线性计算输出的值的数量级应该在1（1.7所在的数量级）左右。这时如果输入较大，就意味着权值必须较小，一个较大，一个较小，两者相乘，就引起数值问题了。
    
5. 梯度更新
    
    若果输出层的数量级很大，会引起损失函数的数量级很大，这样做反向传播时的梯度也就很大，这时会给梯度的更新带来数值问题。
    
6. 学习率
   
    如果梯度非常大，学习率就必须非常小，因此，学习率（学习率初始值）的选择需要参考输入的范围，不如直接将数据标准化，这样学习率就不必再根据数据范围作调整。对 $w_1$ 适合的学习率，可能相对于 $w_2$ 来说会太小，若果使用适合 $w_1$ 的学习率，会导致在 $w_2$ 方向上步进非常慢，从而消耗非常多的时间；而使用适合 $w_2$ 的学习率，对 $w_1$ 来说又太大，搜索不到适合 $w_1$ 的解。

#### 标准化的常用方法

- Min-Max标准化（离差标准化），将数据映射到 $[0,1]$ 区间

$$x_{new}=\frac{x-x_{min}}{x_{max} - x_{min}} \tag{1}$$

- 平均值标准化，将数据映射到[-1,1]区间
   
$$x_{new} = \frac{x - \bar{x}}{x_{max} - x_{min}} \tag{2}$$

- 对数转换
$$x_{new}=\ln(x_i) \tag{3}$$

- 反正切转换
$$x_{new}=\frac{2}{\pi}\arctan(x_i) \tag{4}$$

- Z-Score法

把每个特征值中的所有数据，变成平均值为0，标准差为1的数据，最后为正态分布。Z-Score规范化（标准差标准化 / 零均值标准化，其中std是标准差）：

$$x_{new} = \frac{x_i - \bar{x}}{std} \tag{5}$$

- 中心化，平均值为0，无标准差要求
  
$$x_{new} = x_i - \bar{x} \tag{6}$$

- 比例法，要求数据全是正值

$$
x_{new} = \frac{x_k}{\sum_{i=1}^m{x_i}} \tag{7}
$$
注意：

1. 我们并没有标准化样本的标签Y数据。
2. 我们是对两列特征值分别做标准化处理的。

### 四、还原参数值
这里直接进行代码实现会更容易理解：
#### 代码实现

下面的代码实现了上述公式：

```Python
# get real weights and bias
def DeNormalizeWeightsBias(net, dataReader):
    W_real = np.zeros_like(net.W)
    X_Norm = dataReader.GetDataRange()
    for i in range(W_real.shape[0]):
        W_real[i,0] = net.W[i,0] / X_Norm[i,1]
        B_real = net.B - W_real[0,0]*X_Norm[0,0] - W_real[1,0]*X_Norm[1,0]
    return W_real, B_real
```

`X_Norm`是我们在做标准化时保留下来的样本的两个特征向量的最小值和数值范围（最大值减去最小值）。

修改主程序如下：

```Python
if __name__ == '__main__':
    ......
    net.train(reader, checkpoint=0.1)
    # inference
    W_real, B_real = DeNormalizeWeightsBias(net, reader)
    print("W_real=", W_real)
    print("B_real=", B_real)

    x1 = 15
    x2 = 93
    x = np.array([x1,x2]).reshape(1,2)
    z = np.dot(x, W_real) + B_real
    print("Z=", z)

    ShowResult(net, reader)
```
在`net.train()`方法返回之后，训练好的`W`和`B`的值就保存在`NeuralNet`类的属性里了。然后通过调用`DeNormalizeWeightsBias()`函数，把它们转换成真实的`W_real`和`B_real`值，就好比我们不做标准化而能训练出来的权重值一样。

最后在推理预测时，我们直接使用了`np.dot()`公式，而没有使用`net.inference()`方法，是因为在`net`实例中的`W`和`B`是还原前的值，做前向计算时还是会有问题，所以我们直接把前向计算公式拿出来，代入`W_real`和`B_real`，就可以得到真实的预测值了。

#### 运行结果

运行上述代码，观察最后部分的打印输出：

```
......
499 99 380.5934686827507 
[[-40.23261123] [399.36389489]] [[244.39118797]]
W= [[-40.23261123]
 [399.36389489]]
B= [[244.39118797]]
W_real= [[-2.01737219]
 [ 5.05523918]]
B_real= [[46.26647363]]
Z= [[486.14313417]]
```

把结果与正规方程的结果对比一下，如下表。

表 还原后的值与正规方程结果的比较

||w1|w2|b|
|---|---|---|---|
|正规方程|-2.018|5.055|46.235|
|神经网络|-2.017|5.055|46.266|

### 正确的推理方法
要确保数据标准化的准确性来进行推理

### 标准化标签值
#### 代码实现

参照X的标准化方法，对Y的标准化公式如下：

$$y_{new} = \frac{y-y_{min}}{y_{max}-y_{min}} \tag{2}$$

在`SimpleDataReader`类中增加新方法如下：

```Python
class SimpleDataReader(object):
    def NormalizeY(self):
        self.Y_norm = np.zeros((1,2))
        max_value = np.max(self.YRaw)
        min_value = np.min(self.YRaw)
        # min value
        self.Y_norm[0, 0] = min_value 
        # range value
        self.Y_norm[0, 1] = max_value - min_value 
        y_new = (self.YRaw - min_value) / self.Y_norm[0, 1]
        self.YTrain = y_new
```

原始数据中，$Y$ 的数值范围是：

  - 最大值：674.37
  - 最小值：181.38
  - 平均值：420.64

标准化后，$Y$ 的数值范围是：
  - 最大值：1.0
  - 最小值：0.0
  - 平均值：0.485

注意，我们同样记住了`Y_norm`的值便于以后使用。

修改主程序代码，增加对 $Y$ 标准化的方法调用`NormalizeY()`：

```Python
# main
if __name__ == '__main__':
    # data
    reader = SimpleDataReader()
    reader.ReadData()
    reader.NormalizeX()
    reader.NormalizeY()
    ......
```

总结一下从本章中学到的正确的方法：

1. $X$ 必须标准化，否则无法训练；
2. $Y$ 值不在 $[0,1]$ 之间时，要做标准化，好处是迭代次数少；
3. 如果 $Y$ 做了标准化，对得出来的预测结果做关于 $Y$ 的反标准化

### 总结

归纳总结一下前面遇到的困难及解决办法：

1. 样本不做标准化的话，网络发散，训练无法进行；
2. 训练样本标准化后，网络训练可以得到结果，但是预测结果有问题；
3. 还原参数值后，预测结果正确，但是此还原方法并不能普遍适用；
4. 标准化测试样本，而不需要还原参数值，可以保证普遍适用；
5. 标准化标签值，可以使得网络训练收敛快，但是在预测时需要把结果反标准化，以便得到真实值。
