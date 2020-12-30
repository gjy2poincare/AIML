>>>>>>>>># Step2预习
## 单入单出的单层神经网络 - 单变量线性回归
#### 一元线性回归模型
回归分析是一种数学模型。当因变量和自变量为线性关系时，它是一种特殊的线性模型。
最简单的情形是一元线性回归，由大体上有线性关系的一个自变量和一个因变量组成，模型是：
$$Y=a+bX+\varepsilon \tag{1}$$
$X$ 是自变量，$Y$ 是因变量，$\varepsilon$ 是随机误差，$a$ 和 $b$ 是参数，在线性回归模型中，$a,b$ 是我们要通过算法学习出来的。
对于线性回归模型，有如下一些概念需要了解：

* 通常假定随机误差 $\varepsilon$ 的均值为 $0$，方差为$σ^2$（$σ^2>0$，$σ^2$ 与 $X$ 的值无关）
* 若进一步假定随机误差遵从正态分布，就叫做正态线性模型
* 一般地，若有 $k$ 个自变量和 $1$ 个因变量（即公式1中的 $Y$），因变量的值分为两部分：一部分由自变量影响，即表示为它的函数，函数形式已知且含有未知参数；另一部分由其他的未考虑因素和随机性影响，即随机误差
* 当函数为参数未知的线性函数时，称为线性回归分析模型
* 当函数为参数未知的非线性函数时，称为非线性回归分析模型
* 当自变量个数大于 $1$ 时称为多元回归
* 当因变量个数大于 $1$ 时称为多重回归
  线性回归和非线性回归的区别(如下图)
  ![](01.png)
  公式形态
这里要解释一下线性公式中 $W$ 和 $X$ 的顺序问题。在很多教科书中，我们可以看到下面的公式：

$$Y = W^{\top}X+B \tag{1}$$

或者：

$$Y = W \cdot X + B \tag{2}$$

而我们在本书中使用：

$$Y = X \cdot W + B \tag{3}$$

这三者的主要区别是样本数据 $X$ 的形状定义，相应地会影响到 $W$ 的形状定义。举例来说，如果 $X$ 有三个特征值，那么 $W$ 必须有三个权重值与特征值对应，则：

公式1的矩阵形式
$X$ 是列向量：

$$ X= \begin{pmatrix} x_{1} \\ x_{2} \\ x_{3} \end{pmatrix} $$

$W$ 也是列向量：

$$ W= \begin{pmatrix} w_{1} \\ w_{2} \\ w_{3} \end{pmatrix} $$ $$ Y=W^{\top}X+B= \begin{pmatrix} w_1 & w_2 & w_3 \end{pmatrix} \begin{pmatrix} x_{1} \\ x_{2} \\ x_{3} \end{pmatrix} +b $$ $$ =w_1 \cdot x_1 + w_2 \cdot x_2 + w_3 \cdot x_3 + b \tag{4} $$

$W$ 和 $X$ 都是列向量，所以需要先把 $W$ 转置后，再与 $X$ 做矩阵乘法。

公式2的矩阵形式
公式2与公式1的区别是 $W$ 的形状，在公式2中，$W$ 是个行向量：

$$ W= \begin{pmatrix} w_{1} & w_{2} & w_{3} \end{pmatrix} $$

而 $X$ 的形状仍然是列向量：

$$ X= \begin{pmatrix} x_{1} \\ x_{2} \\ x_{3} \end{pmatrix} $$

这样相乘之前不需要做矩阵转置了：

$$ Y=W \cdot X+B= \begin{pmatrix} w_1 & w_2 & w_3 \end{pmatrix} \begin{pmatrix} x_{1} \\ x_{2} \\ x_{3} \end{pmatrix} +b $$ $$ =w_1 \cdot x_1 + w_2 \cdot x_2 + w_3 \cdot x_3 + b \tag{5} $$

公式3的矩阵形式
$X$ 是个行向量：

$$ X= \begin{pmatrix} x_{1} & x_{2} & x_{3} \end{pmatrix} $$

$W$ 是列向量：

$$ W= \begin{pmatrix} w_{1} \\ w_{2} \\ w_{3} \end{pmatrix} $$

所以 $X$ 在前，$W$ 在后：

$$ Y=X \cdot W+B= \begin{pmatrix} x_1 & x_2 & x_3 \end{pmatrix} \begin{pmatrix} w_{1} \\ w_{2} \\ w_{3} \end{pmatrix} +b $$ $$ =x_1 \cdot w_1 + x_2 \cdot w_2 + x_3 \cdot w_3 + b \tag{6} $$

比较公式4，5，6，其实最后的运算结果是相同的。

我们再分析一下前两种形式的 $X$ 矩阵，由于 $X$ 是个列向量，意味着特征由行表示，当有2个样本同时参与计算时，$X$ 需要增加一列，变成了如下形式：

$$ X= \begin{pmatrix} x_{11} & x_{21} \\ x_{12} & x_{22} \\ x_{13} & x_{23} \end{pmatrix} $$

$x_{ij}$ 的第一个下标 $i$ 表示样本序号，第二个下标 $j$ 表示样本特征，所以 $x_{21}$ 是第2个样本的第1个特征。看 $x_{21}$ 这个序号很别扭，一般我们都是认为行在前、列在后，但是 $x_{21}$ 却是处于第1行第2列，和习惯正好相反。

如果采用第三种形式，则两个样本的 $X$ 矩阵是：

$$ X= \begin{pmatrix} x_{11} & x_{12} & x_{13} \\ x_{21} & x_{22} & x_{23} \end{pmatrix} $$

第1行是第1个样本的3个特征，第2行是第2个样本的3个特征，这与常用的阅读习惯正好一致，第1个样本的第2个特征在矩阵的第1行第2列，因此我们在本书中一律使用第三种形式来描述线性方程。

另外一个原因是，在很多深度学习库的实现中，确实是把 $X$ 放在 $W$ 前面做矩阵运算的，同时 $W$ 的形状也是从左向右看，比如左侧有2个样本的3个特征输入（$2\times 3$ 表示2个样本3个特征值），右侧需要一维的输出，则 $W$ 的形状就是 $3\times 1$，这样矩阵运算的结果是 $(2 \times 3) \times (3 \times 1)=(2 \times 1)$。否则的话就需要倒着看，$W$ 的形状成为了 $1\times 3$，而 $X$ 变成了 $3\times 2$，很别扭。

对于 $B$ 来说，它永远是1行，列数与 $W$ 的列数相等。比如 $W$ 是 $3\times 1$ 的矩阵，则 $B$ 是 $1\times 1$ 的矩阵。如果 $W$ 是 $3\times 2$ 的矩阵，意味着3个特征输入到2个神经元上，则 $B$ 是 $1\times 2$ 的矩阵，每个神经元分配1个bias。

----
####最小二乘法
#####数学原理
线性回归试图学得：

$$z_i=w \cdot x_i+b \tag{1}$$

使得：

$$z_i \simeq y_i \tag{2}$$

其中，$x_i$ 是样本特征值，$y_i$ 是样本标签值，$z_i$ 是模型预测值。

如何学得 $w$ 和 $b$ 呢？均方差(MSE - mean squared error)是回归任务中常用的手段： $$ J = \frac{1}{2m}\sum_{i=1}^m(z_i-y_i)^2 = \frac{1}{2m}\sum_{i=1}^m(y_i-wx_i-b)^2 \tag{3} $$

$J$ 称为损失函数。实际上就是试图找到一条直线，使所有样本到直线上的残差的平方和最小。
### 梯度下降法

#### 4.2.1 数学原理

在下面的公式中，我们规定 $x$ 是样本特征值（单特征），$y$ 是样本标签值，$z$ 是预测值，下标 $i$ 表示其中一个样本。

##### 预设函数（Hypothesis Function）

线性函数：

$$z_i = x_i \cdot w + b \tag{1}$$

##### 损失函数（Loss Function）

均方误差：

$$loss_i(w,b) = \frac{1}{2} (z_i-y_i)^2 \tag{2}$$


与最小二乘法比较可以看到，梯度下降法和最小二乘法的模型及损失函数是相同的，都是一个线性模型加均方差损失函数，模型用于拟合，损失函数用于评估效果。

区别在于，最小二乘法从损失函数求导，直接求得数学解析解，而梯度下降以及后面的神经网络，都是利用导数传递误差，再通过迭代方式一步一步（用近似解）逼近真实解。
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

在梯度下降法中，我们简单讲述了一下神经网络做线性拟合的原理，即：

1. 初始化权重值
2. 根据权重值放出一个解
3. 根据均方差函数求误差
4. 误差反向传播给线性计算部分以调整权重值
5. 是否满足终止条件？不满足的话跳回2

一个不恰当的比喻就是穿糖葫芦：桌子上放了一溜儿12个红果，给你一个足够长的竹签子，选定一个角度，在不移动红果的前提下，想办法用竹签子穿起最多的红果。

最开始你可能会任意选一个方向，用竹签子比划一下，数数能穿到几个红果，发现是5个；然后调整一下竹签子在桌面上的水平角度，发现能穿到6个......最终你找到了能穿10个红果的的角度。

###  定义神经网络结构

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

### 反向传播

由于我们使用了和上一节中的梯度下降法同样的数学原理，所以反向传播的算法也是一样的，细节请查看4.2.2。

#### 计算 $w$ 的梯度

$$
{\partial{loss} \over \partial{w}} = \frac{\partial{loss}}{\partial{z_i}}\frac{\partial{z_i}}{\partial{w}}=(z_i-y_i)x_i
$$

#### 计算 $b$ 的梯度

$$
\frac{\partial{loss}}{\partial{b}} = \frac{\partial{loss}}{\partial{z_i}}\frac{\partial{z_i}}{\partial{b}}=z_i-y_i
$$

为了简化问题，在本小节中，反向传播使用单样本方式，在下一小节中，我们将介绍多样本方式。

###  代码实现

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

### 4.3.4 运行结果可视化

打印输出结果：

```
w=1.716290,b=3.196841
result= [3.79067723]
```

最终我们得到了 $w$ 和 $b$ 的值，对应的直线方程是 $y=1.71629x+3.196841$。推理预测时，已知有346台服务器，先要除以1000，因为横坐标是以$K$(千台)服务器为单位的，代入前向计算函数，得到的结果是3.74千瓦。

结果显示函数：

```Python
def ShowResult(net, dataReader):
    ......
```

对于初学神经网络的人来说，可视化的训练过程及结果，可以极大地帮助理解神经网络的原理，`Python`的`Matplotlib`库提供了非常丰富的绘图功能。

在上面的函数中，先获得所有样本点数据，把它们绘制出来。然后在 $[0,1]$ 之间等距设定 $10$ 个点做为 $x$ 值，用 $x$ 值通过网络推理方法`net.inference()`获得每个点的 $y$ 值，最后把这些点连起来，就可以画出图4-5中的拟合直线。

<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/4/result.png" ch="500" />

图4-5 拟合效果

可以看到红色直线虽然穿过了蓝色点阵，但是好像不是处于正中央的位置，应该再逆时针旋转几度才会达到最佳的位置。我们后面小节中会讲到如何提高训练结果的精度问题。

####  工作原理

就单纯地看待这个线性回归问题，其原理就是先假设样本点是呈线性分布的，注意这里的线性有可能是高维空间的，而不仅仅是二维平面上的。但是高维空间人类无法想象，所以我们不妨用二维平面上的问题来举例。

在4.2的梯度下降法中，首先假设这个问题是个线性问题，因而有了公式 $z=xw+b$，用梯度下降的方式求解最佳的 $w,b$ 的值。

在本节中，用神经元的编程模型把梯度下降法包装了一下，这样就进入了神经网络的世界，从而可以有成熟的方法论可以解决更复杂的问题，比如多个神经元协同工作、多层神经网络的协同工作等等。

如图4-5所示，样本点摆在那里，位置都是固定的了，神经网络的任务就是找到一根直线（注意我们首先假设这是线性问题），让该直线穿过样本点阵，并且所有样本点到该直线的距离的平方的和最小。

可以想象成每一个样本点都有一根橡皮筋连接到直线上，连接点距离该样本点最近，所有的橡皮筋形成一个合力，不断地调整该直线的位置。该合力具备两种调节方式：

1. 如果上方的拉力大一些，直线就会向上平移一些，这相当于调节 $b$ 值；
2. 如果侧方的拉力大一些，直线就会向侧方旋转一些，这相当于调节 $w$ 值。

直到该直线处于平衡位置时，也就是线性拟合的最佳位置了。

如果样本点不是呈线性分布的，可以用直线拟合吗？

答案是“可以的”，只是最终的效果不太理想，误差可以做到在线性条件下的最小，但是误差值本身还是比较大的。比如一个半圆形的样本点阵，用直线拟合可以达到误差值最小为 $1.2$（不妨假设这个值的单位是厘米），已经尽力了但能力有限。如果用弧线去拟合，可以达到误差值最小为 $0.3$。

所以，当使用线性回归的效果不好时，即判断出一个问题不是线性问题时，我们会用第9章的方法来解决。

### 前向计算

由于有多个样本同时计算，所以我们使用 $x_i$ 表示第 $i$ 个样本，$X$ 是样本组成的矩阵，$Z$ 是计算结果矩阵，$w$ 和 $b$ 都是标量：

$$
Z = X \cdot w + b \tag{1}
$$

把它展开成3个样本（3行，每行代表一个样本）的形式：

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
\end{pmatrix} \tag{2}
$$

$z_1,z_2,z_3$ 是三个样本的计算结果。根据公式1和公式2，我们的前向计算`Python`代码可以写成：

```Python
    def __forwardBatch(self, batch_x):
        Z = np.dot(batch_x, self.w) + self.b
        return Z
```
`Python`中的矩阵乘法命名有些问题，`np.dot()`并不是矩阵点乘，而是矩阵叉乘，请读者习惯。

### 4.4.2 损失函数

用传统的均方差函数，其中，$z$ 是每一次迭代的预测输出，$y$ 是样本标签数据。我们使用 $m$ 个样本参与计算，因此损失函数为：

$$J(w,b) = \frac{1}{2m}\sum_{i=1}^{m}(z_i - y_i)^2$$

其中的分母中有个2，实际上是想在求导数时把这个2约掉，没有什么原则上的区别。

我们假设每次有3个样本参与计算，即 $m=3$，则损失函数实例化后的情形是：

$$
\begin{aligned}
J(w,b) &= \frac{1}{2\times3}[(z_1-y_1)^2+(z_2-y_2)^2+(z_3-y_3)^2] \\\\
&=\frac{1}{2\times3}\sum_{i=1}^3[(z_i-y_i)^2]
\end{aligned} 
\tag{3}
$$

公式3中大写的 $Z$ 和 $Y$ 都是矩阵形式，用代码实现：

```Python
    def __checkLoss(self, dataReader):
        X,Y = dataReader.GetWholeTrainSamples()
        m = X.shape[0]
        Z = self.__forwardBatch(X)
        LOSS = (Z - Y)**2
        loss = LOSS.sum()/m/2
        return loss
```
`Python`中的矩阵减法运算，不需要对矩阵中的每个对应的元素单独做减法，而是整个矩阵相减即可。做求和运算时，也不需要自己写代码做遍历每个元素，而是简单地调用求和函数即可。

### 4.4.3 求 $w$ 的梯度

我们用 $J$ 的值作为基准，去求 $w$ 对它的影响，也就是 $J$ 对 $w$ 的偏导数，就可以得到 $w$ 的梯度了。从公式3看 $J$ 的计算过程，$z_1,z_2,z_3$都对它有贡献；再从公式2看 $z_1,z_2,z_3$ 的生成过程，都有 $w$ 的参与。所以，$J$ 对 $w$ 的偏导应该是这样的：

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
\end{aligned} \tag{4}
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

公式4中最后两个等式其实是等价的，只不过倒数第二个公式用求和方式计算每个样本，最后一个公式用矩阵方式做一次性计算。
## 4.5 梯度下降的三种形式

我们比较一下目前我们用三种方法得到的 $w$ 和 $b$ 的值，见表4-2。

表4-2 三种方法的结果比较

|方法|$w$|$b$|
|----|----|----|
|最小二乘法|2.056827|2.965434|
|梯度下降法|1.71629006|3.19684087|
|神经网络法|1.71629006|3.19684087|

这个问题的原始值是可能是 $w=2,b=3$，由于样本噪音的存在，使用最小二乘法得到了 $2.05,2.96$ 这样的非整数解，这是完全可以接受的。但是使用梯度下降和神经网络两种方式，都得到 $1.71,3.19$ 这样的值，准确程度很低。从图4-6的神经网络的训练结果来看，拟合直线是斜着穿过样本点区域的，并没有在正中央的骨架上。

<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/4/result.png" ch="500" />

图4-6 拟合效果

难道是神经网络方法有什么问题吗？

初次使用神经网络，一定有水土不服的地方。最小二乘法可以得到数学解析解，所以它的结果是可信的。梯度下降法和神经网络法实际是一回事儿，只是梯度下降没有使用神经元模型而已。所以，接下来我们研究一下如何调整神经网络的训练过程，先从最简单的梯度下降的三种形式说起。

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

### 4.5.1 单样本随机梯度下降

SGD(Stochastic Gradient Descent)

样本访问示意图如图4-7所示。
  
<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/4/SingleSample-example.png" />

图4-7 单样本访问方式

#### 计算过程

假设一共100个样本，每次使用1个样本：

***

$repeat \\{ \\\\ $
$\quad for \quad i=1,2,3,...,100\{ \\\\ $
$\quad \quad z_i = x_i \cdot w + b\\\\ $
$\quad \quad dw= x_i \cdot (z_i - y_i)\\\\ $
$\quad \quad db= z_i - y_i \\\\ $
$\quad \quad w=w-\eta \cdot dw \\\\ $
$\quad \quad db=b-\eta \cdot db \\\\ $
$\\}$

***

#### 特点
  
  - 训练样本：每次使用一个样本数据进行一次训练，更新一次梯度，重复以上过程。
  - 优点：训练开始时损失值下降很快，随机性大，找到最优解的可能性大。
  - 缺点：受单个样本的影响最大，损失函数值波动大，到后期徘徊不前，在最优解附近震荡。不能并行计算。

#### 运行结果

设置`batch_size=1`，即单样本方式：

```Python
if __name__ == '__main__':
    sdr = SimpleDataReader()
    sdr.ReadData()
    params = HyperParameters(1, 1, eta=0.1, max_epoch=100, batch_size=1, eps = 0.02)
    net = NeuralNet(params)
    net.train(sdr)
```    

表4-3 单样本方式的训练情况

|损失函数值|梯度下降过程|
|---|---|
|<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/4/SingleSample-Loss.png"/>|<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/4/SingleSample-Trace.png"/>|

表4-3的左图，由于我们使用了限定的停止条件，即当损失函数值小于等于 $0.02$ 时停止训练，所以，单样本方式迭代了300次后达到了精度要求。

右图是 $w$ 和 $b$ 共同构成的损失函数等高线图。梯度下降时，开始收敛较快，稍微有些弯曲地向中央地带靠近。到后期波动较大，找不到准确的前进方向，曲折地达到中心附近。

### 4.5.2 小批量样本梯度下降

Mini-Batch Gradient Descent

样本访问示意图如图4-8所示。

<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/4/MiniBatch-example.png" />

图4-8 小批量样本访问方式

#### 计算过程

假设一共100个样本，每个小批量5个样本：

***

$repeat \\{ \\\\ $
$\quad for \quad i=1,6,11,...,96\{\\\\ $
$\quad \quad z_i = x_i \cdot w + b \\\\ $
$\quad \quad z_{i+1} = x_{i+1} \cdot w + b \\\\ $
$\quad \quad \dots \\\\ $
$\quad \quad z_{i+4} = x_{i+4} \cdot w + b \\\\ $
$\quad \quad dw= {1 \over 5}\sum_{k=i}^{i+4} x_k \cdot (z_k - y_k) \\\\ $
$\quad \quad db= {1 \over 5}\sum_{k=i}^{i+4} (z_k - y_k) \\\\ $
$\quad \quad w=w-\eta \cdot dw \\\\ $
$\quad \quad db=b-\eta \cdot db \\\\ $
$\\}$

***

上述算法中，循环体中的前5行分别计算了 $z_i, z_{i+1}, ..., z_{i+4}$，可以换成一次性的矩阵运算。

#### 特点

  - 训练样本：选择一小部分样本进行训练，更新一次梯度，然后再选取另外一小部分样本进行训练，再更新一次梯度。
  - 优点：不受单样本噪声影响，训练速度较快。
  - 缺点：batch size的数值选择很关键，会影响训练结果。


#### 运行结果

设置`batch_size=10`：

```Python
if __name__ == '__main__':
    sdr = SimpleDataReader()
    sdr.ReadData()
    params = HyperParameters(1, 1, eta=0.3, max_epoch=100, batch_size=10, eps = 0.02)
    net = NeuralNet(params)
    net.train(sdr)   
```

表4-4 小批量样本方式的训练情况

|损失函数值|梯度下降过程|
|---|---|
|<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/4/MiniBatch-Loss.png"/>|<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/4/MiniBatch-Trace.png"/>|

表4-4的右图，梯度下降时，在接近中心时有小波动。图太小看不清楚，可以用matplot工具放大局部来观察。和单样本方式比较，在中心区的波动已经缓解了很多。

小批量的大小通常由以下几个因素决定：

- 更大的批量会计算更精确的梯度，但是回报却是小于线性的。
- 极小批量通常难以充分利用多核架构。这决定了最小批量的数值，低于这个值的小批量处理不会减少计算时间。
- 如果批量处理中的所有样本可以并行地处理，那么内存消耗和批量大小成正比。对于多硬件设施，这是批量大小的限制因素。
- 某些硬件上使用特定大小的数组时，运行时间会更少，尤其是GPU，通常使用2的幂数作为批量大小可以更快，如`32,64,128,256`，大模型时尝试用`16`。
- 可能是由于小批量在学习过程中加入了噪声，会带来一些正则化的效果。泛化误差通常在批量大小为1时最好。因为梯度估计的高方差，小批量使用较小的学习率，以保持稳定性，但是降低学习率会使迭代次数增加。

在实际工程中，我们通常使用小批量梯度下降形式。

### 4.5.3 全批量样本梯度下降 

Full Batch Gradient Descent

样本访问示意图如图4-9所示。

<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/4/FullBatch-example.png" />

图4-9  全批量样本访问方式

#### 计算过程

假设一共100个样本，每次使用全部样本：

***

$repeat \\{ \\\\ $
$\quad z_1 = x_1 \cdot w + b \\\\ $
$\quad z_2 = x_2 \cdot w + b \\\\ $
$\quad \dots \\\\ $
$\quad z_{100} = x_{100} \cdot w + b \\\\ $
$\quad dw= {1 \over 100}\sum_{i=1}^{100} x_i \cdot (z_i - y_i) \\\\ $
$\quad db= {1 \over 100}\sum_{i=1}^{100} (z_i - y_i) \\\\ $
$\quad w=w-\eta \cdot dw \\\\ $
$\quad db=b-\eta \cdot db \\\\ $
$\\}$

***

上述算法中，循环体中的前100行分别计算了 $z_1, z_2, ..., z_{100}$，可以换成一次性的矩阵运算。

#### 特点

  - 训练样本：每次使用全部数据集进行一次训练，更新一次梯度，重复以上过程。
  - 优点：受单个样本的影响最小，一次计算全体样本速度快，损失函数值没有波动，到达最优点平稳。方便并行计算。
  - 缺点：数据量较大时不能实现（内存限制），训练过程变慢。初始值不同，可能导致获得局部最优解，并非全局最优解。

#### 运行结果

```Python
if __name__ == '__main__':
    sdr = SimpleDataReader()
    sdr.ReadData()
    params = HyperParameters(1, 1, eta=0.5, max_epoch=1000, batch_size=-1, eps = 0.02)
    net = NeuralNet(params)
    net.train(sdr)
```

设置`batch_size=-1`，即是全批量的意思。

表4-5 全批量样本方式的训练情况

|损失函数值|梯度下降过程|
|---|---|
|<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/4/FullBatch-Loss.png"/>|<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/4/FullBatch-Trace.png"/>|

表4-5中的右图，梯度下降时，在整个过程中只拐了一个弯儿，就直接到达了中心点。

### 4.5.4 三种方式的比较

表4-6 三种方式的比较

||单样本|小批量|全批量|
|---|---|---|---|
|梯度下降过程图解|<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/4/SingleSample-Trace.png"/>|<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/4/MiniBatch-Trace.png"/>|<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/4/FullBatch-Trace.png"/>|
|批大小|1|10|100|
|学习率|0.1|0.3|0.5|
|迭代次数|304|110|60|
|epoch|3|10|60|
|结果|w=2.003, b=2.990|w=2.006, b=2.997|w=1.993, b=2.998|

表4-6比较了三种方式的结果，从结果看，都接近于 $w=2,b=3$ 的原始解。最后的可视化结果图如图4-10，可以看到直线已经处于样本点比较中间的位置。

<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/4/mbgd-result.png" ch="500" />

图4-10 较理想的拟合效果图

相关的概念：

- Batch Size：批大小，一次训练的样本数量。
- Iteration：迭代，一次正向 + 一次反向。
- Epoch：所有样本被使用了一次，叫做一个Epoch，中文的翻译比较杂乱，所以干脆就用原文比较清楚。

假设一共有样本1000个，batch size=20，则一个Epoch中，需要1000/20=50次Iteration才能训练完所有样本。
## 正规方程解法

英文名是 Normal Equations。

对于线性回归问题，除了前面提到的最小二乘法可以解决一元线性回归的问题外，也可以解决多元线性回归问题。

对于多元线性回归，可以用正规方程来解决，也就是得到一个数学上的解析解。它可以解决下面这个公式描述的问题：

$$y=a_0+a_1x_1+a_2x_2+\dots+a_kx_k \tag{1}$$

###  简单的推导方法

在做函数拟合（回归）时，我们假设函数 $H$ 为：

$$H(w,b) = b + x_1 w_1+x_2 w_2+ \dots +x_n w_n \tag{2}$$

令 $b=w_0$，则：

$$H(W) = w_0 + x_1 \cdot w_1 + x_2 \cdot w_2 + \dots + x_n \cdot w_n\tag{3}$$

公式3中的 $x$ 是一个样本的 $n$ 个特征值，如果我们把 $m$ 个样本一起计算，将会得到下面这个矩阵：

$$H(W) = X \cdot W \tag{4}$$

公式5中的 $X$ 和 $W$ 的矩阵形状如下：

$$
X = 
\begin{pmatrix} 
1 & x_{1,1} & x_{1,2} & \dots & x_{1,n} \\\\
1 & x_{2,1} & x_{2,2} & \dots & x_{2,n} \\\\
\vdots & \vdots & \vdots & \ddots & \vdots \\\\
1 & x_{m,1} & x_{m,2} & \dots & x_{m,n}
\end{pmatrix} \tag{5}
$$

$$
W= \begin{pmatrix}
w_0 \\\\
w_1 \\\\
\vdots \\\\
 w_n
\end{pmatrix}  \tag{6}
$$

然后我们期望假设函数的输出与真实值一致，则有：

$$H(W) = X \cdot W = Y \tag{7}$$

其中，Y的形状如下：

$$
Y= \begin{pmatrix}
y_1 \\\\
y_2 \\\\
\vdots \\\\
y_m
\end{pmatrix}  \tag{8}
$$


直观上看，$W = Y/X$，但是这里三个值都是矩阵，而矩阵没有除法，所以需要得到 $X$ 的逆矩阵，用 $Y$ 乘以 $X$ 的逆矩阵即可。但是又会遇到一个问题，只有方阵才有逆矩阵，而 $X$ 不一定是方阵，所以要先把左侧变成方阵，就可能会有逆矩阵存在了。所以，先把等式两边同时乘以 $X$ 的转置矩阵，以便得到 $X$ 的方阵：

$$X^{\top} X W = X^{\top} Y \tag{9}$$

其中，$X^{\top}$ 是 $X$ 的转置矩阵，$ X^{\top}X$ 一定是个方阵，并且假设其存在逆矩阵，把它移到等式右侧来：

$$W = (X^{\top} X)^{-1}{X^{\top} Y} \tag{10}$$

至此可以求出 $W$ 的正规方程。

###  复杂的推导方法

我们仍然使用均方差损失函数（略去了系数$\frac{1}{2m}$）：

$$J(w,b) = \sum_{i=1}^m (z_i - y_i)^2 \tag{11}$$

把 $b$ 看作是一个恒等于 $1$ 的feature，并把 $Z=XW$ 计算公式带入，并变成矩阵形式：

$$J(W) = \sum_{i=1}^m \left(\sum_{j=0}^nx_{ij}w_j -y_i\right)^2=(XW - Y)^{\top} \cdot (XW - Y) \tag{12}$$

对 $W$ 求导，令导数为 $0$，可得到 $W$ 的最小值解：

$$
\begin{aligned}
\frac{\partial J(W)}{\partial W} &= \frac{\partial}{\partial W}[(XW - Y)^{\top} \cdot (XW - Y)] \\\\
&=\frac{\partial}{\partial W}[(W^{\top}X^{\top} - Y^{\top}) \cdot (XW - Y)] \\\\
&=\frac{\partial}{\partial W}[(W^{\top}X^{\top}XW -W^{\top}X^{\top}Y - Y^{\top}XW + Y^{\top}Y)] 
\end{aligned}
\tag{13}
$$

求导后（请参考矩阵/向量求导公式）：

第一项的结果是：$2X^{\top}XW$（分母布局，denominator layout）

第二项的结果是：$X^{\top}Y$（分母布局方式，denominator layout）

第三项的结果是：$X^{\top}Y$（分子布局方式，numerator layout，需要转置$Y^{\top}X$）

第四项的结果是：$0$

再令导数为 $0$：

$$
\frac{\partial J}{\partial W}=2X^{\top}XW - 2X^{\top}Y=0 \tag{14}
$$
$$
X^{\top}XW = X^{\top}Y \tag{15}
$$
$$
W=(X^{\top}X)^{-1}X^{\top}Y \tag{16}
$$

结论和公式10一样。

以上推导的基本公式可以参考第0章的公式60-69。

逆矩阵 $(X^{\top}X)^{-1}$ 可能不存在的原因是：

1. 特征值冗余，比如 $x_2=x^2_1$，即正方形的边长与面积的关系，不能作为两个特征同时存在；
2. 特征数量过多，比如特征数 $n$ 比样本数 $m$ 还要大。

以上两点在我们这个具体的例子中都不存在。

### 代码实现

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

### 运行结果

```
w1= -2.0184092853092226
w2= 5.055333475112755
b= 46.235258613837644
z= 486.1051325196855
```

我们得到了两个权重值和一个偏移值，然后得到房价预测值 $z=486$ 万元。
### 5.2.1 定义神经网络结构

我们定义一个如图5-1所示的一层的神经网络，输入层为2或者更多，反正大于2了就没区别。这个一层的神经网络的特点是：

1. 没有中间层，只有输入项和输出层（输入项不算做一层）；
2. 输出层只有一个神经元；
3. 神经元有一个线性输出，不经过激活函数处理，即在下图中，经过 $\Sigma$ 求和得到 $Z$ 值之后，直接把 $Z$ 值输出。

<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/5/setup.png" ch="500" />

图5-1 多入单出的单层神经元结构

与上一章的神经元相比，这次仅仅是多了一个输入，但却是质的变化，即，一个神经元可以同时接收多个输入，这是神经网络能够处理复杂逻辑的根本。

#### 输入层

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

#### 权重 $W$ 和 $B$

由于输入层是两个特征，输出层是一个变量，所以 $W$ 的形状是 $2\times 1$，而 $B$ 的形状是 $1\times 1$。

$$
W=
\begin{pmatrix}
w_1 \\\\ w_2
\end{pmatrix}
$$

$$B=(b)$$

$B$ 是个单值，因为输出层只有一个神经元，所以只有一个bias，每个神经元对应一个bias，如果有多个神经元，它们都会有各自的b值。

#### 输出层

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

#### 损失函数

因为是线性回归问题，所以损失函数使用均方差函数。

$$loss_i(W,B) = \frac{1}{2} (z_i-y_i)^2 \tag{1}$$

其中，$z_i$ 是样本预测值，$y_i$ 是样本的标签值。

### 5.2.2 反向传播

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

### 5.2.3 代码实现

公式6和第4.4节中的公式5一样，所以我们依然采用第四章中已经写好的`HelperClass`目录中的那些类，来表示我们的神经网络。虽然此次神经元多了一个输入，但是不用改代码就可以适应这种变化，因为在前向计算代码中，使用的是矩阵乘的方式，可以自动适应`x`的多个列的输入，只要对应的`w`的矩阵形状是正确的即可。

但是在初始化时，我们必须手动指定`x`和`W`的形状，如下面的代码所示：

```Python
if __name__ == '__main__':
    # net
    params = HyperParameters(2, 1, eta=0.1, max_epoch=100, batch_size=1, eps = 1e-5)
    net = NeuralNet(params)
    net.train(reader)
    # inference
    x1 = 15
    x2 = 93
    x = np.array([x1,x2]).reshape(1,2)
    print(net.inference(x))
```

在参数中，指定了学习率`0.1`，最大循环次数`100`轮，批大小`1`个样本，以及停止条件损失函数值`1e-5`。

在神经网络初始化时，指定了`input_size=2`，且`output_size=1`，即一个神经元可以接收两个输入，最后是一个输出。

最后的`inference`部分，是把两个条件（15公里，93平方米）代入，查看输出结果。

在下面的神经网络的初始化代码中，`W`的初始化是根据`input_size`和`output_size`的值进行的。

```Python
class NeuralNet(object):
    def __init__(self, params):
        self.params = params
        self.W = np.zeros((self.params.input_size, self.params.output_size))
        self.B = np.zeros((1, self.params.output_size))
```

#### 正向计算的代码

```Python
class NeuralNet(object):
    def __forwardBatch(self, batch_x):
        Z = np.dot(batch_x, self.W) + self.B
        return Z
```

#### 误差反向传播的代码

```Python
class NeuralNet(object):
    def __backwardBatch(self, batch_x, batch_y, batch_z):
        m = batch_x.shape[0]
        dZ = batch_z - batch_y
        dB = dZ.sum(axis=0, keepdims=True)/m
        dW = np.dot(batch_x.T, dZ)/m
        return dW, dB
```

### 运行结果

在Visual Studio 2017中，可以使用Ctrl+F5运行Level2的代码，但是，会遇到一个令人沮丧的打印输出：

```
epoch=0
NeuralNet.py:32: RuntimeWarning: invalid value encountered in subtract
  self.W = self.W - self.params.eta * dW
0 500 nan
epoch=1
1 500 nan
epoch=2
2 500 nan
epoch=3
3 500 nan
......
```

减法怎么会出问题？什么是`nan`？

`nan`的意思是数值异常，导致计算溢出了，出现了没有意义的数值。现在是每500个迭代监控一次，我们把监控频率调小一些，再试试看：

```
epoch=0
0 10 6.838664338516814e+66
0 20 2.665505502247752e+123
0 30 1.4244204612680962e+179
0 40 1.393993758296751e+237
0 50 2.997958629609441e+290
NeuralNet.py:76: RuntimeWarning: overflow encountered in square
  LOSS = (Z - Y)**2
0 60 inf
...
0 110 inf
NeuralNet.py:32: RuntimeWarning: invalid value encountered in subtract
  self.W = self.W - self.params.eta * dW
0 120 nan
0 130 nan
```

前10次迭代，损失函数值已经达到了`6.83e+66`，而且越往后运行值越大，最后终于溢出了。下面的损失函数历史记录也表明了这一过程。

<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/5/wrong_loss.png" ch="500" />

图5-2 训练过程中损失函数值的变化

###  寻找失败的原因

我们可以在`NeuralNet.py`文件中，在图5-3代码行上设置断点，跟踪一下训练过程，以便找到问题所在。

<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/5/debug.png" />

图5-3 在VisualStudio中Debug

在VS2017中用F5运行debug模式，看第50行的结果：

```
batch_x
array([[ 4.96071728, 41.        ]])
batch_y
array([[244.07856544]])
```

返回的样本数据是正常的。再看下一行：

```
batch_z
array([[0.]])
```

第一次运行前向计算，由于`W`和`B`初始值都是`0`，所以`z`也是`0`，这是正常的。再看下一行：

```
dW
array([[ -1210.80475712],
       [-10007.22118309]])
dB
array([[-244.07856544]])
```

`dW`和`dB`的值都非常大，这是因为图5-4所示这行代码。

<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/5/backward.png" />

------
##Step3
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

比如：

- input=2时，output=0.88，而0.88>0.5，算作正例
- input=-1时，output=0.27，而0.27<0.5，算作负例

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

###  代码实现

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

###  结果比较

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

以上两点说明神经网络还是很聪明的，它会尽可能优美而鲁棒地找出那条分割线。





