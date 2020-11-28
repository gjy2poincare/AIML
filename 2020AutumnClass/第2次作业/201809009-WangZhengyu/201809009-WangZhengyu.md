# Step2、3笔记总结

## 学号：201809009  姓名：王征宇

## 本文将会通过以下四个要求书写笔记

时间：2020年10月6日 
要求：
 
1. 必须采用markdown格式，图文并茂，并且务必在github网站上能正确且完整的显示；

2. 详细描述ai-edu中Step2、Step3的学习过程；

3. 必须包含对代码的理解，以及相应分析和测试过程；

4. 必须包含学习总结和心得体会。

## Step2 线性回归

### 单入单出的单层神经网络

#### 一元线性回归模型

回归分析是一种数学模型。当因变量和自变量为线性关系时，它是一种特殊的线性模型。

最简单的情形是一元线性回归，由大体上有线性关系的一个自变量和一个因变量组成，模型是：

$$Y=a+bX+ε \tag{1}$$

X是自变量，Y是因变量，ε是随机误差，a和b是参数，在线性回归模型中，a和b是我们要通过算法学习出来的。
对于线性回归模型，有如下一些概念需要了解：

- 通常假定随机误差的均值为0，方差为σ^2（σ^2﹥0，σ^2与X的值无关）
- 若进一步假定随机误差遵从正态分布，就叫做正态线性模型
- 一般地，若有k个自变量和1个因变量（即公式1中的Y），则因变量的值分为两部分：一部分由自变量影响，即表示为它的函数，函数形式已知且含有未知参数；另一部分由其他的未考虑因素和随机性影响，即随机误差
- 当函数为参数未知的线性函数时，称为线性回归分析模型
- 当函数为参数未知的非线性函数时，我们通过对数据的观察，可以大致认为它符合线性回归模型的条件，于是列出了公式1，不考虑随机误差的话，我们的任务就是找到合适的a和b，这就是线性回归的任务。


![](./Images/regression.png)

如上图所示，左侧为线性模型，可以看到直线穿过了一组三角形所形成的区域的中心线，并不要求这条直线穿过每一个三角形。右侧为非线性模型，一条曲线穿过了一组矩形所形成的区域的中心线。就好比是一条街道，可能是直的，也可能有弯曲，街道的两旁是各种建筑。称为非线性回归分析模型

- 当自变量个数大于1时称为多元回归
- 当因变量个数大于1时称为多重回归


#### ②最小二乘法

####（1）数学原理

线性回归试图学得：

$$z(x_i)=w \cdot x_i+b \tag{1}$$

使得：

$$z(x_i) \simeq y_i \tag{2}$$

其中，$x_i$是样本特征值，$y_i$是样本标签值，$z_i$是模型预测值。

如何学得w和b呢？均方差(MSE - mean squared error)是回归任务中常用的手段：
$$
J = \sum_{i=1}^m(z(x_i)-y_i)^2 = \sum_{i=1}^m(y_i-wx_i-b)^2 \tag{3}
$$

$J$称为损失函数。实际上就是试图找到一条直线，使所有样本到直线上的残差的平方和最小：

![](./Images/mse.png)

#### (2)Python代码来实现推导过程及分析

  
```Python

# 计算w值
def method1(X,Y,m):
    x_mean = X.mean() # mean() 计算平均值
    p = sum(Y*(X-x_mean)) #sum() 计算数组的总值
    q = sum(X*X) - sum(X)*sum(X)/m
    w = p/q
    return w

def method2(X,Y,m):
    x_mean = X.mean()# mean() 计算平均值
    y_mean = Y.mean()
    p = sum(X*(Y-y_mean))#sum() 计算数组的总值
    q = sum(X*X) - x_mean*sum(X)
    w = p/q
    return w

def method3(X,Y,m):
    p = m*sum(X*Y) - sum(X)*sum(Y)
    q = m*sum(X*X) - sum(X)*sum(X)#sum()计算数组的总值
    w = p/q
    return w
```


```Python
# 计算b值
def calculate_b_1(X,Y,w,m):
    b = sum(Y-w*X)/m #sum()计算数组的总值
    return b
def calculate_b_2(X,Y,w):
    b = Y.mean() - w * X.mean()
    return b
```
####（3）运算结果

```Python
if __name__ == '__main__':

    reader = SimpleDataReader() #创建
    reader.ReadData()#读取数据
    X,Y = reader.GetWholeTrainSamples()#读取X和Y值
    m = X.shape[0]#X矩阵的行数
    w1 = method1(X,Y,m)
    b1 = calculate_b_1(X,Y,w1,m)#提取b值

    w2 = method2(X,Y,m)
    b2 = calculate_b_2(X,Y,w2)

    w3 = method3(X,Y,m)
    b3 = calculate_b_1(X,Y,w3,m)

    print("w1=%f, b1=%f" % (w1,b1))
    print("w2=%f, b2=%f" % (w2,b2))
    print("w3=%f, b3=%f" % (w3,b3))
```
用以上几种方法，最后得出的结果都是一致的，可以起到交叉验证的作用：

#### 程序运行测试结果

```
w1=2.056827, b1=2.965434
w2=2.056827, b2=2.965434
w3=2.056827, b3=2.965434
```

#### ③ 梯度下降法

#### （1）数学原理

在下面的公式中，我们规定x是样本特征值（单特征），y是样本标签值，z是预测值，下标 $i$ 表示其中一个样本。

#### 预设函数（Hypothesis Function）

为一个线性函数：

$$z_i = x_i \cdot w + b \tag{1}$$

#### 损失函数（Loss Function）

为均方差函数：
$$loss(w,b) = \frac{1}{2} (z_i-y_i)^2 \tag{2}$$
与最小二乘法比较可以看到，梯度下降法和最小二乘法的模型及损失函数是相同的，都是一个线性模型加均方差损失函数，模型用于拟合，损失函数用于评估效果。
梯度计算

#### 计算z的梯度

根据公式2：
$$
{\partial loss \over \partial z_i}=z_i - y_i \tag{3}
$$

#### 计算w的梯度

我们用loss的值作为误差衡量标准，通过求w对它的影响，也就是loss对w的偏导数，来得到w的梯度。由于loss是通过公式2->公式1间接地联系到w的，所以我们使用链式求导法则，通过单个样本来求导。

根据公式1和公式3：

$$
{\partial{loss} \over \partial{w}} = \frac{\partial{loss}}{\partial{z_i}}\frac{\partial{z_i}}{\partial{w}}=(z_i-y_i)x_i \tag{4}
$$

#### 计算b的梯度
$$
\frac{\partial{loss}}{\partial{b}} = \frac{\partial{loss}}{\partial{z_i}}\frac{\partial{z_i}}{\partial{b}}=z_i-y_i \tag{5}
$$

#### （2）代码实现及分析

```Python
if __name__ == '__main__':

    reader = SimpleDataReader()#同上一个例子
    reader.ReadData()
    X,Y = reader.GetWholeTrainSamples()
    eta = 0.1 #学习率为0.1
    w, b = 0.0, 0.0 #初始化w,b
    for i in range(reader.num_train):#循环 得到w和b
        # get x and y value for one sample
        xi = X[i]
        yi = Y[i]
        zi = xi * w + b
        dz = zi - yi
        dw = dz * xi
        db = dz
        # update w,b
        w = w - eta * dw
        b = b - eta * db
    print("w=", w)    
    print("b=", b)
```

#### 代码推导

大家可以看到，在代码中，我们完全按照公式推导实现了代码，所以，大名鼎鼎的梯度下降，其实就是把推导的结果转化为数学公式和代码，直接放在迭代过程里！另外，我们并没有直接计算损失函数值，而只是把它融入在公式推导中。

#### （3） 运行测试结果
```
w= [1.71629006]
b= [3.19684087]
```

#### ④神经网络法

#### （1）神经网络结构

我们是首次尝试建立神经网络，先用一个最简单的单层单点神经元：

![](./Images/Setup.png)

#### 输入层

此神经元在输入层只接受一个输入特征，经过参数w,b的计算后，直接输出结果。这样一个简单的“网络”，只能解决简单的一元线性回归问题，而且由于是线性的，我们不需要定义激活函数，这就大大简化了程序，而且便于大家循序渐进地理解各种知识点。

严格来说输入层在神经网络中并不能称为一个层。

#### 权重w/b

因为是一元线性问题，所以w/b都是一个标量。

#### 输出层

输出层1个神经元，线性预测公式是：

$$z_i = x_i \cdot w + b$$

z是模型的预测输出，y是实际的样本标签值，下标 $i$ 为样本。

#### 损失函数

因为是线性回归问题，所以损失函数使用均方差函数。

$$loss(w,b) = \frac{1}{2} (z_i-y_i)^2$$

#### (2)python程序实现及分析

#### 定义类

```Python
class NeuralNet(object):
    def __init__(self, eta):
        self.eta = eta
        self.w = 0
        self.b = 0
```

#### 分析

NeuralNet类从object类派生，并具有初始化函数，其参数是eta，也就是学习率，需要调用者指定。另外两个成员变量是w和b，初始化为0。

#### 前向计算

```Python
    def __forward(self, x):
        z = x * self.w + self.b
        return z
```

#### 分析

这是一个私有方法，所以前面有两个下划线，只在NeuralNet类中被调用，不对外公开。

#### 反向传播

下面的代码是通过梯度下降法中的公式推导而得的，也设计成私有方法：

```Python
    def __backward(self, x,y,z):
        dz = z - y
        db = dz
        dw = x * dz
        return dw, db
```

#### 分析


dz是中间变量，避免重复计算。dz又可以写成delta_Z，是当前层神经网络的反向误差输入。

#### 梯度更新

```Python
    def __update(self, dw, db):
        self.w = self.w - self.eta * dw
        self.b = self.b - self.eta * db
```

#### （3）主程序

```Python
if __name__ == '__main__':
    # 读数据
    sdr = SimpleDataReader()
    sdr.ReadData()
    # 创建net
    eta = 0.1
    net = NeuralNet(eta)
    net.train(sdr)
    # 结果
    print("w=%f,b=%f" %(net.w, net.b))
    # 预测
    result = net.inference(0.346)
    print("result=", result)
    ShowResult(net, sdr)
```
#### （4）运行测试结果

打印输出结果：
```
w=1.716290,b=3.196841
result= [3.79067723]
```

## 二 、多入单出的单层神经网络

#### ①多变量线性回归问题

#### （1）多元线性回归模型

准则：

1. 自变量对因变量必须有显著的影响，并呈密切的线性相关；
2. 自变量与因变量之间的线性相关必须是真实的，而不是形式上的；
3. 自变量之间应具有一定的互斥性，即自变量之间的相关程度不应高于自变量与因变量之因的相关程度；
4. 自变量应具有完整的统计数据，其预测值容易确定。

|方法|正规方程|梯度下降|
|---|-----|-----|
|原理|几次矩阵运算|多次迭代|
|特殊要求|$X^TX$的逆矩阵存在|需要确定学习率|
|复杂度|$O(n^3)$|$O(n^2)$|
|适用样本数|$m \lt 10000$|$m \ge 10000$|

#### （2）正规方程解法（Normal Equations)

#### ①推导方法

在做函数拟合（回归）时，我们假设函数H为：

$$h(w,b) = b + x_1 w_1+x_2 w_2+...+x_n w_n \tag{2}$$

令$b=w_0$，则：

$$h(w) = w_0 + x_1 \cdot w_1 + x_2 \cdot w_2+...+ x_n \cdot w_n\tag{3}$$

公式3中的x是一个样本的n个特征值，如果我们把m个样本一起计算，将会得到下面这个矩阵：

$$H(w) = X \cdot W \tag{4}$$

公式5中的X和W的矩阵形状如下：

$$
X^{(m \times (n+1))} = 
\begin{pmatrix} 
1 & x_{1,1} & x_{1,2} & \dots & x_{1,n} \\
1 & x_{2,1} & x_{2,2} & \dots & x_{2,n} \\
\dots \\
1 & x_{m,1} & x_{m,2} & \dots & x_{m,n}
\end{pmatrix} \tag{5}
$$

$$
W^{(n+1)}= \begin{pmatrix}
w_0 \\
w_1 \\
\dots \\
 w_n
\end{pmatrix}  \tag{6}
$$

然后我们期望假设函数的输出与真实值一致，则有：

$$H(w) = X \cdot W = Y \tag{7}$$

其中，Y的形状如下：

$$
Y^{(m)}= \begin{pmatrix}
y_1 \\
y_2 \\
\dots \\
y_m
\end{pmatrix}  \tag{8}
$$


直观上看，W = Y/X，但是这里三个值都是矩阵，而矩阵没有除法，所以需要得到X的逆矩阵，用Y乘以X的逆矩阵即可。但是又会遇到一个问题，只有方阵才有逆矩阵，而X不一定是方阵，所以要先把左侧变成方阵，就可能会有逆矩阵存在了。所以，先把等式两边同时乘以X的转置矩阵，以便得到X的方阵：

$$X^T X W = X^T Y \tag{9}$$

其中，$X^T$是X的转置矩阵，$X^T X$一定是个方阵，并且假设其存在逆矩阵，把它移到等式右侧来：

$$W = (X^T X)^{-1}{X^T Y} \tag{10}$$

至此可以求出W的正规方程。

#### ②代码实现及分析

```Python
if __name__ == '__main__':
    #读取
    reader = SimpleDataReader()
    reader.ReadData()
    X,Y = reader.GetWholeTrainSamples()
    num_example = X.shape[0] #读X的行数
    one = np.ones((num_example,1)) #将第一列的X个行数全为1
    x = np.column_stack((one, (X[0:num_example,:])))#将2个矩阵按行合并
    a = np.dot(x.T, x) # 矩阵乘以矩阵
    # need to convert to matrix, because np.linalg.inv only works on matrix instead of array
    b = np.asmatrix(a) # 变为矩阵
    c = np.linalg.inv(b) # 矩阵转置
    d = np.dot(c, x.T) # 矩阵相乘
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

#### ② 运行测试结果

```
w1= -2.0184092853092226
w2= 5.055333475112755
b= 46.235258613837644
z= 486.1051325196855
```

#### ③神经网络解法

#### （1）定义神经网络结构

我们定义一个一层的神经网络，输入层为2或者更多，反正大于2了就没区别。这个一层的神经网络的特点是：

1. 没有中间层，只有输入项和输出层（输入项不算做一层），
2. 输出层只有一个神经元，
3. 神经元有一个线性输出，不经过激活函数处理，即在下图中，经过$\Sigma$求和得到Z值之后，直接把Z值输出。

与上一章的神经元相比，这次仅仅是多了一个输入，但却是质的变化，即，一个神经元可以同时接收多个输入，这是神经网络能够处理复杂逻辑的根本。

![](./Images/setup.png)

####（2）代码实现

公式6和第4.4节中的公式5一模一样，所以我们依然采用第四章中已经写好的HelperClass目录中的那些类，来表示我们的神经网络。虽然此次神经元多了一个输入，但是不用改代码就可以适应这种变化，因为在前向计算代码中，使用的是矩阵乘的方式，可以自动适应x的多个列的输入，只要对应的w的矩阵形状是正确的即可。

但是在初始化时，我们必须手动指定x和w的形状，如下面的代码所示：

```Python
from HelperClass.SimpleDataReader import *

if __name__ == '__main__':
    # 读数据
    reader = SimpleDataReader()
    reader.ReadData()
    # 创建net
    params = HyperParameters(2, 1, eta=0.1, max_epoch=100, batch_size=1, eps = 1e-5)
    net = NeuralNet(params)
    net.train(reader)
    # inference
    x1 = 15
    x2 = 93
    x = np.array([x1,x2]).reshape(1,2)
    print(net.inference(x))
```

在参数中，指定了学习率0.1，最大循环次数100轮，批大小1个样本，以及停止条件损失函数值1e-5。

在神经网络初始化时，指定了input_size=2，且output_size=1，即一个神经元可以接收两个输入，最后是一个输出。

最后的inference部分，是把两个条件（15公里，93平方米）代入，查看输出结果。

#### 分析 

在下面的神经网络的初始化代码中，W的初始化是根据input_size和output_size的值进行的。

```Python
class NeuralNet(object):#初始化
    def __init__(self, params):
        self.params = params
        self.W = np.zeros((self.params.input_size, self.params.output_size)) # 矩阵各项为0
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
        dB = dZ.sum(axis=0, keepdims=True)/m #横轴上 矩阵规格不变
        dW = np.dot(batch_x.T, dZ)/m
        return dW, dB
```

#### （3） 运行测试结果

在Visual Studio 2019中，可以使用Ctrl+F5运行Level2的代码，但是，会遇到一个令人沮丧的打印输出：

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

#### ④样本特征数据归一化
#### （1）基本概念

有三个类似的概念，归一化，标准化，中心化。

#### 归一化

把数据线性地变成[0,1]或[-1,1]之间的小数，把带单位的数据（比如米，公斤）变成无量纲的数据，区间缩放。

归一化有三种方法:

1. Min-Max归一化：
$$x_{new}={x-x_{min} \over x_{max} - x_{min}} \tag{1}$$

2. 平均值归一化
   
$$x_{new} = {x - \bar{x} \over x_{max} - x_{min}} \tag{2}$$

3. 非线性归一化

对数转换：
$$y=log(x) \tag{3}$$

反余切转换：
$$y=atan(x) \cdot 2/π  \tag{4}$$

#### 标准化

把每个特征值中的所有数据，变成平均值为0，标准差为1的数据，最后为正态分布。Z-score规范化（标准差标准化 / 零均值标准化，其中std是标准差）：

$$x_{new} = (x - \bar{x})／std \tag{5}$$

#### 中心化

平均值为0，无标准差要求：
$$x_{new} = x - \bar{x} \tag{6}$$

#### （2）代码实现及分析

在HelperClass目录的SimpleDataReader.py文件中，给该类增加一个方法：

```Python
    def NormalizeX(self):
        X_new = np.zeros(self.XRaw.shape)
        num_feature = self.XRaw.shape[1]
        self.X_norm = np.zeros((2,num_feature))
        # 按列归一化,即所有样本的同一特征值分别做归一化
        for i in range(num_feature):
            # get one feature from all examples
            col_i = self.XRaw[:,i]
            max_value = np.max(col_i)
            min_value = np.min(col_i)
            # 最小值
            self.X_norm[0,i] = min_value 
            # 值范围
            self.X_norm[1,i] = max_value - min_value 
            new_col = (col_i - self.X_norm[0,i])/(self.X_norm[1,i])
            X_new[:,i] = new_col
        #end for
        self.XTrain = X_new
```
#### （3）运行结果

运行上述代码，看打印结果：

```
epoch=9
9 0 391.75978721600353
9 100 387.79811202735783
9 200 502.9576560855685
9 300 395.63883403610765
9 400 417.61092908059885
9 500 404.62859838907883
9 600 398.0285538622818
9 700 469.12489440138637
9 800 380.78054509441193
9 900 575.5617634691969
W= [[-41.71417524]
 [395.84701164]]
B= [[242.15205099]]
z= [[37366.53336103]]
```


## 一、 Step3 线性分类--线性二分类

### 1、多入单出的单层神经网路

#### （1）线性二分类

①二分类函数
- 公式

$$a(z) = \frac{1}{1 + e^{-z}}$$

- 导数

$$a^{'}(z) = a(z)(1 - a(z))$$

具体求导过程可以参考8.1节。

- 输入值域

$$(-\infty, \infty)$$

- 输出值域

$$(0,1)$$

- 函数图像


![](./Images/logistic.png)

#### (2) 正向传播

#### 矩阵运算

$$
z=x \cdot w + b \tag{1}
$$

#### 分类计算

$$
a = Logistic(z)={1 \over 1 + e^{-z}} \tag{2}
$$

#### 损失函数计算

二分类交叉熵损失函数：

$$
loss(w,b) = -[y \ln a+(1-y)\ln(1-a)] \tag{3}
$$

#### (3) 反向传播

#### 求损失函数loss对a的偏导

$$
\frac{\partial loss}{\partial a}=-[{y \over a}+{-(1-y) \over 1-a}]=\frac{a-y}{a(1-a)} \tag{4}
$$

#### 求损失函数a对z的偏导

$$
\frac{\partial a}{\partial z}= a(1-a) \tag{5}
$$

#### 求损失函数loss对z的偏导

使用链式法则链接公式4和公式5：

$$
\frac{\partial loss}{\partial z}=\frac{\partial loss}{\partial a}\frac{\partial a}{\partial z}
$$
$$
=\frac{a-y}{a(1-a)} \cdot a(1-a)=a-y \tag{6}
$$

### 2、线性二分类的神经网络实现

#### （1）原理实现

#### 输入层

输入经度(x1)和纬度(x2)两个特征：

$$
x=\begin{pmatrix}
x_{1} & x_{2}
\end{pmatrix}
$$

#### 权重矩阵

输入是2个特征，输出一个数，则W的尺寸就是2x1：

$$
w=\begin{pmatrix}
w_{1} \\ w_{2}
\end{pmatrix}
$$

B的尺寸是1x1，行数永远是1，列数永远和W一样。

$$
b=\begin{pmatrix}
b_{1}
\end{pmatrix}
$$

#### 输出层

$$
z = x \cdot w + b
=\begin{pmatrix}
    x_1 & x_2
\end{pmatrix}
\begin{pmatrix}
    w_1 \\ w_2
\end{pmatrix}
$$
$$
=x_1 \cdot w_1 + x_2 \cdot w_2 + b \tag{1}
$$
$$a = Logistic(z) \tag{2}$$

#### 损失函数

二分类交叉熵函损失数：

$$
loss(w,b) = -[yln a+(1-y)ln(1-a)] \tag{3}
$$

#### （2）代码实现


#### 代码理解

由于以前我们的神经网络只会做线性回归，现在多了一个做分类的技能，所以我们加一个枚举类型，可以让调用者通过指定参数来控制神经网络的功能。

```Python
class NetType(Enum):
    Fitting = 1,
    BinaryClassifier = 2,
    MultipleClassifier = 3,
```

#### 代码理解

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

#### 代码理解

再增加一个Logistic分类函数：

```Python
class Logistic(object):
    def forward(self, z):
        a = 1.0 / (1.0 + np.exp(-z))
        return a
```

#### 代码理解 

新建一个类便于管理：

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

主过程：

```Python
if __name__ == '__main__':
    # data
    reader = SimpleDataReader()
    reader.ReadData()
    # net
    params = HyperParameters(eta=0.1, max_epoch=100, batch_size=10, eps=1e-3, net_type=NetType.BinaryClassifier)
    input = 2
    output = 1
    net = NeuralNet(params, input, output)
    net.train(reader, checkpoint=1)
    # inference
    x_predicate = np.array([0.58,0.92,0.62,0.55,0.39,0.29]).reshape(3,2)
    a = net.inference(x_predicate)
    print("A=", a)    
```

#### （3）运行测试结果

损失函数值记录很平稳地下降，说明网络收敛了：

![](./images/binary_loss.png)

### 3、线性二分类原理

几何原理

我们再观察一下下面这张分类正确的图：

![](./images/linear_binary_analysis.png)

假设绿色方块为正类：标签值$y=1$，红色三角形为负类：标签值$y=0$。

从几何关系上理解，如果我们有一条直线，其公式为：$z = w \cdot x_1+b$，如图中的虚线所示，则所有正类的样本的x2都大于z，而所有的负类样本的x2都小于z，那么这条直线就是我们需要的分割线。用正例的样本来表示：

$$
x_2 > z，即正例满足条件：x_2 > w \cdot x_1 + b \tag{4}
$$

那么神经网络用矩阵运算+分类函数+损失函数这么复杂的流程，其工作原理是什么呢？

经典机器学习中的SVM确实就是用这种思路来解决这个问题的，即一个类别的所有样本在分割线的一侧，而负类样本都在线的另一侧。神经网络的正向公式如公式1，2所示，当a>0.5时，判为正类。当a<0.5时，判为负类。z=0即a=0.5时为分割线。

### 4、二分类结果可视化

#### (1)代码实现及分析

主程序：

``` Python
# 主程序
if __name__ == '__main__':
    # 读数据
    reader = SimpleDataReader()
    reader.ReadData()
    # 创建net
    # 超参数
    params = HyperParameters(eta=0.1, max_epoch=100, batch_size=10, eps=1e-3, net_type=NetType.BinaryClassifier)
    input = 2
    output = 1
    net = NeuralNet(params, input, output)
    # 训练
    net.train(reader, checkpoint=1)

    # 展示结果
    draw_source_data(net, reader)
    draw_predicate_data(net)
    draw_split_line(net)
    plt.show()
```

#### （2）运行结果

下图为结果：
![](./images/binary_result.png)

## 二、线性多分类---多入单出的单层神经网路
### 1、线性多分类问题

多分类问题一共有三种解法：

1. 一对一
   
每次先只保留两个类别的数据，训练一个分类器。如果一共有N个类别，则需要训练$C^2_N$个分类器。以N=3时举例，需要训练(A|B)，(B|C)，(A|C)三个分类器。

![](./images/one_vs_one.png)

如上图最左侧所示，这个二分类器只关心蓝色和绿色样本的分类，而不管红色样本的情况，也就是说在训练时，只把蓝色和绿色样本输入网络。
   
推理时，(A|B)分类器告诉你是A类时，需要到(A|C)分类器再试一下，如果也是A类，则就是A类。如果(A|C)告诉你是C类，则基本是C类了，不可能是B类，不信的话可以到(B|C)分类器再去测试一下。

2. 一对多
   
如下图，处理一个类别时，暂时把其它所有类别看作是一类，这样对于三分类问题，可以得到三个分类器。

![](./images/one_vs_multiple.png)

如最左图，这种情况是在训练时，把红色样本当作一类，把蓝色和绿色样本混在一起当作另外一类。

推理时，同时调用三个分类器，再把三种结果组合起来，就是真实的结果。比如，第一个分类器告诉你是“红类”，那么它确实就是红类；如果告诉你是非红类，则需要看第二个分类器的结果，绿类或者非绿类；依此类推。

3. 多对多

假设有4个类别ABCD，我们可以把AB算作一类，CD算作一类，训练一个分类器1；再把AC算作一类，BD算作一类，训练一个分类器2。
    
推理时，第1个分类器告诉你是AB类，第二个分类器告诉你是BD类，则做“与”操作，就是B类。

### 2、多分类函数- Softmax

#### （1）定义
Softmax加了个"soft"来模拟max的行为，但同时又保留了相对大小的信息。

$$
a_j = \frac{e^{z_j}}{\sum\limits_{i=1}^m e^{z_i}}=\frac{e^{z_j}}{e^{z_1}+e^{z_2}+\dots+e^{z_m}}
$$

上式中:

- $z_j$是对第 j 项的分类原始值，即矩阵运算的结果
- $z_i$是参与分类计算的每个类别的原始值
- m 是总的分类数
- $a_j$是对第 j 项的计算结果


#### (2)python实现和分析

#### 公式法
```Python
def Softmax1(x):
    e_x = np.exp(x) #e的x次方
    v = np.exp(x) / np.sum(e_x)
    return v
```
#### 修改法

```Python
class Softmax(object):
    def forward(self, z):
        shift_z = z - np.max(z, axis=1, keepdims=True) #列+矩阵维度不变+最大值
        exp_z = np.exp(shift_z)
        a = exp_z / np.sum(exp_z, axis=1, keepdims=True)
        return a

```
#### （3）线性多分类的神经网络实现  
#### 输入层

输入经度(x1)和纬度(x2)两个特征：

$$
x=\begin{pmatrix}
x_1 & x_2
\end{pmatrix}
$$
#### 权重矩阵w/b
W权重矩阵的尺寸，可以从后往前看，比如：输出层是3个神经元，输入层是2个特征，则W的尺寸就是3x2。

$$
w=\begin{pmatrix}
w_{11} & w_{12} & w_{13}\\
w_{21} & w_{22} & w_{23} 
\end{pmatrix}
$$
b的尺寸是1x3，列数永远和神经元的数量一样，行数永远是1。
$$
b=\begin{pmatrix}
b_1 & b_2 & b_3 
\end{pmatrix}
$$
#### 输出层
输出层三个神经元，再加上一个Softmax计算，最后有A1,A2,A3三个输出，写作：

$$
z = \begin{pmatrix}z_1 & z_2 & z_3 \end{pmatrix}
$$
$$
a = \begin{pmatrix}a_1 & a_2 & a_3 \end{pmatrix}
$$

其中，$Z=X \cdot W+B，A = Softmax(Z)$

#### （4) 代码实现

#### 主程序

```Python
if __name__ == '__main__':
    num_category = 3
    #读数据
    reader = SimpleDataReader()
    reader.ReadData()
    reader.NormalizeX()#X标准化
    reader.ToOneHot(num_category, base=1)#OneHot编码

    num_input = 2
    # 超参数
    params = HyperParameters(num_input, num_category, eta=0.1, max_epoch=100, batch_size=10, eps=1e-3, net_type=NetType.MultipleClassifier)
    net = NeuralNet(params)
    net.train(reader, checkpoint=1)

    inference(net, reader)
```
#### (5)运行结果

![](./images/loss.png)

### 3、 线性多分类原理

#### (1)多分类的几何原理

在前面的二分类原理中，很容易理解为我们用一条直线分开两个部分。对于多分类问题，是否可以沿用二分类原理中的几何解释呢？答案是肯定的，只不过需要单独判定每一个类别。

![](./images/source_data.png)

如上图，假设一共有三类样本，蓝色为1，红色为2，绿色为3，那么Softmax的形式应该是：

$$
a_j = \frac{e^{z_j}}{\sum\limits_{i=1}^3 e^{z_i}}=\frac{e^{z_j}}{e^{z_1}+e^{z_2}+^{z_3}}
$$
把三张图综合分析在一起，应该是这个样子：
![](./images/z123.png)

## 慕课第二章学习

### 如何理解深度学习

深度学习指的是一种表征学习方法，其中的模型是由一连串的模块组成的（一般都会堆成一个多层的或者金字塔形的模型，这也就是「深度」的由来），而其中的每一个模块分别拿出来训练之后都可以作为独立的特征提取器。不需要指定固定的某种学习机制（比如反向传播），也不需要指定固定的使用方式（比如监督学习还是强化学习），而且也不是一定要做联合的端到端学习（和贪婪学习相反）。一定要用成串连起来的特征提取器做表征学习，这才是深度学习。它的本质在于通过深度层次化的特征来描述输入数据，而这些特征都是从数据里学习到的。

### 深度学习在排序方式。

排序方式：回复时间投票数
深度学习时其实就决定了输出就是两类问题，要么是良性的要么是恶性的。这里我们使用了一个类似AlexNet网络的结构，从性能上来说肯定没有目前的ResNet或者DenseNet的性能好，但这样有利于我们观察和分析中间的数据结构流向。训练结束后的误差和准确度的收敛状态。

### 深度学习的典型应用

深度学习的典型应用：图像着色。那是不是古代的水墨画都能拥有色彩呢？自动翻译机器。自动翻译文本和图片，一国语言也可以走遍天下了呢对照片中的物体进行分类和检测。我觉得对生物病理诊断的帮助会比较大哎！自动书写生成。临摹古人书法呀！还有鉴定书法真品还是赝品；自动生成抓人眼球的标题。让你轻松写出好标题；将素描转化为照片。

### 深度学习在皮肤病诊断中的应用情况

排序方式：回复时间投票数
1）深度学习系统 (DLS) 能够模仿临床医生的思维方式，根据皮肤症状排列出可能的皮肤病，从而对患者进行快速分诊、诊断和治疗。（2）DLS 可协助临床医生（包括皮肤科医生）考虑原本不在其鉴别诊断表中的可能情况，从而提高诊断准确率并改善病情管理。（3）DLS 的准确率也会随着图像数的增加而有所提高。如果缺少元数据（例如病历），模型便无法有出色的表现。

### 深度学习有哪些工业应用

概率统计推断：分类/回归（CTR预估，推荐系统）聚类(用户群分析，异常检测)。平面构成分析：图像识别（包括人脸），计算机视觉时间序列分析：NLP（包括语音），计算机听觉。

### 深度学习如何实现工业机器视觉

随着制造商需要更智能，准确和可重复的视觉系统，深度学习软件越来越受欢迎。终端用户最收益的是软件可以在几分钟内自动编程视觉系统。深度学习最适合涉及可变形对象而非刚性对象的应用。另一个好的应用是验证在装配体中存在颜色和纹理变化的许多部件。此外，传统软件要求被检部件具有特定的公差范围，而深度学习最好由最大且最清晰标记的好的和坏的部分图像数据集提供。虽然深度学习通常被认为是化妆品检验应用，但Petry说，它也非常擅长确认试剂盒中存在多个物品。

### 深度学习在医学上的成功案例

其实除了文中介绍的工业视觉方面的应用，深度学习还有许多在医学方面即为广泛的应用，例如心跳检测，根据心电图预测一些疾病；胎儿性别预测；掌静脉身份识别，等等


## 总结及心得体会：

在Step2和Step3中，我学习到了
神经网络线性回归中的很多算法，比如最小二乘法、梯度下降法、神经网络法等等。这样在解决回归问题可以用这些方法解决。当然还学习了多变量线性回归，有正规方程法和神经网络法。还学习了样本特征数据归一化，并知道了归一化的后遗症和正确的推理方法。而且还学习到了关于线性分类和线性多分类。线性分类中学习到了二分类的函数的知识，二分类中也学习到了激活函数和分类函数。并在此基础上学习到它的工作原理和数学原理。还有他的可视化，经过多次的代码实现，得到完美的二分类，而且还学习了神经网络的多线性分类，在此之前我已经了解到线性二分类的概念和基本原理，学习多线性分类之后，我收获到很多知识，也认识到之间没有学到的知识.在学习的过程中，我学到了神经网络线性多分类，首先了解了分类函数和各个运算公式，并在最后用神经网络实现，然后又学到了多分类结果的可视化，也知道了理想中一对多的方式还有现实中的情况。

在本次学习中:

1. 更加熟练的掌握了markdown的阅读与书写规则
2. 逐渐理解掌握了基于Python代码的神经网络代码
3. 学习了python的基本语法
4. 掌握了线性回归和线性分类的思想
5. 掌握了通过mingw64从GitHub网站上拷贝到本地 代码：git clone + 网址 , #更新本地 git pull 
6. 掌握了GitHub Desktop APP的应用方法，使得自己的作业可以通过本地传送到自己的网址上，再自己GitHub的作业上传到老师的账户上