# 深度神经网络
## 搭建深度神经网络框架

###  功能/模式分析
前向计算中：

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

###  抽象与设计

``迷你框架设计``
<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/14/class.png" />

#### NeuralNet

首先需要一个`NeuralNet`类，来包装基本的神经网络结构和功能：

- `Layers` - 神经网络各层的容器，按添加顺序维护一个列表
- `Parameters` - 基本参数，包括普通参数和超参
- `Loss Function` - 提供计算损失函数值，存储历史记录并最后绘图的功能
- `LayerManagement()` - 添加神经网络层
- `ForwardCalculation()` - 调用各层的前向计算方法
- `BackPropagation()` - 调用各层的反向传播方法
- `PreUpdateWeights()` - 预更新各层的权重参数
- `UpdateWeights()` - 更新各层的权重参数
- `Train()` - 训练
- `SaveWeights()` - 保存各层的权重参数
- `LoadWeights()` - 加载各层的权重参数

#### Layer

是一个抽象类，以及更加需要增加的实际类，包括：

- Fully Connected Layer
- Classification Layer
- Activator Layer
- Dropout Layer
- Batch Norm Layer

将来还会包括：

- Convolution Layer
- Max Pool Layer

每个Layer都包括以下基本方法：
 - `ForwardCalculation()` - 调用本层的前向计算方法
 - `BackPropagation()` - 调用本层的反向传播方法
 - `PreUpdateWeights()` - 预更新本层的权重参数
 - `UpdateWeights()` - 更新本层的权重参数
 - `SaveWeights()` - 保存本层的权重参数
 - `LoadWeights()` - 加载本层的权重参数

#### Activator Layer

激活函数和分类函数：

- `Identity` - 直传函数，即没有激活处理
- `Sigmoid`
- `Tanh`
- `Relu`

#### Classification Layer

分类函数，包括：

- `Sigmoid`二分类
- `Softmax`多分类


 #### Parameters

 基本神经网络运行参数：

 - 学习率
 - 最大`epoch`
 - `batch size`
 - 损失函数定义
 - 初始化方法
 - 优化器类型
 - 停止条件
 - 正则类型和条件

#### LossFunction

损失函数及帮助方法：

- 均方差函数
- 交叉熵函数二分类
- 交叉熵函数多分类
- 记录损失函数
- 显示损失函数历史记录
- 获得最小函数值时的权重参数

#### Optimizer

优化器：

- `SGD`
- `Momentum`
- `Nag`
- `AdaGrad`
- `AdaDelta`
- `RMSProp`
- `Adam`

#### WeightsBias

权重矩阵，仅供全连接层使用：

- 初始化 
  - `Zero`, `Normal`, `MSRA` (`HE`), `Xavier`
  - 保存初始化值
  - 加载初始化值
- `Pre_Update` - 预更新
- `Update` - 更新
- `Save` - 保存训练结果值
- `Load` - 加载训练结果值

#### DataReader

样本数据读取器：

- `ReadData` - 从文件中读取数据
- `NormalizeX` - 归一化样本值
- `NormalizeY` - 归一化标签值
- `GetBatchSamples` - 获得批数据
- `ToOneHot` - 标签值变成OneHot编码用于多分类
- `ToZeroOne` - 标签值变成0/1编码用于二分类
- `Shuffle` - 打乱样本顺序

从中派生出两个数据读取器：

- `MnistImageDataReader` - 读取MNIST数据
- `CifarImageReader` - 读取Cifar10数据


### 搭建模型

``` 完成拟合任务的抽象模型```

<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/14/ch09_net.png" />

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

1. 输入层1个神经元，因为只有一个`x`值
2. 隐层4个神经元，对于此问题来说应该是足够了，因为特征很少
3. 输出层1个神经元，因为是拟合任务
4. 学习率=0.5
5. 最大`epoch=10000`轮
6. 批量样本数=10
7. 拟合网络类型
8. Xavier初始化
9. 绝对损失停止条件=0.001


### 训练结果   

```训练过程中损失函数值和准确率的变化```
<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/14/ch09_loss.png" />

```拟合结果```
<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/14/ch09_result.png" />

#### 数据字段解读

- id：唯一id
- date：售出日期
- price：售出价格（标签值）
- bedrooms：卧室数量
- bathrooms：浴室数量
- sqft_living：居住面积
- sqft_lot：停车场面积
- floors：楼层数
- waterfront：泳池
- view：有多少次看房记录
- condition：房屋状况
- grade：评级
- sqft_above：地面上的面积
- sqft_basement：地下室的面积
- yr_built：建筑年份
- yr_renovated：翻修年份
- zipcode：邮政编码
- lat：维度
- long：经度
- sqft_living15：2015年翻修后的居住面积
- sqft_lot15：2015年翻修后的停车场面积

``完成房价预测任务的抽象模型``
<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/14/non_linear_regression.png" />

```Python
def model():
    dr = LoadData()

    num_input = dr.num_feature
    num_hidden1 = 32
    num_hidden2 = 16
    num_hidden3 = 8
    num_hidden4 = 4
    num_output = 1

    max_epoch = 1000
    batch_size = 16
    learning_rate = 0.1

    params = HyperParameters_4_0(
        learning_rate, max_epoch, batch_size,
        net_type=NetType.Fitting,
        init_method=InitialMethod.Xavier,
        stopper=Stopper(StopCondition.StopDiff, 1e-7))

    net = NeuralNet_4_0(params, "HouseSingle")

    fc1 = FcLayer_1_0(num_input, num_hidden1, params)
    net.add_layer(fc1, "fc1")
    r1 = ActivationLayer(Relu())
    net.add_layer(r1, "r1")
    ......
    fc5 = FcLayer_1_0(num_hidden4, num_output, params)
    net.add_layer(fc5, "fc5")

    net.train(dr, checkpoint=10, need_test=True)
    
    output = net.inference(dr.XTest)
    real_output = dr.DeNormalizeY(output)
    mse = np.sum((dr.YTestRaw - real_output)**2)/dr.YTest.shape[0]/10000
    print("mse=", mse)
    
    net.ShowLossHistory()

    ShowResult(net, dr)
```

1. 学习率=0.1
2. 最大`epoch=1000`
3. 批大小=16
4. 拟合网络
5. 初始化方法Xavier
6. 停止条件为相对误差`1e-7`

``训练过程中损失函数值和准确率的变化``
<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/14/house_loss.png" />

## 二分类任务功能测试

``完成非线性二分类教学案例的抽象模型``
<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/14/ch10_net.png" />

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

1. 输入层神经元数为2
2. 隐层的神经元数为3，使用Sigmoid激活函数
3. 由于是二分类任务，所以输出层只有一个神经元，用Logistic做二分类函数
4. 最多训练1000轮
5. 批大小=5
6. 学习率=0.1
7. 绝对误差停止条件=0.02


``训练过程中损失函数值和准确率的变化``
<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/14/ch10_loss.png" />

``分类效果``
<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/14/ch10_result.png" ch="500" />

## 模型
``完成非线性多分类教学案例的抽象模型``
<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/14/ch11_net_sigmoid.png" />

```Python
def model_sigmoid(num_input, num_hidden, num_output, hp):
    net = NeuralNet_4_0(hp, "chinabank_sigmoid")

    fc1 = FcLayer_1_0(num_input, num_hidden, hp)
    net.add_layer(fc1, "fc1")
    s1 = ActivationLayer(Sigmoid())
    net.add_layer(s1, "Sigmoid1")

    fc2 = FcLayer_1_0(num_hidden, num_output, hp)
    net.add_layer(fc2, "fc2")
    softmax1 = ClassificationLayer(Softmax())
    net.add_layer(softmax1, "softmax1")

    net.train(dataReader, checkpoint=50, need_test=True)
    net.ShowLossHistory()
    
    ShowResult(net, hp.toString())
    ShowData(dataReader)
```

1. 隐层8个神经元
2. 最大`epoch=5000`
3. 批大小=10
4. 学习率0.1
5. 绝对误差停止条件=0.08
6. 多分类网络类型
7. 初始化方法为Xavier

`net.train()`函数是一个阻塞函数，只有当训练完毕后才返回。

### 运行结果
``训练过程中损失函数值和准确率的变化``
<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/14/ch11_loss_sigmoid.png" />

``分类效果图``
<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/14/ch11_result_sigmoid.png" ch="500" />

#### 模型
``使用ReLU函数抽象模型``
<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/14/ch11_net_relu.png" />

```Python
def model_relu(num_input, num_hidden, num_output, hp):
    net = NeuralNet_4_0(hp, "chinabank_relu")

    fc1 = FcLayer_1_0(num_input, num_hidden, hp)
    net.add_layer(fc1, "fc1")
    r1 = ActivationLayer(Relu())
    net.add_layer(r1, "Relu1")

    fc2 = FcLayer_1_0(num_hidden, num_hidden, hp)
    net.add_layer(fc2, "fc2")
    r2 = ActivationLayer(Relu())
    net.add_layer(r2, "Relu2")

    fc3 = FcLayer_1_0(num_hidden, num_output, hp)
    net.add_layer(fc3, "fc3")
    softmax = ClassificationLayer(Softmax())
    net.add_layer(softmax, "softmax")

    net.train(dataReader, checkpoint=50, need_test=True)
    net.ShowLossHistory()
    
    ShowResult(net, hp.toString())
    ShowData(dataReader)    
```

1. 隐层8个神经元
2. 最大`epoch=5000`
3. 批大小=10
4. 学习率0.1
5. 绝对误差停止条件=0.08
6. 多分类网络类型
7. 初始化方法为MSRA

### 运行结果
``训练过程中损失函数值和准确率的变化``
<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/14/ch11_loss_relu.png" />

``分类效果图``
<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/14/ch11_result_relu.png" ch="500" />

使用不同的激活函数的分类结果比较

|Sigmoid|ReLU|
|---|---|
|<img src='../Images/14/ch11_result_sigmoid.png'/>|<img src='../Images/14/ch11_result_relu.png'/>|

Relu能直则直，对方形边界适用；Sigmoid能弯则弯，对圆形边界适用。


# 网络优化

随着网络的加深，训练变得越来越困难，时间越来越长，原因可能是：

- 参数多
- 数据量大
- 梯度消失
- 损失函数坡度平缓

为了解决上面这些问题，科学家们在深入研究网络表现的前提下，发现在下面这些方向上经过一些努力，可以给深度网络的训练带来或多或少的改善：

- 权重矩阵初始化
- 批量归一化
- 梯度下降优化算法
- 自适应学习率算法

###  零初始化
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

### 15.1.2 标准初始化

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

``标准初始化在Sigmoid激活函数上的表现``
<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/15/init_normal_sigmoid.png" ch="500" />

###  Xavier初始化方法
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

``Xavier初始化在Sigmoid激活函数上的表现``
<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/15/init_xavier_sigmoid.png" ch="500" />

随机初始化和Xavier初始化的各层激活值与反向传播梯度比较

| |各层的激活值|各层的反向传播梯度|
|---|---|---|
| 随机初始化 |<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images\Images\15\forward_activation1.png"><br/>激活值分布渐渐集中|<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images\Images\15\backward_activation1.png"><br/>反向传播力度逐层衰退|
| Xavier初始化 |<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images\Images\15\forward_activation2.png"><br/>激活值分布均匀|<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images\Images\15\backward_activation2.png"><br/>反向传播力度保持不变|

``Xavier初始化在ReLU激活函数上的表现``
<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/15/init_xavier_relu.png" ch="500" />

###  MSRA初始化方法

MSRA初始化方法$^{[2]}$，又叫做He方法，因为作者姓何

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

``MSRA初始化在ReLU激活函数上的表现``
<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/15/init_msra_relu.png" ch="500" />

几种初始化方法的应用场景

|ID|网络深度|初始化方法|激活函数|说明|
|---|---|---|---|---|
|1|单层|零初始化|无|可以|
|2|双层|零初始化|Sigmoid|错误，不能进行正确的反向传播|
|3|双层|随机初始化|Sigmoid|可以|
|4|多层|随机初始化|Sigmoid|激活值分布成凹形，不利于反向传播|
|5|多层|Xavier初始化|Tanh|正确|
|6|多层|Xavier初始化|ReLU|激活值分布偏向0，不利于反向传播|
|7|多层|MSRA初始化|ReLU|正确|

### 随机梯度下降 SGD

``随机梯度下降算法的梯度搜索轨迹示意图``
<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/15/sgd_algorithm.png" />

学习率对SGD的影响

|学习率|损失函数与准确率|
|---|---|
|0.1|<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images\Images\15\op_sgd_ch09_loss_01.png">|
|0.3|<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images\Images\15\op_sgd_ch09_loss_03.png">|


###  动量算法 Momentum

``动量算法的前进方向``
<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/15/momentum_algorithm.png" />

#### 输入和参数
- $\eta$ - 全局学习率
- $\alpha$ - 动量参数，一般取值为0.5, 0.9, 0.99
- $v_t$ - 当前时刻的动量，初值为0

#### 实际效果

  SGD和动量法的比较

|算法|损失函数和准确率|
|---|---|
|SGD|<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images\Images\15\op_sgd_ch09_loss_01.png">|
|Momentum|<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images\Images\15\op_momentum_ch09_loss_01.png">|

###  梯度加速算法 NAG
Nesterov Accelerated Gradient，或者叫做Nesterov Momentum。
在小球向下滚动的过程中，我们希望小球能够提前知道在哪些地方坡面会上升，这样在遇到上升坡面之前，小球就开始减速。这方法就是Nesterov Momentum，其在凸优化中有较强的理论保证收敛。并且，在实践中Nesterov Momentum也比单纯的Momentum 的效果好。

#### 输入和参数

- $\eta$ - 全局学习率
- $\alpha$ - 动量参数，缺省取值0.9
- $v$ - 动量，初始值为0

动量法和NAG法的比较

|算法|损失函数和准确率|
|---|---|
|Momentum|<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images\Images\15\op_momentum_ch09_loss_01.png">|
|NAG|<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images\Images\15\op_nag_ch09_loss_01.png">|


###  AdaGrad

Adaptive subgradient method.$^{[1]}$

AdaGrad是一个基于梯度的优化算法，它的主要功能是：它对不同的参数调整学习率，具体而言，对低频出现的参数进行大的更新，对高频出现的参数进行小的更新。因此，他很适合于处理稀疏数据。

在这之前，我们对于所有的参数使用相同的学习率进行更新。但 Adagrad 则不然，对不同的训练迭代次数t，AdaGrad 对每个参数都有一个不同的学习率。这里开方、除法和乘法的运算都是按元素运算的。这些按元素运算使得目标函数自变量中每个元素都分别拥有自己的学习率。

AdaGrad算法的学习率设置

|初始学习率|损失函数值变化|
|---|---|
|eta=0.3|<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images\Images\15\op_adagrad_ch09_loss_03.png">|
|eta=0.5|<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images\Images\15\op_adagrad_ch09_loss_05.png">|
|eta=0.7|<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images\Images\15\op_adagrad_ch09_loss_07.png">|

### AdaDelta

Adaptive Learning Rate Method. $^{[2]}$

AdaDelta法是AdaGrad 法的一个延伸，它旨在解决它学习率不断单调下降的问题。相比计算之前所有梯度值的平方和，AdaDelta法仅计算在一个大小为w的时间区间内梯度值的累积和。

但该方法并不会存储之前梯度的平方值，而是将梯度值累积值按如下的方式递归地定义：关于过去梯度值的衰减均值，当前时间的梯度均值是基于过去梯度均值和当前梯度值平方的加权平均，其中是类似上述动量项的权值。

#### 输入和参数

- $\epsilon$ - 用于数值稳定的小常数，建议缺省值为1e-5
- $\alpha \in [0,1)$ - 衰减速率，建议0.9
- $s$ - 累积变量，初始值0
- $r$ - 累积变量变化量，初始为0


AdaDelta法的学习率设置

|初始学习率|损失函数值|
|---|---|
|eta=0.1|<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images\Images\15\op_adadelta_ch09_loss_01.png">|
|eta=0.01|<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images\Images\15\op_adadelta_ch09_loss_001.png">|

### 均方根反向传播 RMSProp

Root Mean Square Prop。$^{[3]}$

RMSprop 是由 Geoff Hinton 在他 Coursera 课程中提出的一种适应性学习率方法，至今仍未被公开发表。RMSprop法要解决AdaGrad的学习率缩减问题。


#### 输入和参数

- $\eta$ - 全局学习率，建议设置为0.001
- $\epsilon$ - 用于数值稳定的小常数，建议缺省值为1e-8
- $\alpha$ - 衰减速率，建议缺省取值0.9
- $r$ - 累积变量矩阵，与$\theta$尺寸相同，初始化为0


RMSProp的学习率设置

|初始学习率|损失函数值|
|---|---|
|eta=0.1|<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images\Images\15\op_rmsprop_ch09_loss_01.png">|<
||迭代了10000次，损失值一直在0.005下不来，说明初始学习率太高了，需要给一个小一些的初值|
|eta=0.01|<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images\Images\15\op_rmsprop_ch09_loss_001.png">|
||合适的学习率初值设置||
|eta=0.005|<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images\Images\15\op_rmsprop_ch09_loss_0005.png">|


###  Adam - Adaptive Moment Estimation

计算每个参数的自适应学习率，相当于RMSProp + Momentum的效果，Adam$^{[4]}$算法在RMSProp算法基础上对小批量随机梯度也做了指数加权移动平均。和AdaGrad算法、RMSProp算法以及AdaDelta算法一样，目标函数自变量中每个元素都分别拥有自己的学习率。

#### 输入和参数

- $t$ - 当前迭代次数
- $\eta$ - 全局学习率，建议缺省值为0.001
- $\epsilon$ - 用于数值稳定的小常数，建议缺省值为1e-8
- $\beta_1, \beta_2$ - 矩估计的指数衰减速率，$\in[0,1)$，建议缺省值分别为0.9和0.999


Adam法的学习率设置

|初始学习率|损失函数值|
|---|---|
|eta=0.1|<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images\Images\15\op_adam_ch09_loss_01.png">|
||迭代了10000次，但是损失值没有降下来，因为初始学习率0.1太高了|
|eta=0.01|<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images\Images\15\op_adam_ch09_loss_001.png">|
||比较合适的学习率|
|eta=0.005|<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images\Images\15\op_adam_ch09_loss_0005.png">|
||学习率较低|
|eta=0.001|<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images\Images\15\op_adam_ch09_loss_0001.png">|
||初始学习率太低，收敛到目标损失值的速度慢|


###  模拟效果比较

``不同梯度下降优化算法的模拟比较``
<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/15/Optimizers_sample.png" ch="500" />

- SGD算法，每次迭代完全受当前梯度的控制，所以会以折线方式前进。
- Momentum算法，学习率只有0.1，每次继承上一次的动量方向，所以会以比较平滑的曲线方式前进，不会出现突然的转向。
- RMSProp算法，有历史梯度值参与做指数加权平均，所以可以看到比较平缓，不会波动太大，都后期步长越来越短也是符合学习规律的。
- Adam算法，因为可以被理解为Momentum和RMSProp的组合，所以比Momentum要平缓一些，比RMSProp要平滑一些。

### 真实效果比较

用`Python`代码实现的前向计算、反向计算、损失函数计算的函数：

```Python
def ForwardCalculationBatch(W,B,batch_x):
    Z = np.dot(W, batch_x) + B
    return Z

def BackPropagationBatch(batch_x, batch_y, batch_z):
    m = batch_x.shape[1]
    dZ = batch_z - batch_y
    dB = dZ.sum(axis=1, keepdims=True)/m
    dW = np.dot(dZ, batch_x.T)/m
    return dW, dB

def CheckLoss(W, B, X, Y):
    m = X.shape[1]
    Z = np.dot(W, X) + B
    LOSS = (Z - Y)**2
    loss = LOSS.sum()/m/2
    return loss
```

各种算法的效果比较

|||
|---|---|
|<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images\Images\15\op_sgd_ch04.png">|<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images\Images\15\op_sgd2_ch04.png">|
|SGD当学习率为0.1时，需要很多次迭代才能逐渐向中心靠近|SGD当学习率为0.5时，会比较快速地向中心靠近，但是在中心的附近有较大震荡|
|<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images\Images\15\op_momentum_ch04.png">|<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images\Images\15\op_nag_ch04.png">|
|Momentum由于惯性存在，一下子越过了中心点，但是很快就会得到纠正|Nag是Momentum的改进，有预判方向功能|
|<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images\Images\15\op_adagrad_ch04.png">|<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images\Images\15\op_adadelta_ch04.png">|
|AdaGrad的学习率在开始时可以设置大一些，因为会很快衰减|AdaDelta即使把学习率设置为0，也不会影响，因为有内置的学习率策略|
|<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images\Images\15\op_rmsprop_ch04.png">|<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images\Images\15\op_adam_ch04.png">|
|RMSProp解决AdaGrad的学习率下降问题，即使学习率设置为0.1，收敛也会快|Adam到达中点的路径比较直接|

``放大后各优化器的训练轨迹``
<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/15/Optimizers_zoom.png" ch="500" />

- SGD：接近中点的过程很曲折，步伐很慢，甚至有反方向的，容易陷入局部最优。
- Momentum：快速接近中点，但中间跳跃较大。
- RMSProp：接近中点很曲折，但是没有反方向的，用的步数比SGD少，跳动较大，有可能摆脱局部最优解的。
- Adam：快速接近中点，难怪很多人喜欢用这个优化器。


##  批量归一化的原理
###  深度神经网络的挑战

在深度神经网络中，我们可以将每一层视为对输入的信号做了一次变换：

$$
Z = W \cdot X + B \tag{5}
$$

``标准正态分布的数值密度占比``
<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/15/bn2.png" ch="500" />

``偏移后的数据分布区域和Sigmoid激活函数``
<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/15/bn3.png" ch="500" />

``ReLU函数曲线``
<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/15/bn4.png" ch="500" />

###  批量归一化

``数据处理过程``
<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/15/bn6.png" ch="500" />

###  前向计算
各个参数的含义和数据形状 

|符号|数据类型|数据形状|
|:---------:|:-----------:|:---------:|
|$X$| 输入数据矩阵 | [m, n] |
|$x_i$|输入数据第i个样本| [1, n] |
|$N$| 经过归一化的数据矩阵 | [m, n] |
|$n_i$| 经过归一化的单样本 | [1, n] |
|$\mu_B$| 批数据均值 | [1, n] |
|$\sigma^2_B$| 批数据方差 | [1, n] |
|$m$|批样本数量| [1] |
|$\gamma$|线性变换参数| [1, n] |
|$\beta$|线性变换参数| [1, n] |
|$Z$|线性变换后的矩阵| [1, n] |
|$z_i$|线性变换后的单样本| [1, n] |
|$\delta$| 反向传入的误差 | [m, n] |

#  正则化
正则化的英文为Regularization，用于防止过拟合。

###  拟合程度比较
``回归任务中的欠拟合、正确的拟合、过拟合``
<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/16/fitting.png" />

``分类任务中的欠拟合、正确的拟合、过拟合``
<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/16/classification.png" />

出现过拟合的原因：

1. 训练集的数量和模型的复杂度不匹配，样本数量级小于模型的参数
2. 训练集和测试集的特征分布不一致
3. 样本噪音大，使得神经网络学习到了噪音，正常样本的行为被抑制
4. 迭代次数过多，过分拟合了训练数据，包括噪音部分和一些非重要特征

既然模型过于复杂，那么我们简化模型不就行了吗？为什么要用复杂度不匹配的模型呢？有两个原因：

1. 因为有的模型以及非常成熟了，比如VGG16，可以不调参而直接用于你自己的数据训练，此时如果你的数据数量不够多，但是又想使用现有模型，就需要给模型加正则项了。
2. 使用相对复杂的模型，可以比较快速地使得网络训练收敛，以节省时间。


### 16.1.3 偏差-方差分解

 符号含义

|符号|含义|
|---|---|
|$x$|测试样本|
|$D$|数据集|
|$y$|x的真实标记|
|$y_D$|x在数据集中标记(可能有误差)|
|$f$|从数据集D学习的模型|
|$f_{x;D}$|从数据集D学习的模型对x的预测输出|
|$f_x$|模型f对x的期望预测输出|

学习算法期望的预测：
$$f_x=E[f_{x;D}] \tag{1}$$
不同的训练集/验证集产生的预测方差：
$$var(x)=E[(f_{x;D}-f_x)^2] \tag{2}$$
噪声：
$$\epsilon^2=E[(y_D-y)^2] \tag{3}$$
期望输出与真实标记的偏差：
$$bias^2(x)=(f_x-y)^2 \tag{4}$$
算法的期望泛化误差：

$$
\begin{aligned}
E(f;D)&=E[(f_{x;D}-y_D)^2]=E[(f_{x;D}-f_x+f_x-y_D)^2] \\\\
&=E[(f_{x;D}-f_x)^2]+E[(f_x-y_D)^2]+E[2(f_{x;D}-f_x)(f_x-y_D)]=E[(f_{x;D}-f_x)^2]+E[(f_x-y_D)^2] \\\\
&=E[(f_{x;D}-f_x)^2]+E[(f_x-y+y-y_D)^2]=E[(f_{x;D}-f_x)^2]+E[(f_x-y)^2]+E(y-y_D)^2]+E[2(f_x-y)(y-y_D)] \\\\
&=E[(f_{x;D}-f_x)^2]+(f_x-y)^2+E[(y-y_D)^2]=var(x) + bias^2(x) + \epsilon^2
\end{aligned}
$$

``训练过程中的偏差和方差变化``
<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/16/error.png" width="600" ch="500" />

###  理论基础

``损失函数值的等高线图``
<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/16/regular0.png" />

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

在`TrainingTrace`类中，增加以下成员以支持早停机制：

- `early_stop`：True表示激活早停机制判断
- `patience`：忍耐次数上限，缺省值为5次
- `patience_counter`：忍耐次数计数器
- `last_vld_loss`：到目前为止最小的验证集损失值

```Python
class TrainingTrace(object):
    def __init__(self, need_earlyStop = False, patience = 5):
        ......
        # for early stop
        self.early_stop = need_earlyStop
        self.patience = patience
        self.patience_counter = 0
        self.last_vld_loss = float("inf")

    def Add(self, epoch, total_iteration, loss_train, accuracy_train, loss_vld, accuracy_vld):
        ......
        if self.early_stop:
            if loss_vld < self.last_vld_loss:
                self.patience_counter = 0
                self.last_vld_loss = loss_vld
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.patience:
                    return True     # need to stop
            # end if
        return False
```


##  集成学习 Ensemble Learning

``集成学习的示意图``
<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/16/ensemble.png" ch="500" />

#### Individual Learner 个体学习器

如果所有的个体学习器都是同一类型的学习器，即同质模式，比如都用神经网路，称为“基学习器”（base learner），相应的学习算法称为“基学习算法”（base learning algorithm）。

在传统的机器学习中，个体学习器可以是不同的，比如用决策树、支持向量机等，此时称为异质模式。

#### Aggregator 结合模块

个体学习器的输出，通过一定的结合策略，在结合模块中有机结合在一起，可以形成一个能力较强的学习器，所以有时称为强学习器，而相应地称个体学习器为弱学习器。

###  Bagging法集成学习的基本流程

``Bagging集成学习示意图``
<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/16/bagging.png" />

1. 首先是数据集的使用，采用自助采样法（Bootstrap Sampling）。假设原始数据集Training Set中有1000个样本，我们从中随机取一个样本的拷贝放到Training Set-1中，此样本不会从原始数据集中被删除，原始数据集中还有1000个样本，而不是999个，这样下次再随机取样本时，此样本还有可能被再次选到。如此重复m次（此例m=1000），我们可以生成Training Set-1。一共重复N次（此例N=9），可以得到N个数据集。
2. 然后搭建一个神经网络模型，可以参数相同。在N个数据集上，训练出N个模型来。
3. 最后再进入Aggregator。N值不能太小，否则无法提供差异化的模型，也不能太大而带来训练模型的时间花销，一般来说取5到10就能满足要求。


###  生成数据集

```Python
def GenerateDataSet(count=9):
    mdr = MnistImageDataReader(train_image_file, train_label_file, test_image_file, test_label_file, "vector")
    mdr.ReadLessData(1000)
    
    for i in range(count):
        X = np.zeros_like(mdr.XTrainRaw)
        Y = np.zeros_like(mdr.YTrainRaw)
        list = np.random.choice(1000,1000)
        k=0
        for j in list:
            X[k] = mdr.XTrainRaw[j]
            Y[k] = mdr.YTrainRaw[j]
            k = k+1
        # end for
        np.savez("level6_" + str(i)+".npz", data=X, label=Y)
    # end for
```
在上面的代码中，我们假设只有1000个手写数据样本，用`np.random.choice(1000,1000)`函数来可重复地选取1000个数字，分别取出对应的图像数据X和标签数据Y，命名为`level6_N.npz`，`N=[1,9]`，保存到9个`npz`文件中。

假设数据集中有m个样本，这样的采样方法，某个样本在第一次被选择的概率是1/m，那么不被选择的概率就是1-1/m，则选择m次后，不被采样到的概率是$(1-\frac{1}{m})^m$，取极限值：

$$\lim_{m \rightarrow \infty} (1-\frac{1}{m})^m \simeq \frac{1}{e} = 0.368$$

即，对于一个新生成的训练数据集来说，原始数据集中的样本没有被使用的概率为36.8%。