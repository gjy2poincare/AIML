# 第14章 搭建深度神经网络框架

## 14.0 深度神经网络框架设计

### 14.0.1 抽象与设计

图14-1是迷你框架的模块化设计，下面对各个模块做功能点上的解释。

<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/14/class.png" />

图14-1 迷你框架设计

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

## 14.1 回归任务功能测试

### 14.1.1 搭建模型

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
### 14.1.2 训练结果

<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/14/ch09_loss.png" />

图14-3 训练过程中损失函数值和准确率的变化

如图14-3所示，损失函数值在一段平缓期过后，开始陡降，这种现象在神经网络的训练中是常见的，最有可能的是当时处于一个梯度变化的平缓地带，算法在艰难地寻找下坡路，然后忽然就找到了。这种情况同时也带来一个弊端：我们会经常遇到缓坡，到底要不要还继续训练？是不是再坚持一会儿就能找到出路呢？抑或是模型能力不够，永远找不到出路呢？这个问题没有准确答案，只能靠试验和经验了。

<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/14/ch09_result.png" />

图14-4 拟合结果

图14-4左侧子图是拟合的情况，绿色点是测试集数据，红色点是神经网路的推理结果，可以看到除了最左侧开始的部分，其它部分都拟合的不错。

## 14.2 二分类任务功能测试

### 14.2.1 搭建模型

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

### 14.2.2 运行结果

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

图14-9 分类效果
## 14.3 多分类功能测试

## 14.3.1 搭建模型一

#### 模型

使用Sigmoid做为激活函数的两层网络，如图14-12。

<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/14/ch11_net_sigmoid.png" />

图14-12 完成非线性多分类教学案例的抽象模型

#### 代码

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
#### 运行结果

训练过程如图14-13所示，分类效果如图14-14所示。

<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/14/ch11_loss_sigmoid.png" />

图14-13 训练过程中损失函数值和准确率的变化

<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/14/ch11_result_sigmoid.png" ch="500" />

图14-14 分类效果图

### 14.3.2 搭建模型二

#### 模型

使用ReLU做为激活函数的三层网络，如图14-15。

<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/14/ch11_net_relu.png" />

图14-15 使用ReLU函数抽象模型

用两层网络也可以实现，但是使用ReLE函数时，训练效果不是很稳定，用三层比较保险。

#### 代码

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

#### 运行结果

训练过程如图14-16所示，分类效果如图14-17所示。

<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/14/ch11_loss_relu.png" />

图14-16 训练过程中损失函数值和准确率的变化

<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/14/ch11_result_relu.png" ch="500" />

图14-17 分类效果图

### 14.3.3 比较

表14-1比较一下使用不同的激活函数的分类效果图。

表14-1 使用不同的激活函数的分类结果比较

|Sigmoid|ReLU|
|---|---|
|<img src='../Images/14/ch11_result_sigmoid.png'/>|<img src='../Images/14/ch11_result_relu.png'/>|

用一句简单的话来描述二者的差别：Relu能直则直，对方形边界适用；Sigmoid能弯则弯，对圆形边界适用。

# 第15章 网络优化

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

## 15.1 权重矩阵初始化

### 15.1.1 零初始化

即把所有层的`W`值的初始值都设置为0。

$$
W = 0
$$

但是对于多层网络来说，绝对不能用零初始化，否则权重值不能学习到合理的结果。

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

当目标问题较为简单时，网络深度不大，所以用标准初始化就可以了。但是当使用深度网络时，会遇到如图15-1所示的问题。

<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/15/init_normal_sigmoid.png" ch="500" />

图15-1 标准初始化在Sigmoid激活函数上的表现
### 15.1.3 Xavier初始化方法

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

其中的W为权重矩阵，N表示正态分布（Normal Distribution），U表示均匀分布（Uniform Distribution)。

图15-2是深度为6层的网络中的表现情况，可以看到，后面几层的激活函数输出值的分布仍然基本符合正态分布，利于神经网络的学习。

<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/15/init_xavier_sigmoid.png" ch="500" />

图15-2 Xavier初始化在Sigmoid激活函数上的表现

表15-1 随机初始化和Xavier初始化的各层激活值与反向传播梯度比较

| |各层的激活值|各层的反向传播梯度|
|---|---|---|
| 随机初始化 |<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images\Images\15\forward_activation1.png"><br/>激活值分布渐渐集中|<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images\Images\15\backward_activation1.png"><br/>反向传播力度逐层衰退|
| Xavier初始化 |<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images\Images\15\forward_activation2.png"><br/>激活值分布均匀|<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images\Images\15\backward_activation2.png"><br/>反向传播力度保持不变|

## 15.2 梯度下降优化算法
### 15.2.1 随机梯度下降 SGD

#### 输入和参数

- $\eta$ - 全局学习率

#### 算法

---

计算梯度：$g_t = \nabla_\theta J(\theta_{t-1})$

更新参数：$\theta_t = \theta_{t-1}  - \eta \cdot g_t$

---
#### 实际效果

表15-3 学习率对SGD的影响

|学习率|损失函数与准确率|
|---|---|
|0.1|<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images\Images\15\op_sgd_ch09_loss_01.png">|
|0.3|<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images\Images\15\op_sgd_ch09_loss_03.png">|

### 15.2.2 动量算法 Momentum
#### 输入和参数

- $\eta$ - 全局学习率
- $\alpha$ - 动量参数，一般取值为0.5, 0.9, 0.99
- $v_t$ - 当前时刻的动量，初值为0
  
#### 算法

---

计算梯度：$g_t = \nabla_\theta J(\theta_{t-1})$

计算速度更新：$v_t = \alpha \cdot v_{t-1} + \eta \cdot g_t$ (公式1)
 
更新参数：$\theta_t = \theta_{t-1}  - v_t$ (公式2)

---
#### 实际效果

表15-4 SGD和动量法的比较

|算法|损失函数和准确率|
|---|---|
|SGD|<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images\Images\15\op_sgd_ch09_loss_01.png">|
|Momentum|<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images\Images\15\op_momentum_ch09_loss_01.png">|

## 15.3 自适应学习率算法

### 15.3.1 AdaGrad

它的主要功能是：它对不同的参数调整学习率，具体而言，对低频出现的参数进行大的更新，对高频出现的参数进行小的更新。因此，他很适合于处理稀疏数据。

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
#### 实际效果

表15-6 AdaGrad算法的学习率设置

|初始学习率|损失函数值变化|
|---|---|
|eta=0.3|<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images\Images\15\op_adagrad_ch09_loss_03.png">|
|eta=0.5|<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images\Images\15\op_adagrad_ch09_loss_05.png">|
|eta=0.7|<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images\Images\15\op_adagrad_ch09_loss_07.png">|

### 15.3.2 AdaDelta

Adaptive Learning Rate Method. $^{[2]}$

AdaDelta法是AdaGrad 法的一个延伸，它旨在解决它学习率不断单调下降的问题。
#### 输入和参数

- $\epsilon$ - 用于数值稳定的小常数，建议缺省值为1e-5
- $\alpha \in [0,1)$ - 衰减速率，建议0.9
- $s$ - 累积变量，初始值0
- $r$ - 累积变量变化量，初始为0
 
#### 算法

---

计算梯度：$g_t = \nabla_\theta J(\theta_{t-1})$

累积平方梯度：$s_t = \alpha \cdot s_{t-1} + (1-\alpha) \cdot g_t \odot g_t$

计算梯度更新：$\Delta \theta = \sqrt{r_{t-1} + \epsilon \over s_t + \epsilon} \odot g_t$

更新梯度：$\theta_t = \theta_{t-1} - \Delta \theta$

更新变化量：$r = \alpha \cdot r_{t-1} + (1-\alpha) \cdot \Delta \theta \odot \Delta \theta$

---

#### 实际效果

表15-7 AdaDelta法的学习率设置

|初始学习率|损失函数值|
|---|---|
|eta=0.1|<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images\Images\15\op_adadelta_ch09_loss_01.png">|
|eta=0.01|<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images\Images\15\op_adadelta_ch09_loss_001.png">|
### 15.3.3 均方根反向传播 RMSProp

Root Mean Square Prop。$^{[3]}$
#### 输入和参数

- $\eta$ - 全局学习率，建议设置为0.001
- $\epsilon$ - 用于数值稳定的小常数，建议缺省值为1e-8
- $\alpha$ - 衰减速率，建议缺省取值0.9
- $r$ - 累积变量矩阵，与$\theta$尺寸相同，初始化为0
  
#### 算法

---

计算梯度：$g_t = \nabla_\theta J(\theta_{t-1})$

累计平方梯度：$r = \alpha \cdot r + (1-\alpha)(g_t \odot g_t)$

计算梯度更新：$\Delta \theta = {\eta \over \sqrt{r + \epsilon}} \odot g_t$

更新参数：$\theta_{t}=\theta_{t-1} - \Delta \theta$

---
## 15.4 算法在等高线图上的效果比较

我们依次测试4种方法：

- 普通SGD, 学习率0.95
- 动量Momentum, 学习率0.1
- RMPSProp，学习率0.5
- Adam，学习率0.5

每种方法都迭代20次，记录下每次反向过程的(x,y)坐标点，绘制图15-8如下。

<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/15/Optimizers_sample.png" ch="500" />

图15-8 不同梯度下降优化算法的模拟比较

- SGD算法，每次迭代完全受当前梯度的控制，所以会以折线方式前进。
- Momentum算法，学习率只有0.1，每次继承上一次的动量方向，所以会以比较平滑的曲线方式前进，不会出现突然的转向。
- RMSProp算法，有历史梯度值参与做指数加权平均，所以可以看到比较平缓，不会波动太大，都后期步长越来越短也是符合学习规律的。
- Adam算法，因为可以被理解为Momentum和RMSProp的组合，所以比Momentum要平缓一些，比RMSProp要平滑一些。

## 15.5 批量归一化的原理
### 15.5.1 批量归一化

既然可以把原始训练样本做归一化，那么如果在深度神经网络的每一层，都可以有类似的手段，也就是说把层之间传递的数据移到0点附近，那么训练效果就应该会很理想。这就是批归一化BN的想法的来源。

具体的数据处理过程如图15-14所示。

<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/15/bn6.png" ch="500" />
### 15.5.2 批量归一化的优点

1. 可以选择比较大的初始学习率，让你的训练速度提高。
   
    以前还需要慢慢调整学习率，甚至在网络训练到一定程度时，还需要想着学习率进一步调小的比例选择多少比较合适，现在我们可以采用初始很大的学习率，因为这个算法收敛很快。当然这个算法即使你选择了较小的学习率，也比以前的收敛速度快，因为它具有快速训练收敛的特性；

2. 减少对初始化的依赖
   
    一个不太幸运的初始化，可能会造成网络训练实际很长，甚至不收敛。

3. 减少对正则的依赖
   
   学习正则化知识，以增强网络的泛化能力。采用BN算法后，我们会逐步减少对正则的依赖，比如令人头疼的dropout、L2正则项参数的选择问题，或者可以选择更小的L2正则约束参数了，因为BN具有提高网络泛化能力的特性；

# 第16章 正则化

正则化的英文为Regularization，用于防止过拟合。

## 16.0 过拟合

### 16.0.1 拟合程度比较
神经网络的两大功能：回归和分类。这两类任务，都会出现欠拟合和过拟合现象，如图16-1和16-2所示。

<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/16/fitting.png" />

图16-1 回归任务中的欠拟合、正确的拟合、过拟合

图16-1是回归任务中的三种情况，依次为：欠拟合、正确的拟合、过拟合。

<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/16/classification.png" />

图16-2 分类任务中的欠拟合、正确的拟合、过拟合

图16-2是分类任务中的三种情况，依次为：分类欠妥、正确的分类、分类过度。由于分类可以看作是对分类边界的拟合，所以经常也统称其为拟合。
出现过拟合的原因：

1. 训练集的数量和模型的复杂度不匹配，样本数量级小于模型的参数
2. 训练集和测试集的特征分布不一致
3. 样本噪音大，使得神经网络学习到了噪音，正常样本的行为被抑制
4. 迭代次数过多，过分拟合了训练数据，包括噪音部分和一些非重要特征
### 16.0.2 解决过拟合问题

如何解决过拟合问题：

1. 数据扩展
2. 正则
3. 丢弃法
4. 早停法
5. 集成学习法
6. 特征工程（属于传统机器学习范畴，不在此处讨论）
7. 简化模型，减小网络的宽度和深度

## 16.1 偏差与方差
## 16.1.1 直观的解释

先用一个直观的例子来理解偏差和方差。比如打靶，如图16-9所示。

<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/16/variance_bias.png" width="600" ch="500" />

图16-9 打靶中的偏差和方差

总结一下，不同偏差和方差反映的射手的特点如表16-1所示。

表16-1 不同偏差和方差的射手特点

||低偏差|高偏差|
|---|---|---|
|低方差|射手很稳，枪的准星也很准。|射手很稳，但是枪的准星有问题，所有子弹都固定地偏向一侧。|
|高方差|射手不太稳，但枪的准星没问题，虽然弹着点分布很散，但没有整体偏移。|射手不稳，而且枪的准星也有问题，弹着点分布很散且有规律地偏向一侧。|
### 16.1.2 偏差-方差分解
表16-3 符号含义

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

所以，各个项的含义是：

- 偏差：度量了学习算法的期望与真实结果的偏离程度，即学习算法的拟合能力。
- 方差：训练集与验证集的差异造成的模型表现的差异。
- 噪声：当前数据集上任何算法所能到达的泛化误差的下线，即学习问题本身的难度。

想当然地，我们希望偏差与方差越小越好，但实际并非如此。一般来说，偏差与方差是有冲突的，称为偏差-方差窘境 (bias-variance dilemma)。
## 16.2 L2正则
### 16.2.1 L2正则化

假设：

- W参数服从高斯分布，即：$w_j \sim N(0,\tau^2)$
- Y服从高斯分布，即：$y_i \sim N(w^Tx_i,\sigma^2)$

贝叶斯最大后验估计：

$$
\arg\max_wL(w) = \ln \prod_i^n \frac{1}{\sigma\sqrt{2 \pi}}\exp(-(\frac{y_i-w^Tx_i}{\sigma})^2/2) \cdot \prod_j^m{\frac{1}{\tau\sqrt{2\pi}}\exp(-(\frac{w_j}{\tau})^2/2)}
$$

$$
=-\frac{1}{2\sigma^2}\sum_i^n(y_i-w^Tx_i)^2-\frac{1}{2\tau^2}\sum_j^m{w_j^2}-n\ln\sigma\sqrt{2\pi}-m\ln \tau\sqrt{2\pi} \tag{3}
$$

因为$\sigma,b,n,\pi,m$等都是常数，所以损失函数$J(w)$的最小值可以简化为：

$$
\arg\min_wJ(w) = \sum_i^n(y_i-w^Tx_i)^2+\lambda\sum_j^m{w_j^2} \tag{4}
$$
<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/16/regular2.png" ch="500" />

图16-13 L2正则区与损失函数等高线示意图

黄色的圆形，就是正则项所处的区域。这个区域的大小，是由参数$\lambda$所控制的，该值越大，黄色圆形区域越小，对w的惩罚力度越大（距离椭圆中心越远）。比如图16-13中分别标出了该值为0.7、0.8、0.9的情况。

## 16.3 L1正则
### 16.3.1 L1正则化

假设：

- W参数服从拉普拉斯分布，即$w_j \sim Laplace(0,b)$
- Y服从高斯分布，即$y_i \sim N(w^Tx_i,\sigma^2)$

贝叶斯最大后验估计：
$$
\begin{aligned}
\arg\max_wL(w) = &\ln \prod_i^n \frac{1}{\sigma\sqrt{2 \pi}}\exp(-\frac{1}{2}(\frac{y_i-w^Tx_i}{\sigma})^2) 
\cdot \prod_j^m{\frac{1}{2b}\exp(-\frac{\lvert w_j \rvert}{b})}
\\\\
=&-\frac{1}{2\sigma^2}\sum_i^n(y_i-w^Tx_i)^2-\frac{1}{2b}\sum_j^m{\lvert w_j \rvert}
-n\ln\sigma\sqrt{2\pi}-m\ln b\sqrt{2\pi} 
\end{aligned}
\tag{1}
$$

因为$\sigma,b,n,\pi,m$等都是常数，所以损失函数$J(w)$的最小值可以简化为：

$$
\arg\min_wJ(w) = \sum_i^n(y_i-w^Tx_i)^2+\lambda\sum_j^m{\lvert w_j \rvert} \tag{2}
$$

我们仍以两个参数为例，公式2的后半部分的正则形式为：

$$L_1 = \lvert w_1 \rvert + \lvert w_2 \rvert \tag{3}$$

因为$w_1,w_2$有可能是正数或者负数，我们令$x=|w_1|,y=|w_2|,c=L_1$，则公式3可以拆成以下4个公式的组合：

$$
y=-x+c \quad (当w_1 \gt 0, w_2 \gt 0时)
$$
$$
y=\quad x+c \quad (当w_1 \lt 0, w_2 \gt 0时)
$$
$$
y=\quad x-c \quad (当w_1 \gt 0, w_2 \lt 0时)
$$
$$
y=-x-c \quad (当w_1 \lt 0, w_2 \lt 0时)
$$

所以上述4个公式（4条直线）会组成一个二维平面上的一个菱形。

图16-17中三个菱形，是因为惩罚因子的数值不同而形成的，越大的话，菱形面积越小，惩罚越厉害。

<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/16/regular1.png" ch="500" />

图16-17 L1正则区与损失函数等高线示意图

以最大的那个菱形区域为例，它与损失函数等高线有多个交点，都可以作为此问题的解，但是其中红色顶点是损失函数值最小的，因此它是最优解。
## 16.4 早停法 Early Stopping

### 16.4.1 算法

一般的做法是，在训练的过程中，记录到目前为止最好的validation 准确率，当连续N次Epoch（比如N=10或者更多次）没达到最佳准确率时，则可以认为准确率不再提高了。此时便可以停止迭代了（Early Stopping）。这种策略也称为“No-improvement-in-N”，N即Epoch的次数，可以根据实际情况取，如10、20、30……

### 16.4.2 实现

首先，在`TrainingTrace`类中，增加以下成员以支持早停机制：

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
最后的拟合效果如图16-23所示。

<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/16/EarlyStop_sin_result.png" ch="500" />

图16-23 拟合后的曲线与训练数据的分布图

蓝点是样本，绿点是理想的拟合效果，红线是实际的拟合效果。

## 16.5 丢弃法 Dropout
Dropout可以作为训练深度神经网络的一种正则方法供选择。在每个训练批次中，通过忽略一部分的神经元（让其隐层节点值为0），可以明显地减少过拟合现象。这种方式可以减少隐层节点间的相互作用，高层的神经元需要低层的神经元的输出才能发挥作用，如果高层神经元过分依赖某个低层神经元，就会有过拟合发生。在一次正向/反向的过程中，通过随机丢弃一些神经元，迫使高层神经元和其它的一些低层神经元协同工作，可以有效地防止神经元因为接收到过多的同类型参数而陷入过拟合的状态，来提高泛化程度。
#### 代码实现

```Python
class DropoutLayer(CLayer):
    def __init__(self, input_size, ratio=0.5):
        self.dropout_ratio = ratio
        self.mask = None
        self.input_size = input_size
        self.output_size = input_size

    def forward(self, input, train=True):
        assert(input.ndim == 2)
        if train:
            self.mask = np.random.rand(*input.shape) > self.dropout_ratio
            self.z = input * self.mask
        else:
            self.z = input * (1.0 - self.dropout_ratio)

        return self.z
       
    def backward(self, delta_in, idx):
        delta_out = self.mask * delta_in
        return delta_out
```
```Python
def Model_Dropout(dataReader, num_input, num_hidden, num_output, params):
    net = NeuralNet41(params, "overfitting")

    fc1 = FcLayer(num_input, num_hidden, params)
    net.add_layer(fc1, "fc1")
    s1 = ActivatorLayer(Sigmoid())
    net.add_layer(s1, "s1")
    
    d1 = DropoutLayer(num_hidden, 0.1)
    net.add_layer(d1, "d1")

    fc2 = FcLayer(num_hidden, num_hidden, params)
    net.add_layer(fc2, "fc2")
    t2 = ActivatorLayer(Tanh())
    net.add_layer(t2, "t2")

    #d2 = DropoutLayer(num_hidden, 0.2)
    #net.add_layer(d2, "d2")

    fc3 = FcLayer(num_hidden, num_hidden, params)
    net.add_layer(fc3, "fc3")
    t3 = ActivatorLayer(Tanh())
    net.add_layer(t3, "t3")

    d3 = DropoutLayer(num_hidden, 0.2)
    net.add_layer(d3, "d3")
    
    fc4 = FcLayer(num_hidden, num_output, params)
    net.add_layer(fc4, "fc4")

    net.train(dataReader, checkpoint=100, need_test=True)
    net.ShowLossHistory(XCoordinate.Epoch)
    
    return net
``` 

运行程序，最后可以得到这样的损失函数图和验证结果，如图16-28所示。

<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/16/dropout_sin_loss.png" />

图16-28 训练过程中损失函数值和准确率的变化曲线

可以提高精确率到98.17%。

拟合效果如图16-29所示。

<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/16/dropout_sin_result.png" ch="500" />

图16-29 拟合后的曲线与训练数据的分布图
## 16.6 数据增强 Data Augmentation
### 16.6.1 图像数据增强

#### 旋转
定义图片中心和旋转角度，进行微小的旋转。

<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/16/data_rotate.png" />

图16-30 原始图片与旋转后的图片

#### 缩放

<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/16/data_stretch.png" ch="500" />
#### 平移和添加噪音

<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/16/data_translate.png" ch="500" />

## 16.7 集成学习 Ensemble Learning

### 16.7.1 集成学习的概念

当数据集有问题，或者网络学习能力不足，或准确度不够时，我们可以采取集成学习的方法，来提升性能。说得通俗一些，就是发挥团队的智慧，根据团队中不同背景、不同能力的成员的独立意见，通过某种决策方法来解决一个问题。所以集成学习也称为多分类器系统(multi-classifier system)、基于委员会的学习(committee-based learning)等。

图16-36是一个简单的集成学习的示意图。

<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/16/ensemble.png" ch="500" />

图16-36 集成学习的示意图
图中有两个组件：

#### Individual Learner 个体学习器

如果所有的个体学习器都是同一类型的学习器，即同质模式，比如都用神经网路，称为“基学习器”（base learner），相应的学习算法称为“基学习算法”（base learning algorithm）。

在传统的机器学习中，个体学习器可以是不同的，比如用决策树、支持向量机等，此时称为异质模式。

#### Aggregator 结合模块

个体学习器的输出，通过一定的结合策略，在结合模块中有机结合在一起，可以形成一个能力较强的学习器，所以有时称为强学习器，而相应地称个体学习器为弱学习器。

#### 学习心得
学习了深度神经网络，包括如何搭建深度神经网络框架，回归任务、二分类任务和多分类任务。网络优化的一些算法，包括权重矩阵初始化、梯度下降优化算法、自适应学习率算法，批量归一化，通过算法比较，知道了每种算法的优劣和适应场景。还学习了偏差与方差、L2正则，L1正则，早停法、丢弃法等各种方法，通过简单的实例和实际应用更加深入的理解各部分知识，每一部分知识都需要有前面学过的，高等数学，概率论等知识为基础才能更好理解各个算法思想。