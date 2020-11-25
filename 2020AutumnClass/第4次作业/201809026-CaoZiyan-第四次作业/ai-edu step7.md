# 十四、 搭建深度神经网络框架

1.深度神经网络框架设计

> 迷你框架的模块化设计

<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/14/class.png" />

> NeuralNet类,来包装基本的神经网络结构和功能：

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

> Layer是一个抽象类，以及更加需要增加的实际类，包括：

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

> Activator Layer

激活函数和分类函数：

- `Identity` - 直传函数，即没有激活处理
- `Sigmoid`
- `Tanh`
- `Relu`

> Classification Layer

分类函数，包括：

- `Sigmoid`二分类
- `Softmax`多分类


> Parameters

 基本神经网络运行参数：

 - 学习率
 - 最大`epoch`
 - `batch size`
 - 损失函数定义
 - 初始化方法
 - 优化器类型
 - 停止条件
 - 正则类型和条件

> LossFunction

损失函数及帮助方法：

- 均方差函数
- 交叉熵函数二分类
- 交叉熵函数多分类
- 记录损失函数
- 显示损失函数历史记录
- 获得最小函数值时的权重参数

> Optimizer

优化器：

- `SGD`
- `Momentum`
- `Nag`
- `AdaGrad`
- `AdaDelta`
- `RMSProp`
- `Adam`

> WeightsBias

权重矩阵，仅供全连接层使用：

- 初始化 
  - `Zero`, `Normal`, `MSRA` (`HE`), `Xavier`
  - 保存初始化值
  - 加载初始化值
- `Pre_Update` - 预更新
- `Update` - 更新
- `Save` - 保存训练结果值
- `Load` - 加载训练结果值

> DataReader

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

2.二分类任务功能测试
> 搭建模型

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

超参数说明：

1. 输入层神经元数为2
2. 隐层的神经元数为3，使用Sigmoid激活函数
3. 由于是二分类任务，所以输出层只有一个神经元，用Logistic做二分类函数
4. 最多训练1000轮
5. 批大小=5
6. 学习率=0.1
7. 绝对误差停止条件=0.02

> 运行结果

<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/14/ch10_loss.png" />

上图为训练过程中损失函数值和准确率的变化，再看下面的打印输出结果：

```
......
epoch=419, total_iteration=30239
loss_train=0.010094, accuracy_train=1.000000
loss_valid=0.019141, accuracy_valid=1.000000
time used: 2.149379253387451
testing...
1.0
```
最后的testing...的结果是1.0，表示100%正确，这初步说明mini框架在这个基本case上工作得很好。下图所示的分类效果也不错。

<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/14/ch10_result.png" ch="500" />

3.多分类功能测试

> 模型（Sigmoid做为激活函数）

使用Sigmoid做为激活函数的两层网络，如下图。

<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/14/ch11_net_sigmoid.png" />

> 代码

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

> 超参数说明

1. 隐层8个神经元
2. 最大`epoch=5000`
3. 批大小=10
4. 学习率0.1
5. 绝对误差停止条件=0.08
6. 多分类网络类型
7. 初始化方法为Xavier

`net.train()`函数是一个阻塞函数，只有当训练完毕后才返回。

> 运行结果
> 
<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/14/ch11_loss_sigmoid.png" />

上图为 训练过程中损失函数值和准确率的变化 
下图为 分类效果图

<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/14/ch11_result_sigmoid.png" ch="500" />




表14-1比较一下使用不同的激活函数的分类效果图。

表14-1 使用不同的激活函数的分类结果比较

|Sigmoid|ReLU|
|---|---|
|<img src='../Images/14/ch11_result_sigmoid.png'/>|<img src='../Images/14/ch11_result_relu.png'/>|

可以看到左图中的边界要平滑许多，这也就是ReLU和Sigmoid的区别，ReLU是用分段线性拟合曲线，Sigmoid有真正的曲线拟合能力。但是Sigmoid也有缺点，看分类的边界，使用ReLU函数的分类边界比较清晰，而使用Sigmoid函数的分类边界要平缓一些，过渡区较宽。

用一句简单的话来描述二者的差别：Relu能直则直，对方形边界适用；Sigmoid能弯则弯，对圆形边界适用。

> 模型(ReLU做为激活函数)

使用ReLU做为激活函数的三层网络，如下图。

<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/14/ch11_net_relu.png" />

用两层网络也可以实现，但是使用ReLE函数时，训练效果不是很稳定，用三层比较保险。

> 代码

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

> 超参数说明

1. 隐层8个神经元
2. 最大`epoch=5000`
3. 批大小=10
4. 学习率0.1
5. 绝对误差停止条件=0.08
6. 多分类网络类型
7. 初始化方法为MSRA

> 运行结果


<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/14/ch11_loss_relu.png" />

上图为 训练过程中损失函数值和准确率的变化
下图为 分类效果图

<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/14/ch11_result_relu.png" ch="500" />


- 比较

比较一下使用不同的激活函数的分类效果图如下表。

|Sigmoid|ReLU|
|---|---|
|<img src='../Images/14/ch11_result_sigmoid.png'/>|<img src='../Images/14/ch11_result_relu.png'/>|

可以看到左图中的边界要平滑许多，这也就是ReLU和Sigmoid的区别，ReLU是用分段线性拟合曲线，Sigmoid有真正的曲线拟合能力。但是Sigmoid也有缺点，看分类的边界，使用ReLU函数的分类边界比较清晰，而使用Sigmoid函数的分类边界要平缓一些，过渡区较宽。

用一句简单的话来描述二者的差别：Relu能直则直，对方形边界适用；Sigmoid能弯则弯，对圆形边界适用。

# 第十五章、网络优化

1.随着网络的加深，训练变得越来越困难，时间越来越长，原因可能是：

- 参数多
- 数据量大
- 梯度消失
- 损失函数坡度平缓

为了解决上面这些问题，科学家们在深入研究网络表现的前提下，发现在下面这些方向上经过一些努力，可以给深度网络的训练带来或多或少的改善：

- 权重矩阵初始化
- 批量归一化
- 梯度下降优化算法
- 自适应学习率算法

2.标准初始化

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

当目标问题较为简单时，网络深度不大，所以用标准初始化就可以了。但是当使用深度网络时，会遇到如下图所示的问题。

<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/15/init_normal_sigmoid.png" ch="500" />

上图是一个6层的深度网络，使用全连接层+Sigmoid激活函数，图中表示的是各层激活函数的直方图。可以看到各层的激活值严重向两侧[0,1]靠近，从Sigmoid的函数曲线可以知道这些值的导数趋近于0，反向传播时的梯度逐步消失。处于中间地段的值比较少，对参数学习非常不利。

3.Xavier初始化方法

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

其中的W为权重矩阵，N表示正态分布（Normal Distribution），U表示均匀分布（Uniform Distribution)。下同。

假设激活函数关于0对称，且主要针对于全连接神经网络。适用于tanh和softsign。

即权重矩阵参数应该满足在该区间内的均匀分布。其中的W是权重矩阵，U是Uniform分布，即均匀分布。

Xavier初始化方法比直接用高斯分布进行初始化W的优势所在： 

一般的神经网络在前向传播时神经元输出值的方差会不断增大，而使用Xavier等方法理论上可以保证每层神经元输入输出方差一致。 

下图是深度为6层的网络中的表现情况，可以看到，后面几层的激活函数输出值的分布仍然基本符合正态分布，利于神经网络的学习。

<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/15/init_xavier_sigmoid.png" ch="500" />


随机初始化和Xavier初始化的各层激活值与反向传播梯度比较

| |各层的激活值|各层的反向传播梯度|
|---|---|---|
| 随机初始化 |<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images\Images\15\forward_activation1.png"><br/>激活值分布渐渐集中|<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images\Images\15\backward_activation1.png"><br/>反向传播力度逐层衰退|
| Xavier初始化 |<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images\Images\15\forward_activation2.png"><br/>激活值分布均匀|<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images\Images\15\backward_activation2.png"><br/>反向传播力度保持不变|

但是，随着深度学习的发展，人们觉得Sigmoid的反向力度受限，又发明了ReLU激活函数。下图显示了Xavier初始化在ReLU激活函数上的表现。

<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/15/init_xavier_relu.png" ch="500" />

可以看到，随着层的加深，使用ReLU时激活值逐步向0偏向，同样会导致梯度消失问题。于是He Kaiming等人研究出了MSRA初始化法，又叫做He初始化法。

4. MSRA初始化方法（又叫做He方法，因为作者姓何。）

条件：正向传播时，状态值的方差保持不变；反向传播时，关于激活值的梯度的方差保持不变。

网络初始化是一件很重要的事情。但是，传统的固定方差的高斯分布初始化，在网络变深的时候使得模型很难收敛。VGG团队是这样处理初始化的问题的：他们首先训练了一个8层的网络，然后用这个网络再去初始化更深的网络。

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

下图中的激活值从0到1的分布，在各层都非常均匀，不会由于层的加深而梯度消失，所以，在使用ReLU时，推荐使用MSRA法初始化。

<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/15/init_msra_relu.png" ch="500" />


对于Leaky ReLU：

$$
W \sim N \begin{bmatrix} 0, \sqrt{\frac{2}{(1+\alpha^2) \hat n_i}} \end{bmatrix}
\\\\ \hat n_i = h_i \cdot w_i \cdot d_i
\\\\ h_i: 卷积核高度，w_i: 卷积核宽度，d_i: 卷积核个数
$$

5.几种初始化方法的应用场景

|ID|网络深度|初始化方法|激活函数|说明|
|---|---|---|---|---|
|1|单层|零初始化|无|可以|
|2|双层|零初始化|Sigmoid|错误，不能进行正确的反向传播|
|3|双层|随机初始化|Sigmoid|可以|
|4|多层|随机初始化|Sigmoid|激活值分布成凹形，不利于反向传播|
|5|多层|Xavier初始化|Tanh|正确|
|6|多层|Xavier初始化|ReLU|激活值分布偏向0，不利于反向传播|
|7|多层|MSRA初始化|ReLU|正确|

6.随机梯度下降 SGD

先回忆一下随机梯度下降的基本算法，便于和后面的各种算法比较。下图中的梯度搜索轨迹为示意图。

<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/15/sgd_algorithm.png" />

- 输入和参数

- $\eta$ - 全局学习率

- 算法

---

计算梯度：$g_t = \nabla_\theta J(\theta_{t-1})$

更新参数：$\theta_t = \theta_{t-1}  - \eta \cdot g_t$

---

- 实际效果

学习率对SGD的影响

|学习率|损失函数与准确率|
|---|---|
|0.1|<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images\Images\15\op_sgd_ch09_loss_01.png">|
|0.3|<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images\Images\15\op_sgd_ch09_loss_03.png">|

SGD的另外一个缺点就是收敛速度慢，见表15-3，在学习率为0.1时，训练10000个epoch不能收敛到预定损失值；学习率为0.3时，训练5000个epoch可以收敛到预定水平。

7. 动量算法 Momentum

- 动量算法的前进方向

<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/15/momentum_algorithm.png" />

上图中，第一次的梯度更新完毕后，会记录$v_1$的动量值。在“求梯度点”进行第二次梯度检查时，得到2号方向，与$v_1$的动量组合后，最终的更新为2'方向。这样一来，由于有$v_1$的存在，会迫使梯度更新方向具备“惯性”，从而可以减小随机样本造成的震荡。

- 输入和参数

- $\eta$ - 全局学习率
- $\alpha$ - 动量参数，一般取值为0.5, 0.9, 0.99
- $v_t$ - 当前时刻的动量，初值为0
  
- 算法

---

计算梯度：$g_t = \nabla_\theta J(\theta_{t-1})$

计算速度更新：$v_t = \alpha \cdot v_{t-1} + \eta \cdot g_t$ (公式1)
 
更新参数：$\theta_t = \theta_{t-1}  - v_t$ (公式2)

---

- 实际效果

 SGD和动量法的比较

|算法|损失函数和准确率|
|---|---|
|SGD|<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images\Images\15\op_sgd_ch09_loss_01.png">|
|Momentum|<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images\Images\15\op_momentum_ch09_loss_01.png">|

从表的比较可以看到，使用同等的超参数设置，普通梯度下降算法经过epoch=10000次没有到达预定0.001的损失值；动量算法经过2000个epoch迭代结束。

在损失函数历史数据图中，中间有一大段比较平坦的区域，梯度值很小，或者是随机梯度下降算法找不到合适的方向前进，只能慢慢搜索。而下侧的动量法，利用惯性，判断当前梯度与上次梯度的关系，如果方向相同，则会加速前进；如果不同，则会减速，并趋向平衡。所以很快地就达到了停止条件。

当我们将一个小球从山上滚下来时，没有阻力的话，它的动量会越来越大，但是如果遇到了阻力，速度就会变小。加入的这一项，可以使得梯度方向不变的维度上速度变快，梯度方向有所改变的维度上的更新速度变慢，这样就可以加快收敛并减小震荡。

8. 梯度加速算法 NAG（Nesterov Accelerated Gradient / Nesterov Momentum）

- 输入和参数

- $\eta$ - 全局学习率
- $\alpha$ - 动量参数，缺省取值0.9
- $v$ - 动量，初始值为0
  
- 算法

---

临时更新：$\hat \theta = \theta_{t-1} - \alpha \cdot v_{t-1}$

前向计算：$f(\hat \theta)$

计算梯度：$g_t = \nabla_{\hat\theta} J(\hat \theta)$

计算速度更新：$v_t = \alpha \cdot v_{t-1} + \eta \cdot g_t$

更新参数：$\theta_t = \theta_{t-1}  - v_t$

---

- 实际效果

动量法和NAG法的比较

|算法|损失函数和准确率|
|---|---|
|Momentum|<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images\Images\15\op_momentum_ch09_loss_01.png">|
|NAG|<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images\Images\15\op_nag_ch09_loss_01.png">|

表显示，使用动量算法经过2000个epoch迭代结束，NAG算法是加速的动量法，因此只用1400个epoch迭代结束。

NAG 可以使 RNN 在很多任务上有更好的表现。
