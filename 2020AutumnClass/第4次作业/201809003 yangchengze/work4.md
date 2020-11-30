## 1 深度神经网络框架设计

### 1 功能/模式分析

比较第12章中的三层神经网络的代码，我们可以看到大量的重复之处，比如前向计算中：

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

因为三层网络比两层网络多了一层，所以会在初始化、前向、反向、更新参数等四个环节有所不同，但却是有规律的。再加上前面章节中，为了实现一些辅助功能，我们已经写了很多类。所以，现在可以动手搭建一个深度学习的迷你框架了。

### 2 抽象与设计

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
## 回归任务功能测试

在第九章中，我们用一个两层的神经网络，验证了万能近似定理。当时是用hard code方式写的，现在我们用迷你框架来搭建一下。

### 1 搭建模型

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

超参数说明：

1. 输入层1个神经元，因为只有一个`x`值
2. 隐层4个神经元，对于此问题来说应该是足够了，因为特征很少
3. 输出层1个神经元，因为是拟合任务
4. 学习率=0.5
5. 最大`epoch=10000`轮
6. 批量样本数=10
7. 拟合网络类型
8. Xavier初始化
9. 绝对损失停止条件=0.001

### 2 训练结果

<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/14/ch09_loss.png" />

图14-3 训练过程中损失函数值和准确率的变化

如图14-3所示，损失函数值在一段平缓期过后，开始陡降，这种现象在神经网络的训练中是常见的，最有可能的是当时处于一个梯度变化的平缓地带，算法在艰难地寻找下坡路，然后忽然就找到了。这种情况同时也带来一个弊端：我们会经常遇到缓坡，到底要不要还继续训练？是不是再坚持一会儿就能找到出路呢？抑或是模型能力不够，永远找不到出路呢？这个问题没有准确答案，只能靠试验和经验了。

<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/14/ch09_result.png" />

图14-4 拟合结果

图14-4左侧子图是拟合的情况，绿色点是测试集数据，红色点是神经网路的推理结果，可以看到除了最左侧开始的部分，其它部分都拟合的不错。注意，这里我们不是在讨论过拟合、欠拟合的问题，我们在这个章节的目的就是更好地拟合一条曲线。

图14-4右侧的子图是用下面的代码生成的：

```Python
    y_test_real = net.inference(dr.XTest)
    axes.scatter(y_test_real, y_test_real-dr.YTestRaw, marker='o')
```

以测试集的真实值为横坐标，以真实值和预测值的差为纵坐标。最理想的情况是所有点都在y=0处排成一条横线。从图上看，真实值和预测值二者的差异明显，但是请注意横坐标和纵坐标的间距相差一个数量级，所以差距其实不大。

再看打印输出的最后部分：

```
epoch=4999, total_iteration=449999
loss_train=0.000920, accuracy_train=0.968329
loss_valid=0.000839, accuracy_valid=0.962375
time used: 28.002626419067383
save parameters
total weights abs sum= 45.27530164993504
total weights = 8
little weights = 0
zero weights = 0
testing...
0.9817814550687021
0.9817814550687021
```

由于我们设置了`eps=0.001`，所以在5000多个`epoch`时便达到了要求，训练停止。最后用测试集得到的准确率为98.17%，已经非常不错了。如果训练更多的轮，可以得到更好的结果。


#### 数据处理

原始数据只有一个数据集，所以需要我们自己把它分成训练集和测试集，比例大概为4:1。此数据集为`csv`文件格式，为了方便，我们把它转换成了两个扩展名为`npz`的`numpy`压缩形式：

- `house_Train.npz`，训练数据集
- `house_Test.npz`，测试数据集

#### 加载数据

与上面第一个例子的代码相似，但是房屋数据属性繁杂，所以需要做归一化，房屋价格也是至少6位数，所以也需要做归一化。

这里有个需要注意的地方，即训练集和测试集的数据，需要合并在一起做归一化，然后再分开使用。为什么要先合并呢？假设训练集样本中的房屋面积的范围为150到220，而测试集中的房屋面积有可能是160到230，两者不一致。分别归一化的话，150变成0，160也变成0，这样预测就会产生误差。

最后还需要在训练集中用`GenerateValidaionSet(k=10)`分出一个1:9的验证集。

### 3 搭建模型

在不知道一个问题的实际复杂度之前，我们不妨把模型设计得复杂一些。如下图所示，这个模型包含了四组全连接层-Relu层的组合，最后是一个单输出做拟合。

<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/14/non_linear_regression.png" />

图14-5 完成房价预测任务的抽象模型

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

超参数说明：

1. 学习率=0.1
2. 最大`epoch=1000`
3. 批大小=16
4. 拟合网络
5. 初始化方法Xavier
6. 停止条件为相对误差`1e-7`

net.train()函数是一个阻塞函数，只有当训练完毕后才返回。

在train后面的部分，是用测试集来测试该模型的准确度，使用了数据城堡(Data Castle)的官方评测方法，用均方差除以10000，得到的数字越小越好。一般的模型大概是一个7位数的结果，稍微好一些的是6位数。

### 4 训练结果

<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/14/house_loss.png" />

图14-6 训练过程中损失函数值和准确率的变化

由于标签数据也做了归一化，变换为都是0至1间的小数，所以均方差的数值很小，需要观察小数点以后的第4位。从图14-6中可以看到，损失函数值很快就降到了0.0002以下，然后就很缓慢地下降。而精度值在不断的上升，相信更多的迭代次数会带来更高的精度。

再看下面的打印输出部分，用R2_Score法得到的值为0.841，而用数据城堡官方的评测标准，得到的MSE值为2384411，还比较大，说明模型精度还应该有上升的空间。

```
......
epoch=999, total_iteration=972999
loss_train=0.000079, accuracy_train=0.740406
loss_valid=0.000193, accuracy_valid=0.857289
time used: 193.5549156665802
testing...
0.8412989144927305
mse= 2384411.5840510926
```
##  二分类任务功能测试

### 1 搭建模型

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

超参数说明：

1. 输入层神经元数为2
2. 隐层的神经元数为3，使用Sigmoid激活函数
3. 由于是二分类任务，所以输出层只有一个神经元，用Logistic做二分类函数
4. 最多训练1000轮
5. 批大小=5
6. 学习率=0.1
7. 绝对误差停止条件=0.02

### 2 运行结果

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
##  多分类功能测试

在第11章里，我们讲解了如何使用神经网络做多分类。在本节我们将会用Mini框架重现那个教学案例，然后使用一个真实的案例验证多分类的用法。

### 1 搭建模型一

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

#### 超参数说明

1. 隐层8个神经元
2. 最大`epoch=5000`
3. 批大小=10
4. 学习率0.1
5. 绝对误差停止条件=0.08
6. 多分类网络类型
7. 初始化方法为Xavier

`net.train()`函数是一个阻塞函数，只有当训练完毕后才返回。

#### 运行结果

训练过程如图14-13所示，分类效果如图14-14所示。

<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/14/ch11_loss_sigmoid.png" />

图14-13 训练过程中损失函数值和准确率的变化

<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/14/ch11_result_sigmoid.png" ch="500" />

图14-14 分类效果图

### 2 搭建模型二

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

#### 超参数说明

1. 隐层8个神经元
2. 最大`epoch=5000`
3. 批大小=10
4. 学习率0.1
5. 绝对误差停止条件=0.08
6. 多分类网络类型
7. 初始化方法为MSRA

#### 运行结果

训练过程如图14-16所示，分类效果如图14-17所示。

<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/14/ch11_loss_relu.png" />

图14-16 训练过程中损失函数值和准确率的变化

<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/14/ch11_result_relu.png" ch="500" />

图14-17 分类效果图


##  多分类任务 - MNIST手写体识别

### 1 数据读取

MNIST数据本身是图像格式的，我们用`mode="vector"`去读取，转变成矢量格式。

```Python
def LoadData():
    print("reading data...")
    dr = MnistImageDataReader(mode="vector")
    ......
```

### 2 搭建模型

一共4个隐层，都用ReLU激活函数连接，最后的输出层接Softmax分类函数。

<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/14/mnist_net.png" />

图14-18 完成MNIST分类任务的抽象模型

以下是主要的参数设置：

```Python
if __name__ == '__main__':
    dataReader = LoadData()
    num_feature = dataReader.num_feature
    num_example = dataReader.num_example
    num_input = num_feature
    num_hidden1 = 128
    num_hidden2 = 64
    num_hidden3 = 32
    num_hidden4 = 16
    num_output = 10
    max_epoch = 10
    batch_size = 64
    learning_rate = 0.1

    params = HyperParameters_4_0(
        learning_rate, max_epoch, batch_size,
        net_type=NetType.MultipleClassifier,
        init_method=InitialMethod.MSRA,
        stopper=Stopper(StopCondition.StopLoss, 0.12))

    net = NeuralNet_4_0(params, "MNIST")

    fc1 = FcLayer_1_0(num_input, num_hidden1, params)
    net.add_layer(fc1, "fc1")
    r1 = ActivationLayer(Relu())
    net.add_layer(r1, "r1")
    ......
    fc5 = FcLayer_1_0(num_hidden4, num_output, params)
    net.add_layer(fc5, "fc5")
    softmax = ClassificationLayer(Softmax())
    net.add_layer(softmax, "softmax")

    net.train(dataReader, checkpoint=0.05, need_test=True)
    net.ShowLossHistory(xcoord=XCoordinate.Iteration)
```

### 3 运行结果

我们设计的停止条件是绝对Loss值达到0.12时，所以迭代到6个epoch时，达到了0.119的损失值，就停止训练了。

<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/14/mnist_loss.png" />

图14-19 训练过程中损失函数值和准确率的变化

图14-19是训练过程图示，下面是最后几行的打印输出。

```
......
epoch=6, total_iteration=5763
loss_train=0.005559, accuracy_train=1.000000
loss_valid=0.119701, accuracy_valid=0.971667
time used: 17.500738859176636
save parameters
testing...
0.9697
```

最后用测试集得到的准确率为96.97%。
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
##  权重矩阵初始化

权重矩阵初始化是一个非常重要的环节，是训练神经网络的第一步，选择正确的初始化方法会带了事半功倍的效果。这就好比攀登喜马拉雅山，如果选择从南坡登山，会比从北坡容易很多。而初始化权重矩阵，相当于下山时选择不同的道路，在选择之前并不知道这条路的难易程度，只是知道它可以抵达山下。这种选择是随机的，即使你使用了正确的初始化算法，每次重新初始化时也会给训练结果带来很多影响。

比如第一次初始化时得到权重值为(0.12847，0.36453)，而第二次初始化得到(0.23334，0.24352)，经过试验，第一次初始化用了3000次迭代达到精度为96%的模型，第二次初始化只用了2000次迭代就达到了相同精度。这种情况在实践中是常见的。

### 1 零初始化

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

可以看到`W1`、`B1`、`W2`内部3个单元的值都一样，这是因为初始值都是0，所以梯度均匀回传，导致所有`W`的值都同步更新，没有差别。这样的话，无论多少轮，最终的结果也不会正确。

### 2 标准初始化

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

图15-1是一个6层的深度网络，使用全连接层+Sigmoid激活函数，图中表示的是各层激活函数的直方图。可以看到各层的激活值严重向两侧[0,1]靠近，从Sigmoid的函数曲线可以知道这些值的导数趋近于0，反向传播时的梯度逐步消失。处于中间地段的值比较少，对参数学习非常不利。

### 3 Xavier初始化方法

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

其中的W为权重矩阵，N表示正态分布（Normal Distribution），U表示均匀分布（Uniform Distribution)。下同。

假设激活函数关于0对称，且主要针对于全连接神经网络。适用于tanh和softsign。

即权重矩阵参数应该满足在该区间内的均匀分布。其中的W是权重矩阵，U是Uniform分布，即均匀分布。

论文摘要：神经网络在2006年之前不能很理想地工作，很大原因在于权重矩阵初始化方法上。Sigmoid函数不太适合于深度学习，因为会导致梯度饱和。基于以上原因，我们提出了一种可以快速收敛的参数初始化方法。

Xavier初始化方法比直接用高斯分布进行初始化W的优势所在： 

一般的神经网络在前向传播时神经元输出值的方差会不断增大，而使用Xavier等方法理论上可以保证每层神经元输入输出方差一致。 

图15-2是深度为6层的网络中的表现情况，可以看到，后面几层的激活函数输出值的分布仍然基本符合正态分布，利于神经网络的学习。

<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/15/init_xavier_sigmoid.png" ch="500" />

图15-2 Xavier初始化在Sigmoid激活函数上的表现

表15-1 随机初始化和Xavier初始化的各层激活值与反向传播梯度比较

| |各层的激活值|各层的反向传播梯度|
|---|---|---|
| 随机初始化 |<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images\Images\15\forward_activation1.png"><br/>激活值分布渐渐集中|<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images\Images\15\backward_activation1.png"><br/>反向传播力度逐层衰退|
| Xavier初始化 |<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images\Images\15\forward_activation2.png"><br/>激活值分布均匀|<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images\Images\15\backward_activation2.png"><br/>反向传播力度保持不变|

但是，随着深度学习的发展，人们觉得Sigmoid的反向力度受限，又发明了ReLU激活函数。图15-3显示了Xavier初始化在ReLU激活函数上的表现。

<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/15/init_xavier_relu.png" ch="500" />

图15-3 Xavier初始化在ReLU激活函数上的表现

可以看到，随着层的加深，使用ReLU时激活值逐步向0偏向，同样会导致梯度消失问题。于是He Kaiming等人研究出了MSRA初始化法，又叫做He初始化法。

### 4 MSRA初始化方法

MSRA初始化方法$^{[2]}$，又叫做He方法，因为作者姓何。

条件：正向传播时，状态值的方差保持不变；反向传播时，关于激活值的梯度的方差保持不变。

网络初始化是一件很重要的事情。但是，传统的固定方差的高斯分布初始化，在网络变深的时候使得模型很难收敛。VGG团队是这样处理初始化的问题的：他们首先训练了一个8层的网络，然后用这个网络再去初始化更深的网络。

“Xavier”是一种相对不错的初始化方法，但是，Xavier推导的时候假设激活函数在零点附近是线性的，显然我们目前常用的ReLU和PReLU并不满足这一条件。所以MSRA初始化主要是想解决使用ReLU激活函数后，方差会发生变化，因此初始化权重的方法也应该变化。

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

图15-4中的激活值从0到1的分布，在各层都非常均匀，不会由于层的加深而梯度消失，所以，在使用ReLU时，推荐使用MSRA法初始化。

<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/15/init_msra_relu.png" ch="500" />

图15-4 MSRA初始化在ReLU激活函数上的表现

对于Leaky ReLU：

$$
W \sim N \begin{bmatrix} 0, \sqrt{\frac{2}{(1+\alpha^2) \hat n_i}} \end{bmatrix}
\\\\ \hat n_i = h_i \cdot w_i \cdot d_i
\\\\ h_i: 卷积核高度，w_i: 卷积核宽度，d_i: 卷积核个数
$$
 dW_1$
6. $W_2 = W_1 - v_2 = W_1 - (\alpha v_1 +\eta dW_1) = W_1 - \alpha \cdot \eta \cdot dW_0 - \eta \cdot dW_1$
7. $dW_2 = \nabla J(w)$
8. $v_3=\alpha v_2 + \eta dW_2$
9. $W_3 = W_2 - v_3=W_2-(\alpha v_2 + \eta dW_2) = W_2 - \alpha^2 \eta dW_0 - \alpha \eta dW_1 - \eta dW_2$


根据公式(3)(4)有：

0. $v_0 = 0$
1. $dW_0 = \nabla J(w)$
2. $v_1 = \alpha v_0 - \eta \cdot dW_0 = -\eta \cdot dW_0$
3. $W_1 = W_0 + v_1=W_0 - \eta \cdot dW_0$
4. $dW_1 = \nabla J(w)$
5. $v_2 = \alpha v_1 - \eta dW_1$
6. $W_2 = W_1 + v_2 = W_1 + (\alpha v_1 - \eta dW_1) = W_1 - \alpha \cdot \eta \cdot dW_0 - \eta \cdot dW_1$
7. $dW_2 = \nabla J(w)$
8. $v_3=\alpha v_2 - \eta dW_2$
9. $W_3 = W_2 + v_3=W_2 + (\alpha v_2 - \eta dW_2) = W_2 - \alpha^2 \eta dW_0 - \alpha \eta dW_1-\eta dW_2$

通过手工推导迭代，我们得到两个结论：

1. 可以看到两种方式的第9步结果是相同的，即公式(1)(2)等同于(3)(4)
2. 与普通SGD的算法$W_3 = W_2 - \eta dW_2$相比，动量法不但每次要减去当前梯度，还要减去历史梯度$W_0,W_1$乘以一个不断减弱的因子$\alpha$，因为$\alpha$小于1，所以$\alpha^2$比$\alpha$小，$\alpha^3$比$\alpha^2$小。这种方式的学名叫做指数加权平均。

#### 实际效果

表15-4 SGD和动量法的比较

|算法|损失函数和准确率|
|---|---|
|SGD|<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images\Images\15\op_sgd_ch09_loss_01.png">|
|Momentum|<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images\Images\15\op_momentum_ch09_loss_01.png">|

从表15-4的比较可以看到，使用同等的超参数设置，普通梯度下降算法经过epoch=10000次没有到达预定0.001的损失值；动量算法经过2000个epoch迭代结束。

在损失函数历史数据图中，中间有一大段比较平坦的区域，梯度值很小，或者是随机梯度下降算法找不到合适的方向前进，只能慢慢搜索。而下侧的动量法，利用惯性，判断当前梯度与上次梯度的关系，如果方向相同，则会加速前进；如果不同，则会减速，并趋向平衡。所以很快地就达到了停止条件。

当我们将一个小球从山上滚下来时，没有阻力的话，它的动量会越来越大，但是如果遇到了阻力，速度就会变小。加入的这一项，可以使得梯度方向不变的维度上速度变快，梯度方向有所改变的维度上的更新速度变慢，这样就可以加快收敛并减小震荡。

###  梯度加速算法 NAG

Nesterov Accelerated Gradient，或者叫做Nesterov Momentum。

在小球向下滚动的过程中，我们希望小球能够提前知道在哪些地方坡面会上升，这样在遇到上升坡面之前，小球就开始减速。这方法就是Nesterov Momentum，其在凸优化中有较强的理论保证收敛。并且，在实践中Nesterov Momentum也比单纯的Momentum 的效果好。

#### 输入和参数

- $\eta$ - 全局学习率
- $\alpha$ - 动量参数，缺省取值0.9
- $v$ - 动量，初始值为0
  
#### 算法

---

临时更新：$\hat \theta = \theta_{t-1} - \alpha \cdot v_{t-1}$

前向计算：$f(\hat \theta)$

计算梯度：$g_t = \nabla_{\hat\theta} J(\hat \theta)$

计算速度更新：$v_t = \alpha \cdot v_{t-1} + \eta \cdot g_t$

更新参数：$\theta_t = \theta_{t-1}  - v_t$

---

其核心思想是：注意到 momentum 方法，如果只看 $\alpha \cdot v_{t-1}$ 项，那么当前的θ经过momentum的作用会变成 $\theta - \alpha \cdot v_{t-1}$。既然我们已经知道了下一步的走向，我们不妨先走一步，到达新的位置”展望”未来，然后在新位置上求梯度, 而不是原始的位置。

所以，同Momentum相比，梯度不是根据当前位置θ计算出来的，而是在移动之后的位置$\theta - \alpha \cdot v_{t-1}$计算梯度。理由是，既然已经确定会移动$\theta - \alpha \cdot v_{t-1}$，那不如之前去看移动后的梯度。

图15-7是NAG的前进方向。

<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/15/nag_algorithm.png" ch="500" />

图15-7 梯度加速算法的前进方向

这个改进的目的就是为了提前看到前方的梯度。如果前方的梯度和当前梯度目标一致，那我直接大步迈过去； 如果前方梯度同当前梯度不一致，那我就小心点更新。

#### 实际效果

表15-5 动量法和NAG法的比较

|算法|损失函数和准确率|
|---|---|
|Momentum|<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images\Images\15\op_momentum_ch09_loss_01.png">|
|NAG|<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images\Images\15\op_nag_ch09_loss_01.png">|

表15-9显示，使用动量算法经过2000个epoch迭代结束，NAG算法是加速的动量法，因此只用1400个epoch迭代结束。

NAG 可以使 RNN 在很多任务上有更好的表现。
## 3 自适应学习率算法

### 1 AdaGrad

Adaptive subgradient method.$^{[1]}$

AdaGrad是一个基于梯度的优化算法，它的主要功能是：它对不同的参数调整学习率，具体而言，对低频出现的参数进行大的更新，对高频出现的参数进行小的更新。因此，他很适合于处理稀疏数据。

在这之前，我们对于所有的参数使用相同的学习率进行更新。但 Adagrad 则不然，对不同的训练迭代次数t，AdaGrad 对每个参数都有一个不同的学习率。这里开方、除法和乘法的运算都是按元素运算的。这些按元素运算使得目标函数自变量中每个元素都分别拥有自己的学习率。

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

从AdaGrad算法中可以看出，随着算法不断迭代，$r$会越来越大，整体的学习率会越来越小。所以，一般来说AdaGrad算法一开始是激励收敛，到了后面就慢慢变成惩罚收敛，速度越来越慢。$r$值的变化如下：

0. $r_0 = 0$
1. $r_1=g_1^2$
2. $r_2=g_1^2+g_2^2$
3. $r_3=g_1^2+g_2^2+g_3^2$

在SGD中，随着梯度的增大，我们的学习步长应该是增大的。但是在AdaGrad中，随着梯度$g$的增大，$r$也在逐渐的增大，且在梯度更新时$r$在分母上，也就是整个学习率是减少的，这是为什么呢？

这是因为随着更新次数的增大，我们希望学习率越来越慢。因为我们认为在学习率的最初阶段，我们距离损失函数最优解还很远，随着更新次数的增加，越来越接近最优解，所以学习率也随之变慢。

但是当某个参数梯度较小时，累积和也会小，那么更新速度就大。

经验上已经发现，对于训练深度神经网络模型而言，从训练开始时积累梯度平方会导致有效学习率过早和过量的减小。AdaGrad在某些深度学习模型上效果不错，但不是全部。

#### 实际效果

表15-6 AdaGrad算法的学习率设置

|初始学习率|损失函数值变化|
|---|---|
|eta=0.3|<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images\Images\15\op_adagrad_ch09_loss_03.png">|
|eta=0.5|<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images\Images\15\op_adagrad_ch09_loss_05.png">|
|eta=0.7|<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images\Images\15\op_adagrad_ch09_loss_07.png">|

表15-6表明，我们设定不同的初始学习率，分别为0.3、0.5、0.7，可以看到学习率为0.7时，收敛得最快，只用1750个epoch；学习率为0.5时用了3000个epoch；学习率为0.3时用了8000个epoch。所以，对于AdaGrad来说，可以在开始时把学习率的值设置大一些，因为它会衰减得很快。

### 2 AdaDelta

Adaptive Learning Rate Method. $^{[2]}$

AdaDelta法是AdaGrad 法的一个延伸，它旨在解决它学习率不断单调下降的问题。相比计算之前所有梯度值的平方和，AdaDelta法仅计算在一个大小为w的时间区间内梯度值的累积和。

但该方法并不会存储之前梯度的平方值，而是将梯度值累积值按如下的方式递归地定义：关于过去梯度值的衰减均值，当前时间的梯度均值是基于过去梯度均值和当前梯度值平方的加权平均，其中是类似上述动量项的权值。

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

从表15-7可以看到，初始学习率设置为0.1或者0.01，对于本算法来说都是一样的，这是因为算法中用r来代替学习率。

### 3 均方根反向传播 RMSProp

Root Mean Square Prop。$^{[3]}$

RMSprop 是由 Geoff Hinton 在他 Coursera 课程中提出的一种适应性学习率方法，至今仍未被公开发表。RMSprop法要解决AdaGrad的学习率缩减问题。

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

RMSprop也将学习率除以了一个指数衰减的衰减均值。为了进一步优化损失函数在更新中存在摆动幅度过大的问题，并且进一步加快函数的收敛速度，RMSProp算法对权重$W$和偏置$b$的梯度使用了微分平方加权平均数，这种做法有利于消除了摆动幅度大的方向，用来修正摆动幅度，使得各个维度的摆动幅度都较小。另一方面也使得网络函数收敛更快。

其中，$r$值的变化如下：

0. $r_0 = 0$
1. $r_1=0.1g_1^2$
2. $r_2=0.9r_1+0.1g_2^2=0.09g_1^2+0.1g_2^2$
3. $r_3=0.9r_2+0.1g_3^2=0.081g_1^2+0.09g_2^2+0.1g_3^2$
 
与AdaGrad相比，$r_3$要小很多，那么计算出来的学习率也不会衰减的太厉害。注意，在计算梯度更新时，分母开始时是个小于1的数，而且非常小，所以如果全局学习率设置过大的话，比如0.1，将会造成开始的步子迈得太大，而且久久不能收缩步伐，损失值也降不下来。

#### 实际效果

表15-8 RMSProp的学习率设置

|初始学习率|损失函数值|
|---|---|
|eta=0.1|<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images\Images\15\op_rmsprop_ch09_loss_01.png">|<
||迭代了10000次，损失值一直在0.005下不来，说明初始学习率太高了，需要给一个小一些的初值|
|eta=0.01|<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images\Images\15\op_rmsprop_ch09_loss_001.png">|
||合适的学习率初值设置||
|eta=0.005|<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images\Images\15\op_rmsprop_ch09_loss_0005.png">|
||初值稍微小了些，造成迭代次数增加才能到达精度要求||

从上面的试验可以看出，0.01是本示例最好的设置。


### 4 Adam - Adaptive Moment Estimation

计算每个参数的自适应学习率，相当于RMSProp + Momentum的效果，Adam$^{[4]}$算法在RMSProp算法基础上对小批量随机梯度也做了指数加权移动平均。和AdaGrad算法、RMSProp算法以及AdaDelta算法一样，目标函数自变量中每个元素都分别拥有自己的学习率。

#### 输入和参数

- $t$ - 当前迭代次数
- $\eta$ - 全局学习率，建议缺省值为0.001
- $\epsilon$ - 用于数值稳定的小常数，建议缺省值为1e-8
- $\beta_1, \beta_2$ - 矩估计的指数衰减速率，$\in[0,1)$，建议缺省值分别为0.9和0.999

#### 算法

---

计算梯度：$g_t = \nabla_\theta J(\theta_{t-1})$

计数器加一：$t=t+1$

更新有偏一阶矩估计：$m_t = \beta_1 \cdot m_{t-1} + (1-\beta_1) \cdot g_t$

更新有偏二阶矩估计：$v_t = \beta_2 \cdot v_{t-1} + (1-\beta_2)(g_t \odot g_t)$

修正一阶矩的偏差：$\hat m_t = m_t / (1-\beta_1^t)$

修正二阶矩的偏差：$\hat v_t = v_t / (1-\beta_2^t)$

计算梯度更新：$\Delta \theta = \eta \cdot \hat m_t /(\epsilon + \sqrt{\hat v_t})$

更新参数：$\theta_t=\theta_{t-1} - \Delta \theta$

---

#### 实际效果

表15-9 Adam法的学习率设置

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

由于Adam继承了RMSProp的传统，所以学习率不宜设置太高，从表15-9的比较可以看到，初始学习率设置为0.01时比较理想。
# 第16章 正则化

正则化的英文为Regularization，用于防止过拟合。

## 过拟合

### 1 拟合程度比较

在深度神经网络中，我们遇到的另外一个挑战，就是网络的泛化问题。所谓泛化，就是模型在测试集上的表现要和训练集上一样好。经常有这样的例子：一个模型在训练集上千锤百炼，能到达99%的准确率，拿到测试集上一试，准确率还不到90%。这说明模型过度拟合了训练数据，而不能反映真实世界的情况。解决过度拟合的手段和过程，就叫做泛化。

神经网络的两大功能：回归和分类。这两类任务，都会出现欠拟合和过拟合现象，如图16-1和16-2所示。

<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/16/fitting.png" />

图16-1 回归任务中的欠拟合、正确的拟合、过拟合

图16-1是回归任务中的三种情况，依次为：欠拟合、正确的拟合、过拟合。

<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/16/classification.png" />

图16-2 分类任务中的欠拟合、正确的拟合、过拟合

图16-2是分类任务中的三种情况，依次为：分类欠妥、正确的分类、分类过度。由于分类可以看作是对分类边界的拟合，所以我们经常也统称其为拟合。

上图中对于“深入敌后”的那颗绿色点样本，正确的做法是把它当作噪音看待，而不要让它对网络产生影响。而对于上例中的欠拟合情况，如果简单的（线性）模型不能很好地完成任务，我们可以考虑使用复杂的（非线性或深度）模型，即加深网络的宽度和深度，提高神经网络的能力。

但是如果网络过于宽和深，就会出现第三张图展示的过拟合的情况。

出现过拟合的原因：

1. 训练集的数量和模型的复杂度不匹配，样本数量级小于模型的参数
2. 训练集和测试集的特征分布不一致
3. 样本噪音大，使得神经网络学习到了噪音，正常样本的行为被抑制
4. 迭代次数过多，过分拟合了训练数据，包括噪音部分和一些非重要特征

既然模型过于复杂，那么我们简化模型不就行了吗？为什么要用复杂度不匹配的模型呢？有两个原因：

1. 因为有的模型以及非常成熟了，比如VGG16，可以不调参而直接用于你自己的数据训练，此时如果你的数据数量不够多，但是又想使用现有模型，就需要给模型加正则项了。
2. 使用相对复杂的模型，可以比较快速地使得网络训练收敛，以节省时间。

### 2. 过拟合的例子一

充分理解过拟合的原因之后，我们先制作一个数据集，让其符合上面的第三条：制造样本噪音。但是如何制作一个合理的噪音呢？这让笔者想起了一篇讲解傅里叶变换的文章，一个复合的傅里叶变换公式可以是这样的：

$$
y = \frac{4 \sin (\theta)}{\pi} + \frac{4 \sin (5\theta)}{5\pi} \tag{1}
$$

这个公式可以在$[0,2\pi]$之间制作出图16-3。

<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/16/sin_data.png" ch="500" />

图16-3 公式1的函数图

其中，绿色的点是公式1的第一部分的结果，蓝色的点是整个公式1的结果。我们可以把绿色的点作为测试/验证基线，可以看到它是一条标准的正弦曲线。而蓝色的点作为带噪音的训练样本，该训练样本只有25个数据。

然后我们使用MiniFramework，可以很方便地搭建起下面这个模型，如图16-4。

<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/16/overfitting_net_1.png" />

图16-4 用于拟合公式1的模型结构

这个模型的复杂度要比训练样本级大很多，所以可以重现过拟合的现象，当然还需要设置好合适的参数，代码片段如下：

```Python
def SetParameters():
    num_hidden = 16
    max_epoch = 20000
    batch_size = 5
    learning_rate = 0.1
    eps = 1e-6
    
    hp = HyperParameters41(
        learning_rate, max_epoch, batch_size, eps,        
        net_type=NetType.Fitting,
        init_method=InitialMethod.Xavier, 
        optimizer_name=OptimizerName.SGD)

    return hp, num_hidden
```

我们故意把最大`epoch`次数设置得比较大，以充分展示过拟合效果。训练结束后，首先看损失函数值和精度值的变化曲线，如图16-5所示。

<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/16/overfitting_sin_loss.png" />

图16-5 损失函数值和精度值的变化曲线

蓝色为训练集，红色为验证集。可以看到，训练集上的损失函数值很快降低到极点，精确度很快升高到极点，而验证集上的表现正好相反。说明网络对训练集很适应，但是越来越不适应验证集数据，出现了严重的过拟合。验证集的精确度为0.9605。

再看下图的拟合情况，如图16-6所示。

<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/16/overfitting_sin_result.png" ch="500" />

图16-6 模型的拟合情况

红色拟合曲线严丝合缝地拟合了每一个样本点，也就是说模型学习到了样本的误差。绿色点所组成的曲线，才是我们真正想要的拟合结果。

### 3 过拟合的例子二

我们将要使用MNIST数据集做例子，模拟出令一个过拟合（分类）的情况。从上面的过拟合出现的4点原因分析，第2点和第3点对于MNIST数据集来说并不成立，MNIST数据集有60000个样本，这足以保证它的特征分布的一致性，少数样本的噪音也会被大多数正常的数据所淹没。但是如果我们只选用其中的很少一部分的样本，则特征分布就可能会有偏差，而且独立样本的噪音会变得突出一些。

再看过拟合原因中的第1点和第4点，我们利用第14章中的已有知识和代码，搭建一个复杂网络很容易，而且迭代次数完全可以由代码来控制。

首先，只使用1000个样本来做训练，如下面的代码所示，调用一个`ReadLessData(1000)`函数，并且用`GenerateValidationSet(k=10)`函数把1000个样本分成900和100两部分，分别做为训练集和验证集：

```Python
def LoadData():
    mdr = MnistImageDataReader(train_image_file, train_label_file, test_image_file, test_label_file, "vector")
    mdr.ReadLessData(1000)
    mdr.Normalize()
    mdr.GenerateDevSet(k=10)
    return mdr
```

然后，我们搭建一个深度网络，如图16-7所示。

<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/16/overfit_net.png" />

图16-7 过拟合例子二的深度网络模型结构

这个网络有5个全连接层，前4个全连接层后接ReLU激活函数层，最后一个全连接层接Softmax分类函数做10分类。由于我们在第14章就已经搭建好了深度神经网络的Mini框架，所以可以简单地搭建这个网络，如下代码所示：

```Python
def Net(dateReader, num_input, num_hidden, num_output, params):
    net = NeuralNet(params)

    fc1 = FcLayer(num_input, num_hidden, params)
    net.add_layer(fc1, "fc1")
    relu1 = ActivatorLayer(Relu())
    net.add_layer(relu1, "relu1")

    fc2 = FcLayer(num_hidden, num_hidden, params)
    net.add_layer(fc2, "fc2")
    relu2 = ActivatorLayer(Relu())
    net.add_layer(relu2, "relu2")

    fc3 = FcLayer(num_hidden, num_hidden, params)
    net.add_layer(fc3, "fc3")
    relu3 = ActivatorLayer(Relu())
    net.add_layer(relu3, "relu3")

    fc4 = FcLayer(num_hidden, num_hidden, params)
    net.add_layer(fc4, "fc4")
    relu4 = ActivatorLayer(Relu())
    net.add_layer(relu4, "relu4")

    fc5 = FcLayer(num_hidden, num_output, params)
    net.add_layer(fc5, "fc5")
    softmax = ActivatorLayer(Softmax())
    net.add_layer(softmax, "softmax")

    net.train(dataReader, checkpoint=1)
    
    net.ShowLossHistory()
```

`net.train(dataReader, checkpoint=1)`函数的参数`checkpoint`的含义是，每隔1个`epoch`记录一次训练过程中的损失值和准确率。可以设置成大于1的数字，比如10，意味着每10个`epoch`检查一次。也可以设置为小于1大于0的数比如0.5，假设在一个`epoch`中要迭代100次，则每50次检查一次。

在`main`过程中，设置一些超参数，然后调用刚才建立的`Net`进行训练：

```Python
if __name__ == '__main__':

    dataReader = LoadData()
    num_feature = dataReader.num_feature
    num_example = dataReader.num_example
    num_input = num_feature
    num_hidden = 30
    num_output = 10
    max_epoch = 200
    batch_size = 100
    learning_rate = 0.1
    eps = 1e-5

    params = CParameters(
      learning_rate, max_epoch, batch_size, eps,
      LossFunctionName.CrossEntropy3, 
      InitialMethod.Xavier, 
      OptimizerName.SGD)

    Net(dataReader, num_input, num_hidden, num_hidden, num_hidden, num_hidden, num_output, params)
```

在超参数中，我们指定了：

1. 每个隐层30个神经元（4个隐层在Net函数里指定）
2. 最多训练200个`epoch`
3. 批大小为100个样本
4. 学习率为0.1
5. 多分类交叉熵损失函数(CrossEntropy3)
6. Xavier权重初始化方法
7. 随机梯度下降算法

最终我们可以得到如图16-8所示的训练曲线。

<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/16/overfit_result.png" />

图16-8 过拟合例子二的训练曲线

在训练集上（蓝色曲线），很快就达到了损失函数值趋近于0，准确度100%的程度。而在验证集上（红色曲线），损失函数值却越来越大，准确度也在下降。这就造成了一个典型的过拟合网络，即所谓U型曲线，无论是损失函数值和准确度，都呈现出了这种分化的特征。

我们再看打印输出部分：
```
epoch=199, total_iteration=1799
loss_train=0.0015, accuracy_train=1.000000
loss_valid=0.9956, accuracy_valid=0.860000
time used: 5.082462787628174
total weights abs sum= 1722.470655813152
total weights = 26520
little weights = 2815
zero weights = 27
testing...
rate=8423 / 10000 = 0.8423
```

结果说明：

1. 第199个`epoch`上（从0开始计数，所以实际是第200个`epoch`），训练集的损失为0.0015，准确率为100%。测试集损失值0.9956，准确率86%。过拟合线性很严重。
2. `total weights abs sum = 1722.4706`，实际上是把所有全连接层的权重值先取绝对值，再求和。这个值和下面三个值在后面会有比较说明。
3. `total weights = 26520`，一共26520个权重值，偏移值不算在内。
4. `little weights = 2815`，一共2815个权重值小于0.01。
5. `zero weights = 27`，是权重值中接近于0的数量（小于0.0001）。
6. 测试准确率为84.23%

在着手解决过拟合的问题之前，我们先来学习一下关于偏差与方差的知识，以便得到一些理论上的指导，虽然神经网络是一门实验学科。

### 4 解决过拟合问题

有了直观感受和理论知识，下面我们看看如何解决过拟合问题：

1. 数据扩展
2. 正则
3. 丢弃法
4. 早停法
5. 集成学习法
6. 特征工程（属于传统机器学习范畴，不在此处讨论）
7. 简化模型，减小网络的宽度和深度
##  早停法 Early Stopping

### 1 想法的由来

从图16-20来看，如果我们在第2500次迭代时就停止训练，就应该是验证集的红色曲线的最佳取值位置了，因为此时损失值最小，而准确率值最大。

<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/16/overfitting_sin_loss.png" />

图16-20 训练过程中损失函数值和准确率的变化曲线

这种做法很符合直观感受，因为准确率都不再提高了，损失值反而上升了，再继续训练也是无益的，只会浪费训练的时间。那么该做法的一个重点便是怎样才认为验证集不再提高了呢？并不是说准确率一降下来便认为不再提高了，因为可能在这个Epoch上，准确率降低了，但是随后的Epoch准确率又升高了，所以不能根据一两次的连续降低就判断不再提高。

对模型进行训练的过程即是对模型的参数进行学习更新的过程，这个参数学习的过程往往会用到一些迭代方法，如梯度下降（Gradient descent）学习算法。Early stopping便是一种迭代次数截断的方法来防止过拟合的方法，即在模型对训练数据集迭代收敛之前停止迭代来防止过拟合。

### 2 理论基础

早停法，实际上也是一种正则化的策略，可以理解为在网络训练不断逼近最优解的过程种（实际上这个最优解是过拟合的），在梯度等高线的外围就停止了训练，所以其原理上和L2正则是一样的，区别在于得到解的过程。

我们把图16-21再拿出来讨论一下。

<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/16/regular0.png" />

图16-21 损失函数值的等高线图

图中所示的等高线图，是当前带噪音的样本点所组成梯度图，并不代表测试集数据，所以其中心位置也不代表这个问题的最优解。我们假设红线是最优解，则早停法的目的就是在到达红线附近时停止训练。

### 3 算法

一般的做法是，在训练的过程中，记录到目前为止最好的validation 准确率，当连续N次Epoch（比如N=10或者更多次）没达到最佳准确率时，则可以认为准确率不再提高了。此时便可以停止迭代了（Early Stopping）。这种策略也称为“No-improvement-in-N”，N即Epoch的次数，可以根据实际情况取，如10、20、30……

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

#### 要注意的问题

1. 门限值`patience`不能太小，比如小于5，因为很可能在5个`epoch`之外，损失函数值又会再次下降
2. `patience`不能太大，比如大于30，因为在这30个`epoch`之内，由于样本数量少和数据`shuffle`的关系，很可能某个`epoch`的损失函数值会比上一次低，这样忍耐次数计数器`counter`就清零了，从而不能及时停止。
3. 当样本数量少时，为了获得平滑的变化曲线，可以考虑使用加权平均的方式处理当前和历史损失函数值，以避免某一次的高低带来的影响。

### 4 实现

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
接下来在Add()函数的代码中，如果激活了early_stop机制，则：

1. 判断loss_vld是否小于last_vld_loss，如果是，清零计数器，保存最新loss值
2. 如果否，计数器加1，判断是否达到门限值，是的话返回True，否则返回False

在main过程中，设置超参时指定正则项为RegularMethod.EarlyStop，并且value=8 (即门限值为8)。

```Python
from Level0_OverFitNet import *

if __name__ == '__main__':
    dr = LoadData()
    hp, num_hidden = SetParameters()
    hp.regular_name = RegularMethod.EarlyStop
    hp.regular_value = 8
    net = Model(dr, 1, num_hidden, 1, hp)
    ShowResult(net, dr, hp.toString())
```

注意，我们仍然使用和过拟合试验中一样的神经网络，宽度深度不变，只是增加了早停逻辑。

运行程序后，训练只迭代了2500多次就停止了，和我们预想的一样，损失值和准确率的曲线如图16-22所示。

<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/16/EarlyStop_sin_loss.png" />

图16-22 训练过程中损失函数值和准确率的变化曲线

早停法并不会提高准确率，而只是在最高的准确率上停止训练（前提是知道后面的训练会造成过拟合），从上图可以看到，最高的准确率是99.07%，达到了我们的目的。

最后的拟合效果如图16-23所示。

<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/16/EarlyStop_sin_result.png" ch="500" />

图16-23 拟合后的曲线与训练数据的分布图

蓝点是样本，绿点是理想的拟合效果，红线是实际的拟合效果。
##  丢弃法 Dropout

### 1 基本原理

2012年，Alex、Hinton在其论文《ImageNet Classification with Deep Convolutional Neural Networks》中用到了Dropout算法，用于防止过拟合。

我们假设原来的神经网络是这个结构，最后输出三分类结果，如图16-24所示。

<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/16/dropout_before.png" />

图16-24 输出三分类的神经网络结构图

Dropout可以作为训练深度神经网络的一种正则方法供选择。在每个训练批次中，通过忽略一部分的神经元（让其隐层节点值为0），可以明显地减少过拟合现象。这种方式可以减少隐层节点间的相互作用，高层的神经元需要低层的神经元的输出才能发挥作用，如果高层神经元过分依赖某个低层神经元，就会有过拟合发生。在一次正向/反向的过程中，通过随机丢弃一些神经元，迫使高层神经元和其它的一些低层神经元协同工作，可以有效地防止神经元因为接收到过多的同类型参数而陷入过拟合的状态，来提高泛化程度。

丢弃后的结果如图16-25所示。

<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/16/dropout_after.png" />

图16-25 使用丢弃法的神经网络结构图

其中有叉子的神经元在本次迭代训练中被暂时的封闭了，在下一次迭代训练中，再随机地封闭一些神经元，同一个神经元也许被连续封闭两次，也许一次都没有被封闭，完全随机。封闭多少个神经元是由一个超参来控制的，叫做丢弃率。

### 2 算法与实现

#### 前向计算

正常的隐层计算公式是：

$$
Z = W \cdot X + B \tag{1}
$$

加入随机丢弃步骤后，变成了：

$$
r \sim Bernoulli(p) \tag{2}
$$
$$Y = r \cdot X \tag{3}$$
$$Z = Y \cdot W + B \tag{4}
$$

公式2是得到一个分布概率为p的伯努利分布，伯努利分布在这里可以简单地理解为0-1分布，$p=0.5$时，会以相同概率产生0、1，假设一共10个数，则：
$$
r=[0,0,1,1,0,1,0,1,1,0]
$$
或者
$$
r=[0,1,1,0,0,1,0,1,0,1]
$$
或者其它一些分布。

从公式3，Y将会是X经过r的mask的结果，1的位置保留原x值，0的位置相乘后为0。
### 3 多样本合成法

#### SMOTE

SMOTE,Synthetic Minority Over-sampling Technique$^{[1]}$，通过人工合成新样本来处理样本不平衡问题，提升分类器性能。

类不平衡现象是数据集中各类别数量不近似相等。如果样本类别之间相差很大，会影响分类器的分类效果。假设小样本数据数量极少，仅占总体的1%，所能提取的相应特征也极少，即使小样本被错误地全部识别为大样本，在经验风险最小化策略下的分类器识别准确率仍能达到99%，但在验证环节分类效果不佳。

基于插值的SMOTE方法为小样本类合成新的样本，主要思路为：

1. 定义好特征空间，将每个样本对应到特征空间中的某一点，根据样本不平衡比例确定采样倍率N；
2. 对每一个小样本类样本$(x,y)$，按欧氏距离找K个最近邻样本，从中随机选取一个样本点，假设选择的近邻点为$(x_n,y_n)$。在特征空间中样本点与最近邻样本点的连线段上随机选取一点作为新样本点，满足以下公式:

$$(x_{new},y_{new})=(x,y)+rand(0,1)\times ((x_n-x),(y_n-y))$$

3. 重复选取取样，直到大、小样本数量平衡。

在`python`中，SMOTE算法已经封装到了`imbalanced-learn`库中。

#### SamplePairing

SamplePairing$^{[2]}$方法的处理流程如图16-35所示，从训练集中随机抽取两张图片分别经过基础数据增强操作（如随机翻转等）处理后经像素取平均值的形式叠加合成一个新的样本，标签为原样本标签中的一种。

<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/16/sample_pairing.png" />

图16-35 SamplePairing方法的处理流程

经SamplePairing处理后可使训练集的规模从N扩增到N*N，在CPU上也能完成处理。

训练过程是交替禁用与使用SamplePairing处理操作的结合：

1. 使用传统的数据增强训练网络，不使用SamplePairing 数据增强训练。
2. 在ILSVRC数据集上完成一个epoch或在其他数据集上完成100个epoch后，加入SamplePairing 数据增强训练。
3. 间歇性禁用 SamplePairing。对于 ILSVRC 数据集，为其中的300000 个图像启用SamplePairing，然后在接下来的100000个图像中禁用它。对于其他数据集，在开始的8个epoch中启用，在接下来的2个epoch中禁止。
4. 在训练损失函数和精度稳定后进行微调，禁用SamplePairing。

实验结果表明，因SamplePairing数据增强操作可能引入不同标签的训练样本，导致在各数据集上使用SamplePairing训练的误差明显增加，而在检测误差方面使用SamplePairing训练的验证误差有较大幅度降低。

尽管SamplePairing思路简单，性能上提升效果可观，符合奥卡姆剃刀原理，遗憾的是的可解释性不强，目前尚缺理论支撑。目前仅有图片数据的实验，还需下一步的实验与解读。

#### Mixup

Mixup$^{[3]}$是基于邻域风险最小化（VRM）原则的数据增强方法，使用线性插值得到新样本数据。在邻域风险最小化原则下，根据特征向量线性插值将导致相关目标线性插值的先验知识，可得出简单且与数据无关的mixup公式：

$$
x_n=\lambda x_i + (1-\lambda)x_j \\\\
y_n=\lambda y_i + (1-\lambda)y_j
$$

其中$(x_n，y_n)$是插值生成的新数据，$(x_i,y_i)$和$(x_j，y_j)$是训练集中随机选取的两个数据，λ的取值满足贝塔分布，取值范围介于0到1，超参数α控制特征目标之间的插值强度。

Mixup的实验丰富，实验结果表明可以改进深度学习模型在ImageNet数据集、CIFAR数据集、语音数据集和表格数据集中的泛化误差，降低模型对已损坏标签的记忆，增强模型对对抗样本的鲁棒性和训练对抗生成网络的稳定性。

Mixup处理实现了边界模糊化，提供平滑的预测效果，增强模型在训练数据范围之外的预测能力。随着超参数α增大，实际数据的训练误差就会增加，而泛化误差会减少。说明Mixup隐式地控制着模型的复杂性。随着模型容量与超参数的增加，训练误差随之降低。

尽管有着可观的效果改进，但mixup在偏差—方差平衡方面尚未有较好的解释。在其他类型的有监督学习、无监督、半监督和强化学习中，Mixup还有很大的发展空间。

#### 小结

Mixup、SMOTE、SamplePairing三者思路上有相同之处，都是试图将离散样本点连续化来拟合真实样本分布，但所增加的样本点在特征空间中仍位于已知小样本点所围成的区域内。但在特征空间中，小样本数据的真实分布可能并不限于该区域中，在给定范围之外适当插值，也许能实现更好的数据增强效果。
## 学习心得
回想一下BP神经网络。BP网络每一层节点是一个线性的一维排列状态，层与层的网络节点之间是全连接的。这样设想一下，如果BP网络中层与层之间的节点连接不再是全连接，而是局部连接的。这样，就是一种最简单的一维卷积网络。如果我们把上述这个思路扩展到二维，这就是我们在大多数参考资料上看到的卷积神经网络。
根据BP网络信号前向传递过程，我们可以很容易计算网络节点的输出。例如，对于上图中被标注为红色节点的净输入，就等于所有与红线相连接的上一层神经元节点值与红色线表示的权值之积的累加。这样的计算过程，很多书上称其为卷积。