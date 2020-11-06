# Step7学习笔记

## 搭建深度神经网络框架
* 通过分析三层神经网络的代码可发现，在前向计算（def forward）中，每层代码形式为矩阵运算+激活/分类函数，在反向传播（def backward）中，每层规律为计算本层的dZ，再根据dZ计算dW和dB。
* 迷你框架设计：
![avatar](https://note.youdao.com/yws/api/personal/file/EA71C58E429B4A278296268FB71F80BD?method=download&shareKey=bbcedb0500c4544e595a1e1844fd7b51)
1. NeuralNet:包装基本的神经网络结构和功能。
2. Layer:一个抽象类，以及更加需要增加的实际类。
3. Loss Function：提供计算损失函数值，存储历史记录并最后绘图的功能
4. Parameters：基本参数，包括普通参数和超参。
5. Optimizer:优化器。
6. WeightsBias：权重矩阵，仅供全连接层使用。
7. DataReader：样本数据读取器。

## 搭建模型
### 回归任务功能测试
* 
```Python
  params = HyperParameters_4_0(
        learning_rate, max_epoch, batch_size,
        net_type=NetType.Fitting,
        init_method=InitialMethod.Xavier,
        stopper=Stopper(StopCondition.StopLoss, 0.001))
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

* 
  ![avatar](https://note.youdao.com/yws/api/personal/file/BB9F50FC7BBA49A088F3F1F65CF8AD75?method=download&shareKey=ecd5f4ed4b18b73d41cf4f1aa8ed0c42)
  ![avatar](https://note.youdao.com/yws/api/personal/file/D8EE9BE4CCAD4E17AEEE13A7EBDFED04?method=download&shareKey=f698e17d02967c4e11c33d9d661db583)
  将batch的值变小，相对的增加了每个epoch的训练量，可发现损失函数下降的更快。
* 
  ![avatar](https://note.youdao.com/yws/api/personal/file/BB1F6F70DC2E43DFA14DF7BE063FA0EB?method=download&shareKey=e3b2cbba9ff407dc2559d72ac47ba6bc)
  ![avatar](https://note.youdao.com/yws/api/personal/file/9DA4D4DD6A194A90B4EFB3F91945E106?method=download&shareKey=209ec0b12d3d1c3adc90b4a1ba687389)
  将epoch从10000调整到20000可发现拟合效果变得更好。

### 二分类任务功能测试
相比于回归任务功能测试，同样是一个双层神经网络，但是最后一层要接一个Logistic二分类函数来完成二分类任务。
    
  ```python
    
    fc2 = FcLayer_1_0(num_hidden, num_output, params)
    net.add_layer(fc2, "fc2")
    logistic = ClassificationLayer(Logistic())
    net.add_layer(logistic, "logistic")
  ```
![avatar](https://note.youdao.com/yws/api/personal/file/7AB2200D10F54E8B8656558024D45DF7?method=download&shareKey=075b3cd1e1e1d1418051529a9ffaed4e)
![avatar](https://note.youdao.com/yws/api/personal/file/922929F717EC47559E19D8827ABE87BB?method=download&shareKey=1cd442d7038f620ea1d9cab0ba8eb903)

PS:数据处理的方法对于连续值，我们可以直接使用原始数据。对于枚举型，我们需要把它们转成连续值。以性别举例，Female=0，Male=1即可。对于其它枚举型，都可以用从0开始的整数编码。一个小技巧是利用python的list功能，取元素下标，即可以作为整数编码：
```Python
sex_list = ["Female", "Male"]
array_x[0,9] = sex_list.index(row[9].strip())
```
`strip()`是trim掉前面的空格，因为是`csv`格式，读出来会是这个样子："_Female"，前面总有个空格。`index`是取列表下标，这样对于字符串"Female"取出的下标为0，对于字符串"Male"取出的下标为1。

把所有数据按行保存到`numpy`数组中，最后用`npz`格式存储：
```Python
np.savez(data_npz, data=self.XData, label=self.YData)
```

### 多分类功能测试
* 使用Sigmoid作为激活函数的二层网络。最后用softmax输出。
* 第二类模型，使用ReLU作为激活函数的三层网络。
  ![avatar](https://note.youdao.com/yws/api/personal/file/BC743B98C1104AE9928267A1F75E4967?method=download&shareKey=b9f7c1a8b7c666a72c431f8e9687cf5d)
PS:两种模型的比较结论，Relu能直则直，对方形边界适用；Sigmoid能弯则弯，对圆形边界适用。

## 网络优化

### 权重矩阵初始化
* 零初始化：即把所有层的W值的初始值都设置为0。（适用于单层网络）
* 标准初始化：
$$
W \sim N \begin{bmatrix} 0, 1 \end{bmatrix}
$$
当目标问题较为简单，网络深度不大，可以用标准初始化
* Xavier初始化：
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
条件：正向传播时，激活值的方差保持不变；反向传播时，关于状态值的梯度的方差保持不变。
W表示权重矩阵，U表示均匀分布，N表示正态分布。

* Xavier初始化方法比直接用高斯分布进行初始化W的优势所在：
  一般的神经网络在前向传播时神经元输出值的方差会不断增大，而使用Xavier等方法理论上可以保证每层神经元输入输出方差一致。
* MSRA初始化方法：
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
  MSRA初始化是一个均值为0，方差为2/n的高斯分布.
  条件：正向传播时，状态值的方差保持不变；反向传播时，关于激活值的梯度的方差保持不变.
 

![avatar](https://note.youdao.com/yws/api/personal/file/EE1EC655D0304CF287DBBEF53A5D6315?method=download&shareKey=dff66b9b94b9519a5234f794a0c0bbe9)

## 梯度下降优化算法
### 随机梯度下降 SGD
缺点：到中点附近时，由于样本误差或者学习率问题，会发生来回徘徊的现象，很可能会错过最优解且收敛速度慢。

### 动量算法 Momentum

原理：第一次的梯度更新完毕后，会记录$v_1$的动量值。在“求梯度点”进行第二次梯度检查时，得到2号方向，与$v_1$的动量组合后，最终的更新为2'方向。这样一来，由于有$v_1$的存在，会迫使梯度更新方向具备“惯性”，从而可以减小随机样本造成的震荡。
![avatar](https://note.youdao.com/yws/api/personal/file/65948CF9ED204B19980449D8233C16F3?method=download&shareKey=70a297541c1c55e5db31974a4a479ea8)

### 梯度加速算法 NAG
原理：同Momentum相比，梯度不是根据当前位置θ计算出来的，而是在移动之后的位置$\theta - \alpha \cdot v_{t-1}$计算梯度。理由是，既然已经确定会移动$\theta - \alpha \cdot v_{t-1}$，那不如之前去看移动后的梯度。
![avatar](https://note.youdao.com/yws/api/personal/file/9F1DA077DF8E4F46953836C8204AF96F?method=download&shareKey=df61081e21bec323983fd48cff576183)

### 自适应学习率算法
#### AdaGrad
功能：它对不同的参数调整学习率，具体而言，对低频出现的参数进行大的更新，对高频出现的参数进行小的更新。因此，他很适合于处理稀疏数据。

#### AdaDelta
功能：AdaDelta法是AdaGrad 法的一个延伸，它旨在解决它学习率不断单调下降的问题。相比计算之前所有梯度值的平方和，AdaDelta法仅计算在一个大小为w的时间区间内梯度值的累积和。

#### 均方根反向传播 RMSProp
功能：解决AdaGrad的学习率缩减问题。

#### Adam - Adaptive Moment Estimation
功能：计算每个参数的自适应学习率，相当于RMSProp + Momentum的效果，Adam
算法在RMSProp算法基础上对小批量随机梯度也做了指数加权移动平均。和AdaGrad算法、RMSProp算法以及AdaDelta算法一样，目标函数自变量中每个元素都分别拥有自己的学习率。

### 算法在等高线图上的效果比较
- SGD：接近中点的过程很曲折，步伐很慢，甚至有反方向的，容易陷入局部最优。
- Momentum：快速接近中点，但中间跳跃较大。
- RMSProp：接近中点很曲折，但是没有反方向的，用的步数比SGD少，跳动较大，有可能摆脱局部最优解的。
- Adam：快速接近中点。

### 批量归一化的原理
* Batch Normalization，简称为BatchNorm，或BN
* 训练过程中网络中间层数据分布的改变称之为内部协变量偏移（Internal Covariate Shift）。BN的提出，就是要解决在训练过程中，中间层数据分布发生改变的情况
* BN就是在深度神经网络训练过程中使得每一层神经网络的输入保持相同的分布，致力于将每一层的输入数据正则化成N(0,1)N(0,1)的分布。因次，每次训练的数据必须是mini-batch形式，一般取32，64等数值。
* 使用批量归一化后，迭代速度提升，但是花费时间多了2秒，这是因为批量归一化的正向和反向计算过程还是比较复杂的，需要花费一些时间，但是批量归一化确实可以帮助网络快速收敛。
![avatar](https://note.youdao.com/yws/api/personal/file/59BE1AC5298D4EFE9D782E5A2E43DD12?method=download&shareKey=aed16df6e77233fc593968bb62d1afb7)

## 正则化
* 过拟合的例子：
![avatar](https://note.youdao.com/yws/api/personal/file/CB6EEB6A162A477088B197687A456F45?method=download&shareKey=a395e00026717f132b4516096f2a0232)
红色拟合曲线严丝合缝地拟合了每一个样本点，也就是说模型学习到了样本的误差。
* 不同偏差和方差反映的四种情况
![avatar](https://note.youdao.com/yws/api/personal/file/FBE1EDD1056247F99648022DE25CA718?method=download&shareKey=c0de1e1446100e6cdb83465122b76c48)

### L2正则化
* 直接在原来的损失函数基础上加上权重参数的平方和:
![avatar](https://note.youdao.com/yws/api/personal/file/24B77FE535804519B671F2DD7FDF50D0?method=download&shareKey=9f67f2458d288aca787c8bfe66dfbb9d)
Ein 是未包含正则化项的训练样本误差，λ 是正则化参数

### L1正则化
* 直接在原来的损失函数基础上加上权重参数的绝对值:
![avatar](https://note.youdao.com/yws/api/personal/file/69E7AA546AC4424097CE673EA3496FE2?method=download&shareKey=aa29200fd2ebae64802970485214bb23)

### L1 与 L2 解的稀疏性
![avatar](https://note.youdao.com/yws/api/personal/file/423631C7531C4FD0A260D4AA2282FCE5?method=download&shareKey=4b185bcd7c3ef8d1c13a20de8067fca8)
对于L1来说，限定区域是正方形，方形与蓝色区域相交的交点是顶点的概率很大，这从视觉和常识上来看是很容易理解的。也就是说，方形的凸点会更接近 Ein最优解对应的wlin位置，而凸点处必有w1或w2为0.这样，得到的解w1或w2为零的概率就很大了。所以，L1正则化的解具有稀疏性。

### 早停法
主要步骤：
1. 将原始的训练数据集划分成训练集和验证集
2. 只在训练集上进行训练，并每个一个周期计算模型在验证集上的误差，例如，每15次epoch（mini batch训练中的一个周期）
3. 当模型在验证集上的误差比上一次训练结果差的时候停止训练
4. 使用上一次迭代结果中的参数作为模型的最终参数
作用：在训练中计算模型在验证集上的表现，当模型在验证集上的表现开始下降的时候，停止训练，这样就能避免继续训练导致过拟合的问题。

### Dropout
Dropout可以作为训练深度神经网络的一种正则方法供选择。在每个训练批次中，通过忽略一部分的神经元（让其隐层节点值为0），可以明显地减少过拟合现象。这种方式可以减少隐层节点间的相互作用，高层的神经元需要低层的神经元的输出才能发挥作用，如果高层神经元过分依赖某个低层神经元，就会有过拟合发生。在一次正向/反向的过程中，通过随机丢弃一些神经元，迫使高层神经元和其它的一些低层神经元协同工作，可以有效地防止神经元因为接收到过多的同类型参数而陷入过拟合的状态，来提高泛化程度。
![avatar](https://note.youdao.com/yws/api/personal/file/D5A8EB5CFC2045C58455D9C86D3DA232?method=download&shareKey=bfa98a4c09010bd4cc0291f73c14a57f)

### 数据扩展
Mixup、SMOTE、SamplePairing三者思路上有相同之处，都是试图将离散样本点连续化来拟合真实样本分布，但所增加的样本点在特征空间中仍位于已知小样本点所围成的区域内。但在特征空间中，小样本数据的真实分布可能并不限于该区域中，在给定范围之外适当插值，也许能实现更好的数据增强效果。

### 集成学习 Ensemble Learning
Bagging集成学习：
![avatar](https://note.youdao.com/yws/api/personal/file/B723162A855A4DBB8F6F48084247CDF4?method=download&shareKey=48d32a484f34707dd2422f85423c2851)
1. 首先是数据集的使用，采用自助采样法（Bootstrap Sampling）。假设原始数据集Training Set中有1000个样本，我们从中随机取一个样本的拷贝放到Training Set-1中，此样本不会从原始数据集中被删除，原始数据集中还有1000个样本，而不是999个，这样下次再随机取样本时，此样本还有可能被再次选到。如此重复m次（此例m=1000），我们可以生成Training Set-1。一共重复N次（此例N=9），可以得到N个数据集。
2. 然后搭建一个神经网络模型，可以参数相同。在N个数据集上，训练出N个模型来。
3. 最后再进入Aggregator。N值不能太小，否则无法提供差异化的模型，也不能太大而带来训练模型的时间花销，一般来说取5到10就能满足要求。

## 学习体会
在多分类功能测试任务中测试了MNIST手写体识别，与在慕课中所讲到的利用卷积识别相对应，观察可发现在多分类任务中耗时17秒达到了准确率96%，而在卷积深度学习中，利用卷积核提取特征，大大缩小了训练量，提高了准确率。在第15部分学习了网络优化的一些算法，这些算法的演替过程，使随机梯度算法的过程更加优化，但同时也应该明白，没有一个算法是可以顾及到每个方面。最后是正则化，也是对过程中数据的归一，解决过拟合的问题，了解了L1，L2正则化，最后也学习了早停法，丢弃法等方法。对于整个优化过程有了更全面概念的认识。