


# 第10章 多入单出的双层神经网络 - 非线性二分类


### 10.0.3 二分类模型的评估标准

#### 准确率 Accuracy

也可以称之为精度，我们在本书中混用这两个词。

对于二分类问题，假设测试集上一共1000个样本，其中550个正例，450个负例。测试一个模型时，得到的结果是：521个正例样本被判断为正类，435个负例样本被判断为负类，则正确率计算如下：

$$Accuracy=(521+435)/1000=0.956$$

即正确率为95.6%。这种方式对多分类也是有效的，即三类中判别正确的样本数除以总样本数，即为准确率。

但是这种计算方法丢失了很多细节，比如：是正类判断的精度高还是负类判断的精度高呢？因此，我们还有如下一种评估标准。

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

在二分类器中，如果使用Logistic函数作为分类函数，可以设置一系列不同的阈值，比如[0.1,0.2,0.3...0.9]，把测试样本输入，从而得到一系列的TP、FP、TN、FN，然后就可以绘制如下曲线，如图10-4。

<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/10/ROC.png"/>

图10-4 ROC曲线图

图中红色的曲线就是ROC曲线，曲线下的面积就是AUC值，取值区间为$[0.5,1.0]$，面积越大越好。

- ROC曲线越靠近左上角，该分类器的性能越好。
- 对角线表示一个随机猜测分类器。
- 若一个学习器的ROC曲线被另一个学习器的曲线完全包住，则可判断后者性能优于前者。
- 若两个学习器的ROC曲线没有包含关系，则可以判断ROC曲线下的面积，即AUC，谁大谁好。

当然在实际应用中，取决于阈值的采样间隔，红色曲线不会这么平滑，由于采样间隔会导致该曲线呈阶梯状。

既然已经这么多标准，为什么还要使用ROC和AUC呢？因为ROC曲线有个很好的特性：当测试集中的正负样本的分布变换的时候，ROC曲线能够保持不变。在实际的数据集中经常会出现样本类不平衡，即正负样本比例差距较大，而且测试数据中的正负样本也可能随着时间变化。

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

#### Mean absolute error 和 Root mean squared error 

平均绝对误差和均方根误差，用来衡量分类器预测值和实际结果的差异，越小越好。

#### Relative absolute error 和 Root relative squared error 

相对绝对误差和相对均方根误差，有时绝对误差不能体现误差的真实大小，而相对误差通过体现误差占真值的比重来反映误差大小。

## 10.1 为什么必须用双层神经网络

### 10.1.1 分类

我们先回忆一下各种分类的含义：

- 从复杂程度上分，有线性/非线性之分；
- 从样本类别上分，有二分类/多分类之分。

从直观上理解，这几个概念应该符合表10-2中的示例。

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

### 10.1.2 简单证明异或问题的不可能性

用单个感知机或者单层神经网络，是否能完成异或任务呢？我们自己做个简单的证明。先看样本数据，如表10-4。

表10-4 异或的样本数据

|样本|$x_1$|$x_2$|$y$|
|---|---|---|---|
|1|0|0|0|
|2|0|1|1|
|3|1|0|1|
|4|1|1|0|

用单个神经元（感知机）的话，就是表10-5中两种技术的组合。

表10-5 神经元结构与二分类函数

|神经元|分类函数Logistic|
|--|--|
|<img src='../Images/10/xor_prove.png' width="400"/>|<img src='../Images/8/sigmoid_seperator.png' width="430"/>|

前向计算公式：

$$z = x_1  w_1 + x_2  w_2 + b \tag{1}$$
$$a = Logistic(z) \tag{2}$$

- 对于第一个样本数据

$x_1=0,x_2=0,y=0$。如果需要$a=y$的话，从Logistic函数曲线看，需要$z<0$，于是有：

$$x_1 w_1 + x_2  w_2 + b < 0$$

因为$x_1=0,x_2=0$，所以只剩下$b$项：

$$b < 0 \tag{3}$$

- 对于第二个样本数据

$x_1=0,x_2=1,y=1$。如果需要$a=y$，则要求$z>0$，不等式为：

$$x_1w_1 + x_2w_2+b=w_2+b > 0 \tag{4}$$

- 对于第三个样本数据

$x_1=1,x_2=0,y=1$。如果需要$a=y$，则要求$z>0$，不等式为：

$$x_1w_1 + x_2w_2+b=w_1+b > 0 \tag{5}$$

- 对于第四个样本

$x_1=1,x_2=1,y=0$。如果需要$a=y$，则要求$z<0$，不等式为：

$$x_1w_1 + x_2w_2+b=w_1+w_2+b < 0 \tag{6}$$

把公式6两边都加$b$，并把公式3接续：

$$(w_1 + b) + (w_2 + b) < b < 0 \tag{7}$$

再看公式4、5，不等式左侧括号内的两个因子都大于0，其和必然也大于0，不可能小于$b$。因此公式7不成立，无论如何也不能满足所有的4个样本的条件，所以单个神经元做异或运算是不可能的。

### 10.1.3 非线性的可能性

我们前边学习过如何实现与、与非、或、或非，我们看看如何用已有的逻辑搭建异或门，如图10-5所示。

<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/10/xor_gate.png" />

图10-5 用基本逻辑单元搭建异或运算单元

表10-6 组合运算的过程

|样本与计算|1|2|3|4|
|----|----|----|----|----|
|$x_1$|0|0|1|1|
|$x_2$|0|1|0|1|
|$s_1=x_1$ NAND $x_2$|1|1|1|0|
|$s_2=x_1$ OR $x_2$|0|1|1|1|
|$y=s_1$ AND $s_2$|0|1|1|0|

经过表10-6所示的组合运算后，可以看到$y$的输出与$x_1,x_2$的输入相比，就是异或逻辑了。所以，实践证明两层逻辑电路可以解决问题。另外，我们在地四步中学习了非线性回归，使用双层神经网络可以完成一些神奇的事情，比如复杂曲线的拟合，只需要6、7个参数就搞定了。我们可以模拟这个思路，用两层神经网络搭建模型，来解决非线性分类问题。


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

在二分类问题中，$Y,A2$都是一个单一的数值，而非矩阵，但是为了前后统一，我们可以把它们看作是一个$1\times 1$的矩阵。

### 10.2.3 反向传播

图10-9展示了反向传播的过程。

<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/10/binary_backward.png" />

图10-9 反向传播过程

#### 求损失函数对输出层的反向误差

对损失函数求导，可以得到损失函数对输出层的梯度值，即图10-9中的$Z2$部分。

根据公式12，求$A2$和$Z2$的导数（此处$A2,Z2,Y$可以看作是标量，以方便求导）：

$$
\begin{aligned}
\frac{\partial loss}{\partial Z2}&=\frac{\partial loss}{\partial A2}\frac{\partial A2}{\partial Z2} \\\\
&=\frac{A2-Y}{A2(1-A2)} \cdot A2(1-A2) \\\\
&=A2-Y \rightarrow dZ2
\end{aligned}
\tag{13}
$$

#### 求$W2$和$B2$的梯度

$$
\begin{aligned}
\frac{\partial loss}{\partial W2}&=\begin{pmatrix}
  \frac{\partial loss}{\partial w2_{11}} \\\\
  \frac{\partial loss}{\partial w2_{21}}
\end{pmatrix}
=\begin{pmatrix}
  \frac{\partial loss}{\partial Z2}\frac{\partial z2}{\partial w2_{11}} \\\\
  \frac{\partial loss}{\partial Z2}\frac{\partial z2}{\partial w2_{21}}
\end{pmatrix}
\\\\
&=\begin{pmatrix}
  dZ2 \cdot a1_{1} \\\\
  dZ2 \cdot a1_{2} 
\end{pmatrix}
=\begin{pmatrix}
  a1_{1} \\\\ a1_{2}
\end{pmatrix}dZ2
\\\\
&=A1^{\top} \cdot dZ2 \rightarrow dW2  
\end{aligned}
\tag{14}
$$
$$\frac{\partial{loss}}{\partial{B2}}=dZ2 \rightarrow dB2 \tag{15}$$

#### 求损失函数对隐层的反向误差

$$
\begin{aligned}  
\frac{\partial{loss}}{\partial{A1}} &= \begin{pmatrix}
  \frac{\partial loss}{\partial a1_{1}} & \frac{\partial loss}{\partial a1_{2}} 
\end{pmatrix}
\\\\
&=\begin{pmatrix}
\frac{\partial{loss}}{\partial{Z2}} \frac{\partial{Z2}}{\partial{a1_{1}}} & \frac{\partial{loss}}{\partial{Z2}}  \frac{\partial{Z2}}{\partial{a1_{2}}}  
\end{pmatrix}
\\\\
&=\begin{pmatrix}
dZ2 \cdot w2_{11} & dZ2 \cdot w2_{21}
\end{pmatrix}
\\\\
&=dZ2 \cdot \begin{pmatrix}
  w2_{11} & w2_{21}
\end{pmatrix}
\\\\
&=dZ2 \cdot W2^{\top}
\end{aligned}
\tag{16}
$$

$$
\frac{\partial A1}{\partial Z1}=A1 \odot (1-A1) \rightarrow dA1\tag{17}
$$

所以最后到达$Z1$的误差矩阵是：

$$
\begin{aligned}
\frac{\partial loss}{\partial Z1}&=\frac{\partial loss}{\partial A1}\frac{\partial A1}{\partial Z1}
\\\\
&=dZ2 \cdot W2^{\top} \odot dA1 \rightarrow dZ1 
\end{aligned}
\tag{18}
$$

有了$dZ1$后，再向前求$W1$和$B1$的误差，就和第5章中一样了，我们直接列在下面：

$$
dW1=X^{\top} \cdot dZ1 \tag{19}
$$
$$
dB1=dZ1 \tag{20}
$$


## 10.3 实现逻辑异或门

### 10.3.1 代码实现

#### 准备数据

异或数据比较简单，只有4个记录，所以就hardcode在此，不用再建立数据集了。这也给读者一个机会了解如何从`DataReader`类派生出一个全新的子类`XOR_DataReader`。

比如在下面的代码中，我们覆盖了父类中的三个方法：

- `init()` 初始化方法：因为父类的初始化方法要求有两个参数，代表train/test数据文件
- `ReadData()`方法：父类方法是直接读取数据文件，此处直接在内存中生成样本数据，并且直接令训练集等于原始数据集（不需要归一化），令测试集等于训练集
- `GenerateValidationSet()`方法，由于只有4个样本，所以直接令验证集等于训练集

因为`NeuralNet2`中的代码要求数据集比较全，有训练集、验证集、测试集，为了已有代码能顺利跑通，我们把验证集、测试集都设置成与训练集一致，对于解决这个异或问题没有什么影响。

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

与第6章中的逻辑与门和或门一样，我们需要神经网络的运算结果达到一定的精度，也就是非常的接近0，1两端，而不是说勉强大于0.5就近似为1了，所以精度要求是误差绝对值小于`1e-2`。

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

此处的代码有几个需要强调的细节：

- `n_input = dataReader.num_feature`，值为2，而且必须为2，因为只有两个特征值
- `n_hidden=2`，这是人为设置的隐层神经元数量，可以是大于2的任何整数
- `eps`精度=0.005是后验知识，笔者通过测试得到的停止条件，用于方便案例讲解
- 网络类型是`NetType.BinaryClassifier`，指明是二分类网络
- 最后要调用`Test`函数验证精度

### 10.3.2 运行结果

经过快速的迭代后，会显示训练过程如图10-10所示。

<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/10/xor_loss.png" />

图10-10 训练过程中的损失函数值和准确率值的变化

可以看到二者的走势很理想。

同时在控制台会打印一些信息，最后几行如下：

```
......
epoch=5799, total_iteration=23199
loss_train=0.005553, accuracy_train=1.000000
loss_valid=0.005058, accuracy_valid=1.000000
epoch=5899, total_iteration=23599
loss_train=0.005438, accuracy_train=1.000000
loss_valid=0.004952, accuracy_valid=1.000000
W= [[-7.10166559  5.48008579]
 [-7.10286572  5.48050039]]
B= [[ 2.91305831 -8.48569781]]
W= [[-12.06031599]
 [-12.26898815]]
B= [[5.97067802]]
testing...
1.0
None
testing...
A2= [[0.00418973]
 [0.99457721]
 [0.99457729]
 [0.00474491]]
True
```
一共用了5900个`epoch`，达到了指定的`loss`精度（0.005），`loss_valid`是0.004991，刚好小于0.005时停止迭代。

我们特意打印出了`A2`值，即网络推理结果，如表10-7所示。

表10-7 异或计算值与神经网络推理值的比较

|x1|x2|XOR|Inference|diff|
|---|---|---|---|---|
|0|0|0|0.0041|0.0041|
|0|1|1|0.9945|0.0055|
|1|0|1|0.9945|0.0055|
|1|1|0|0.0047|0.0047|

表中第四列的推理值与第三列的`XOR`结果非常的接近，继续训练的话还可以得到更高的精度，但是一般没这个必要了。由此我们再一次认识到，神经网络只可以得到无限接近真实值的近似解。



## 10.4 逻辑异或门的工作原理

上一节课的内容从实践上证明了两层神经网络是可以解决异或问题的，下面让我们来理解一下神经网络在这个异或问题的上工作原理，此原理可以扩展到更复杂的问题空间，但是由于高维空间无法可视化，给我们的理解带来了困难。

### 10.4.1 可视化分类结果

为了辅助理解异或分类的过程，我们增加一些可视化函数来帮助理解。

#### 显示原始数据

```Python
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

from Level1_XorGateClassifier import *

def ShowSourceData(dataReader):
    DrawSamplePoints(dataReader.XTrain[:,0],dataReader.XTrain[:,1],dataReader.YTrain, "XOR Source Data", "x1", "x2")

def DrawSamplePoints(x1, x2, y, title, xlabel, ylabel, show=True):
    assert(x1.shape[0] == x2.shape[0])
    fig = plt.figure(figsize=(6,6))
    count = x1.shape[0]
    for i in range(count):
        if y[i,0] == 0:
            plt.scatter(x1[i], x2[i], marker='^', color='r', s=200, zorder=10)
        else:
            plt.scatter(x1[i], x2[i], marker='o', color='b', s=200, zorder=10)
        #end if
    #end for
    plt.grid()
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if show:
        plt.show()
```

1. 首先是从Level_XorGateClassifier中导入所有内容，省去了我们重新写数据准备部分的代码的麻烦
2. 获得所有分类为1的训练样本，用红色叉子显示在画板上
3. 获得所有分类为0的训练样本，用蓝色圆点显示在画板上

由此我们会得到样本如图10-11所示。

<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/10/xor_source_data.png" ch="500" />

图10-11 异或样本数据

异或问题的四个点分布在[0,1]空间的四个角上，红色点是正类，蓝色点是负类。

#### 显示推理的中间结果

由于是双层神经网络，回忆一下其公式：$Z1 = X \cdot W1 +B1,A1=Sigmoid(Z1),Z2=A1 \cdot W2+B2,A2=Logistic(A2)$，所以会有$Z1,A1,Z2,A2$等中间运算结果。我们把它们用图形方式显示出来帮助读者理解推理过程。

```Python
def ShowProcess2D(net, dataReader):
    net.inference(dataReader.XTest)
    # show z1    
    DrawSamplePoints(net.Z1[:,0], net.Z1[:,1], dataReader.YTest, "net.Z1", "Z1[0]", "Z1[1]")
    # show a1
    DrawSamplePoints(net.A1[:,0], net.A1[:,1], dataReader.YTest, "net.A1", "A1[0]", "A1[1]")
    # show sigmoid
    DrawSamplePoints(net.Z2, net.A2, dataReader.YTrain, "Z2->A2", "Z2", "A2", show=False)
    x = np.linspace(-6,6)
    a = Sigmoid().forward(x)
    plt.plot(x,a)
    plt.show()
```

1. 先用测试样本做一次推理；
2. Z1是第一层神经网络线性变换的结果，由于Z1是一个4行两列的数组，我们以Z1的第1列作为x1，以Z1的第2列作为x2，画出4个点来；
3. A1是Z1经过激活函数后的结果，同Z1一样作为4个点画出来；
4. Z2是第二层神经网络线性变换的结果，A2是Z2的Logistic Function的运算结果，以Z2为x1，A2为x2，画出4个点来，并叠加Logistic函数图像，看是否吻合。

于是我们得到下面三张图，放入表10-8中（把原始图作为对比，放在第一个位置）。

表10-8 XOR问题的推理过程

|||
|---|---|
|<img src='../Images/10/xor_source_data.png'/>|<img src='../Images/10/xor_z1.png'/>|
|原始样本|Z1是第一层网络线性计算结果|
|<img src='../Images/10/xor_a1.png'/>|<img src='../Images/10/xor_z2_a2.png'/>|
|A1是Z1的激活函数计算结果|Z2是第二层线性计算结果，A2是二分类结果|

- Z1：通过线性变换，把原始数据蓝色点移动到两个对角上，把红色点向中心移动，接近重合。图中的红色点看上去好像是一个点，实际上是两个点重合在了一起，可以通过在原画板上放大的方式来看细节
- A1：通过Sigmoid运算，把Z1的值压缩到了[0,1]空间内，使得蓝色点的坐标向[0,1]和[1,0]接近，红色点的坐标向[0,0]靠近
- Z2->A2：再次通过线性变换，把两类点都映射到横坐标轴上，并把蓝点向负方向移动，把红点向正方向移动，再Logistic分类，把两类样本点远远地分开到[0,1]的两端，从而完成分类任务

我们把中间计算结果显示在表10-9中，便于观察比较。

表10-9 中间计算结果

||1（蓝点1）|2（红点1）|3（红点2）|4（蓝点2）|
|---|---|---|---|---|
|x1|0|0|1|1|
|x2|0|1|0|1|
|y|0|1|1|0|
|Z1|2.868856|-4.142354|-4.138914|-11.150125|
||-8.538638|-3.024127|-3.023451|2.491059|
|A1|0.946285|0.015637|0.015690|0.000014|
||0.000195|0.046347|0.046377|0.923512|
|Z2|-5.458510|5.203479|5.202473|-5.341711|
|A2|0.004241|0.994532|0.994527|0.004764|

#### 显示最后结果

到目前位置，我们只知道神经网络完成了异或问题，但它究竟是如何画分割线的呢？

也许读者还记得在第四步中学习线性分类的时候，我们成功地通过公式推导画出了分割直线，但是这一次不同了，这里使用了两层神经网络，很难再通过公式推导来解释W和B权重矩阵的含义了，所以我们换个思路。

思考一下，神经网络最后是通过什么方式判定样本的类别呢？在前向计算过程中，最后一个公式是Logistic函数，它把$(-\infty, +\infty)$压缩到了(0,1)之间，相当于计算了一个概率值，然后通过概率值大于0.5与否，判断是否属于正类。虽然异或问题只有4个样本点，但是如果：

1. 我们在[0,1]正方形区间内进行网格状均匀采样，这样每个点都会有坐标值；
2. 再把坐标值代入神经网络进行推理，得出来的应该会是一个网格状的结果；
3. 每个结果都是一个概率值，肯定处于(0,1)之间，所以不是大于0.5，就是小于0.5；
4. 我们把大于0.5的网格涂成粉色，把小于0.5的网格涂成黄色，就应该可以画出分界线来了。

好，有了这个令人激动人心的想法，我们立刻实现：

```Python
def ShowResult2D(net, dr, title):
    print("please wait for a while...")
    DrawSamplePoints(dr.XTest[:,0], dr.XTest[:,1], dr.YTest, title, "x1", "x2", show=False)
    count = 50
    x1 = np.linspace(0,1,count)
    x2 = np.linspace(0,1,count)
    for i in range(count):
        for j in range(count):
            x = np.array([x1[i],x2[j]]).reshape(1,2)
            output = net.inference(x)
            if output[0,0] >= 0.5:
                plt.plot(x[0,0], x[0,1], 's', c='m', zorder=1)
            else:
                plt.plot(x[0,0], x[0,1], 's', c='y', zorder=1)
            # end if
        # end for
    # end for
    plt.title(title)
    plt.show()
```

在上面的代码中，横向和竖向各取了50个点，形成一个50x50的网格，然后依次推理，得到output值后染色。由于一共要计算2500次，所以花费的时间稍长，我们打印"please wait for a while..."让程序跑一会儿。最后得到图10-12。

<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/10/xor_result_2d.png" ch="500" />

图10-12 分类结果的分割图

第一次看到这张图是不是很激动！从此我们不再靠画线过日子了，而是上升到了染色的层次！请忽略图中的锯齿，因为我们取了50x50的网格，所以会有马赛克，如果取更密集的网格点，会缓解这个问题，但是计算速度要慢很多倍。

可以看到，两类样本点被分在了不同颜色的区域内，这让我们恍然大悟，原来神经网络可以同时画两条分割线的，更准确的说法是“可以画出两个分类区域”。

### 10.4.2 更直观的可视化结果

#### 3D图

神经网络真的可以同时画两条分割线吗？这颠覆了笔者的认知，因为笔者一直认为最后一层的神经网络只是一个线性单元，它能做的事情有限，所以它的行为就是线性的行为，画一条线做拟合或分割，......，稍等，为什么只能是一条线呢？难道不可以是一个平面吗？

这让笔者想起了在第5章里，曾经用一个平面拟合了空间中的样本点，如表10-10所示。

表10-10 平面拟合的可视化结果

|正向|侧向|
|---|---|
|<img src='../Images/5/level3_result_1.png'/>|<img src='../Images/5/level3_result_2.png'/>|

那么这个异或问题的解是否可能是个立体空间呢？有了这个更激动人心的想法，我们立刻写代码：

```Python
def Prepare3DData(net, count):
    x = np.linspace(0,1,count)
    y = np.linspace(0,1,count)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros((count, count))
    input = np.hstack((X.ravel().reshape(count*count,1),Y.ravel().reshape(count*count,1)))
    output = net.inference(input)
    Z = output.reshape(count,count)
    return X,Y,Z

def ShowResult3D(net, dr):
    fig = plt.figure(figsize=(6,6))
    ax = Axes3D(fig)
    X,Y,Z = Prepare3DData(net, 50)
    ax.plot_surface(X,Y,Z,cmap='rainbow')
    ax.set_zlim(0,1)
    # draw sample data in 3D space
    for i in range(dr.num_train):
        if dataReader.YTrain[i,0] == 0:
            ax.scatter(dataReader.XTrain[i,0],dataReader.XTrain[i,1],dataReader.YTrain[i,0],marker='^',c='r',s=200)
        else:
            ax.scatter(dataReader.XTrain[i,0],dataReader.XTrain[i,1],dataReader.YTrain[i,0],marker='o',c='b',s=200)

    plt.show()
```

函数Prepare3DData()用于准备一个三维坐标系内的数据：

1. x坐标在[0,1]空间分成50份
2. y坐标在[0,1]空间分成50份
3. np.meshgrid(x,y)形成网格式点阵X和Y，它们各有2500个记录，每一行的X必须和对应行的Y组合使用形成网点
4. np.hstack()把X,Y合并成2500x2的样本矩阵
5. net.inference()做推理，得到结果output
6. 把结果再转成50x50的形状并赋值给Z，与X、Y的50x50的网格点匹配
7. 最后返回三维点阵XYZ
8. 函数ShowResult3D()使用ax.plot_surface()函数绘制空间曲面
9. 然后在空间中绘制4个样本点，X和Y值就是原始的样本值x1和x2，Z值是原始的标签值y，即0或1

最后得到表10-11的结果。

表10-11 异或分类结果可视化

|斜侧视角|顶视角|
|---|---|
|<img src='../Images/10/xor_result_3D_1.png'/>|<img src='../Images/10/xor_result_3D_2.png'/>|

这下子我们立刻就明白了神经网络都做了些什么事情：它通过样本点，推算出了平面上每个坐标点的分类结果概率，形成空间曲面，然后拦腰一刀（一个切面），这样神经网络就可以在Z=0.5出画一个平面，完美地分开对角顶点。如果看顶视图，与我们在前面生成的2D区域染色图极为相似，它的红色区域的概率值接近于1，蓝色区域的概率值接近于0，在红蓝之间的颜色，代表了从0到1的渐变值。

平面上分割两类的直线，只是我们的想象：使用0.5为门限值像国界一样把两部分数据分开。但实际上，神经网络的输出是个概率，即，它可以告诉你某个点属于某个类别的概率是多少，我们人为地设定为当概率大于0.5时属于正类，小于0.5时属于负类。在空间曲面中，可以把过渡区也展示出来，让大家更好地理解。

#### 2.5D图

3D图虽然有趣，但是2D图已经能表达分类的意思了，只是不完美，那我们想办法做一个2.5D图吧。

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

2.5D图通过颜色来表示不同区域的概率值，可以看到红色区和蓝色区分别是概率接近于0和1的区域，对应着两类样本点。我们后面将会使用这种方式继续研究分类问题。

但是神经网络真的可以聪明到用升维的方式来解决问题吗？我们只是找到了一种能自圆其说的解释，但是不能确定神经网络就是这样工作的。下面我们会通过探查神经网络的训练过程，来理解它究竟是怎样学习的。

### 10.4.3 探查训练的过程

随着迭代次数的增加，对异或二分类问题的分类结果也越来越精确，我们不妨观察一下训练过程中的几个阶段，来试图理解神经网络的训练过程。

在下面的试验中，我们指定500、2000、6000三个迭代次数，来查看各个阶段的分类情况。

表10-12 异或分类训练过程中Z1和A1的值的演变

|迭代次数|Z1的演变|A1的演变|
|---|---|---|
|500次|<img src='../Images/10/xor_z1_500.png'/>|<img src='../Images/10/xor_a1_500.png'/>|
|2000次|<img src='../Images/10/xor_z1_2000.png'/>|<img src='../Images/10/xor_a1_2000.png'/>|
|6000次|<img src='../Images/10/xor_z1_6000.png'/>|<img src='../Images/10/xor_a1_6000.png'/>|

从上图Z1演变过程看，神经网络试图使得两个红色的点重合，而两个蓝色的点距离越远越好，但是中心对称的。

从A1的演变过程看，和Z1差不多，但最后的目的是使得红色点处于[0,1]空间两个顶点（和原始数据一样的位置），蓝色点重合于一个角落。从A1的演变过程最后一张图来看，两个红色点已经被挤压到了一起，所以完全可以有一根分割线把二者分开，如图10-14所示。

<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/10/xor_a1_6000_line.png" ch="500" />

图10-14 经过空间变换后的两类样本数据

也就是说到了这一步，神经网络已经不需要做升维演算了，在二维平面上就可以解决分类问题了。从笔者个人的观点出发，更愿意相信这才是神经网络的工作原理。

下面我们再看看最后的分类结果的演变过程，如表10-13所示。

表10-13 异或分类训练过程中分类函数值和结果的演变

|迭代次数|分类函数值的演变|分类结果的演变|
|---|---|---|
|500次|<img src='../Images/10/xor_logistic_500.png'/>|<img src='../Images/10/xor_result_500.png'/>|
|2000次|<img src='../Images/10/xor_logistic_2000.png'/>|<img src='../Images/10/xor_result_2000.png'/>|
|6000次|<img src='../Images/10/xor_logistic_6000.png'/>|<img src='../Images/10/xor_result_6000.png'/>|

从分类函数情况看，开始时完全分不开两类点，随着学习过程的加深，两类点逐步地向两端移动，直到最后尽可能地相距很远。从分类结果的2.5D图上，可以看出这个方形区域内的每个点的概率变化，由于样本点的对称分布，最后形成了带状的概率分布图。

### 10.4.4 隐层神经元数量的影响

一般来说，隐层的神经元数量要大于等于输入特征的数量，在本例中特征值数量是2。出于研究目的，笔者使用了6种数量的神经元配置来试验神经网络的工作情况，请看表10-14中的比较图。

表10-14 隐层神经元数量对分类结果的影响

|||
|---|---|
|<img src='../Images/10/xor_n1.png'/>|<img src='../Images/10/xor_n2.png'/>|
|1个神经元，无法完成分类任务|2个神经元，迭代6200次到达精度要求|
|<img src='../Images/10/xor_n3.png'/>|<img src='../Images/10/xor_n4.png'/>|
|3个神经元，迭代4900次到达精度要求|4个神经元，迭代4300次到达精度要求|
|<img src='../Images/10/xor_n8.png'/>|<img src='../Images/10/xor_n16.png'/>|
|8个神经元，迭代4400次到达精度要求|16个神经元，迭代4500次到达精度要求|

以上各情况的迭代次数是在Xavier初始化的情况下测试一次得到的数值，并不意味着神经元越多越好，合适的数量才好。总结如下：

- 2个神经元肯定是足够的；
- 4个神经元肯定要轻松一些，用的迭代次数最少。
- 而更多的神经元也并不是更轻松，比如8个神经元，杀鸡用了宰牛刀，由于功能过于强大，出现了曲线的分类边界；
- 而16个神经元更是事倍功半地把4个样本分到了4个区域上，当然这也给了我们一些暗示：神经网络可以做更强大的事情！
- 表中图3的分隔带角度与前面几张图相反，但是红色样本点仍处于蓝色区，蓝色样本点仍处于红色区，这个性质没有变。这只是初始化参数不同造成的神经网络的多个解，与神经元数量无关。


## 10.5 实现双弧形二分类

逻辑异或问题的成功解决，可以带给我们一定的信心，但是毕竟只有4个样本，还不能发挥出双层神经网络的真正能力。下面让我们一起来解决问题二，复杂的二分类问题。

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

蓝色的线条是小批量训练样本的曲线，波动相对较大，不必理会，因为批量小势必会造成波动。红色曲线是验证集的走势，可以看到二者的走势很理想，经过一小段时间的磨合后，从第200个`epoch`开始，两条曲线都突然找到了突破的方向，然后只用了50个`epoch`，就迅速达到指定精度。

同时在控制台会打印一些信息，最后几行如下：

```
......
epoch=259, total_iteration=18719
loss_train=0.092687, accuracy_train=1.000000
loss_valid=0.074073, accuracy_valid=1.000000
W= [[ 8.88189429  6.09089509]
 [-7.45706681  5.07004428]]
B= [[ 1.99109895 -7.46281087]]
W= [[-9.98653838]
 [11.04185384]]
B= [[3.92199463]]
testing...
1.0
```
一共用了260个`epoch`，达到了指定的loss精度（0.08）时停止迭代。看测试集的情况，准确度1.0，即100%分类正确。


## 10.6 双弧形二分类的工作原理

在异或问题中，我们知道了如果使用三维坐标系来分析平面上任意复杂的分类问题，都可以迎刃而解：只要把不同的类别的点通过三维线性变换把它们向上升起，就很容易地分开不同类别的样本。但是这种解释有些牵强，笔者不认为神经网络已经聪明到这个程度了。

所以，笔者试图在二维平面上继续研究，寻找真正的答案，恰巧读到了关于流式学习的一些资料，于是做了下述试验，来验证神经网络到底在二维平面上做了什么样的空间变换。

### 10.6.1 两层神经网络的可视化

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
接收`max_epoch`做为参数，控制神经网络训练迭代的次数，以此来观察中间结果。我们使用了如下超参：

- `n_input=2`，输入的特征值数量
- `n_hidden=2`，隐层的神经元数
- `n_output=1`，输出为二分类
- `eta=0.1`，学习率
- `batch_size=5`，批量样本数为5
- `eps=0.01`，停止条件
- `NetType.BinaryClassifier`，二分类网络
- `InitialMethod.Xavier`，初始化方法为Xavier

每迭代5次做一次损失值计算，打印一次结果。最后显示中间状态图和分类结果图。

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

分析表10-15中各列图片的变化，我们可以得到以下结论：

1. 在第一层的线性变换中，原始样本被斜侧拉伸，角度渐渐左倾到40度，并且样本间距也逐渐拉大，原始样本归一化后在[0,1]之间，最后已经拉到了[-5,15]的范围。这种侧向拉伸实际上是为激活函数做准备。
2. 在激活函数计算中，由于激活函数的非线性，所以空间逐渐扭曲变形，使得红色样本点逐步向右下角移动，并变得稠密；而蓝色样本点逐步向左上方扩撒，相信它的极限一定是[0,1]空间的左边界和上边界；另外一个值得重点说明的就是，通过空间扭曲，红蓝两类之间可以用一条直线分割了！这是一件非常神奇的事情。
3. 最后的分类结果，从毫无头绪到慢慢向上拱起，然后是宽而模糊的分类边界，最后形成非常锋利的边界。

似乎到了这里，我们可以得出结论了：神经网络通过空间变换的方式，把线性不可分的样本变成了线性可分的样本，从而给最后的分类变得很容易。

<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/10/sin_a1_line.png" ch="500" />

图10-17 经过空间变换后的样本数据

如图10-17中的那条绿色直线，很轻松就可以完成二分类任务。这条直线如果还原到原始样本图片中，将会是上表中第四列的分类结果的最后一张图的样子。



# 第11章 多入多出的双层神经网络 - 非线性多分类

## 11.0 非线性多分类问题

### 11.0.1 提出问题：铜钱孔形问题

前面用异或问题和弧形样本学习了二分类，现在我们看看如何用它来做非线性多分类。

我们有如表11-1所示的1000个样本和标签。

表11-1 多分类问题数据样本

|样本|$x_1$|$x_2$|$y$|
|---|---|---|---|
|1|0.22825111|-0.34587097|2|
|2|0.20982606|0.43388447|3|
|...|...|...|...|
|1000|0.38230143|-0.16455377|2|

还好这个数据只有两个特征，所以我们可以用可视化的方法展示，如图11-1。

<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/11/data.png" ch="500" />

图11-1 可视化样本数据

一共有3个类别：

1. 蓝色方点
2. 红色叉点
3. 绿色圆点

样本组成了一个貌似铜钱的形状，我们就把这个问题叫做“铜钱孔形分类”问题吧，后面还要再提到。

**问题：如何用两层神经网络实现这个铜钱孔三分类问题？**

三种颜色的点有规律地占据了一个单位平面内$(-0.5,0.5)$的不同区域，从图中可以明显看出，这不是线性可分问题，而单层神经网络只能做线性分类，如果想做非线性分类，需要至少两层神经网络来完成。

红绿两色是圆形边界分割，红蓝两色是个矩形边界，都是有规律的。但是，学习神经网络，要忘记“规律”这个词，对于神经网络来说，数学上的“有规律”或者“无规律”是没有意义的，对于它来说一概都是无规律，训练难度是一模一样的。

另外，边界也是无意义的，要用概率来理解：没有一条非0即1的分界线来告诉我们哪些点应该属于哪个区域，我们可以得到的是处于某个位置的点属于三个类别的概率有多大，然后我们从中取概率最大的那个类别作为最终判断结果。

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
 
总体的准确率是89.67%。三类的精确率是90%、84%、95%。实际上表11-2也是混淆矩阵在二分类基础上的扩展形式，其特点是在对角线上的值越大越好。

当然也可以计算每个类别的Precision和Recall，但是只在需要时才去做具体计算。比如，当第2类和第3类混淆比较严重时，为了记录模型训练的历史情况，才会把第2类和第3类单独拿出来分析。

我们在本书中，只使用总体的准确率来衡量多分类器的好坏。
