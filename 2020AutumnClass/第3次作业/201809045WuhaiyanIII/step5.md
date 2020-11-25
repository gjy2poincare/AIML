# 第五步  非线性分类

## 摘要
我们将会在使用双层网络解决异或问题。

异或问题是个简单的二分类问题，因为毕竟只有4个样本数据，我们会用更复杂的数据样本来学习非线性多分类问题，并理解其工作原理。

然后我们将会用一个稍微复杂些的二分类例子，来说明在二维平面上，神经网络是通过怎样的神奇的线性变换加激活函数压缩，把线性不可分的问题转化为线性可分问题的。

解决完二分类问题，我们将学习如何解决更复杂的三分类问题，由于样本的复杂性，必须在隐层使用多个神经元才能完成分类任务。

最后我们将搭建一个三层神经网络，来解决MNIST手写数字识别问题，并学习使用梯度检查来帮助我们测试反向传播代码的正确性。

# 第10章 多入单出的双层神经网络 - 非线性二分类

## 10.0 非线性二分类问题

### 10.0.1 提出问题一：异或问题

<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/10/xor_source_data.png" ch="500" />

图10-1 异或问题的样本数据

从图10-1看，两类样本点（红色叉子和蓝色圆点）交叉分布在[0,1]空间的四个角上，用一条直线无法分割开两类样本。神经网络是建立在感知器的基础上的，那么我们用神经网络如何解决异或问题呢？

### 10.0.2 提出问题二：双弧形问题

我们给出一个比异或问题要稍微复杂些的问题，如图10-2所示。

<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/10/sin_data_source.png" ch="500" />

图10-2 呈弧线分布的两类样本数据

平面上有两类样本数据，都成弧形分布，由于弧度的存在，使得我们无法使用一根直线来分开红蓝两种样本点，那么神经网络能用一条曲线来分开它们吗？

### 10.0.3 二分类模型的评估标准

#### 准确率 Accuracy

也可以称之为精度，我们在本书中混用这两个词。

对于二分类问题，假设测试集上一共1000个样本，其中550个正例，450个负例。测试一个模型时，得到的结果是：521个正例样本被判断为正类，435个负例样本被判断为负类，则正确率计算如下：

$$Accuracy=(521+435)/1000=0.956$$

即正确率为95.6%。这种方式对多分类也是有效的，即三类中判别正确的样本数除以总样本数，即为准确率。

但是这种计算方法丢失了很多细节，比如：是正类判断的精度高还是负类判断的精度高呢？

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

ROC曲线有个很好的特性：当测试集中的正负样本的分布变换的时候，ROC曲线能够保持不变。在实际的数据集中经常会出现样本类不平衡，即正负样本比例差距较大，而且测试数据中的正负样本也可能随着时间变化。

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

我们可以模拟这个思路，用两层神经网络搭建模型，来解决非线性分类问题。


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

在前向计算过程中，最后一个公式是Logistic函数，它把$(-\infty, +\infty)$压缩到了(0,1)之间，相当于计算了一个概率值，然后通过概率值大于0.5与否，判断是否属于正类。虽然异或问题只有4个样本点，但是如果：

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

可以看到，两类样本点被分在了不同颜色的区域内，这让我们恍然大悟，原来神经网络可以同时画两条分割线的，更准确的说法是“可以画出两个分类区域”。

### 10.4.2 更直观的可视化结果

#### 3D图

用一个平面拟合了空间中的样本点，如表10-10所示。

表10-10 平面拟合的可视化结果

|正向|侧向|
|---|---|
|[![DVIPC4.jpg](https://s3.ax1x.com/2020/11/17/DVIPC4.jpg)](https://imgchr.com/i/DVIPC4)|[![DVIl2d.jpg](https://s3.ax1x.com/2020/11/17/DVIl2d.jpg)](https://imgchr.com/i/DVIl2d)|

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
|[![DVoio8.jpg](https://s3.ax1x.com/2020/11/17/DVoio8.jpg)](https://imgchr.com/i/DVoio8)|[![DVo3YF.jpg](https://s3.ax1x.com/2020/11/17/DVo3YF.jpg)](https://imgchr.com/i/DVo3YF)|

它通过样本点，推算出了平面上每个坐标点的分类结果概率，形成空间曲面，然后拦腰一刀（一个切面），这样神经网络就可以在Z=0.5出画一个平面，完美地分开对角顶点。如果看顶视图，与我们在前面生成的2D区域染色图极为相似，它的红色区域的概率值接近于1，蓝色区域的概率值接近于0，在红蓝之间的颜色，代表了从0到1的渐变值。

实际上，神经网络的输出是个概率，即，它可以告诉你某个点属于某个类别的概率是多少，我们人为地设定为当概率大于0.5时属于正类，小于0.5时属于负类。在空间曲面中，可以把过渡区也展示出来，让大家更好地理解。

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

2.5D图通过颜色来表示不同区域的概率值，可以看到红色区和蓝色区分别是概率接近于0和1的区域，对应着两类样本点。

### 10.4.3 探查训练的过程

在下面的试验中，我们指定500、2000、6000三个迭代次数，来查看各个阶段的分类情况。

表10-12 异或分类训练过程中Z1和A1的值的演变

|迭代次数|Z1的演变|A1的演变|
|---|---|---|
|500次|[![DVT64U.jpg](https://s3.ax1x.com/2020/11/17/DVT64U.jpg)](https://imgchr.com/i/DVT64U)|[![DVTBBq.jpg](https://s3.ax1x.com/2020/11/17/DVTBBq.jpg)](https://imgchr.com/i/DVTBBq)|
|2000次|[![DVT4D1.jpg](https://s3.ax1x.com/2020/11/17/DVT4D1.jpg)](https://imgchr.com/i/DVT4D1)|[![DVTgCF.jpg](https://s3.ax1x.com/2020/11/17/DVTgCF.jpg)](https://imgchr.com/i/DVTgCF)|
|6000次|[![DVTbCD.jpg](https://s3.ax1x.com/2020/11/17/DVTbCD.jpg)](https://imgchr.com/i/DVTbCD)|[![DVToE6.jpg](https://s3.ax1x.com/2020/11/17/DVToE6.jpg)](https://imgchr.com/i/DVToE6)|

从上图Z1演变过程看，神经网络试图使得两个红色的点重合，而两个蓝色的点距离越远越好，但是中心对称的。

从A1的演变过程看，和Z1差不多，但最后的目的是使得红色点处于[0,1]空间两个顶点（和原始数据一样的位置），蓝色点重合于一个角落。从A1的演变过程最后一张图来看，两个红色点已经被挤压到了一起，所以完全可以有一根分割线把二者分开，如图10-14所示。

<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/10/xor_a1_6000_line.png" ch="500" />

图10-14 经过空间变换后的两类样本数据

也就是说到了这一步，神经网络已经不需要做升维演算了，在二维平面上就可以解决分类问题了。

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
|[![DVjyp6.jpg](https://s3.ax1x.com/2020/11/17/DVjyp6.jpg)](https://imgchr.com/i/DVjyp6)|[![DVjgXD.jpg](https://s3.ax1x.com/2020/11/17/DVjgXD.jpg)](https://imgchr.com/i/DVjgXD)|
|1个神经元，无法完成分类任务|2个神经元，迭代6200次到达精度要求|
|[![DVjW0H.jpg](https://s3.ax1x.com/2020/11/17/DVjW0H.jpg)](https://imgchr.com/i/DVjW0H)|[![DVj5tI.jpg](https://s3.ax1x.com/2020/11/17/DVj5tI.jpg)](https://imgchr.com/i/DVj5tI)|
|3个神经元，迭代4900次到达精度要求|4个神经元，迭代4300次到达精度要求|
|[![DVjLng.jpg](https://s3.ax1x.com/2020/11/17/DVjLng.jpg)](https://imgchr.com/i/DVjLng)|[![DVjLng.jpg](https://s3.ax1x.com/2020/11/17/DVjLng.jpg)](https://imgchr.com/i/DVjLng)|
|8个神经元，迭代4400次到达精度要求|16个神经元，迭代4500次到达精度要求|

以上各情况的迭代次数是在Xavier初始化的情况下测试一次得到的数值，并不意味着神经元越多越好，合适的数量才好。总结如下：

- 2个神经元肯定是足够的；
- 4个神经元肯定要轻松一些，用的迭代次数最少。
- 而更多的神经元也并不是更轻松，比如8个神经元，杀鸡用了宰牛刀，由于功能过于强大，出现了曲线的分类边界；
- 而16个神经元更是事倍功半地把4个样本分到了4个区域上，当然这也给了我们一些暗示：神经网络可以做更强大的事情！
- 表中图3的分隔带角度与前面几张图相反，但是红色样本点仍处于蓝色区，蓝色样本点仍处于红色区，这个性质没有变。这只是初始化参数不同造成的神经网络的多个解，与神经元数量无关。


## 10.5 实现双弧形二分类

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


## 11.1 非线性多分类

### 11.1.1 定义神经网络结构

先设计出能完成非线性多分类的网络结构，如图11-2所示。

<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/11/nn.png" />

图11-2 非线性多分类的神经网络结构图

- 输入层两个特征值$x_1, x_2$
$$
x=
\begin{pmatrix}
    x_1 & x_2
\end{pmatrix}
$$
- 隐层$2\times 3$的权重矩阵$W1$
$$
W1=
\begin{pmatrix}
    w1_{11} & w1_{12} & w1_{13} \\\\
    w1_{21} & w1_{22} & w1_{23}
\end{pmatrix}
$$

- 隐层$1\times 3$的偏移矩阵$B1$

$$
B1=\begin{pmatrix}
    b1_1 & b1_2 & b1_3 
\end{pmatrix}
$$

- 隐层由3个神经元构成
- 输出层$3\times 3$的权重矩阵$W2$
$$
W2=\begin{pmatrix}
    w2_{11} & w2_{12} & w2_{13} \\\\
    w2_{21} & w2_{22} & w2_{23} \\\\
    w2_{31} & w2_{32} & w2_{33} 
\end{pmatrix}
$$

- 输出层$1\times 1$的偏移矩阵$B2$

$$
B2=\begin{pmatrix}
    b2_1 & b2_2 & b2_3 
  \end{pmatrix}
$$

- 输出层有3个神经元使用Softmax函数进行分类

### 11.1.2 前向计算

根据网络结构，可以绘制前向计算图，如图11-3所示。

<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/11/multiple_forward.png" />

图11-3 前向计算图

#### 第一层

- 线性计算

$$
z1_1 = x_1 w1_{11} + x_2 w1_{21} + b1_1
$$
$$
z1_2 = x_1 w1_{12} + x_2 w1_{22} + b1_2
$$
$$
z1_3 = x_1 w1_{13} + x_2 w1_{23} + b1_3
$$
$$
Z1 = X \cdot W1 + B1
$$

- 激活函数

$$
a1_1 = Sigmoid(z1_1) 
$$
$$
a1_2 = Sigmoid(z1_2) 
$$
$$
a1_3 = Sigmoid(z1_3) 
$$
$$
A1 = Sigmoid(Z1)
$$

#### 第二层

- 线性计算

$$
z2_1 = a1_1 w2_{11} + a1_2 w2_{21} + a1_3 w2_{31} + b2_1
$$
$$
z2_2 = a1_1 w2_{12} + a1_2 w2_{22} + a1_3 w2_{32} + b2_2
$$
$$
z2_3 = a1_1 w2_{13} + a1_2 w2_{23} + a1_3 w2_{33} + b2_3
$$
$$
Z2 = A1 \cdot W2 + B2
$$

- 分类函数

$$
a2_1 = \frac{e^{z2_1}}{e^{z2_1} + e^{z2_2} + e^{z2_3}}
$$
$$
a2_2 = \frac{e^{z2_2}}{e^{z2_1} + e^{z2_2} + e^{z2_3}}
$$
$$
a2_3 = \frac{e^{z2_3}}{e^{z2_1} + e^{z2_2} + e^{z2_3}}
$$
$$
A2 = Softmax(Z2)
$$

#### 损失函数

使用多分类交叉熵损失函数：
$$
loss = -(y_1 \ln a2_1 + y_2 \ln a2_2 + y_3 \ln a2_3)
$$
$$
J(w,b) = -\frac{1}{m} \sum^m_{i=1} \sum^n_{j=1} y_{ij} \ln (a2_{ij})
$$

$m$为样本数，$n$为类别数。

### 11.1.3 反向传播

根据前向计算图，可以绘制出反向传播的路径如图11-4。

<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/11/multiple_backward.png" />

图11-4 反向传播图

在第7.1中学习过了Softmax与多分类交叉熵配合时的反向传播推导过程，最后是一个很简单的减法：

$$
\frac{\partial loss}{\partial Z2}=A2-y \rightarrow dZ2
$$

从Z2开始再向前推的话，和10.2节是一模一样的，所以直接把结论拿过来：

$$
\frac{\partial loss}{\partial W2}=A1^{\top} \cdot dZ2 \rightarrow dW2
$$
$$\frac{\partial{loss}}{\partial{B2}}=dZ2 \rightarrow dB2$$
$$
\frac{\partial A1}{\partial Z1}=A1 \odot (1-A1) \rightarrow dA1
$$
$$
\frac{\partial loss}{\partial Z1}=dZ2 \cdot W2^{\top} \odot dA1 \rightarrow dZ1 
$$
$$
dW1=X^{\top} \cdot dZ1
$$
$$
dB1=dZ1
$$

### 11.1.4 代码实现

绝大部分代码都在`HelperClass2`目录中的基本类实现，这里只有主过程：

```Python
if __name__ == '__main__':
    ......
    n_input = dataReader.num_feature
    n_hidden = 3
    n_output = dataReader.num_category
    eta, batch_size, max_epoch = 0.1, 10, 5000
    eps = 0.1
    hp = HyperParameters2(n_input, n_hidden, n_output, eta, max_epoch, batch_size, eps, NetType.MultipleClassifier, InitialMethod.Xavier)
    # create net and train
    net = NeuralNet2(hp, "Bank_233")
    net.train(dataReader, 100, True)
    net.ShowTrainingTrace()
    # show result
    ......
```

过程描述：

1. 读取数据文件
2. 显示原始数据样本分布图
3. 其它数据操作：归一化、打乱顺序、建立验证集
4. 设置超参
5. 建立神经网络开始训练
6. 显示训练结果

### 11.1.5 运行结果

训练过程如图11-5所示。

<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/11/loss.png" />

图11-5 训练过程中的损失函数值和准确率值的变化

迭代了5000次，没有到达损失函数小于0.1的条件。

分类结果如图11-6所示。

<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/11/result.png" ch="500" />

图11-6 分类效果图

因为没达到精度要求，所以分类效果一般。从分类结果图上看，外圈圆形差不多拟合住了，但是内圈的方形还差很多。

打印输出：

```
......
epoch=4999, total_iteration=449999
loss_train=0.225935, accuracy_train=0.800000
loss_valid=0.137970, accuracy_valid=0.960000
W= [[ -8.30315494   9.98115605   0.97148346]
 [ -5.84460922  -4.09908698 -11.18484376]]
B= [[ 4.85763475 -5.61827538  7.94815347]]
W= [[-32.28586038  -8.60177788  41.51614172]
 [-33.68897413  -7.93266621  42.09333288]
 [ 34.16449693   7.93537692 -41.19340947]]
B= [[-11.11937314   3.45172617   7.66764697]]
testing...
0.952
```
最后的测试分类准确率为0.952。


## 11.2 非线性多分类的工作原理

### 11.2.1 隐层神经元数量的影响

下面列出了隐层神经元为2、4、8、16、32、64的情况，设定好最大epoch数为10000，以比较它们各自能达到的最大精度值的区别。每个配置只测试一轮，所以测出来的数据有一定的随机性。

表11-3展示了隐层神经元数与分类结果的关系。

表11-3 神经元数与网络能力及分类结果的关系

|神经元数|损失函数|分类结果|
|---|---|---|
|2|![](https://i.loli.net/2020/11/17/siU4YjkVKXd3zfh.jpg)|![](https://i.loli.net/2020/11/17/Ux6HhySXmZIdLgA.jpg)|
||测试集准确度0.618，耗时49秒，损失函数值0.795。类似这种曲线的情况，损失函数值降不下去，准确度值升不上去，主要原因是网络能力不够。|没有完成分类任务|
|4|![](https://i.loli.net/2020/11/17/lAmJ1OaTgoseCXY.jpg)|![](https://i.loli.net/2020/11/17/BG3Ll7Xj1xQbHkh.jpg)|
||测试准确度0.954，耗时51秒，损失函数值0.132。虽然可以基本完成分类任务，网络能力仍然不够。|基本完成，但是边缘不够清晰|
|8|![](https://i.loli.net/2020/11/17/psFubz92Jr376XC.jpg)|![](https://i.loli.net/2020/11/17/VgXWR4vzZ6lLi2D.jpg)|
||测试准确度0.97，耗时52秒，损失函数值0.105。可以先试试在后期衰减学习率，如果再训练5000轮没有改善的话，可以考虑增加网络能力。|基本完成，但是边缘不够清晰|
|16|![](https://i.loli.net/2020/11/17/G8CynBRiSaITegH.jpg)|![](https://i.loli.net/2020/11/17/LmlRidyJeu8aWSx.jpg)|
||测试准确度0.978，耗时53秒，损失函数值0.094。同上，可以同时试着使用优化算法，看看是否能收敛更快。|较好地完成了分类任务|
|32|![](https://i.loli.net/2020/11/17/s3Ru7ICqTjKXioY.jpg)|![](https://i.loli.net/2020/11/17/oEAd6YqBKkPbe7r.jpg)|
||测试准确度0.974，耗时53秒，损失函数值0.085。网络能力够了，从损失值下降趋势和准确度值上升趋势来看，可能需要更多的迭代次数。|较好地完成了分类任务|
|64|![](https://i.loli.net/2020/11/17/TgXU9hMqYjQv4GW.jpg)|![](https://i.loli.net/2020/11/17/tYezONUFJIadlpv.jpg)|
||测试准确度0.972，耗时64秒，损失函数值0.075。网络能力足够。|较好地完成了分类任务|

### 11.2.2 三维空间内的变换过程
如果隐层是两个神经元，我们可以把两个神经元的输出数据看作横纵坐标，在二维平面上绘制中间结果；由于使用了3个神经元，使得隐层的输出结果为每个样本一行三列，我们必须在三维空间中绘制中间结果。

```Python
def Show3D(net, dr):
    ......
```

上述代码首先使用测试集（500个样本点）在已经训练好的网络上做一次推理，得到了隐层的计算结果，然后分别用`net.Z1`和`net.A1`的三列数据做为XYZ坐标值，绘制三维点图，列在表11-4中做比较。



#### 3D分类结果图

更高维的空间无法展示，所以当隐层神经元数量为4或8或更多时，基本无法理解空间变换的样子了。但是有一个方法可以近似地解释高维情况：在三维空间时，蓝色点会被推挤到一个角落形成一个三角形，那么在N（N>3）维空间中，蓝色点也会被推挤到一个角落。由于N很大，所以一个多边形会近似成一个圆形，也就是我们下面要生成的这些立体图的样子。

我们延续9.2节的3D效果体验，但是，多分类的实际实现方式是1对多的，所以我们只能一次显示一个类别的分类效果图，列在表11-5中。

表11-5 分类效果图

|||
|---|---|
|<img src='../Images/11/multiple_3d_c1_1.png'/>|<img src='../Images/11/multiple_3d_c2_1.png'/>|
|红色：类别1样本区域|红色：类别2样本区域|
|<img src='../Images/11/multiple_3d_c3_1.png'/>|<img src='../Images/11/multiple_3d_c1_c2_1.png'/>|
|红色：类别3样本区域|红色：类别1，青色：类别2，紫色：类别3|

上表中最后一行的图片显示类别1和2的累加效果。由于最后的结果都被Softmax归一为$[0,1]$之间，所以我们可以简单地把类别1的数据乘以2，再加上类别2的数据即可。


## 11.3 分类样本不平衡问题

### 11.3.1 什么是样本不平衡

英文名叫做Imbalanced Data。

在一般的分类学习方法中都有一个假设，就是不同类别的训练样本的数量相对平衡。

以二分类为例，比如正负例都各有1000个左右。如果是1200:800的比例，也是可以接受的，但是如果是1900:100，就需要有些措施来解决不平衡问题了，否则最后的训练结果很大可能是忽略了负例，将所有样本都分类为正类了。

如果是三分类，假设三个类别的样本比例为：1000:800:600，这是可以接受的；但如果是1000:300:100，就属于不平衡了。它带来的结果是分类器对第一类样本过拟合，而对其它两个类别的样本欠拟合，测试效果一定很糟糕。

类别不均衡问题是现实中很常见的问题，大部分分类任务中，各类别下的数据个数基本上不可能完全相等，但是一点点差异是不会产生任何影响与问题的。在现实中有很多类别不均衡问题，它是常见的，并且也是合理的，符合人们期望的。

在前面，我们使用准确度这个指标来评价分类质量，可以看出，在类别不均衡时，准确度这个评价指标并不能work。比如一个极端的例子是，在疾病预测时，有98个正例，2个反例，那么分类器只要将所有样本预测为正类，就可以得到98%的准确度，则此分类器就失去了价值。

### 11.3.2 如何解决样本不平衡问题

#### 平衡数据集

有一句话叫做“更多的数据往往战胜更好的算法”。所以一定要先想办法扩充样本数量少的类别的数据，比如目前的正负类样本数量是1000:100，则可以再搜集2000个数据，最后得到了2800:300的比例，此时可以从正类样本中丢弃一些，变成500:300，就可以训练了。

一些经验法则：

- 考虑对大类下的样本（超过1万、十万甚至更多）进行欠采样，即删除部分样本；
- 考虑对小类下的样本（不足1万甚至更少）进行过采样，即添加部分样本的副本；
- 考虑尝试随机采样与非随机采样两种采样方法；
- 考虑对各类别尝试不同的采样比例，比一定是1:1，有时候1:1反而不好，因为与现实情况相差甚远；
- 考虑同时使用过采样（over-sampling）与欠采样（under-sampling）。

#### 尝试其它评价指标 

从前面的分析可以看出，准确度这个评价指标在类别不均衡的分类任务中并不能work，甚至进行误导（分类器不work，但是从这个指标来看，该分类器有着很好的评价指标得分）。因此在类别不均衡分类任务中，需要使用更有说服力的评价指标来对分类器进行评价。如何对不同的问题选择有效的评价指标参见这里。 

常规的分类评价指标可能会失效，比如将所有的样本都分类成大类，那么准确率、精确率等都会很高。这种情况下，AUC是最好的评价指标。

#### 尝试产生人工数据样本 

一种简单的人工样本数据产生的方法便是，对该类下的所有样本每个属性特征的取值空间中随机选取一个组成新的样本，即属性值随机采样。你可以使用基于经验对属性值进行随机采样而构造新的人工样本，或者使用类似朴素贝叶斯方法假设各属性之间互相独立进行采样，这样便可得到更多的数据，但是无法保证属性之前的线性关系（如果本身是存在的）。 

有一个系统的构造人工数据样本的方法SMOTE(Synthetic Minority Over-sampling Technique)。SMOTE是一种过采样算法，它构造新的小类样本而不是产生小类中已有的样本的副本，即该算法构造的数据是新样本，原数据集中不存在的。该基于距离度量选择小类别下两个或者更多的相似样本，然后选择其中一个样本，并随机选择一定数量的邻居样本对选择的那个样本的一个属性增加噪声，每次处理一个属性。这样就构造了更多的新生数据。具体可以参见原始论文。 

使用命令：

```
pip install imblearn
```
可以安装SMOTE算法包，用于实现样本平衡。

#### 尝试一个新的角度理解问题 

我们可以从不同于分类的角度去解决数据不均衡性问题，我们可以把那些小类的样本作为异常点(outliers)，因此该问题便转化为异常点检测(anomaly detection)与变化趋势检测问题(change detection)。 

异常点检测即是对那些罕见事件进行识别。如通过机器的部件的振动识别机器故障，又如通过系统调用序列识别恶意程序。这些事件相对于正常情况是很少见的。 

变化趋势检测类似于异常点检测，不同在于其通过检测不寻常的变化趋势来识别。如通过观察用户模式或银行交易来检测用户行为的不寻常改变。 

将小类样本作为异常点这种思维的转变，可以帮助考虑新的方法去分离或分类样本。这两种方法从不同的角度去思考，让你尝试新的方法去解决问题。

#### 修改现有算法

- 设超大类中样本的个数是极小类中样本个数的L倍，那么在随机梯度下降（SGD，stochastic gradient descent）算法中，每次遇到一个极小类中样本进行训练时，训练L次。
- 将大类中样本划分到L个聚类中，然后训练L个分类器，每个分类器使用大类中的一个簇与所有的小类样本进行训练得到。最后对这L个分类器采取少数服从多数对未知类别数据进行分类，如果是连续值（预测），那么采用平均值。
- 设小类中有N个样本。将大类聚类成N个簇，然后使用每个簇的中心组成大类中的N个样本，加上小类中所有的样本进行训练。

无论你使用前面的何种方法，都对某个或某些类进行了损害。为了不进行损害，那么可以使用全部的训练集采用多种分类方法分别建立分类器而得到多个分类器，采用投票的方式对未知类别的数据进行分类，如果是连续值（预测），那么采用平均值。
在最近的ICML论文中，表明增加数据量使得已知分布的训练集的误差增加了，即破坏了原有训练集的分布，从而可以提高分类器的性能。这篇论文与类别不平衡问题不相关，因为它隐式地使用数学方式增加数据而使得数据集大小不变。但是，我认为破坏原有的分布是有益的。

#### 集成学习

一个很好的方法去处理非平衡数据问题，并且在理论上证明了。这个方法便是由Robert E. Schapire于1990年在Machine Learning提出的”The strength of weak learnability” ，该方法是一个boosting算法，它递归地训练三个弱学习器，然后将这三个弱学习器结合起形成一个强的学习器。我们可以使用这个算法的第一步去解决数据不平衡问题。 

1. 首先使用原始数据集训练第一个学习器L1；
2. 然后使用50%在L1学习正确和50%学习错误的那些样本训练得到学习器L2，即从L1中学习错误的样本集与学习正确的样本集中，循环一边采样一个；
3. 接着，使用L1与L2不一致的那些样本去训练得到学习器L3；
4. 最后，使用投票方式作为最后输出。 

那么如何使用该算法来解决类别不平衡问题呢？ 

假设是一个二分类问题，大部分的样本都是true类。让L1输出始终为true。使用50%在L1分类正确的与50%分类错误的样本训练得到L2，即从L1中学习错误的样本集与学习正确的样本集中，循环一边采样一个。因此，L2的训练样本是平衡的。L使用L1与L2分类不一致的那些样本训练得到L3，即在L2中分类为false的那些样本。最后，结合这三个分类器，采用投票的方式来决定分类结果，因此只有当L2与L3都分类为false时，最终结果才为false，否则true。 


# 第12章 多入多出的三层神经网络 - 深度非线性多分类

## 12.0 多变量非线性多分类

### 12.0.1 提出问题

手写识别是人工智能的重要课题之一。MNIST数字手写体识别图片集，大家一定不陌生，下面就是一些样本。

<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/12/Mnist.png" />

图12-1 MNIST数据集样本示例

由于这是从欧美国家和地区收集的数据，从图中可以看出有几点和中国人的手写习惯不一样：

- 数字2，下面多一个圈
- 数字4，很多横线不出头
- 数字6，上面是直的
- 数字7，中间有个横杠

不过这些细节不影响咱们学习课程，正好还可以验证一下中国人的手写习惯是否能够被正确识别。

由于不是让我们识别26个英文字母或者3500多个常用汉字，所以问题还算是比较简单，不需要图像处理知识，也暂时不需要卷积神经网络的参与。咱们可以试试用一个三层的神经网络解决此问题，把每个图片的像素都当作一个向量来看，而不是作为点阵。

#### 图片数据归一化

在第5章中，我们学习了数据归一化，是针对数据的特征值做的处理，也就是针对样本数据的列做的处理。

本章中，要处理的对象是图片，需要把整张图片看作一个样本，因此使用下面这段代码做数据归一化方法：

```Python
    def __NormalizeData(self, XRawData):
        X_NEW = np.zeros(XRawData.shape)
        x_max = np.max(XRawData)
        x_min = np.min(XRawData)
        X_NEW = (XRawData - x_min)/(x_max-x_min)
        return X_NEW
```


## 12.1 三层神经网络的实现

### 12.1.1 定义神经网络

为了完成MNIST分类，我们需要设计一个三层神经网络结构，如图12-2所示。

<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/12/nn3.png" ch="500" />

图12-2 三层神经网络结构

#### 输入层

共计$28\times 28=784$个特征值：

$$
X=\begin{pmatrix}
    x_1 & x_2 & \cdots & x_{784}
  \end{pmatrix}
$$

#### 隐层1

- 权重矩阵$W1$形状为$784\times 64$

$$
W1=\begin{pmatrix}
    w1_{1,1} & w1_{1,2} & \cdots & w1_{1,64} \\\\
    \vdots & \vdots & \cdots & \vdots \\\\
    w1_{784,1} & w1_{784,2} & \cdots & w1_{784,64} 
  \end{pmatrix}
$$

- 偏移矩阵$B1$的形状为$1\times 64$

$$
B1=\begin{pmatrix}
    b1_{1} & b1_{2} & \cdots & b1_{64}
  \end{pmatrix}
$$

- 隐层1由64个神经元构成，其结果为$1\times 64$的矩阵

$$
Z1=\begin{pmatrix}
    z1_{1} & z1_{2} & \cdots & z1_{64}
  \end{pmatrix}
$$
$$
A1=\begin{pmatrix}
    a1_{1} & a1_{2} & \cdots & a1_{64}
  \end{pmatrix}
$$

#### 隐层2

- 权重矩阵$w2$形状为$64\times 16$

$$
W2=\begin{pmatrix}
    w2_{1,1} & w2_{1,2} & \cdots & w2_{1,16} \\\\
    \vdots & \vdots & \cdots & \vdots \\\\
    w2_{64,1} & w2_{64,2} & \cdots & w2_{64,16} 
  \end{pmatrix}
$$

- 偏移矩阵#B2#的形状是$1\times 16$

$$
B2=\begin{pmatrix}
    b2_{1} & b2_{2} & \cdots & b2_{16}
  \end{pmatrix}
$$

- 隐层2由16个神经元构成

$$
Z2=\begin{pmatrix}
    z2_{1} & z2_{2} & \cdots & z2_{16}
  \end{pmatrix}
$$
$$
A2=\begin{pmatrix}
    a2_{1} & a2_{2} & \cdots & a2_{16}
  \end{pmatrix}
$$

#### 输出层

- 权重矩阵$W3$的形状为$16\times 10$

$$
W3=\begin{pmatrix}
    w3_{1,1} & w3_{1,2} & \cdots & w3_{1,10} \\\\
    \vdots & \vdots & \cdots & \vdots \\\\
    w3_{16,1} & w3_{16,2} & \cdots & w3_{16,10} 
  \end{pmatrix}
$$

- 输出层的偏移矩阵$B3$的形状是$1\times 10$

$$
B3=\begin{pmatrix}
    b3_{1}& b3_{2} & \cdots & b3_{10}
  \end{pmatrix}
$$

- 输出层有10个神经元使用Softmax函数进行分类

$$
Z3=\begin{pmatrix}
    z3_{1} & z3_{2} & \cdots & z3_{10}
  \end{pmatrix}
$$
$$
A3=\begin{pmatrix}
    a3_{1} & a3_{2} & \cdots & a3_{10}
  \end{pmatrix}
$$

### 12.1.2 前向计算

我们都是用大写符号的矩阵形式的公式来描述，在每个矩阵符号的右上角是其形状。

#### 隐层1

$$Z1 = X \cdot W1 + B1 \tag{1}$$

$$A1 = Sigmoid(Z1) \tag{2}$$

#### 隐层2

$$Z2 = A1 \cdot W2 + B2 \tag{3}$$

$$A2 = Tanh(Z2) \tag{4}$$

#### 输出层

$$Z3 = A2 \cdot W3  + B3 \tag{5}$$

$$A3 = Softmax(Z3) \tag{6}$$

我们的约定是行为样本，列为一个样本的所有特征，这里是784个特征，因为图片高和宽均为28，总共784个点，把每一个点的值做为特征向量。

两个隐层，分别定义64个神经元和16个神经元。第一个隐层用Sigmoid激活函数，第二个隐层用Tanh激活函数。

输出层10个神经元，再加上一个Softmax计算，最后有$a1,a2,...a10$共十个输出，分别代表0-9的10个数字。

### 12.1.3 反向传播

和以前的两层网络没有多大区别，只不过多了一层，而且用了tanh激活函数，目的是想把更多的梯度值回传，因为tanh函数比sigmoid函数稍微好一些，比如原点对称，零点梯度值大。

#### 输出层

$$dZ3 = A3-Y \tag{7}$$
$$dW3 = A2^{\top} \cdot dZ3 \tag{8}$$
$$dB3=dZ3 \tag{9}$$

#### 隐层2

$$dA2 = dZ3 \cdot W3^{\top} \tag{10}$$
$$dZ2 = dA2 \odot (1-A2 \odot A2) \tag{11}$$
$$dW2 = A1^{\top} \cdot dZ2 \tag{12}$$
$$dB2 = dZ2 \tag{13}$$

#### 隐层1

$$dA1 = dZ2 \cdot W2^{\top} \tag{14}$$
$$dZ1 = dA1 \odot A1 \odot (1-A1) \tag{15}$$
$$dW1 = X^{\top} \cdot dZ1 \tag{16}$$
$$dB1 = dZ1 \tag{17}$$

### 12.1.4 代码实现

在`HelperClass3` / `NeuralNet3.py`中，下面主要列出与两层网络不同的代码。

#### 初始化

```Python
class NeuralNet3(object):
    def __init__(self, hp, model_name):
        ...
        self.wb1 = WeightsBias(self.hp.num_input, self.hp.num_hidden1, self.hp.init_method, self.hp.eta)
        self.wb1.InitializeWeights(self.subfolder, False)
        self.wb2 = WeightsBias(self.hp.num_hidden1, self.hp.num_hidden2, self.hp.init_method, self.hp.eta)
        self.wb2.InitializeWeights(self.subfolder, False)
        self.wb3 = WeightsBias(self.hp.num_hidden2, self.hp.num_output, self.hp.init_method, self.hp.eta)
        self.wb3.InitializeWeights(self.subfolder, False)
```
初始化部分需要构造三组`WeightsBias`对象，请注意各组的输入输出数量，决定了矩阵的形状。

#### 前向计算

```Python
    def forward(self, batch_x):
        # 公式1
        self.Z1 = np.dot(batch_x, self.wb1.W) + self.wb1.B
        # 公式2
        self.A1 = Sigmoid().forward(self.Z1)
        # 公式3
        self.Z2 = np.dot(self.A1, self.wb2.W) + self.wb2.B
        # 公式4
        self.A2 = Tanh().forward(self.Z2)
        # 公式5
        self.Z3 = np.dot(self.A2, self.wb3.W) + self.wb3.B
        # 公式6
        if self.hp.net_type == NetType.BinaryClassifier:
            self.A3 = Logistic().forward(self.Z3)
        elif self.hp.net_type == NetType.MultipleClassifier:
            self.A3 = Softmax().forward(self.Z3)
        else:   # NetType.Fitting
            self.A3 = self.Z3
        #end if
        self.output = self.A3
```
前向计算部分增加了一层，并且使用`Tanh()`作为激活函数。

- 反向传播
```Python
    def backward(self, batch_x, batch_y, batch_a):
        # 批量下降，需要除以样本数量，否则会造成梯度爆炸
        m = batch_x.shape[0]

        # 第三层的梯度输入 公式7
        dZ3 = self.A3 - batch_y
        # 公式8
        self.wb3.dW = np.dot(self.A2.T, dZ3)/m
        # 公式9
        self.wb3.dB = np.sum(dZ3, axis=0, keepdims=True)/m 

        # 第二层的梯度输入 公式10
        dA2 = np.dot(dZ3, self.wb3.W.T)
        # 公式11
        dZ2,_ = Tanh().backward(None, self.A2, dA2)
        # 公式12
        self.wb2.dW = np.dot(self.A1.T, dZ2)/m 
        # 公式13
        self.wb2.dB = np.sum(dZ2, axis=0, keepdims=True)/m 

        # 第一层的梯度输入 公式8
        dA1 = np.dot(dZ2, self.wb2.W.T) 
        # 第一层的dZ 公式10
        dZ1,_ = Sigmoid().backward(None, self.A1, dA1)
        # 第一层的权重和偏移 公式11
        self.wb1.dW = np.dot(batch_x.T, dZ1)/m
        self.wb1.dB = np.sum(dZ1, axis=0, keepdims=True)/m 

    def update(self):
        self.wb1.Update()
        self.wb2.Update()
        self.wb3.Update()
```
反向传播也相应地增加了一层，注意要用对应的`Tanh()`的反向公式。梯度更新时也是三组权重值同时更新。

- 主过程

```Python
if __name__ == '__main__':
    ......
    n_input = dataReader.num_feature
    n_hidden1 = 64
    n_hidden2 = 16
    n_output = dataReader.num_category
    eta = 0.2
    eps = 0.01
    batch_size = 128
    max_epoch = 40

    hp = HyperParameters3(n_input, n_hidden1, n_hidden2, n_output, eta, max_epoch, batch_size, eps, NetType.MultipleClassifier, InitialMethod.Xavier)
    net = NeuralNet3(hp, "MNIST_64_16")
    net.train(dataReader, 0.5, True)
    net.ShowTrainingTrace(xline="iteration")
```
超参配置：第一隐层64个神经元，第二隐层16个神经元，学习率0.2，批大小128，Xavier初始化，最大训练40个epoch。

### 12.1.5 运行结果

损失函数值和准确度值变化曲线如图12-3。

<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/12/loss.png" />

图12-3 训练过程中损失函数和准确度的变化

打印输出部分：

```
...
epoch=38, total_iteration=16769
loss_train=0.012860, accuracy_train=1.000000
loss_valid=0.100281, accuracy_valid=0.969400
epoch=39, total_iteration=17199
loss_train=0.006867, accuracy_train=1.000000
loss_valid=0.098164, accuracy_valid=0.971000
time used: 25.697904109954834
testing...
0.9749
```

在测试集上得到的准确度为97.49%，比较理想。


## 12.2 梯度检查

### 12.2.1 为何要做梯度检查？

为了确认代码中反向传播计算的梯度是否正确，可以采用梯度检验（gradient check）的方法。通过计算数值梯度，得到梯度的近似值，然后和反向传播得到的梯度进行比较，若两者相差很小的话则证明反向传播的代码是正确无误的。

### 12.2.2 数值微分

#### 导数概念回忆

$$
f'(x)=\lim_{h \to 0} \frac{f(x+h)-f(x)}{h} \tag{1}
$$

其含义就是$x$的微小变化$h$（$h$为无限小的值），会导致函数$f(x)$的值有多大变化。在`Python`中可以这样实现：

```Python
def numerical_diff(f, x):
    h = 1e-5
    d = (f(x+h) - f(x))/h
    return d
```

因为计算机的舍入误差的原因，`h`不能太小，比如`1e-10`，会造成计算结果上的误差，所以我们一般用`[1e-4,1e-7]`之间的数值。

但是如果使用上述方法会有一个问题，如图12-4所示。

<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/12/grad_check.png" ch="500" />

图12-4 数值微分方法

红色实线为真实的导数切线，蓝色虚线是上述方法的体现，即从$x$到$x+h$画一条直线，来模拟真实导数。但是可以明显看出红色实线和蓝色虚线的斜率是不等的。因此我们通常用绿色的虚线来模拟真实导数，公式变为：

$$
f'(x) = \lim_{h \to 0} \frac{f(x+h)-f(x-h)}{2h} \tag{2}
$$

公式2被称为双边逼近方法。

用双边逼近形式会比单边逼近形式的误差小100~10000倍左右，可以用泰勒展开来证明。

#### 泰勒公式

泰勒公式是将一个在$x=x_0$处具有n阶导数的函数$f(x)$利用关于$(x-x_0)$的n次多项式来逼近函数的方法。若函数$f(x)$在包含$x_0$的某个闭区间$[a,b]$上具有n阶导数，且在开区间$(a,b)$上具有$n+1$阶导数，则对闭区间$[a,b]$上任意一点$x$，下式成立：

$$f(x)=\frac{f(x_0)}{0!} + \frac{f'(x_0)}{1!}(x-x_0)+\frac{f''(x_0)}{2!}(x-x_0)^2 + ...+\frac{f^{(n)}(x_0)}{n!}(x-x_0)^n+R_n(x) \tag{3}$$

其中,$f^{(n)}(x)$表示$f(x)$的$n$阶导数，等号后的多项式称为函数$f(x)$在$x_0$处的泰勒展开式，剩余的$R_n(x)$是泰勒公式的余项，是$(x-x_0)^n$的高阶无穷小。 

利用泰勒展开公式，令$x=\theta + h, x_0=\theta$，我们可以得到：

$$f(\theta + h)=f(\theta) + f'(\theta)h + O(h^2) \tag{4}$$

#### 单边逼近误差

如果用单边逼近，把公式4两边除以$h$后变形：

$$f'(\theta) + O(h)=\frac{f(\theta+h)-f(\theta)}{h} \tag{5}$$

公式5已经和公式1的定义非常接近了，只是左侧多出来的第二项，就是逼近的误差，是个$O(h)$级别的误差项。

#### 双边逼近误差

如果用双边逼近，我们用三阶泰勒展开：

令$x=\theta + h, x_0=\theta$，我们可以得到：

$$f(\theta + h)=f(\theta) + f'(\theta)h + f''(\theta)h^2 + O(h^3) \tag{6}$$

再令$x=\theta - h, x_0=\theta$我们可以得到：

$$f(\theta - h)=f(\theta) - f'(\theta)h + f''(\theta)h^2 - O(h^3) \tag{7}$$

公式6减去公式7，有：

$$f(\theta + h) - f(\theta - h)=2f'(\theta)h + 2O(h^3) \tag{8}$$

两边除以$2h$：

$$f'(\theta) + O(h^2)={f(\theta + h) - f(\theta - h) \over 2h} \tag{9}$$

公式9中，左侧多出来的第二项，就是双边逼近的误差，是个$O(h^2)$级别的误差项，比公式5中的误差项小很多数量级。

### 12.2.3 实例说明

公式2就是梯度检查的理论基础。比如一个函数：

$$f(x) = x^2 + 3x$$

我们看一下它在$x=2$处的数值微分，令$h = 0.001$：

$$
\begin{aligned}
f(x+h) &= f(2+0.001) \\\\
&= (2+0.001)^2 + 3 \times (2+0.001) \\\\
&=10.007001
\end{aligned}
$$
$$
\begin{aligned}
f(x-h) &= f(2-0.001) \\\\
&= (2-0.001)^2 + 3 \times (2-0.001) \\\\
&=9.993001
\end{aligned}
$$
$$
\frac{f(x+h)-f(x-h)}{2h}=\frac{10.007001-9.993001}{2 \times 0.001}=7 \tag{10}
$$

再看它的数学解析解：

$$f'(x)=2x+3$$
$$f'(x|_{x=2})=2 \times 2+3=7 \tag{11}$$

可以看到公式10和公式11的结果一致。当然在实际应用中，一般不可能会完全相等，只要两者的误差小于`1e-5`以下，我们就认为是满足精度要求的。

### 12.2.4 算法实现

在神经网络中，我们假设使用多分类的交叉熵函数，则其形式为：

$$J(w,b) =- \sum_{i=1}^m \sum_{j=1}^n y_{ij} \ln a_{ij}$$

m是样本数，n是分类数。

#### 参数向量化

我们需要检查的是关于$W$和$B$的梯度，而$W$和$B$是若干个矩阵，而不是一个标量，所以在进行梯度检验之前，我们先做好准备工作，那就是把矩阵$W$和$B$向量化，然后把神经网络中所有层的向量化的$W$和$B$连接在一起(concatenate)，成为一个大向量，我们称之为$J(\theta)$，然后对通过back-prop过程得到的W和B求导的结果$d\theta_{real}$也做同样的变。

向量化的$W,B$连接以后，统一称作为$\theta$，按顺序用不同下标区分，于是有$J(\theta)$的表达式为：

$$J(w,b)=J(\theta_1,...,\theta_i,...,\theta_n)$$

对于上式中的每一个向量，我们依次使用公式2的方式做检查，于是有对第i个向量值的梯度检查公式：

$$\frac{\partial J}{\partial \theta_i}=\frac{J(\theta_1,...,\theta_i+h,...,\theta_n) - J(\theta_1,...,\theta_i-h,...,\theta_n)}{2h}$$

因为我们是要比较两个向量的对应分量的差别，这个可以用对应分量差的平方和的开方（欧氏距离）来刻画。但是我们不希望得到一个具体的刻画差异的值，而是希望得到一个比率，这也便于我们得到一个标准的梯度检验的要求。

#### 算法

1. 初始化神经网络的所有矩阵参数（可以使用随机初始化或其它非0的初始化方法）
2. 把所有层的$W,B$都转化成向量，按顺序存放在$\theta$中
3. 随机设置$X$值，最好是归一化之后的值，在[0,1]之间
4. 做一次前向计算，再紧接着做一次反向计算，得到各参数的梯度$d\theta_{real}$
5. 把得到的梯度$d\theta_{real}$变化成向量形式，其尺寸应该和第2步中的$\theta$相同，且一一对应（$W$对应$dW$, $B$对应$dB$）
6. 对2中的$\theta$向量中的每一个值，做一次双边逼近，得到$d\theta_{approx}$
7. 比较$d\theta_{real}$和$d\theta_{approx}$的值，通过计算两个向量之间的欧式距离：
   
$$diff = \frac{\parallel d\theta_{real} - d\theta_{approx}\parallel_2}{\parallel d\theta_{approx}\parallel_2 + \parallel d\theta_{real}\parallel_2}$$

结果判断：

1. $diff > 1e^{-2}$
   
   梯度计算肯定出了问题。

2. $1e^{-2} > diff > 1e^{-4}$
   
   可能有问题了，需要检查。

3. $1e^{-4} \gt diff \gt 1e^{-7}$
   
   不光滑的激励函数来说时可以接受的，但是如果使用平滑的激励函数如 tanh nonlinearities and softmax，这个结果还是太高了。

4. $1e^{-7} \gt diff$

另外要注意的是，随着网络深度的增加会使得误差积累，如果用了10层的网络，得到的相对误差为`1e-2`那么这个结果也是可以接受的。
  
### 12.2.5 注意事项

1. 首先，不要使用梯度检验去训练，即不要使用梯度检验方法去计算梯度，因为这样做太慢了，在训练过程中，我们还是使用backprop去计算参数梯度，而使用梯度检验去调试，去检验backprop的过程是否准确。

2. 其次，如果我们在使用梯度检验过程中发现backprop过程出现了问题，就需要对所有的参数进行计算，以判断造成计算偏差的来源在哪里，它可能是在求解$B$出现问题，也可能是在求解某一层的$W$出现问题，梯度检验可以帮助我们确定发生问题的范围，以帮助我们调试。

3. 别忘了正则化。如果我们添加了二范数正则化，在使用backprop计算参数梯度时，不要忘记梯度的形式已经发生了变化，要记得加上正则化部分，同理，在进行梯度检验时，也要记得目标函数$J$的形式已经发生了变化。

4. 注意，如果我们使用了drop-out正则化，梯度检验就不可用了。为什么呢？因为我们知道drop-out是按照一定的保留概率随机保留一些节点，因为它的随机性，目标函数$J$的形式变得非常不明确，这时我们便无法再用梯度检验去检验backprop。如果非要使用drop-out且又想检验backprop，我们可以先将保留概率设为1，即保留全部节点，然后用梯度检验来检验backprop过程，如果没有问题，我们再改变保留概率的值来应用drop-out。

5. 最后，介绍一种特别少见的情况。在刚开始初始化W和b时，W和b的值都还很小，这时backprop过程没有问题，但随着迭代过程的进行，$W$和$B$的值变得越来越大时，backprop过程可能会出现问题，且可能梯度差距越来越大。要避免这种情况，我们需要多进行几次梯度检验，比如在刚开始初始化权重时进行一次检验，在迭代一段时间之后，再使用梯度检验去验证backprop过程。


## 12.3 学习率与批大小

在梯度下降公式中：

$$
w_{t+1} = w_t - \frac{\eta}{m} \sum_i^m \nabla J(w,b) \tag{1}
$$

其中，$\eta$是学习率，m是批大小。所以，学习率与批大小是对梯度下降影响最大的两个因子。

### 12.3.1 关于学习率的挑战

有一句业内人士的流传的话：如果所有超参中，只需要调整一个参数，那么就是学习率。由此可见学习率是多么的重要，如果读者仔细做了9.6的试验，将会发现，不论你改了批大小或是隐层神经元的数量，总会找到一个合适的学习率来适应上面的修改，最终得到理想的训练结果。

但是学习率是一个非常难调的参数，下面给出具体说明。

前面章节学习过，普通梯度下降法，包含三种形式：

1. 单样本
2. 全批量样本
3. 小批量样本

我们通常把1和3统称为SGD(Stochastic Gradient Descent)。当批量不是很大时，全批量也可以纳入此范围。大的含义是：万级以上的数据量。

使用梯度下降的这些形式时，我们通常面临以下挑战：

1. 很难选择出合适的学习率
   
   太小的学习率会导致网络收敛过于缓慢，而学习率太大可能会影响收敛，并导致损失函数在最小值上波动，甚至出现梯度发散。
   
2. 相同的学习率并不适用于所有的参数更新
   
   如果训练集数据很稀疏，且特征频率非常不同，则不应该将其全部更新到相同的程度，但是对于很少出现的特征，应使用更大的更新率。
   
3. 避免陷于多个局部最小值中。
   
   实际上，问题并非源于局部最小值，而是来自鞍点，即一个维度向上倾斜且另一维度向下倾斜的点。这些鞍点通常被相同误差值的平面所包围，这使得SGD算法很难脱离出来，因为梯度在所有维度上接近于零。

表12-1 鞍点和驻点

|鞍点|驻点|
|---|---|
|<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images\Images\12\saddle_point.png" width="640">|<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images\Images\9\sgd_loss_8.png">|

表12-1中左图就是鞍点的定义，在鞍点附近，梯度下降算法经常会陷入泥潭，从而产生右图一样的历史记录曲线：有一段时间，Loss值随迭代次数缓慢下降，似乎在寻找突破口，然后忽然找到了，就一路下降，最终收敛。

为什么在3000至6000个epoch之间，有很大一段平坦地段，Loss值并没有显著下降？这其实也体现了这个问题的实际损失函数的形状，在这一区域上梯度比较平缓，以至于梯度下降算法并不能找到合适的突破方向寻找最优解，而是在原地徘徊。这一平缓地区就是损失函数的鞍点。

### 12.3.2 初始学习率的选择

我们前面一直使用固定的学习率，比如0.1或者0.05，而没有采用0.5、0.8这样高的学习率。这是因为在接近极小点时，损失函数的梯度也会变小，使用小的学习率时，不会担心步子太大越过极小点。

保证SGD收敛的充分条件是：

$$\sum_{k=1}^\infty \eta_k = \infty \tag{2}$$

且： 

$$\sum_{k=1}^\infty \eta^2_k < \infty \tag{3}$$ 

图12-5是不同的学习率的选择对训练结果的影响。

<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/12/learning_rate.png" ch="500" />

图12-5 学习率对训练的影响

- 黄色：学习率太大，loss值增高，网络发散
- 红色：学习率可以使网络收敛，但值较大，开始时loss值下降很快，但到达极值点附近时，在最优解附近来回跳跃
- 绿色：正确的学习率设置
- 蓝色：学习率值太小，loss值下降速度慢，训练次数长，收敛慢

有一种方式可以帮助我们快速找到合适的初始学习率。

Leslie N. Smith 在2015年的一篇论文[Cyclical Learning Rates for Training Neural Networks](https://arxiv.org/abs/1506.01186)中的描述了一个非常棒的方法来找初始学习率。

这个方法在论文中是用来估计网络允许的最小学习率和最大学习率，我们也可以用来找我们的最优初始学习率，方法非常简单：

1. 首先我们设置一个非常小的初始学习率，比如`1e-5`；
2. 然后在每个`batch`之后都更新网络，计算损失函数值，同时增加学习率；
3. 最后我们可以描绘出学习率的变化曲线和loss的变化曲线，从中就能够发现最好的学习率。

表12-2就是随着迭代次数的增加，学习率不断增加的曲线，以及不同的学习率对应的loss的曲线（理想中的曲线）。

表12-2 试验最佳学习率

|随着迭代次数增加学习率|观察Loss值与学习率的关系|
|---|---|
|<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images\Images\12\lr-select-1.jpg">|<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images\Images\12\lr-select-2.jpg">|

从表12-2的右图可以看到，学习率在0.3左右表现最好，再大就有可能发散了。我们把这个方法用于到我们的代码中试一下是否有效。

首先，设计一个数据结构，做出表12-3。

表12-3 学习率与迭代次数试验设计

|学习率段|0.0001~0.0009|0.001~0.009|0.01~0.09|0.1~0.9|1.0~1.1|
|----|----|----|----|---|---|
|步长|0.0001|0.001|0.01|0.1|0.01|
|迭代|10|10|10|10|10|

对于每个学习率段，在每个点上迭代10次，然后：

$$当前学习率+步长 \rightarrow 下一个学习率$$

以第一段为例，会在0.1迭代100次，在0.2上迭代100次，......，在0.9上迭代100次。步长和迭代次数可以分段设置，得到图12-6。

<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/12/LR_try_1.png" ch="500" />

图12-6 第一轮的学习率测试

横坐标用了`np.log10()`函数来显示对数值，所以横坐标与学习率的对应关系如表12-4所示。

表12-4 横坐标与学习率的对应关系

|横坐标|-1.0|-0.8|-0.6|-0.4|-0.2|0.0|
|--|--|--|--|--|--|--|
|学习率|0.1|0.16|0.25|0.4|0.62|1.0|

前面一大段都是在下降，说明学习率为0.1、0.16、0.25、0.4时都太小了，那我们就继续探查-0.4后的段，得到第二轮测试结果如图12-7。

<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/12/LR_try_2.png" ch="500" />

图12-7 第二轮的学习率测试

到-0.13时（对应学习率0.74）开始，损失值上升，所以合理的初始学习率应该是0.7左右，于是我们再次把范围缩小的0.6，0.7，0.8去做试验，得到第三轮测试结果，如图12-8。

<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/12/LR_try_3.png" ch="500" />

图12-8 第三轮的学习率测试

最后得到的最佳初始学习率是0.8左右。由于loss值是渐渐从下降变为上升的，前面有一个积累的过程，如果想避免由于前几轮迭代带来的影响，可以使用比0.8小一些的数值，比如0.75作为初始学习率。

### 12.3.3 学习率的后期修正

用12.1的MNIST的例子，固定批大小为128时，我们分别使用学习率为0.2，0.3，0.5，0.8来比较一下学习曲线。

<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/12/acc_bs_128.png" ch="500" />

图12-9 不同学习率对应的迭代次数与准确度值的

学习率为0.5时效果最好，虽然0.8的学习率开始时上升得很快，但是到了10个`epoch`时，0.5的曲线就超上来了，最后稳定在0.8的曲线之上。

这就给了我们一个提示：可以在开始时，把学习率设置大一些，让准确率快速上升，损失值快速下降；到了一定阶段后，可以换用小一些的学习率继续训练。用公式表示：

$$
LR_{new}=LR_{current} * DecayRate^{GlobalStep/DecaySteps} \tag{4}
$$

举例来说：

- 当前的LR = 0.1
- DecayRate = 0.9
- DecaySteps = 50

公式变为：

$$lr = 0.1 * 0.9^{GlobalSteps/50}$$

意思是初始学习率为0.1，每训练50轮计算一次新的$lr$，是当前的$0.9^n$倍，其中$n$是正整数，因为一般用$GlobalSteps/50$的结果取整，所以$n=1,2,3,\ldots$

<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/12/lr_decay.png" ch="500" />

图12-10 阶梯状学习率下降法

如果计算一下每50轮的衰减的具体数值，见表12-5。

表12-5 学习率衰减值计算

|迭代|0|50|100|150|200|250|300|...|
|---|---|---|---|---|---|---|---|---|
|学习率|0.1|0.09|0.081|0.073|0.065|0.059|0.053|...|

这样的话，在开始时可以快速收敛，到后来变得很谨慎，小心翼翼地向极值点逼近，避免由于步子过大而跳过去。

上面描述的算法叫做step算法，还有一些其他的算法如下。

<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/12/lr_policy.png" ch="500" />

图12-11 其他各种学习率下降算法

#### fixed

使用固定的学习率，比如全程都用0.1。要注意的是，这个值不能大，否则在后期接近极值点时不易收敛。

#### step

每迭代一个预订的次数后（比如500步），就调低一次学习率。离散型，简单实用。

#### multistep

预设几个迭代次数，到达后调低学习率。与step不同的是，这里的次数可以是不均匀的，比如3000、5500、8000。离散型，简单实用。

#### exp

连续的指数变化的学习率，公式为：

$$lr_{new}=lr_{base} * \gamma^{iteration} \tag{5}$$

由于一般的iteration都很大（训练需要很多次迭代），所以学习率衰减得很快。$\gamma$可以取值0.9、0.99等接近于1的数值，数值越大，学习率的衰减越慢。

#### inv

倒数型变化，公式为：

$$lr_{new}=lr_{base} * \frac{1}{( 1 + \gamma * iteration)^{p}} \tag{6}$$

$\gamma$控制下降速率，取值越大下降速率越快；$p$控制最小极限值，取值越大时最小值越小，可以用0.5来做缺省值。

#### poly

多项式衰减，公式为：

$$lr_{new}=lr_{base} * (1 - {iteration \over iteration_{max}})^p \tag{7}$$

$p=1$时，为线性下降；$p>1$时，下降趋势向上突起；$p<1$时，下降趋势向下凹陷。$p$可以设置为0.9。

### 12.3.4 学习率与批大小的关系

#### 试验结果

我们回到MNIST的例子中，继续做试验。当批大小为32时，还是0.5的学习率最好，如图12-12所示。

<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/12/acc_bs_32.png" ch="500" />

图12-12 批大小为32时的几种学习率的比较

难道0.5是一个全局最佳学习率吗？别着急，继续降低批大小到16时，再观察准确率曲线。由于批大小缩小了一倍，所以要完成相同的`epoch`时，图12-13中的迭代次数会是图12-12中的两倍。

<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/12/acc_bs_16.png" ch="500" />

图12-13 批大小为16时几种学习率的比较

这次有了明显变化，一下子变成了0.1的学习率最好，这说明当批大小小到一定数量级后，学习率要和批大小匹配，较大的学习率配和较大的批量，反之亦然。

#### 原因解释

我们从试验中得到了这个直观的认识：大的批数值应该对应大的学习率，否则收敛很慢；小的批数值应该对应小的学习率，否则会收敛不到最佳点。

一个极端的情况是，当批大小为1时，即单个样本，由于噪音的存在，我们不能确定这个样本所代表的梯度方向就是正确的方向，但是我们又不能忽略这个样本的作用，所以往往采用很小的学习率。这种情况很适合于online-learning的场景，即流式训练。

使用Mini-batch的好处是可以克服单样本的噪音，此时就可以使用稍微大一些的学习率，让收敛速度变快，而不会由于样本噪音问题而偏离方向。从偏差方差的角度理解，单样本的偏差概率较大，多样本的偏差概率较小，而由于I.I.D.（独立同分布）的假设存在，多样本的方差是不会有太大变化的，即16个样本的方差和32个样本的方差应该差不多，那它们产生的梯度的方差也应该相似。

通常当我们增加batch size为原来的N倍时，要保证经过同样的样本后更新的权重相等，按照线性缩放规则，学习率应该增加为原来的m倍。但是如果要保证权重的梯度方差不变，则学习率应该增加为原来的$\sqrt m$倍。

研究表明，衰减学习率可以通过增加batch size来实现类似的效果，这实际上从SGD的权重更新式子就可以看出来两者确实是等价的。对于一个固定的学习率，存在一个最优的batch size能够最大化测试精度，这个batch size和学习率以及训练集的大小正相关。对此实际上是有两个建议：

1. 如果增加了学习率，那么batch size最好也跟着增加，这样收敛更稳定。
2. 尽量使用大的学习率，因为很多研究都表明更大的学习率有利于提高泛化能力。如果真的要衰减，可以尝试其他办法，比如增加batch size，学习率对模型的收敛影响真的很大，慎重调整。

#### 数值理解

如果上述一些文字不容易理解的话，我们用一个最简单的示例来试图说明一下学习率与批大小的正比关系。

<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/12/lr_bs.png" />

图12-14 学习率与批大小关系的数值理解

先看图12-14中的左图：假设有三个蓝色样本点，正确的拟合直线如绿色直线所示，但是目前的拟合结果是红色直线，其斜率为0.5。我们来计算一下它的损失函数值，假设虚线组成的网格的单位值为1。

$$loss = \frac{1}{2m} \sum_i^m (z-y)^2 = (1^2 + 0^2 + 1^2)/2/3=0.333$$

损失函数值可以理解为是反向传播的误差回传力度，也就是说此时需要回传0.333的力度，就可以让红色直线向绿色直线的位置靠近，即，让$W$值变小，斜率从0.5变为0。

注意，我们下面如果用一个不太准确的计算来说明学习率与样本的关系，这个计算并不是真实存在于神经网络的，只是个数值上的直观解释。

我们需要一个学习率$\eta_1$，令：

$$w = w - \eta_1 * 0.333 = 0.5 - \eta_1 * 0.333 = 0$$

则：

$$\eta_1 = 1.5 \tag{8}$$

再看图12-14的右图，样本点变成5个，多了两个橙色的样本点，相当于批大小从3变成了5，计算损失函数值：

$$loss = (1^2+0.5^2+0^2+0.5^2+1^2)/2/5=0.25$$

样本数量增加了，由于样本服从I.I.D.分布，因此新的橙色样本位于蓝色样本之间。也因此损失函数值没有增加，反而从三个样本点的0.333降低到了五个样本点的0.25，此时，如果想回传同样的误差力度，使得w的斜率变为0，则：

$$w = w - \eta_2 * 0.25 = 0.5 - \eta_2 * 0.25 = 0$$

则：

$$\eta_2 = 2 \tag{9}$$

比较公式8和公式9的结果，样本数量增加了，学习率需要相应地增加。

大的batch size可以减少迭代次数，从而减少训练时间；另一方面，大的batch size的梯度计算更稳定，曲线平滑。在一定范围内，增加batch size有助于收敛的稳定性，但是过大的batch size会使得模型的泛化能力下降，验证或测试的误差增加。

batch size的增加可以比较随意，比如从16到32、64、128等等，而学习率是有上限的，从公式2和3知道，学习率不能大于1.0，这一点就如同Sigmoid函数一样，输入值可以变化很大，但很大的输入值会得到接近于1的输出值。因此batch size和学习率的关系可以大致总结如下：

1. 增加batch size，需要增加学习率来适应，可以用线性缩放的规则，成比例放大
2. 到一定程度，学习率的增加会缩小，变成batch size的$\sqrt m$倍
3. 到了比较极端的程度，无论batch size再怎么增加，也不能增加学习率了。
