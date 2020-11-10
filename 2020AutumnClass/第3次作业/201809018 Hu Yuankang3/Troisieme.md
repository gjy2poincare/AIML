# Step4&Step5笔记

## 激活函数的作用

1. 给神经网络增加非线性因素
2. 把公式算出的Z压缩到$[0,1]$ 之间，便于之后的计算。
   
## 激活函数的基本性质：

1. 非线性：线性的激活函数和没有激活函数一样；
2. 可导性：做误差反向传播和梯度下降，必须要保证激活函数的可导性。
3. 单调性：单一的输入会得到单一的输出，较大值的输入得到较大值的输出。

### 挤压型激活函数

#### Logistic函数
* 特点：当输入值域的绝对值较大的时候，其输出在两端是饱和的，都具有S形的函数曲线以及压缩输入值域的作用，所以叫挤压型激活函数，又可以叫饱和型激活函数。

* 对数几率函数（Logistic Function，简称对率函数）
  $$Sigmoid(z) = \frac{1}{1 + e^{-z}} \rightarrow a $$

* 导数推导：
  
  ![avatar](https://note.youdao.com/yws/api/personal/file/72A585958BF142EEA9327B54A623EA8C?method=download&shareKey=8ae3f68843bbf0886962e71666f8188d)

* 值域：
  - 输入值域：$(-\infty, \infty)$
  - 输出值域：$(0,1)$
  - 导数值域：$(0,0.25]$

* 函数导数图像：
  
  ![avatar](https://note.youdao.com/yws/api/personal/file/8586FCBD769546B3B00EDEAF2BB2AE93?method=download&shareKey=8cb387cc8234b4db5c9f0b6aceb4c151)

* 运用公式例题：
  
  ![avatar](https://note.youdao.com/yws/api/personal/file/3201021CDDA54815967ED0FFC9FF620E?method=download&shareKey=bf0f98d8b0e93b52dde34f8a8228a5ab)

#### Tanh函数（双曲正切函数）

* 公式：
  $$Tanh(z) = \frac{e^{z} - e^{-z}}{e^{z} + e^{-z}} = (\frac{2}{1 + e^{-2z}}-1) \rightarrow a $$
  $$Tanh(z) = 2 \cdot Sigmoid(2z) - 1 $$

* 公式及导数推导：
  $$Tanh'(z) = (1 + a)(1 - a)$$
  
  ![avatar](https://note.youdao.com/yws/api/personal/file/4EBEFE195FC24DA18FFDC4E761243CD0?method=download&shareKey=70c6edc7ccc3906debea7fac7e4b462a)

* 值域：
  - 输入值域：$(-\infty,\infty)$
  - 输出值域：$(-1,1)$
  - 导数值域：$(0,1)$

* 函数图像：
  
  ![avatar](https://note.youdao.com/yws/api/personal/file/0ED36C7CB6634D83BC2C9DF0293532E0?method=download&shareKey=69c07ce0e4cc976b7cc0815246df1d08)

* 相较于Sigmoid的优势：
  
  Tanh减少了一个缺点，就是他本身是零均值的，也就是说，在传递过程中，输入数据的均值并不会发生改变，这就使他在很多应用中能表现出比Sigmoid优异一些的效果。 
  
### 半线性激活函数

#### ReLU函数

* 公式：
  $$ReLU(z) = max(0,z) = \begin{cases} 
  z, & z \geq 0 \\\\ 
  0, & z < 0 
  \end{cases}$$

* 导数：
  $$ReLU'(z) = \begin{cases} 1 & z \geq 0 \\\\ 0 & z < 0 \end{cases}$$

* 图像：
  
  ![avatar](https://note.youdao.com/yws/api/personal/file/3EF098F8BE68426B90CB6E0C81C447AA?method=download&shareKey=ea2ccc954682b9a03a441d51efcb6bda)

* 值域：
  - 输入值域：$(-\infty, \infty)$
  - 输出值域：$(0,\infty)$
  - 导数值域：$\\{0,1\\}$

* 优点：
  - 反向导数恒等于1，更加有效率的反向传播梯度值，收敛速度快；
  - 避免梯度消失问题；
  - 计算简单，速度快；
  - 活跃度的分散性使得神经网络的整体计算成本下降。
  
#### Leaky ReLU函数

* 公式：
  $$LReLU(z) = \begin{cases} z & z \geq 0 \\\\ \alpha \cdot z & z < 0 \end{cases}$$
  
* 导数：
     $$LReLU'(z) = \begin{cases} 1 & z \geq 0 \\\\ \alpha & z < 0 \end{cases}$$
* 值域：
  - 输入值域：$(-\infty, \infty)$

  - 输出值域：$(-\infty,\infty)$ 

  - 导数值域：$\\{\alpha,1\\}$

* 较于ReLU函数的优点：
  
  由于给了z<0时一个比较小的梯度a,使得z<0时依旧可以进行梯度传递和更新，可以在一定程度上避免神经元“死”掉的问题。

* 图像：
  
  ![avatar](https://note.youdao.com/yws/api/personal/file/B4188A4AFE7C462FA28B05E8F3C30B76?method=download&shareKey=911eda48e019d0e230cff3a6bfd9bc26)
  
#### Softplus函数

* 公式：
  $$Softplus(z) = \ln (1 + e^z)$$

* 导数：
  $$Softplus'(z) = \frac{e^z}{1 + e^z}$$

* 值域：
  - 输入值域：$(-\infty, \infty)$

  - 输出值域：$(0,\infty)$

  - 导数值域：$(0,1)$ 

* 图像：
  
  ![avatar](https://note.youdao.com/yws/api/personal/file/0EF74B2A6631474690F7212D5FCF2659?method=download&shareKey=5057ac3b65fa703e0f66a6d1f97089b0)

## 回归模型的评估标准:

### 均方差(MSE)

$$MSE = \frac{1}{m} \sum_{i=1}^m (a_i-y_i)^2 $$

由于MSE计算的是误差的平方，所以它对异常值是非常敏感的，因为一旦出现异常值，MSE指标会变得非常大。MSE越小，证明误差越小。

### R平方

$$R^2=1-\frac{\sum (a_i - y_i)^2}{\sum(\bar y_i-y_i)^2}=1-\frac{MSE(a,y)}{Var(y)} \tag{6}$$

R平方是多元回归中的回归平方和（分子）占总平方和（分母）的比例，它是度量多元回归方程中拟合程度的一个统计量。R平方值越接近1，表明回归平方和占总平方和的比例越大，回归线与各观测点越接近，回归的拟合程度就越好。

## 多项式回归法拟合正弦曲线
* 令：$x_1 = x,x_2=x^2,\ldots,x_m=x^m$，则：
$$z = x_1 w_1 + x_2 w_2 + ... + x_m w_m + b $$

* 增加X项数的代码：
```python
X = self.XTrain[:,]**2//二次项
X = self.XTrain[:,0:1]**3//三次项
X = self.XTrain[:,0:1]**4//四次项
self.XTrain = np.hstack((self.XTrain, X))
```
其中主函数中要修改input的个数。

* 通过多项式回归，找到合适的项数和足够的周期可以初步得到理想的拟合结果，但是成果过高。

## 神经网络/深度学习

### 留出法（Hold out）

从训练数据中保留出验证样本集，主要用于解决过拟合情况，这部分数据不用于训练。如果训练数据的准确度持续增长，但是验证数据的准确度保持不变或者反而下降，说明神经网络亦即过拟合了，此时需要停止训练，用测试集做最终测试。

```
for each epoch
    shuffle
    for each iteraion
        获得当前小批量数据
        前向计算
        反向传播
        更新梯度
        if is checkpoint
            用当前小批量数据计算训练集的loss值和accuracy值并记录
            计算验证集的loss值和accuracy值并记录
            如果loss值不再下降，停止训练
            如果accuracy值满足要求，停止训练
        end if
    end for
end for
```
## 双层神经网络实现非线性回归

### 万能近似定理
* 万能近似定理是深度学习最根本的理论依据。它证明了在给定网络具有足够多的隐藏单元的条件下，配备一个线性输出层和一个带有任何“挤压”性质的激活函数（如Sigmoid激活函数）的隐藏层的前馈神经网络，能够以任何想要的误差量近似任何从一个有限维度的空间映射到另一个有限维度空间的Borel可测的函数。
* 正向计算和反向传播路径图：
  
  ![avatar](https://note.youdao.com/yws/api/personal/file/C135CC32D40F4CAFA770ABE15B69FF47?method=download&shareKey=e048abb86da67133e82024fb825a37e3)
  - 蓝色矩形表示数值或矩阵；
  - 蓝色圆形表示计算单元；
  - 蓝色的箭头表示正向计算过程；
  - 红色的箭头表示反向计算过程。

* 使用的方法和第五章中的线性回归类似。

## 曲线拟合

* n_hidden=2，意为隐层神经元数量为2


* 有三个神经元解复合函数
  ```Python
    if __name__ == '__main__':
    ......
    n_input, n_hidden, n_output = 1, 3, 1
    eta, batch_size, max_epoch = 0.5, 10, 10000
    eps = 0.001
    hp = HyperParameters2(n_input, n_hidden, n_output, eta, max_epoch, batch_size, eps, NetType.Fitting, InitialMethod.Xavier)
    net = NeuralNet2(hp, "model_131")
    ......
  ```

## 非线性回归的工作原理

  1. 隐层把x拆成不同的特征，根据问题复杂度决定神经元数量，神经元的数量相当于特征值的数量；
   
  2. 隐层通过激活函数（sigmoid函数）做一次非线性变换；
   
  3. 输出层使用多变量线性回归，把隐层的输出当作输入特征值，再做一次线性变换，得出拟合结果。

  **注意： 与多项式回归不同的是，不需要指定变换参数，而是从训练中学习到参数，这样的话权重值不会大得离谱（补充：多项式回归通过最小二乘法的数学方法得到优化的权重，而神经网络通过训练集，BP算法等找到权重误差最小的取值）**
  1. 双层神经网络法将每个神经元当作一个输出，将原本复杂的函数分解成两个函数，再通过激活函数和将输出值当作特征值运用多变量线性回归法效率更高的拟合了函数。

## 二分类模型的评估

### 混淆矩阵
- 准确率：
$$
\begin{aligned}
Accuracy &= \frac{TP+TN}{TP+TN+FP+FN} \\\\
\end{aligned}
$$
- 精确率/查准率 

分子为被判断为正类并且真的是正类的样本数，分母是被判断为正类的样本数。越大越好。

$$
Precision=\frac{TP}{TP+FP}
$$

- 召回率/查全率 Recall

$$
Recall = \frac{TP}{TP+FN}
$$

分子为被判断为正类并且真的是正类的样本数，分母是真的正类的样本数。越大越好。

- TPR - True Positive Rate 真正例率

$$
TPR = \frac{TP}{TP + FN}=Recall
$$

- FPR - False Positive Rate 假正例率

$$
FPR = \frac{FP}{FP+TN}
$$

分子为被判断为正类的负例样本数，分母为所有负类样本数。越小越好。

- 调和平均值 F1

$$
\begin{aligned}
F1&=\frac{2 \times Precision \times Recall}{recision+Recall}
\end{aligned}
$$

该值越大越好。

## 解决异或问题
* 把异或问题归类成二分类问题，使用二分类交叉熵损失函数： 
$$
loss = -Y \ln A2 + (1-Y) \ln (1-A2) \tag{12}
$$

* 求损失函数对输出层的反向误差，方法和第五章相同，算出了dw和db的值：
$$
dW1=X^{\top} \cdot dZ1 
$$
$$
dB1=dZ1
$$

* 代码实现：
  - `n_input = dataReader.num_feature`，值为2，而且必须为2，因为只有两个特征值
  - `n_hidden=2`，这是人为设置的隐层神经元数量，可以是大于2的任何整数
  - `eps`精度=0.005是后验知识，笔者通过测试得到的停止条件，用于方便案例讲解
  - 网络类型是`NetType.BinaryClassifier`，指明是二分类网络
  - 最后要调用`Test`函数验证精度
![avatar](https://note.youdao.com/yws/api/personal/file/7CEF34F776AF4BAA88BDF39498015551?method=download&shareKey=df28c473a4a25b664cd8b9907e1864ec)

* 逻辑异或门的工作原理：
  神经网络:通过样本点，推算出了平面上每个坐标点的分类结果概率，形成空间曲面，然后拦腰一刀（一个切面），这样神经网络就可以在Z=0.5出画一个平面，
![avatar](https://note.youdao.com/yws/api/personal/file/3C791E8341704DE59BA01B7F95D347D7?method=download&shareKey=bbab96216884c64665498e3666306a96)
![avatar](https://note.youdao.com/yws/api/personal/file/7ACB97644307436A90F0325F5B80F070?method=download&shareKey=750de04a4dbbc48764b75eb7eacb18fe)

* 实现双弧形二分类
![avatar](https://note.youdao.com/yws/api/personal/file/BC5E3D995A5D4FFD912ACEB519736075?method=download&shareKey=157090516144f75c4b2062170e448b72)

* 在激活函数计算中，由于激活函数的非线性，所以空间逐渐扭曲变形，使得红色样本点逐步向右下角移动，并变得稠密；而蓝色样本点逐步向左上方扩撒，通过空间扭曲，红蓝两类之间可以用一条直线分割。
![avatar](https://note.youdao.com/yws/api/personal/file/D1811F90849A490EB84DDDAAEF85B821?method=download&shareKey=1e6de129859cc86d5a02b4f0ed94e54b)

* 类比异或问题，两个红色样本点区域逐渐上移，形成了两个切面，在二维中就是形成了两条直线。

## 非线性多分类
* 隐层神经元的个数决定了精度和损失函数
![avatar](https://note.youdao.com/yws/api/personal/file/DD9EA09617A6484FA88F07AE9E1608A9?method=download&shareKey=c40b5627371c12bcfffe796734605bfa)
![avatar](https://note.youdao.com/yws/api/personal/file/193D55378596484CA89E30C818510021?method=download&shareKey=de5001910825f3ce10993b9cce812b6e)

* 当有3个隐层神经元时，在三维图中观察可视化结果，经过激活函数做非线性变换后的图。由于绿色点比较靠近边缘，所以三维坐标中的每个值在经过Sigmoid激活函数计算后，都有至少一维坐标会是向1靠近的值，所以分散的比较开，形成外围的三角区域；蓝色点正好相反，三维坐标值都趋近于0，所以最后都集中在三维坐标原点的三角区域内；红色点处于前两者之间，因为有很多中间值。
![avatar](https://note.youdao.com/yws/api/personal/file/3761FACCD9A34D5C868A7CEBE3BBD1CF?method=download&shareKey=8bf674692a0f0fa6673b55b9fceff87b)

## 解决分类样本不平衡问题
* 集成学习：
  1. 首先使用原始数据集训练第一个学习器L1；
  2. 然后使用50%在L1学习正确和50%学习错误的那些样本训练得到学习器L2，即从L1中学习错误的样本集与学习正确的样本集中，循环一边采样一个；
  3. 接着，使用L1与L2不一致的那些样本去训练得到学习器L3；
  4. 最后，使用投票方式作为最后输出。 

**假设是一个二分类问题，大部分的样本都是true类。让L1输出始终为true。使用50%在L1分类正确的与50%分类错误的样本训练得到L2，即从L1中学习错误的样本集与学习正确的样本集中，循环一边采样一个。因此，L2的训练样本是平衡的。L使用L1与L2分类不一致的那些样本训练得到L3，即在L2中分类为false的那些样本。最后，结合这三个分类器，采用投票的方式来决定分类结果，因此只有当L2与L3都分类为false时，最终结果才为false，否则true。**

## 深度非线性多分类
* 对数字识别，把每个图片的像素都当作一个向量，把整张图片看作为一个样本：
```Python
    def __NormalizeData(self, XRawData):
        X_NEW = np.zeros(XRawData.shape)
        x_max = np.max(XRawData)
        x_min = np.min(XRawData)
        X_NEW = (XRawData - x_min)/(x_max-x_min)
        return X_NEW
```
* 三层神经网络的实现：
  * 初始化部分需要构造三组WeightsBias对象，注意各组的输入输出数量，决定了矩阵的形状。
  * 前向计算部分增加了一层，并且使用Tanh()作为激活函数。
  * logistic：二分类函数。softmax：归一化函数
  * 反向传播也相应地增加了一层，要用对应的Tanh()的反向公式。梯度更新时也是三组权重值同时更新。
```Python    
        def update(self):
        self.wb1.Update()
        self.wb2.Update()
        self.wb3.Update()
```
  * 测试如图所示：
  ![avatar](https://note.youdao.com/yws/api/personal/file/D9B76E2764AF46CE997C4C37184E764D?method=download&shareKey=5471fc985398457a91112a29b6825915)

  * 当把第一隐层神经神经元改为256时发现，拟合结果基本一致，但训练速度变慢，可见隐层神经元数目不是越多越好。
  ![avatar](https://note.youdao.com/yws/api/personal/file/C769631F391B4747972D979029E91FA3?method=download&shareKey=2ad7b5d5096437daa02afbdc71d83832)
  * 关于隐藏层大小的经验法则是在输入层和输出层之间，为了计算隐藏层大小我们使用一个一般法则：（输入大小+输出大小）*2/3。
  * 隐藏层的大小还取决于训练数据大小，目标中的噪声（响应变量的值），以及特征空间复杂度。

## 梯度检查
* 为了确认代码中反向传播计算的梯度是否正确，可以采用梯度检验的方法。通过计算数值梯度，得到梯度的近似值，然后和反向传播得到的梯度进行比较，若两者相差很小的话则证明反向传播的代码是正确无误的。
* 双边逼近误差法：
  $$f'(\theta) + O(h^2)={f(\theta + h) - f(\theta - h) \over 2h} $$ 
左边第二项即为误差，数量级为平方项，因此很小。

## 学习率的选择
* 初始时用固定的学习率，比如0.1或者0.05，而没有采用0.5、0.8这样高的学习率。这是因为在接近极小点时，损失函数的梯度也会变小，使用小的学习率时，不会担心步子太大越过极小点。
* 训练学利率的方法：
  1. 首先设置一个非常小的初始学习率，比如`1e-5`；
  2. 然后在每个`batch`之后都更新网络，计算损失函数值，同时增加学习率；
  3. 最后描绘出学习率的变化曲线和loss的变化曲线，从中就能够发现最好的学习率。

## 心得体会
可以发现，在简单的多线性回归中，只是通过增加特征值在逼近拟合，而在神经网络中，凸显出了计算机训练的优势，运用激活函数和分解函数的方法来使效率得到提高。一个隐藏层可以解决几乎所有的非线性问题，而出现解决弧形问题或是异或问题出，则考虑多层隐藏层。通过本阶段学习，我感受到了神经网络在三维空间的魅力，特别是在多隐藏层解决非线性问题时，对于概率区域的染色从而直观的观察神经网络训练中的变化，更加清晰的明白了神经网络的作用。