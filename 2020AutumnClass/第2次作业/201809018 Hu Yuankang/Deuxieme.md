#Step2&Step3预习笔记
## 一 单变量线性回归

1. 最小二乘法；
2. 梯度下降法；
3. 简单的神经网络法；
4. 更通用的神经网络算法。 

###公式形态

![avatar](https://note.youdao.com/yws/api/personal/file/BF3B56A256794EDD8AA563F8DC1D3105?method=download&shareKey=d8fcda084013511da348e9438efbf313)

* 对B来说，它永远是1行，列数与w的列数相等。比如w是3×1的矩阵，则b是1×1的矩阵。**如果w是3×2的矩阵，意味着3个特征输入到2个神经元上**，则b是1×2的矩阵，每个神经元分配1个bias。

![avatar](https://note.youdao.com/yws/api/personal/file/2305667C7E44450DAD062F1A54AAC876?method=download&shareKey=b439aadb2e4032c0b0cc7f30960435b4)
* Xij的第一个下标i表示样本序号，第二个下标j表示样本特征。

###最小二乘法
利用均方差公式求w和b
![avartar](https://note.youdao.com/yws/api/personal/file/56B0B3AF7A9E430EBB6482EE06790E8E?method=download&shareKey=571728618829c5e02f9c29986bb6ad33)

如果想让误差的值最小，通过对w和b求导，再令导数为0（到达最小极值），就是w和b的最优解。
def method3(X,Y,m):
    p = m*sum(X*Y) - sum(X)*sum(Y)
    q = m*sum(X*X) - sum(X)*sum(X)
    w = p/q
    return w
![avatar](https://note.youdao.com/yws/api/personal/file/A3CBC43F04734AF19A7C92AB0A737977?method=download&shareKey=78983dab9c477d25e520207473c64911)

def calculate_b_1(X,Y,w,m):
    b = sum(Y-w*X)/m
    return b
![avatar](https://note.youdao.com/yws/api/personal/file/4F7CA1AC23D84F51B7DE2EFFA4935C23?method=download&shareKey=b746a06161291ba6e9fa6e99f0dd407c)

![avatar](https://note.youdao.com/yws/api/personal/file/8E3E5932FE9F476F9DC900871250111D?method=download&shareKey=9ff47f52056e7533d7bc58067fde3e58)
###梯度下降法
与最小二乘法的模型及损失函数相同，都为一个线性模型加均方差的损失函数，区别于最小二乘法，梯度下降利用导数传递误差，而最小二乘法从损失函数求导，直接求得数学解析解。
![avatar](https://note.youdao.com/yws/api/personal/file/ED8AE21FE4D24BC1847B1A682A1728E7?method=download&shareKey=fa0224a8d98034e68dbf3e79fa48a3b9)

梯度下降的意义：把推导的结果转化为数学公式和代码，直接放在迭代过程中。

###神经网络法
* 神经网络法和梯度下降法在本质上是一样的，只不过神经网络法使用一个崭新的编程模型，即以神经元为中心的代码结构设计，这样便于以后的功能扩充。
* 使用面向对象的技术，通过创建一个类来描述神经网络的属性和行为
  **定义类：**
  class NeuralNet(object):
    def __init__(self, eta):
        self.eta = eta
        self.w = 0
        self.b = 0
  **前向计算：**
  def __forward(self, x):
        z = x * self.w + self.b
        return z
  **后向传播：（利用梯度算法）**
  def __backward(self, x,y,z):
        dz = z - y
        db = dz
        dw = x * dz
        return dw, db
  **梯度更新：**
  def __update(self, dw, db):
        self.w = self.w - self.eta * dw
        self.b = self.b - self.eta * db
  ![avatar](https://note.youdao.com/yws/api/personal/file/261C05BEE14C4212B51A236B62099346?method=download&shareKey=a4a36bcfb718678e7905ba249cae7c8f)
###多样本计算
* **定义矩阵：**
  ![avatar](https://note.youdao.com/yws/api/personal/file/1ADD3F2E0CA547D0B652F853BE9CFFC8?method=download&shareKey=b21d014969e4e13b1cd848166ed39302)
  def __forwardBatch(self, batch_x):
        Z = np.dot(batch_x, self.w) + self.b
        return Z
* **利用偏导求出各量的梯度，方法与前面所用类似**
  def __checkLoss(self, dataReader):
        X,Y = dataReader.GetWholeTrainSamples()
        m = X.shape[0]
        Z = self.__forwardBatch(X)
        LOSS = (Z - Y)**2
        loss = LOSS.sum()/m/2
        return loss
**Python中的矩阵减法运算，不需要对矩阵中的每个对应的元素单独做减法，而是整个矩阵相减即可。做求和运算时，也不需要自己写代码做遍历每个元素，而是简单地调用求和函数即可。**
注意在代码中np.dot（）与sum（）函数。

###单样本随机梯度下降（SGD）

#### 特点
  
  - 训练样本：每次使用一个样本数据进行一次训练，更新一次梯度，重复以上过程。
  - 优点：训练开始时损失值下降很快，随机性大，找到最优解的可能性大。
  - 缺点：受单个样本的影响最大，损失函数值波动大，到后期徘徊不前，在最优解附近震荡。不能并行计算。
![avatar](https://note.youdao.com/yws/api/personal/file/0D652B0FAE6B41BCAA3DF562274633FB?method=download&shareKey=af5147257fe95916228232dbd4640881)
![avatar](https://note.youdao.com/yws/api/personal/file/DBD66A7757A1405C96DC698DF44257B9?method=download&shareKey=b1c675be0520a4d11460e8c73c8eae11)

#### 特点

  - 训练样本：选择一小部分样本进行训练，更新一次梯度，然后再选取另外一小部分样本进行训练，再更新一次梯度。
  - 优点：不受单样本噪声影响，训练速度较快。
  - 缺点：batch size的数值选择很关键，会影响训练结果。
![avatar](https://note.youdao.com/yws/api/personal/file/8BEA3B8ED1624ABF82EE1D026DCA8A69?method=download&shareKey=232bd11be803806443134020a47903e8)
![avatar](https://note.youdao.com/yws/api/personal/file/3CC9F37FFE9B46C8817D0AAEEA612AA7?method=download&shareKey=4dccca9c1c5d66f2ef73196945936571)
###全批量样本梯度下降

#### 特点

  - 训练样本：每次使用全部数据集进行一次训练，更新一次梯度，重复以上过程。
  - 优点：受单个样本的影响最小，一次计算全体样本速度快，损失函数值没有波动，到达最优点平稳。方便并行计算。
  - 缺点：数据量较大时不能实现（内存限制），训练过程变慢。初始值不同，可能导致获得局部最优解，并非全局最优解。
![avatar](https://note.youdao.com/yws/api/personal/file/23CEA64A8BE34BAEA70BF9FE502DD191?method=download&shareKey=02bcdc1660cda2839b718055382b6acd)
![avatar](https://note.youdao.com/yws/api/personal/file/FAE04C0644E04222BE607E1BC784AA78?method=download&shareKey=b5615885f7e4eb8cff46aadb4581faf8)

####相关的概念：

- Batch Size：批大小，一次训练的样本数量。
- Iteration：迭代，一次正向 + 一次反向。
- Epoch：所有样本被使用了一次，叫做一个Epoch，

###梯度下降总结
* Batch Size增大了，要到达相同的准确度，必须要增大epoch。
* 通过并行化提高内存的利用率。就是尽量让你的GPU满载运行，提高训练速度。 
* 适当Batch Size使得梯度下降方向更加准确。

##二 多变量线性回归

###正规方程法
![avatar](https://note.youdao.com/yws/api/personal/file/8AA5DEA53DFB407380D2935636B2A19D?method=download&shareKey=e448ad6273a33ed9c5b2fb7750a8000c)
![avatar](https://note.youdao.com/yws/api/personal/file/C81264059E1148EBB8059A2464D24957?method=download&shareKey=ebb15620099eecafc16dbcb8e82ca9b7)

###神经网路解法
* 仅仅是多了一个输入，但却是质的变化，即，一个神经元可以同时接收多个输入，这是神经网络能够处理复杂逻辑的根本。

* 一共有1000个样本，每个样本2个特征值，X就是一个1000×2的矩阵。

* 输入层是两个特征，输出层是一个变量，所以w的形状是2×1。

* 因为输出层只有一个神经元，所以只有一个bias，每个神经元对应一个bias，如果有多个神经元，它们都会有各自的b值。

* 可以直接利用单变量线性回归中创建的面向对象代码
 x = np.array([x1,x2]).reshape(1,2)
 reshape()函数创建二维数组，即X的转置矩阵，一行两列。
 
 
* self.W = np.zeros((self.params.input_size, self.params.output_size))
self.B = np.zeros((1, self.params.output_size))
在神经网络初始化时，指定了input_size=2，且output_size=1，即一个神经元可以接收两个输入，最后是一个输出。

###样本特征数据标准化
1. 样本的各个特征的取值要符合概率分布，即[0,1]。
2. 样本的度量单位要相同。
3. 标准化可以避免一些不必要的数值问题。如梯度更新，学习率的选择。
   若输出层的数量级很大，会引起损失函数的数量级很大，这样做反向传播时的梯度也就很大。
   如果梯度非常大，学习率就必须非常小，因此，学习率（学习率初始值）的选择需要参考输入的范围。
4. 为什么要在[0,1]空间中形成50乘50的网格呢？
   解：NumPy库的np.linspace(0,1)的含义，就是在[0,1]空间中生成50个等距的点，第三个参数不指定时，缺省是50。因为我们前面对样本数据做过标准化，统一到了[0,1]空间中。

###还原参数值
* 结论：W的值和样本特征值的缩放有关系，而且缩放倍数非常相似。
  
###预测数据的标准化
* 利用训练数据的最小值和最大值，在预测时使用它们对预测数据做标准化。前提是预测数据的特征值不能超出训练数据的特征值范围，否则有可能影响准确程度。
* 预测数据看作训练数据的一个记录，先做标准化，再做预测，这样就不需要把权重矩阵还原了。
  
###标签值Y标准化
* Y值不在[0,1]之间时，要做标准化，好处是迭代次数少。
* 如果Y做了标准化，对得出来的预测结果做关于Y的反标准化。

##三 逻辑回归

###逻辑回归理解
* 逻辑回归（Logistic Regression），回归给出的结果是事件成功或失败的概率。
* 逻辑回归的目标是“拟合”0或1两个数值，而不是具体连续数值，所以称为广义线性模型。逻辑回归又称Logistic回归分析，常用于数据挖掘，疾病自动诊断，经济预测等领域。
* 逻辑回归的另外一个名字叫做分类器，分为线性分类器和非线性分类器，而无论是线性还是非线性分类器，又分为两种：二分类问题和多分类问题。。

###二分类函数
对率函数Logistic Function：
在二分类任务中，叫做Logistic函数，而在作为激活函数时，叫做Sigmoid函数。

* Logistic函数公式：
  ![avatar](https://note.youdao.com/yws/api/personal/file/DA91DFF01D7F436DBCBE2CD9F3D013E8?method=download&shareKey=803b7316a7797575303c3839978f2d92)
* 求误差loss对z的偏导：
  ![avatar](https://note.youdao.com/yws/api/personal/file/92ABD0CFD0AB4A63B83D7FE642602040?method=download&shareKey=78c8196354db9a031f25331af94558a2)

* 几率（odds）：
  ![avatar](https://note.youdao.com/yws/api/personal/file/EA58591025C140CFB837693D98335B12?method=download&shareKey=937fef419e14f5477aab348f93d66cc9)
* 对几率取对数就叫做对数几率：对几率取对数，可以得到一组成线性关系的值。
  ![avatar](https://note.youdao.com/yws/api/personal/file/44AF806919C94EF985AFD8CADC02C5E9?method=download&shareKey=84a312033671739e5ca7da276d066afb)

###用神经网络实现线性二分类
* 网络结构图的区别是，这次我们在神经元输出时使用了分类函数，所以输出为A，而不是以往直接输出的Z。
  ![avatar](https://note.youdao.com/yws/api/personal/file/A7641CD3B37141C5B82B637B1913CEBF?method=download&shareKey=c906ce3b15a64faee43f645d1c1d7494)

###线性分类和线性回归的异同：
* ![avatar](https://note.youdao.com/yws/api/personal/file/03C51A8033AC422A82455770C1F45CE5?method=download&shareKey=7894d43b11848ee59133fb64465e8d3a)
* 代数方式：通过一个分类函数计算所有样本点在经过线性变换后的概率值，使得正例样本的概率大于0.5，而负例样本的概率小于0.5。

###线性二分类的工作原理：
* 当 z>0z>0 时，Logistic(z) > 0.5，Logistic(z)>0.5 为正例，反之为负例。
  ![avatar](https://note.youdao.com/yws/api/personal/file/5A092A59FDF34720A2B2FA7D8338102C?method=download&shareKey=6ad4a62041a1839577ea7263b942ee95)
* Logistic函数用来判断**误差的力度**。
* 二分类交叉熵损失函数：
  ![avatar](https://note.youdao.com/yws/api/personal/file/FBC8ECE4B11A49FC8103B1787CB80624?method=download&shareKey=6be499bfa96179bf6aca20a822023855)

###二分类结果可视化
* 实际的工程实践中，一般我们会把样本分成训练集、验证集、测试集，用测试集来测试训练结果的正确性。

###实现逻辑与或非门
* 单层神经网络，又叫做感知机，它可以轻松实现逻辑与、或、非门。由于逻辑与、或门，需要有两个变量输入，而逻辑非门只有一个变量输入。但是它们共同的特点是输入为0或1，可以看作是正负两个类别。
* 四种逻辑门的样本和标签数据：
  ![avatar](https://note.youdao.com/yws/api/personal/file/CDE9E18BD3E947F7B76A0BC7A3AAAF5D?method=download&shareKey=c6768c09e08d751b4d1f0121311d06f8)

###双曲正切函数（Tanh）
* 可以形象地总结出，当使用Tanh函数后，相当于把Logistic的输出值域范围拉伸到2倍，下边界从0变成-1；而对应的交叉熵函数，是把输入值域的范围拉伸到2倍，左边界从0变成-1，完全与分类函数匹配。
* 函数图像：
  ![avatar](https://note.youdao.com/yws/api/personal/file/9CD2E418AC8D49B48F38C34DF8E18246?method=download&shareKey=a3e044dc8dced9fb43b6ad7be4a9efe1)

##线性多分类

**1. 一对一方式：** 每次先只保留两个类别的数据，训练一个分类器。
   ![avatar](https://note.youdao.com/yws/api/personal/file/382F53A021A9488D87C2693C52C122AA?method=download&shareKey=f1a9b435c4d193ae980a0d3e6acaf7d4)
**2. 一对多方式：** 处理一个类别时，暂时把其它所有类别看作是一类，这样对于三分类问题，可以得到三个分类器。
   ![avatar](https://note.youdao.com/yws/api/personal/file/05B0810D431141EBA86135E8D35EC551?method=download&shareKey=acc07d2fb42c0ce8cb3343cf37bb61c4)
**3. 多对多方式：** 假设有4个类别ABCD，我们可以把AB算作一类，CD算作一类，训练一个分类器1；再把AC算作一类，BD算作一类，训练一个分类器2。

###多分类函数
* 函数定义：（softmax）
  ![avatar](https://note.youdao.com/yws/api/personal/file/BD8631562CD341E9BFE0893996536E74?method=download&shareKey=0f7215b011250e4e61654f7b5eec7250)
- $z_j$ 是对第 $j$ 项的分类原始值，即矩阵运算的结果
- $z_i$ 是参与分类计算的每个类别的原始值
- $m$ 是总分类数
- $a_j$ 是对第 $j$ 项的计算结果

###线性多分类的神经网络实现
* ![avatar](https://note.youdao.com/yws/api/personal/file/52AA30B5B4414CDFAB31570C103F1B03?method=download&shareKey=60a3c8109b26139e7b582cbf14421787)
* 与前面的单层网络不同的是，图7-7最右侧的输出层还多出来一个Softmax分类函数，这是多分类任务中的标准配置，可以看作是输出层的激活函数，并不单独成为一层，与二分类中的Logistic函数一样。
* 如果有三个以上的分类同时存在，我们需要对每一类别分配一个神经元，这个神经元的作用是根据前端输入的各种数据，先做线性处理（$Z=WX+B$），然后做一次非线性处理，计算每个样本在每个类别中的预测概率，再和标签中的类别比较，看看预测是否准确，如果准确，则奖励这个预测，给与正反馈；如果不准确，则惩罚这个预测，给与负反馈。两类反馈都反向传播到神经网络系统中去调整参数。

###线性多分类原理
* 更多的分类或任意特征值，比如在ImageNet的图像分类任务中，最后一层全连接层输出给分类器的特征值有成千上万个，分类有1000个。
* ![avatar](https://note.youdao.com/yws/api/personal/file/A39360D11EEA42B8AF231ADBB3340B9C?method=download&shareKey=ffa40bc8a746d810169d67478705e5d0)

###多分类结果可视化
* 训练一对多分类器时，是把蓝色样本当作一类，把红色和绿色样本混在一起当作另外一类。训练一对一分类器时，是把绿色样本扔掉，只考虑蓝色样本和红色样本。而我们在此并没有这样做，三类样本是同时参与训练的。所以我们只能说神经网络从结果上看，是一种一对多的方式。

损失函数特征记录:
class LossFunction(object):
    # fcFunc: feed forward calculation
    def CheckLoss(self, A, Y):
        m = Y.shape[0]
        if self.net_type == NetType.Fitting:
            loss = self.MSE(A, Y, m)
        elif self.net_type == NetType.BinaryClassifier:
            loss = self.CE2(A, Y, m)
        elif self.net_type == NetType.MultipleClassifier:
            loss = self.CE3(A, Y, m)
        #end if
        return loss
    # end def
![avatar](https://note.youdao.com/yws/api/personal/file/D83D3AB5ECEF4D5699E3CF7EE11AF619?method=download&shareKey=91086fb5d64981a198c6844c4e029d07)

##三 学习心得
通过本次对Step2和Step3的学习，对于线性回归和线性分类有了初步的认识，其基础是数学的运算，最小二乘法在数值计算中已经学习，梯度下降法是通过数学的计算，链式法则找到规律并将其转换成代码，神经网络法的特点是加入了一个面向对象的设计，通过面向对象方法可以在其中增添其他的功能，之后对样本特征数据标准化是一个缺一不少的过程，其中又进一步探讨，对于参数值，标签值也要分别标准化。在多分类函数中，理解softmax的用法，还有一对一，一对多算法之间的区别和优劣。














