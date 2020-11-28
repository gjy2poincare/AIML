# 分类

## 神经元

`主要定义神经元的输入个数和输出个数，初始化神经元的权值和偏置也可以使用辅助的类来为神经元的使用添加功能`
```python
class Nuralnet(objcet):
    def __init()__(self):
        self.w = 0
        self.b = 0

```
## 前向计算

`根据样本定义输入的特征数和神经网络层数，若要多样本，小批量则使用矩阵`

$z = w*x + b$
**范例**：
```python
    def forward(self, batch_x):#定义神经网络
        #  层一
        self.Z1 = np.dot(batch_x, self.wb1.W) + self.wb1.B
        self.A1 = Sigmoid().forward(self.Z1)
        # 层二
        self.Z2 = np.dot(self.A1, self.wb2.W) + self.wb2.B
        self.A2 = Logistic().forward(self.Z2)
        #输出
        self.output = self.A2
```

## 激活函数

**要求**：
<font color=red> 
    非线性,可导的,单调性
</font>

**用途**：最后一层不用激活函数，`连接`前后的神经网络

- `二分类函数`
  
    <font face="STCAIYUN">logistic</font>
    **公式**：
    $logistic(z) = \frac1{1 + e^{-z}}$
    **导数**：
    $logisitic(z)' = a(1-a)$
    **描述**：`这是常用的分类函数，通过此函数将利用w,b而计算出的数据压缩在根据设定的阈值分类`

**函数图像**

<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/8/logistic.png" ch="500" />

```python
 def forward(self, z):
        a = 1.0 / (1.0 + np.exp(-z))
        return a   
```

<font face="STCAIYUN">tanh</font>
    
**公式**：
    
$tanh(z) = \frac{2}{1 + e^{-2z}} - 1$

 **导数**：
    $tanh(z) = (1+a)(1-a)$

**函数图像**：
<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/6/tanh_seperator.png">


- `多分类函数`
   <font face="STCAIYUN">softmax</font>
    **公式**：
    $$
    a_j = \frac{e^{z_j}}{\sum\limits_{i=1}^m e^{z_i}}=\frac{e^{z_j}}{e^{z_1}+e^{z_2}+\dots+e^{z_m}}
    $$
   
    **描述**：`由原来的logistic和max改进而来可用于多分类，给予样本在每个类别的概率`
    
    示例：
    
    ```python
    def Softmax2(Z):
    shift_Z = Z - np.max(Z)
    exp_Z = np.exp(shift_Z)
    A = exp_Z / np.sum(exp_Z)
    return A
    ```

## 损失函数

-  <font face="STCAIYUN">交叉熵</font>
  
  **说明**`多样本时用矩阵形式,多分类时应运用求和的方式，求导易实现`
  **反向计算**：`在反向传播时，1.计算z,再计算dz,然后计算dw,db,db如果是多样本，则为矩阵计算，且计算dw,db时要求平均值`
    **公式**：
    $loss(w,b) = -[ylna + (1-y)ln(1-a)]$
    **导数**：
    $\frac {\delta loss(w,b)} {\delta z} = a - y$
    **示例**：
```python
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
```
**图示**：
![](/AIML/2020AutumnClass/第2次作业/201809059_XuZhou/3_1.png)


