### 1单变量线性回归

$$1.1模型表示$$

机器学习的算法的工作原理如下：
1）我们向学习算法提供训练集
2）学习算法的任务通常是输出一个函数，用h代表假设函数
3）假设函数的作用是，把相关的已知变量作为输入变量，而试着输出相应的预测值
当我们设计一个机器学习算法时，我们首先要做的是：决定怎样表达这个假设函数h
例如：有y(x)=a+bx 其中a,b是常量这个函数中只含有一个输入变量所以这样的问题就叫做单变量线性回归问题而且这个函数模型就叫做单变量线性回归。

$$1.2代价函数$$
在上面的函数y(x)=a+bx中a,b两个参数我们要如何确定，也就是输入x时我们预测的值最接近该样本对应的y值的参数a,b因此我们需要一定量的样本进行预测来画图进行求解所以我们就需要训练集，在线性回归中我们要解决的是一个最小化的问题，所以我们就要写出关于函数y(x)=a+bx中a,b两个参数的最小化这里就要用到我们的代价函数了。
![gjy6](gjy6.png)

m表示训练样本数量

(x(i),y(i))代表第i个训练样本

通过训练样本我们就可以绘制支出关于需想x,y的线性图

代码如下图所示：

import numpy as np

import matplotlib.pyplot as plt

def target_function(x):
    y = x*x
    return y

def derivative_function(x):
    return 2*x

def draw_function():
    x = np.linspace(-1.2,1.2)
    y = target_function(x)
    plt.plot(x,y)

def draw_gd(X):
   
    Y = []
    
    for i in range(len(X)):
      
        Y.append(target_function(X[i]))
    
    plt.plot(X,Y)

if __name__ == '__main__':

    x = 1.2
    
    eta = 0.3
    
    error = 1e-3
    
    X = []
    
    X.append(x)
    
    y = target_function(x)
    
    while y > error:
        
        x = x - eta * derivative_function(x)
        
        X.append(x)
        
        y = target_function(x)
        
        print("x=%f, y=%f" %(x,y))

    draw_function()
    
    draw_gd(X)
    
    plt.show()
    运行结果如图所示：
![gjy7](gjy7.png)
![gjy4](gjy4.png)

$$1.3最小二乘法$$

最小二乘法，也叫做最小平方法，它通过最小化误差的平方和寻找数据的最佳函数匹配。利用最小二乘法可以简便地求得未知的数据，并使得这些求得的数据与实际数据之间误差的平方和为最小。最小二乘法还可用于曲线拟合。其他一些优化问题也可通过最小化能量或最小二乘法来表达。

当我们在研究两个变量（x，y）之间的相互关系时，往往会有一系列的数据对[(x1,y1),(x2,y2)... (xm,ym)],那么将这些数据描绘到x-y直系坐标中若发现这些点都在一条直线附近时，那么初始令这条直线方程的表达式：

![gjy8](gjy8.png)


其中  是任意的实数，现在需要让当 x 取值为xi预测值yi与回归方程所预测的 之间的差值平方最小，但是对于整个回归方程而言，就是所有预测值与实际值之间差值平方之和最小。

用预测值与真实值之间的差值,需要比较两个Y值，必须有个不变的因子那就是X，在同一个X下比较两种Y才有意义。两个Y值之间做差值总会有正负的性质

故建立一下方程:

![gjy9](gjy9.png)

Q为关于预测方程中两个参数a0,a1的函数而已，此时将预测方程带入以上公式得到以下方程:


![gjy10](gjy10.png)

要使的方程Q的取值最小，那么需要对函数Q分别对a0,a1求一阶偏导数，并且零偏导之后的值为0。即

![gjy11](gjy11.png)

接下来就需要对两个参数进行变换求解了，经过一顿移项变换操作之后得到两个a0,a1参数关于x和y的表达式。


![gjy12](gjy12.png)

最小二乘法主要代码如下：

根据公式15
def method1(X,Y,m):
    x_mean = X.mean()
    p = sum(Y*(X-x_mean))
    q = sum(X*X) - sum(X)*sum(X)/m
    w = p/q
    return w

根据公式16
def method2(X,Y,m):
    x_mean = X.mean()
    y_mean = Y.mean()
    p = sum(X*(Y-y_mean))
    q = sum(X*X) - x_mean*sum(X)
    w = p/q
    return w

根据公式13
def method3(X,Y,m):
    p = m*sum(X*Y) - sum(X)*sum(Y)
    q = m*sum(X*X) - sum(X)*sum(X)
    w = p/q
    return w

根据公式14
def calculate_b_1(X,Y,w,m):
    b = sum(Y-w*X)/m
    return b

根据公式9
def calculate_b_2(X,Y,w):
    b = Y.mean() - w * X.mean()
    return b

最小二乘法结果如下：

![gjy13](gjy13.png)

多变量线性回归求解方法大致与单变量线性回归差不多在此就不在过多的描述

### 2线性二分类

$$ 2.1分类原理 $$
分类函数

对率函数Logistic Function，本身是激活函数，又可以当作二分类的分类函数。
公式
a(z)=1/(1+e−z)

导数
a′(z)=a(z)(1−a(z))

输出值域
[0,1]

使用方式
此函数实际上是一个概率计算，它把[−∞,∞][−∞,∞]之间的任何数字都压缩到[0,1]之间，返回一个概率值。这就是它的工作原理。

训练时，一个样本x在经过神经网络的最后一层的矩阵运算后的结果作为输入，经过Sigmoid后，输出一个[0,1]之间的预测值。我们假设这个样本的标签值为0，如果其预测值越接近0，就越接近标签值，那么误差越小，反向传播的力度就越小。

推理时，我们预先设定一个阈值，我们设置阈值=0.5，则当推理结果大于0.5时，认为是正类；小于0.5时认为是负类；等于0.5时，根据情况自己定义。阈值也不一定就是0.5，也可以是0.65等等，阈值越大，准确率越高，召回率越低；阈值越小则相反。

正向计算

神经网络计算
zi=wxi+b  (1)

分类计算
ai=1/1+e−zi  (2)

损失函数计算
J(w,b)=−1m∑mi=1yiln(ai)+(1−yi)ln(1−ai) (3)

反向传播
对公式3求导：
∂J/∂ai=−1/m∑mi=1(yi/ai−(1−yi)/(1−ai)) (4)


对公式2求导：
∂ai/∂zi=ai(1−ai)(5)


用链式法则结合公式4,5：

∂J/∂zi=∂J/∂ai*∂ai/∂zi

=−1/m∑mi=1(yi−ai)/ai(1−ai)⋅ai(1−ai)

=−1/m∑mi=1(yi−ai)

=1/m∑mi=1(ai−yi)(6)

至此，我们得到了z的误差，进一步向后传播给w和b，对公式1求导可知：

∂zi/∂w=xi (7)

∂zi/∂b=1 (8)

用链式法则结合公式6，7，8：

∂J/∂w=∂J/∂zi*∂zi/∂w=1/m∑mi=1/(ai−yi)xi (9)

∂J/∂b=∂J/∂zi*∂zi/∂b=1/m∑mi=1(ai−yi)   (10)

总的来说将数据输入到神经网络之后，不管是对权重还是偏置的回归，在二分类问题上，经过数学变换均可以映射到形式为wx+bwx+b的变化，关于对数，则表示样本作为正例的可能性。