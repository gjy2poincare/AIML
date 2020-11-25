<<<<<<< HEAD
人工智能概论第一次作业（step1）
#第1章 概论
##1.0 人工智能的定义
从1956年的达特茅斯会议开始，人工智能作为一个专门的研究领域出现，经历了超过半个世纪的起伏，终于在2007年前后，迎来了又一次大发展。

<img src="https://i.loli.net/2020/10/05/KFkZTtGu6czWhos.png" width="600" />

##1.1人工智能的定义
#### 第一个层面，人们对人工智能的**期待**可以分为：
**智能地把某件特定的事情做好，在某个领域增强人类的智慧，这种方式又叫做智能增强。**
**像人类一样能认知，思考，判断：模拟人类的智能。**

#### 第二个层面，**从技术的特点来看**。

**机器学习：能让运行程序的电脑来学习并自动掌握某些规律。**
**机器学习可以大致地分为下面三种类型：1、监督学习 2、无监督学习 3、强化学习**
**机器学习领域出现了各种模型，其中，神经网络模型是一个重要的方法。**

<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/1/image5.png" width="500" />

图1-1 M-P神经元模型
#### 第三个层面，**从应用的角度来看**，我们看到狭义人工智能在各个领域都取得了很大的成果。
**例如：翻译领域（微软的中英翻译超过人类）、阅读理解（SQuAD 比赛）、下围棋（2016）德州扑克（2019）麻将（2019）**

<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/1/image6.png" />

图1-2 弱人工智能领域中机器学习部分的内容

**在现代软件开发流程中，程序的开发，和AI模型的开发的生命周期应该协作，图下展示了这个协作的过程。**


<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/1/image8.png" />

图1-3 软件开发流程与AI模型开发流程的协作

## 1.2 范式的演化

范式演化的四个阶段
#### 第一阶段：经验
**从几千年前到几百年前，人们描述自然现象，归纳总结一些规律。**

#### 第二阶段：理论
**在理论演算阶段，不但要定性，而且要定量，要通过数学公式严格的推
导得到结论。**

#### 第三阶段：计算仿真
**从二十世纪中期开始，利用电子计算机对科学实验进行模拟仿真的模式得到迅速普及，人们可以对复杂现象通过模拟仿真，推演更复杂的现象，典型案例如模拟核试验、天气预报等。**

#### 第四阶段：数据探索
**在这个阶段，科学家收集数据，分析数据，探索新的规律。在深度学习的浪潮中出现的许多结果就是基于海量数据学习得来的。**
**当人类探索客观世界的时候，大部分情况下，我们是不了解新环境的运行规则的。这个时候，我们可以观察自己的行动和客观世界的反馈，判断得失，再总结出规律。这种学习方法，叫强化学习，可以使用这种方法来找出适合的策略。**

<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/1/image13.png" width="500" />

图1-4 强化学习示意图


## 1.3 神经网络的基本工作原理简介

### 1.3.1 神经元细胞的数学模型

神经网络由基本的神经元组成，图1-13就是一个神经元的数学/计算模型，便于我们用程序来实现。

<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/1/NeuranCell.png" ch="500" />

图1-5 神经元计算模型

#### 输入 input：$(x_1,x_2,x_3)$ 是外界输入信号

#### 权重 weights：$(w_1,w_2,w_3)$ 是每个输入信号的权重值

#### 偏移 bias：
$$w_1 \cdot x_1 + w_2 \cdot x_2 + w_3 \cdot x_3 \geq t$$

时，该神经元细胞才会兴奋。我们把t挪到等式左侧来，变成$(-t)$，然后把它写成 $b$，变成了：

$$w_1 \cdot x_1 + w_2 \cdot x_2 + w_3 \cdot x_3 + b \geq 0$$

于是 $b$ 诞生了！

#### 求和计算 sum：
$
\begin{aligned}
Z &= w_1 \cdot x_1 + w_2 \cdot x_2 + w_3 \cdot x_3 + b \\\\
&= \sum_{i=1}^m(w_i \cdot x_i) + b
\end{aligned}
$$

在上面的例子中 $m=3$。我们把$w_i \cdot x_i$变成矩阵运算的话，就变成了：

$$Z = W \cdot X + b$$

### 神经网络的主要功能

#### 回归（Regression）或者叫做拟合（Fitting）
<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images\Images\1\sgd_result.png">

图1-6 回归/拟合示意图

所谓回归或者拟合，其实就是给出x值输出y值的过程，并且让y值与样本数据形成的曲线的距离尽量小，可以理解为是对样本数据的一种骨架式的抽象。

# 第2章 神经网络中的三个基本概念

## 2.0 通俗地理解三大概念

这三大概念是：反向传播，梯度下降，损失函数。

例子：打靶
小明拿了一支步枪，射击100米外的靶子。这支步枪没有准星，或者是准星有问题，或者是小明眼神儿不好看不清靶子，或者是雾很大，或者风很大，或者由于木星的影响而侧向引力场异常......反正就是遇到各种干扰因素。

第一次试枪后，拉回靶子一看，弹着点偏左了，于是在第二次试枪时，小明就会有意识地向右侧偏几毫米，再看靶子上的弹着点，如此反复几次，小明就会掌握这支步枪的脾气了。图2-1显示了小明的5次试枪过程。

<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/2/target1.png" width="500" ch="500" />

图2-1 打靶的弹着点记录

在这个例子中：

- 目的：打中靶心；
- 初始化：随便打一枪，能上靶就行，但是要记住当时的步枪的姿态；
- 前向计算：让子弹飞一会儿，击中靶子；
- 损失函数：环数，偏离角度；
- 反向传播：把靶子拉回来看；
- 梯度下降：根据本次的偏差，调整步枪的射击角度。

###总结
简单总结一下反向传播与梯度下降的基本工作原理：

1. 初始化；
2. 正向计算；
3. 损失函数为我们提供了计算损失的方法；
4. 梯度下降是在损失函数基础上向着损失最小的点靠近而指引了网络权重调整的方向；
5. 反向传播把损失值反向传给神经网络的每一层，让每一层都根据损失值反向调整权重；
6. 重复正向计算过程，直到精度满足要求（比如损失函数值小于 $0.001$）。

## 2.2 梯度下降

### 2.2.1 从自然现象中理解梯度下降

在自然界中，梯度下降的最好例子，就是泉水下山的过程：

1. 水受重力影响，会在当前位置，沿着最陡峭的方向流动，有时会形成瀑布（梯度下降）；
2. 水流下山的路径不是唯一的，在同一个地点，有可能有多个位置具有同样的陡峭程度，而造成了分流（可以得到多个解）；
3. 遇到坑洼地区，有可能形成湖泊，而终止下山过程（不能得到全局最优解，而是局部最优解）。
### 2.2.2 梯度下降的数学理解

梯度下降的数学公式：

$$\theta_{n+1} = \theta_{n} - \eta \cdot \nabla J(\theta) \tag{1}$$

其中：

- $\theta_{n+1}$：下一个值；
- $\theta_n$：当前值；
- $-$：减号，梯度的反向；
- $\eta$：学习率或步长，控制每一步走的距离，不要太快以免错过了最佳景点，不要太慢以免时间太长；
- $\nabla$：梯度，函数当前位置的最快上升点；
- $J(\theta)$：函数。

#### 梯度下降的三要素

1. 当前点；
2. 方向；
3. 步长。

# 第3章 损失函数

### 3.0.1 概念
在各种材料中经常看到的中英文词汇有：误差，偏差，Error，Cost，Loss，损失，代价......意思都差不多，在本书中，使用“损失函数”和“Loss Function”这两个词汇，具体的损失函数符号用 $J$ 来表示，误差值用 $loss$ 表示。

“损失”就是所有样本的“误差”的总和，亦即（$m$ 为样本数）：

$$损失 = \sum^m_{i=1}误差_i$$

$$J = \sum_{i=1}^m loss_i$$
### 损失函数的作用

损失函数的作用，就是计算神经网络每次迭代的前向计算结果与真实值的差距，从而指导下一步的训练向正确的方向进行。

如何使用损失函数呢？具体步骤：

1. 用随机值初始化前向计算公式的参数；
2. 代入样本，计算输出的预测值；
3. 用损失函数计算预测值和标签值（真实值）的误差；
4. 根据损失函数的导数，沿梯度最小方向将误差回传，修正前向计算公式中的各个权重值；
5. 进入第2步重复, 直到损失函数值达到一个满意的值就停止迭代。


均方差函数，主要用于回归

交叉熵函数，主要用于分类

二者都是非负函数，极值在底部，用梯度下降法可以求解。

# 3.1 均方差函数

MSE - Mean Square Error。

该函数就是最直观的一个损失函数了，计算预测值和真实值之间的欧式距离。预测值和真实值越接近，两者的均方差就越小。

均方差函数常用于线性回归(linear regression)，即函数拟合(function fitting)。公式如下：

$$
loss = {1 \over 2}(z-y)^2 \tag{单样本}
$$

$$
J=\frac{1}{2m} \sum_{i=1}^m (z_i-y_i)^2 \tag{多样本}
$$

- 第一张，用均方差函数计算得到 $Loss=0.53$；
- 第二张，直线向上平移一些，误差计算 $Loss=0.16$，比图一的误差小很多；
- 第三张，又向上平移了一些，误差计算 $Loss=0.048$，此后还可以继续尝试平移（改变 $b$ 值）或者变换角度（改变 $w$ 值），得到更小的损失函数值；
- 第四张，偏离了最佳位置，误差值 $Loss=0.18$，这种情况，算法会让尝试方向反向向下。

<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/3/mse2.png" ch="500" />


## 3.2 交叉熵损失函数

交叉熵（Cross Entropy）是Shannon信息论中一个重要概念，主要用于度量两个概率分布间的差异性信息。在信息论中，交叉熵是表示两个概率分布 $p,q$ 的差异，其中 $p$ 表示真实分布，$q$ 表示预测分布，那么 $H(p,q)$ 就称为交叉熵：

$$H(p,q)=\sum_i p_i \cdot \ln {1 \over q_i} = - \sum_i p_i \ln q_i \tag{1}$$

交叉熵可在神经网络中作为损失函数，$p$ 表示真实标记的分布，$q$ 则为训练后的模型的预测标记分布，交叉熵损失函数可以衡量 $p$ 与 $q$ 的相似性。

**交叉熵函数常用于逻辑回归(logistic regression)，也就是分类(classification)。**



##代码
import numpy as np

def target_function(w,b):
    x = 2*w+3*b
    y=2*b+1
    z=x*y
    return x,y,z

def single_variable(w,b,t):
    print("\nsingle variable: b ----- ")
    error = 1e-5
    while(True):
        x,y,z = target_function(w,b)
        delta_z = z - t
        print("w=%f,b=%f,z=%f,delta_z=%f"%(w,b,z,delta_z))
        if abs(delta_z) < error:
            break
        delta_b = delta_z /63
        print("delta_b=%f"%delta_b)
        b = b - delta_b

    print("done!")
    print("final b=%f"%b)

def single_variable_new(w,b,t):
    print("\nsingle variable new: b ----- ")
    error = 1e-5
    while(True):
        x,y,z = target_function(w,b)
        delta_z = z - t
        print("w=%f,b=%f,z=%f,delta_z=%f"%(w,b,z,delta_z))
        if abs(delta_z) < error:
            break
        factor_b = 2*x+3*y
        delta_b = delta_z/factor_b
        print("factor_b=%f, delta_b=%f"%(factor_b, delta_b))
        b = b - delta_b

    print("done!")
    print("final b=%f"%b)


# this version has a bug
def double_variable(w,b,t):
    print("\ndouble variable: w, b -----")
    error = 1e-5
    while(True):
        x,y,z = target_function(w,b)
        delta_z = z - t
        print("w=%f,b=%f,z=%f,delta_z=%f"%(w,b,z,delta_z))
        if abs(delta_z) < error:
            break
        delta_b = delta_z/63/2
        delta_w = delta_z/18/2
        print("delta_b=%f, delta_w=%f"%(delta_b,delta_w))
        b = b - delta_b
        w = w - delta_w
    print("done!")
    print("final b=%f"%b)
    print("final w=%f"%w)

# this is correct version
def double_variable_new(w,b,t):
    print("\ndouble variable new: w, b -----")
    error = 1e-5
    while(True):
        x,y,z = target_function(w,b)
        delta_z = z - t
        print("w=%f,b=%f,z=%f,delta_z=%f"%(w,b,z,delta_z))
        if abs(delta_z) < error:
            break

        factor_b, factor_w = calculate_wb_factor(x,y)
        delta_b = delta_z/factor_b/2
        delta_w = delta_z/factor_w/2
        print("factor_b=%f, factor_w=%f, delta_b=%f, delta_w=%f"%(factor_b, factor_w, delta_b,delta_w))
        b = b - delta_b
        w = w - delta_w
    print("done!")
    print("final b=%f"%b)
    print("final w=%f"%w)

def calculate_wb_factor(x,y):
    factor_b = 2*x+3*y
    factor_w = 2*y
    return factor_b, factor_w

if __name__ == '__main__':
    w = 3
    b = 4
    t = 150
    single_variable(w,b,t)
    single_variable_new(w,b,t)
    double_variable(w,b,t)
    double_variable_new(w,b,t)

<img src="https://i.loli.net/2020/10/05/Mv8AiqztEx2okCO.png" ch="500" />


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

<img src="https://i.loli.net/2020/10/05/lR6ceD9oQjILWv8.png" ch="500" />

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def target_function(x,y):
    J = x**2 + np.sin(y)**2
    return J

def derivative_function(theta):
    x = theta[0]
    y = theta[1]
    return np.array([2*x,2*np.sin(y)*np.cos(y)])

def show_3d_surface(x, y, z):
    fig = plt.figure()
    ax = Axes3D(fig)
 
    u = np.linspace(-3, 3, 100)
    v = np.linspace(-3, 3, 100)
    X, Y = np.meshgrid(u, v)
    R = np.zeros((len(u), len(v)))
    for i in range(len(u)):
        for j in range(len(v)):
            R[i, j] = X[i, j]**2 + np.sin(Y[i, j])**2

    ax.plot_surface(X, Y, R, cmap='rainbow')
    plt.plot(x,y,z,c='black')
    plt.show()

if __name__ == '__main__':
    theta = np.array([3,1])
    eta = 0.1
    error = 1e-2

    X = []
    Y = []
    Z = []
    for i in range(100):
        print(theta)
        x=theta[0]
        y=theta[1]
        z=target_function(x,y)
        X.append(x)
        Y.append(y)
        Z.append(z)
        print("%d: x=%f, y=%f, z=%f" %(i,x,y,z))
        d_theta = derivative_function(theta)
        print("    ",d_theta)
        theta = theta - eta * d_theta
        if z < error:
            break
    show_3d_surface(X,Y,Z)

   <img src="https://i.loli.net/2020/10/05/qUxdjfzHPCoFmu9.png" ch="500" />

   import numpy as np
import matplotlib.pyplot as plt

def targetFunction(x):
    y = (x-1)**2 + 0.1
    return y

def derivativeFun(x):
    y = 2*(x-1)
    return y

def create_sample():
    x = np.linspace(-1,3,num=100)
    y = targetFunction(x)
    return x,y

def draw_base():
    x,y=create_sample()
    plt.plot(x,y,'.')
    plt.show()
    return x,y
   
def gd(eta):
    x = -0.8
    a = np.zeros((2,10))
    for i in range(10):
        a[0,i] = x
        a[1,i] = targetFunction(x)
        dx = derivativeFun(x)
        x = x - eta*dx
    
    plt.plot(a[0,:],a[1,:],'x')
    plt.plot(a[0,:],a[1,:])
    plt.title("eta=%f" %eta)
    plt.show()

if __name__ == '__main__':

    eta = [1.1,1.,0.8,0.6,0.4,0.2,0.1]

    for e in eta:
        X,Y=create_sample()
        plt.plot(X,Y,'.')
        #plt.show()
        gd(e)

<img src="https://i.loli.net/2020/10/05/fs28ka75WmSAIow.png" ch="500" />

<img src="https://i.loli.net/2020/10/05/K2DAUHgtxv7z5Qy.png" ch="500" />

<img src="https://i.loli.net/2020/10/05/AflQxYXkye748q2.png" ch="500" />


# 心得
=======
人工智能概论第一次作业（step1）
#第1章 概论
##1.0 人工智能的定义
从1956年的达特茅斯会议开始，人工智能作为一个专门的研究领域出现，经历了超过半个世纪的起伏，终于在2007年前后，迎来了又一次大发展。

<img src="https://i.loli.net/2020/10/05/KFkZTtGu6czWhos.png" width="600" />

##1.1人工智能的定义
#### 第一个层面，人们对人工智能的**期待**可以分为：
**智能地把某件特定的事情做好，在某个领域增强人类的智慧，这种方式又叫做智能增强。**
**像人类一样能认知，思考，判断：模拟人类的智能。**

#### 第二个层面，**从技术的特点来看**。

**机器学习：能让运行程序的电脑来学习并自动掌握某些规律。**
**机器学习可以大致地分为下面三种类型：1、监督学习 2、无监督学习 3、强化学习**
**机器学习领域出现了各种模型，其中，神经网络模型是一个重要的方法。**

<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/1/image5.png" width="500" />

图1-1 M-P神经元模型
#### 第三个层面，**从应用的角度来看**，我们看到狭义人工智能在各个领域都取得了很大的成果。
**例如：翻译领域（微软的中英翻译超过人类）、阅读理解（SQuAD 比赛）、下围棋（2016）德州扑克（2019）麻将（2019）**

<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/1/image6.png" />

图1-2 弱人工智能领域中机器学习部分的内容

**在现代软件开发流程中，程序的开发，和AI模型的开发的生命周期应该协作，图下展示了这个协作的过程。**


<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/1/image8.png" />

图1-3 软件开发流程与AI模型开发流程的协作

## 1.2 范式的演化

范式演化的四个阶段
#### 第一阶段：经验
**从几千年前到几百年前，人们描述自然现象，归纳总结一些规律。**

#### 第二阶段：理论
**在理论演算阶段，不但要定性，而且要定量，要通过数学公式严格的推
导得到结论。**

#### 第三阶段：计算仿真
**从二十世纪中期开始，利用电子计算机对科学实验进行模拟仿真的模式得到迅速普及，人们可以对复杂现象通过模拟仿真，推演更复杂的现象，典型案例如模拟核试验、天气预报等。**

#### 第四阶段：数据探索
**在这个阶段，科学家收集数据，分析数据，探索新的规律。在深度学习的浪潮中出现的许多结果就是基于海量数据学习得来的。**
**当人类探索客观世界的时候，大部分情况下，我们是不了解新环境的运行规则的。这个时候，我们可以观察自己的行动和客观世界的反馈，判断得失，再总结出规律。这种学习方法，叫强化学习，可以使用这种方法来找出适合的策略。**

<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/1/image13.png" width="500" />

图1-4 强化学习示意图


## 1.3 神经网络的基本工作原理简介

### 1.3.1 神经元细胞的数学模型

神经网络由基本的神经元组成，图1-13就是一个神经元的数学/计算模型，便于我们用程序来实现。

<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/1/NeuranCell.png" ch="500" />

图1-5 神经元计算模型

#### 输入 input：$(x_1,x_2,x_3)$ 是外界输入信号

#### 权重 weights：$(w_1,w_2,w_3)$ 是每个输入信号的权重值

#### 偏移 bias：
$$w_1 \cdot x_1 + w_2 \cdot x_2 + w_3 \cdot x_3 \geq t$$

时，该神经元细胞才会兴奋。我们把t挪到等式左侧来，变成$(-t)$，然后把它写成 $b$，变成了：

$$w_1 \cdot x_1 + w_2 \cdot x_2 + w_3 \cdot x_3 + b \geq 0$$

于是 $b$ 诞生了！

#### 求和计算 sum：
$
\begin{aligned}
Z &= w_1 \cdot x_1 + w_2 \cdot x_2 + w_3 \cdot x_3 + b \\\\
&= \sum_{i=1}^m(w_i \cdot x_i) + b
\end{aligned}
$$

在上面的例子中 $m=3$。我们把$w_i \cdot x_i$变成矩阵运算的话，就变成了：

$$Z = W \cdot X + b$$

### 神经网络的主要功能

#### 回归（Regression）或者叫做拟合（Fitting）
<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images\Images\1\sgd_result.png">

图1-6 回归/拟合示意图

所谓回归或者拟合，其实就是给出x值输出y值的过程，并且让y值与样本数据形成的曲线的距离尽量小，可以理解为是对样本数据的一种骨架式的抽象。

# 第2章 神经网络中的三个基本概念

## 2.0 通俗地理解三大概念

这三大概念是：反向传播，梯度下降，损失函数。

例子：打靶
小明拿了一支步枪，射击100米外的靶子。这支步枪没有准星，或者是准星有问题，或者是小明眼神儿不好看不清靶子，或者是雾很大，或者风很大，或者由于木星的影响而侧向引力场异常......反正就是遇到各种干扰因素。

第一次试枪后，拉回靶子一看，弹着点偏左了，于是在第二次试枪时，小明就会有意识地向右侧偏几毫米，再看靶子上的弹着点，如此反复几次，小明就会掌握这支步枪的脾气了。图2-1显示了小明的5次试枪过程。

<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/2/target1.png" width="500" ch="500" />

图2-1 打靶的弹着点记录

在这个例子中：

- 目的：打中靶心；
- 初始化：随便打一枪，能上靶就行，但是要记住当时的步枪的姿态；
- 前向计算：让子弹飞一会儿，击中靶子；
- 损失函数：环数，偏离角度；
- 反向传播：把靶子拉回来看；
- 梯度下降：根据本次的偏差，调整步枪的射击角度。

###总结
简单总结一下反向传播与梯度下降的基本工作原理：

1. 初始化；
2. 正向计算；
3. 损失函数为我们提供了计算损失的方法；
4. 梯度下降是在损失函数基础上向着损失最小的点靠近而指引了网络权重调整的方向；
5. 反向传播把损失值反向传给神经网络的每一层，让每一层都根据损失值反向调整权重；
6. 重复正向计算过程，直到精度满足要求（比如损失函数值小于 $0.001$）。

## 2.2 梯度下降

### 2.2.1 从自然现象中理解梯度下降

在自然界中，梯度下降的最好例子，就是泉水下山的过程：

1. 水受重力影响，会在当前位置，沿着最陡峭的方向流动，有时会形成瀑布（梯度下降）；
2. 水流下山的路径不是唯一的，在同一个地点，有可能有多个位置具有同样的陡峭程度，而造成了分流（可以得到多个解）；
3. 遇到坑洼地区，有可能形成湖泊，而终止下山过程（不能得到全局最优解，而是局部最优解）。
### 2.2.2 梯度下降的数学理解

梯度下降的数学公式：

$$\theta_{n+1} = \theta_{n} - \eta \cdot \nabla J(\theta) \tag{1}$$

其中：

- $\theta_{n+1}$：下一个值；
- $\theta_n$：当前值；
- $-$：减号，梯度的反向；
- $\eta$：学习率或步长，控制每一步走的距离，不要太快以免错过了最佳景点，不要太慢以免时间太长；
- $\nabla$：梯度，函数当前位置的最快上升点；
- $J(\theta)$：函数。

#### 梯度下降的三要素

1. 当前点；
2. 方向；
3. 步长。

# 第3章 损失函数

### 3.0.1 概念
在各种材料中经常看到的中英文词汇有：误差，偏差，Error，Cost，Loss，损失，代价......意思都差不多，在本书中，使用“损失函数”和“Loss Function”这两个词汇，具体的损失函数符号用 $J$ 来表示，误差值用 $loss$ 表示。

“损失”就是所有样本的“误差”的总和，亦即（$m$ 为样本数）：

$$损失 = \sum^m_{i=1}误差_i$$

$$J = \sum_{i=1}^m loss_i$$
### 损失函数的作用

损失函数的作用，就是计算神经网络每次迭代的前向计算结果与真实值的差距，从而指导下一步的训练向正确的方向进行。

如何使用损失函数呢？具体步骤：

1. 用随机值初始化前向计算公式的参数；
2. 代入样本，计算输出的预测值；
3. 用损失函数计算预测值和标签值（真实值）的误差；
4. 根据损失函数的导数，沿梯度最小方向将误差回传，修正前向计算公式中的各个权重值；
5. 进入第2步重复, 直到损失函数值达到一个满意的值就停止迭代。


均方差函数，主要用于回归

交叉熵函数，主要用于分类

二者都是非负函数，极值在底部，用梯度下降法可以求解。

# 3.1 均方差函数

MSE - Mean Square Error。

该函数就是最直观的一个损失函数了，计算预测值和真实值之间的欧式距离。预测值和真实值越接近，两者的均方差就越小。

均方差函数常用于线性回归(linear regression)，即函数拟合(function fitting)。公式如下：

$$
loss = {1 \over 2}(z-y)^2 \tag{单样本}
$$

$$
J=\frac{1}{2m} \sum_{i=1}^m (z_i-y_i)^2 \tag{多样本}
$$

- 第一张，用均方差函数计算得到 $Loss=0.53$；
- 第二张，直线向上平移一些，误差计算 $Loss=0.16$，比图一的误差小很多；
- 第三张，又向上平移了一些，误差计算 $Loss=0.048$，此后还可以继续尝试平移（改变 $b$ 值）或者变换角度（改变 $w$ 值），得到更小的损失函数值；
- 第四张，偏离了最佳位置，误差值 $Loss=0.18$，这种情况，算法会让尝试方向反向向下。

<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/3/mse2.png" ch="500" />


## 3.2 交叉熵损失函数

交叉熵（Cross Entropy）是Shannon信息论中一个重要概念，主要用于度量两个概率分布间的差异性信息。在信息论中，交叉熵是表示两个概率分布 $p,q$ 的差异，其中 $p$ 表示真实分布，$q$ 表示预测分布，那么 $H(p,q)$ 就称为交叉熵：

$$H(p,q)=\sum_i p_i \cdot \ln {1 \over q_i} = - \sum_i p_i \ln q_i \tag{1}$$

交叉熵可在神经网络中作为损失函数，$p$ 表示真实标记的分布，$q$ 则为训练后的模型的预测标记分布，交叉熵损失函数可以衡量 $p$ 与 $q$ 的相似性。

**交叉熵函数常用于逻辑回归(logistic regression)，也就是分类(classification)。**



##代码
import numpy as np

def target_function(w,b):
    x = 2*w+3*b
    y=2*b+1
    z=x*y
    return x,y,z

def single_variable(w,b,t):
    print("\nsingle variable: b ----- ")
    error = 1e-5
    while(True):
        x,y,z = target_function(w,b)
        delta_z = z - t
        print("w=%f,b=%f,z=%f,delta_z=%f"%(w,b,z,delta_z))
        if abs(delta_z) < error:
            break
        delta_b = delta_z /63
        print("delta_b=%f"%delta_b)
        b = b - delta_b

    print("done!")
    print("final b=%f"%b)

def single_variable_new(w,b,t):
    print("\nsingle variable new: b ----- ")
    error = 1e-5
    while(True):
        x,y,z = target_function(w,b)
        delta_z = z - t
        print("w=%f,b=%f,z=%f,delta_z=%f"%(w,b,z,delta_z))
        if abs(delta_z) < error:
            break
        factor_b = 2*x+3*y
        delta_b = delta_z/factor_b
        print("factor_b=%f, delta_b=%f"%(factor_b, delta_b))
        b = b - delta_b

    print("done!")
    print("final b=%f"%b)


# this version has a bug
def double_variable(w,b,t):
    print("\ndouble variable: w, b -----")
    error = 1e-5
    while(True):
        x,y,z = target_function(w,b)
        delta_z = z - t
        print("w=%f,b=%f,z=%f,delta_z=%f"%(w,b,z,delta_z))
        if abs(delta_z) < error:
            break
        delta_b = delta_z/63/2
        delta_w = delta_z/18/2
        print("delta_b=%f, delta_w=%f"%(delta_b,delta_w))
        b = b - delta_b
        w = w - delta_w
    print("done!")
    print("final b=%f"%b)
    print("final w=%f"%w)

# this is correct version
def double_variable_new(w,b,t):
    print("\ndouble variable new: w, b -----")
    error = 1e-5
    while(True):
        x,y,z = target_function(w,b)
        delta_z = z - t
        print("w=%f,b=%f,z=%f,delta_z=%f"%(w,b,z,delta_z))
        if abs(delta_z) < error:
            break

        factor_b, factor_w = calculate_wb_factor(x,y)
        delta_b = delta_z/factor_b/2
        delta_w = delta_z/factor_w/2
        print("factor_b=%f, factor_w=%f, delta_b=%f, delta_w=%f"%(factor_b, factor_w, delta_b,delta_w))
        b = b - delta_b
        w = w - delta_w
    print("done!")
    print("final b=%f"%b)
    print("final w=%f"%w)

def calculate_wb_factor(x,y):
    factor_b = 2*x+3*y
    factor_w = 2*y
    return factor_b, factor_w

if __name__ == '__main__':
    w = 3
    b = 4
    t = 150
    single_variable(w,b,t)
    single_variable_new(w,b,t)
    double_variable(w,b,t)
    double_variable_new(w,b,t)

<img src="https://i.loli.net/2020/10/05/Mv8AiqztEx2okCO.png" ch="500" />


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

<img src="https://i.loli.net/2020/10/05/lR6ceD9oQjILWv8.png" ch="500" />

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def target_function(x,y):
    J = x**2 + np.sin(y)**2
    return J

def derivative_function(theta):
    x = theta[0]
    y = theta[1]
    return np.array([2*x,2*np.sin(y)*np.cos(y)])

def show_3d_surface(x, y, z):
    fig = plt.figure()
    ax = Axes3D(fig)
 
    u = np.linspace(-3, 3, 100)
    v = np.linspace(-3, 3, 100)
    X, Y = np.meshgrid(u, v)
    R = np.zeros((len(u), len(v)))
    for i in range(len(u)):
        for j in range(len(v)):
            R[i, j] = X[i, j]**2 + np.sin(Y[i, j])**2

    ax.plot_surface(X, Y, R, cmap='rainbow')
    plt.plot(x,y,z,c='black')
    plt.show()

if __name__ == '__main__':
    theta = np.array([3,1])
    eta = 0.1
    error = 1e-2

    X = []
    Y = []
    Z = []
    for i in range(100):
        print(theta)
        x=theta[0]
        y=theta[1]
        z=target_function(x,y)
        X.append(x)
        Y.append(y)
        Z.append(z)
        print("%d: x=%f, y=%f, z=%f" %(i,x,y,z))
        d_theta = derivative_function(theta)
        print("    ",d_theta)
        theta = theta - eta * d_theta
        if z < error:
            break
    show_3d_surface(X,Y,Z)

   <img src="https://i.loli.net/2020/10/05/qUxdjfzHPCoFmu9.png" ch="500" />

   import numpy as np
import matplotlib.pyplot as plt

def targetFunction(x):
    y = (x-1)**2 + 0.1
    return y

def derivativeFun(x):
    y = 2*(x-1)
    return y

def create_sample():
    x = np.linspace(-1,3,num=100)
    y = targetFunction(x)
    return x,y

def draw_base():
    x,y=create_sample()
    plt.plot(x,y,'.')
    plt.show()
    return x,y
   
def gd(eta):
    x = -0.8
    a = np.zeros((2,10))
    for i in range(10):
        a[0,i] = x
        a[1,i] = targetFunction(x)
        dx = derivativeFun(x)
        x = x - eta*dx
    
    plt.plot(a[0,:],a[1,:],'x')
    plt.plot(a[0,:],a[1,:])
    plt.title("eta=%f" %eta)
    plt.show()

if __name__ == '__main__':

    eta = [1.1,1.,0.8,0.6,0.4,0.2,0.1]

    for e in eta:
        X,Y=create_sample()
        plt.plot(X,Y,'.')
        #plt.show()
        gd(e)

<img src="https://i.loli.net/2020/10/05/fs28ka75WmSAIow.png" ch="500" />

<img src="https://i.loli.net/2020/10/05/K2DAUHgtxv7z5Qy.png" ch="500" />

<img src="https://i.loli.net/2020/10/05/AflQxYXkye748q2.png" ch="500" />


# 心得
>>>>>>> fb92f0def436efb9779b38e243f99b9547ef8ed8
人工智能：像人一样思考，理性的思考；像人一样行动，理性的行动。如果你与一台机器进行对话，他能回答你的问题并且感受不到是机器在回答的话，就说这台机器具有智能。当然并不是通过测试就说明有智能，但现阶段的研究主要还是弱人工智能；模仿人脑的基本功能，感知，记忆、学习和决策等，向着强人工智能以及超级人工智能发展的话还有很长一段路要走，中间有着巨大的鸿沟。通过第一章的学习：了解了人工智能的定义，以及一些算法概念，将现实问题转变为数据再用电脑来实现。概念很多，很抽象很难理解，但我相信我花时间可以明白的更多。有信心学好它。