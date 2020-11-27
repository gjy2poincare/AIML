   ## MOOC总结
	人工神经网络是由简单神经元经过相互连接形成网状结构，通过调节各连接的权重值改变连接的强度，进而实现感知判断。
    反向传播算法的提出进一步推动 了神经网络的发展。目前，神经网络作为一种重要的数据挖掘方法，已在医学诊断、信用卡欺诈识别、手写数字识别以及发动机的故障诊断等领域得到了广泛的应用。
    传统神经网络结构比较简单，训练时随机初始化输入参数并开启循环计算。输出结果与实际结果进行比较，从而得到损失函数并更新变量，使损失函数结果值变小。当达到的误差阈值时，即可停止循环。
    神经网络的训练目的是希望能够学习到一个模型。实现输出一个期望的目标，只是学习的方式是在外界输入样本的刺激下不断改变网络的连接权值。传统神经网络主要分为以下几类。前馈型神经网络，反馈性神经网络和自组织神经网络这几类网络具有不同的学习训练算法，可以归结为监督性学习算法和非监督性学习算法。
    前馈神经网络是一种单向多层的网络结构，即信息是从输入层开始，逐层向一个方向传递，一直到输出层结束。所谓的“前馈”是指输入信号的传播方向为前向，在此过程中并不调整各层的权值参数，而反传播时是将误差逐层向后传递，从而实现使用权值参数对特征的记忆，即通过反向传播(BP)算法来计算各层网络中神经元之,间边的权重。BP算法具有非线性映射能力，理论.上可逼近任意连续函，从而实现对模型的学习。
	BP神经网络也是前馈神经网络，只是它的参数权重值是由反向传播学习算法进行调整的。
 	BP神经网络模型拓扑结构包括输入层、隐层和输出层，利用激活函数来实现从输入到输出的任意非线性映射，从而模拟各层神经元之间的交互
	激活函数须满足处处可导的条件。例如，Sigmoid函数连续可微，求导合适，单调递增，输出值是0~1之间的连续量，这些特点使其适合作为神经网络的激活函数。
    BP神经网络训练过程的基本步骤可以归纳如下
	-初始化网络权值和神经元的阈值，一般通过随机的方式进行初始化
	-前向传播:计算隐层神经元和输出层神经元的输出
	-后向传播:根据目标函数公式修正权值W;
    上述过程反复迭代，通过损失函数和成本函数对前向传播结果进行判定,并通过后向传播过程对权重参数进行修正，起到监督学习的作用，一-直到满足终止条件为止。
	BP神经网络的核心思想是由后层误差推导前层误差，- -层一层的反传，最终获得各层的误差估计，从而得到参数的权重值。由于权值参数的运算量过大，一般采用梯度下降法来实现。
 	所谓梯度下降就是让参数向着梯度的反方向前进一段距离， 不断重复，直到梯度接近零时停止。此时，所有的参数恰好达到使损失函数取得最低值的状态，为了避免局部最优，可以采用随机化梯度下降。
##  神经网络中的矩阵运算

图中是一个两层的神经网络，包含隐藏层和输出层，输入层不算做一层。

<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/1/TwoLayerNN.png" ch="500" />

图1-17 神经网络中的各种符号约定

$$
z1_1 = x_1 \cdot w1_{1,1}+ x_2 \cdot w1_{2,1}+b1_1
$$
$$
z1_2 = x_1 \cdot w1_{1,2}+ x_2 \cdot w1_{2,2}+b1_2
$$
$$
z1_3 = x_1 \cdot w1_{1,3}+ x_2 \cdot w1_{2,3}+b1_3
$$

变成矩阵运算：

$$
z1_1=
\begin{pmatrix}
x_1 & x_2
\end{pmatrix}
\begin{pmatrix}
w1_{1,1} \\\\
w1_{2,1}
\end{pmatrix}
+b1_1
##  通俗地理解三大概念

这三大概念是：反向传播，梯度下降，损失函数。

神经网络训练的最基本的思想就是：先“猜”一个结果，称为预测结果 $a$，看看这个预测结果和事先标记好的训练集中的真实结果 $y$ 之间的差距，然后调整策略，再试一次，这一次就不是“猜”了，而是有依据地向正确的方向靠近。如此反复多次，一直到预测结果和真实结果之间相差无几，亦即 $|a-y|\rightarrow 0$，就结束训练。

在神经网络训练中，我们把“猜”叫做初始化，可以随机，也可以根据以前的经验给定初始值。即使是“猜”，也是有技术含量的。
###  反向传播求解 $b$

#### 求 $b$ 的偏导

这次我们令 $w$ 的值固定为 $3$，变化 $b$ 的值，目标还是让 $z=150$。同上一小节一样，先求 $b$ 的偏导数。

注意，在上一小节中，求 $w$ 的导数只经过了一条路：从 $z$ 到 $x$ 到 $w$。但是求 $b$ 的导数时要经过两条路，如图2-7所示：

1. 从 $z$ 到 $x$ 到 $b$；
2. 从 $z$ 到 $y$ 到 $b$。

<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/2/flow4.png" />

对b的偏导求解过程

从复合导数公式来看，这两者应该是相加的关系，所以有：

$$\frac{\partial{z}}{\partial{b}}=\frac{\partial{z}}{\partial{x}} \cdot \frac{\partial{x}}{\partial{b}}+\frac{\partial{z}}{\partial{y}}\cdot\frac{\partial{y}}{\partial{b}}=y \cdot 3+x \cdot 2=63 \tag{7}$$

其中：

$$\frac{\partial{z}}{\partial{x}}=\frac{\partial{}}{\partial{x}}(x \cdot y)=y=9$$
$$\frac{\partial{z}}{\partial{y}}=\frac{\partial{}}{\partial{y}}(x \cdot y)=x=18$$
$$\frac{\partial{x}}{\partial{b}}=\frac{\partial{}}{\partial{b}}(2w+3b)=3$$
$$\frac{\partial{y}}{\partial{b}}=\frac{\partial{}}{\partial{b}}(2b+1)=2$$

我们不妨再验证一下链式求导的正确性。把公式5再拿过来：

$$z=x \cdot y=(2w+3b)(2b+1)=4wb+2w+6b^2+3b \tag{5}$$

对上式求b的偏导：

$$
\frac{\partial z}{\partial b}=4w+12b+3=12+48+3=63 \tag{8}
$$

##  非线性反向传播

在上面的线性例子中，我们可以发现，误差一次性地传递给了初始值 $w$ 和 $b$，即，只经过一步，直接修改 $w$ 和 $b$ 的值，就能做到误差校正。因为从它的计算图看，无论中间计算过程有多么复杂，它都是线性的，所以可以一次传到底。缺点是这种线性的组合最多只能解决线性问题，不能解决更复杂的问题。这个我们在神经网络基本原理中已经阐述过了，需要有激活函数连接两个线性单元。


<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/2/game.png" ch="500" />
### 梯度下降的三要素

1. 当前点；
2. 方向；
3. 步长。

#### 为什么说是“梯度下降”？

“梯度下降”包含了两层含义：

1. 梯度：函数当前位置的最快上升点；
2. 下降：与导数相反的方向，用数学语言描述就是那个减号。

亦即与上升相反的方向运动，就是下降。

<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/2/gd_concept.png" ch="500" />

梯度下降的步骤

解释了在函数极值点的两侧做梯度下降的计算过程，梯度下降的目的就是使得x值向极值点逼近。


### 损失函数的概念

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

### 损失函数图像理解

#### 用二维函数图像理解单变量对损失函数的影响

<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/3/gd2d.png" />

单变量的损失函数图

#### 用等高线图理解双变量对损失函数影响

<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/3/gd3d.png" />

双变量的损失函数图


###  均方差损失的实际案例

假设有一组数据如图，我们想找到一条拟合的直线。

<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/3/mse1.png" ch="500" />

平面上的样本数据

下图中，前三张显示了一个逐渐找到最佳拟合直线的过程。

- 第一张，用均方差函数计算得到 $Loss=0.53$；
- 第二张，直线向上平移一些，误差计算 $Loss=0.16$，比图一的误差小很多；
- 第三张，又向上平移了一些，误差计算 $Loss=0.048$，此后还可以继续尝试平移（改变 $b$ 值）或者变换角度（改变 $w$ 值），得到更小的损失函数值；
- 第四张，偏离了最佳位置，误差值 $Loss=0.18$，这种情况，算法会让尝试方向反向向下。

<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/3/mse2.png" ch="500" />

 损失函数值与直线位置的关系




### Copyright (c) Microsoft. All rights reserved.
### Licensed under the MIT license. See LICENSE file in the project root for full license information.

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


#### this version has a bug
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

#### this is correct version
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

 <img src="https://i.loli.net/2020/10/05/fsbkij4enmTcJd7.png" />
 <img src="https://i.loli.net/2020/10/05/OX36Ff5uV2wNPd1.png" />

# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for full license information.

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

 <img src="https://i.loli.net/2020/10/05/SdVxHEDzI7vLrK2.png" />

 <img src="https://i.loli.net/2020/10/05/kFIWTQ31VEnYyxt.png" />

 <img src="https://i.loli.net/2020/10/05/Xy3D84mINJ5o7sn.png" />


 # 学习心得
 
人工智能包括了十分广泛的科学，它由不同的领域组成，例如机器学习，计算机视觉等,总的来说，人工智能研究的主要目标是使机器能够胜任一些通常需要人类智能才能完成的复杂工作。人工智能在很多领域得到了发展川在我们的日常生活中发挥了重要的作用。如:机器翻译机器翻译是利用计算机把一种自然语言转换成其他语言的过程。用以完成这一过程的软件 系统叫做机器翻译系统,利用这个系统我们可以很方使的完成一些语言翻译工作。一些大公司在人工智能领域的投入和研究对于推动人工智能的发展起到了很大的作用，最值得一提的就是谷歌。谷歌的免费搜索表面上是为了方便人们的查询，但这款搜索弓|擎推出的初衷就是为了帮助人工智能的深度学习，通过上亿的用户一次又一次地查询，来锻炼人工智能的学习能力。人工智能一定会发展地更远。
