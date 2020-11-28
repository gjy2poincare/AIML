# 《人工智能概论》结课报告

## 基于miniFramework的手写数字识别

代码呈现：
```

from matplotlib import pyplot as plt
import numpy as np
from PIL import Image

from HelperClass2.NeuralNet_3_0 import *

def ReadImage(img_file_name):
    img = Image.open(img_file_name)
    out1 = img.convert('L')
    out2 = out1.resize((28,28))
    a = np.array(out2)
    b = 255 - a
    x_max = np.max(b)
    x_min = np.min(b)
    X_NEW = (b - x_min)/(x_max-x_min)
    plt.cla()
    plt.imshow(X_NEW)
    plt.plot()
    return X_NEW.reshape(1,-1)

def Inference(img_array):
    output = net.inference(img_array)
    n = np.argmax(output)
    print("------recognize result is: -----", n)

def on_key_press(event):
    img_file_name = "handwriting.png"
    print(event.key)
    if event.key == 'enter':
        plt.axis('off')
        plt.savefig(img_file_name)
        plt.axis('on')
        img_array = ReadImage(img_file_name)
        Inference(img_array)
    elif event.key == 'backspace':
        plt.cla()
        plt.axis([0,1,0,1])
        ax.figure.canvas.draw()
    #end if

def on_mouse_press(event):
    global startx, starty, isdraw
    print(isdraw)
    isdraw = True
    startx = event.xdata
    starty = event.ydata
    print("press:{0},{1}", startx, starty)
    
def on_mouse_release(event):
    global isdraw, startx, starty
    print("release:", event.xdata, event.ydata, isdraw)
    isdraw = False

def on_mouse_move(event):
    global isdraw, startx, starty
    if isdraw:
        endx = event.xdata        
        endy = event.ydata        
        x1 = [startx, endx]
        y1 = [starty, endy]
        ax.plot(x1, y1, color='black', linestyle='-', linewidth='40')
        ax.figure.canvas.draw()
        startx = endx
        starty = endy
    # end if

def LoadNet():
    n_input = 784
    n_hidden1 = 64
    n_hidden2 = 16
    n_output = 10
    eta = 0.2
    eps = 0.01
    batch_size = 128
    max_epoch = 40

    hp = HyperParameters_3_0(
        n_input, n_hidden1, n_hidden2, n_output, 
        eta, max_epoch, batch_size, eps, 
        NetType.MultipleClassifier, 
        InitialMethod.Xavier)
    net = NeuralNet_3_0(hp, "MNIST_64_16")
    net.LoadResult()
    return net
   
if __name__ == "__main__":
    isdraw = False
    startx, starty = 0, 0

    print("need to run level3 first to get result")
    print("============================================================================")
    print("handwriting a digit, then press enter to recognize, press backspace to clear")
    print("resize the window to square, say, height == width")
    print("the handwriting should full fill the window")
    print("============================================================================")

    net = LoadNet()

    fig, ax = plt.subplots()
    fig.canvas.mpl_connect('key_press_event', on_key_press)
    fig.canvas.mpl_connect('button_release_event', on_mouse_release)
    fig.canvas.mpl_connect('button_press_event', on_mouse_press)
    fig.canvas.mpl_connect('motion_notify_event', on_mouse_move)
    
    plt.axis([0,1,0,1])
    plt.show()

```

代码结果展示：
![](./images/1.png)
![](./images/2.png)
![](./images/3.png)
![](./images/4.png)

核心代码说明：
```
def LoadNet():
    n_imput = 784
    n_hidden1 = 64 
    n_hidden2 = 16
    n_output = 10
    eta = 0.2
    eps = 0.01
    batch_size = 128
    max-epoch = 40

    hp = HyperParameters_3_0(
        n_input, n_hidden1, n_hidden2, n_output, 
        eta, max_epoch, batch_size, eps, 
        NetType.MultipleClassifier, 
        InitialMethod.Xavier)
    net = NeuralNet_3_0(hp, "MNIST_64_16")
    net.LoadResult()
    return net
```

其中，```HyperParameters```表示超参数，即用于确定模型的一些参数。对于模型而言，不同的超参数意味着不同的模型。

在深度学习中，超参数有：
学习速率
迭代次数
层数
每层神经元的个数
其他

## 前四次作业的内容
# The First-Time Work



Neil 201809023 Form IAE

## 1st AI-EDU Part

### 1.1 对人工智能的定义

现在的人们把人工智能分成了几个层面：


#### 第一个层面 人们的期待

-智能的把某件特定的事情做好，在某个领域增强人类的智慧，这种方式又叫做智能增强。

-像人类一样能认知、思考、判断：模拟人类的智能。

#### 第二个层面 技术特点
机器学习的各种方法是如何从经验中学习的方法我们可以大致地分为下面三种类型：

1. 监督学习（Supervised Learning）

    通过标注的数据来学习，例如，程序通过学习标注了正确答案的手写数字的图像数据，它就能认识其他的手写数字。

2. 无监督学习（Unsupervised Learning）

    通过没有标注的数据来学习。这种算法可以发现数据中自然形成的共同特性（聚类），可以用来发现不同数据之间的联系，例如，买了商品A的顾客往往也购买了商品B。

3. 强化学习（Reinforcement Learning）

    我们可以让程序选择和它的环境互动（例如玩一个游戏），环境给程序的反馈是一些“奖励”（例如游戏中获得高分），程序要学习到一个模型，能在这种环境中得到高的分数，不仅是当前局面要得到高分，而且最终的结果也要是高分才行。

#### 第三个层面 应用的角度
我们也看到了AI取得了达到或超过人类最高水平的成绩：

- 翻译领域（微软的中英翻译超过人类）
- 阅读理解（SQuAD 比赛）
- 下围棋（2016）德州扑克（2019）麻将（2019）

另一种，是AI技术和各种其他技术结合，解决政府，企业，个人用户的需求。在政府方面，把所有计算，数据，云端和物联网终端的设备联系起来，搭建一个能支持智能决定的系统，现代社会的城市管理，金融，医疗，物流和交通管理等等都运行在这样的系统上。专家称之为智能基础建设。

### 范式的演化的不同阶段

#### 第一阶段：经验

从几千年前到几百年前，人们描述自然现象，归纳总结一些规律。

#### 第二阶段：理论

这一阶段，科学家们开始明确定义，速度是什么，质量是什么，化学元素是什么（不再是五行和燃素）……也开始构建各种模型，在模型中尽量撇除次要和无关因素，例如我们在中学的物理实验中，要假设“斜面足够光滑，无摩擦力”，“空气阻力可以忽略不计”，等等。在这个理论演算（Theoretical）阶段，以伽利略为代表的科学家，开启了现代科学之门。他在比萨斜塔做的试验（图1-9）推翻了两千多年来大家想当然的“定律”。

#### 第三阶段：计算仿真

从二十世纪中期开始，利用电子计算机对科学实验进行模拟仿真的模式得到迅速普及，人们可以对复杂现象通过模拟仿真，推演更复杂的现象，典型案例如模拟核试验、天气预报等。这样计算机仿真越来越多地取代实验，逐渐成为科研的常规方法。科学家先定义问题，确认假设，再利用数据进行分析和验证。

#### 第四阶段：数据探索

最后我们到了“数据探索”（Data Exploration）阶段。在这个阶段，科学家收集数据，分析数据，探索新的规律。在深度学习的浪潮中出现的许多结果就是基于海量数据学习得来的。有些数据并不是从现实世界中收集而来，而是由计算机程序自己生成，例如，在AlphaGo算法训练的过程中，它和自己对弈了数百万局，这个数量大大超过了所有记录下来的职业选手棋谱的数量。

### 神经网络的基本工作原理

神经网络由基本的神经元组成，图1-13就是一个神经元的数学/计算模型，便于我们用程序来实现。

<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/1/NeuranCell.png" ch="500" />

图1-13 神经元计算模型

#### 输入 input

$(x_1,x_2,x_3)$ 是外界输入信号，一般是一个训练数据样本的多个属性，比如，我们要预测一套房子的价格，那么在房屋价格数据样本中，$x_1$ 可能代表了面积，$x_2$ 可能代表地理位置，$x_3$ 可能代表朝向。另外一个例子是，$(x_1,x_2,x_3)$ 分别代表了(红,绿,蓝)三种颜色，而此神经元用于识别输入的信号是暖色还是冷色。

#### 权重 weights

$(w_1,w_2,w_3)$ 是每个输入信号的权重值，以上面的 $(x_1,x_2,x_3)$ 的例子来说，$x_1$ 的权重可能是 $0.92$，$x_2$ 的权重可能是 $0.2$，$x_3$ 的权重可能是 $0.03$。当然权重值相加之后可以不是 $1$。

#### 偏移 bias

还有个 $b$ 是怎么来的？一般的书或者博客上会告诉你那是因为 $y=wx+b$，$b$ 是偏移值，使得直线能够沿 $Y$ 轴上下移动。这是用结果来解释原因，并非 $b$ 存在的真实原因。从生物学上解释，在脑神经细胞中，一定是输入信号的电平/电流大于某个临界值时，神经元细胞才会处于兴奋状态，这个 $b$ 实际就是那个临界值。亦即当：

$$w_1 \cdot x_1 + w_2 \cdot x_2 + w_3 \cdot x_3 \geq t$$

时，该神经元细胞才会兴奋。我们把t挪到等式左侧来，变成$(-t)$，然后把它写成 $b$，变成了：

$$w_1 \cdot x_1 + w_2 \cdot x_2 + w_3 \cdot x_3 + b \geq 0$$

于是 $b$ 诞生了！

#### 求和计算 sum

$$
\begin{aligned}
Z &= w_1 \cdot x_1 + w_2 \cdot x_2 + w_3 \cdot x_3 + b \\\\
&= \sum_{i=1}^m(w_i \cdot x_i) + b
\end{aligned}
$$

在上面的例子中 $m=3$。我们把$w_i \cdot x_i$变成矩阵运算的话，就变成了：

$$Z = W \cdot X + b$$

#### 激活函数 activation

求和之后，神经细胞已经处于兴奋状态了，已经决定要向下一个神经元传递信号了，但是要传递多强烈的信号，要由激活函数来确定：

$$A=\sigma{(Z)}$$

如果激活函数是一个阶跃信号的话，会像继电器开合一样咔咔的开启和闭合，在生物体中是不可能有这种装置的，而是一个渐渐变化的过程。所以一般激活函数都是有一个渐变的过程，也就是说是个曲线，如图1-14所示。

<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/1/activation.png" />

图1-14 激活函数图像

至此，一个神经元的工作过程就在电光火石般的一瞬间结束了。

#### 小结

- 一个神经元可以有多个输入。
- 一个神经元只能有一个输出，这个输出可以同时输入给多个神经元。
- 一个神经元的 $w$ 的数量和输入的数量一致。
- 一个神经元只有一个 $b$。
- $w$ 和 $b$ 有人为的初始值，在训练过程中被不断修改。
- $A$ 可以等于 $Z$，即激活函数不是必须有的。
- 一层神经网络中的所有神经元的激活函数必须一致。

#### 神经网络的训练过程

从真正的“零”开始学习神经网络时，我没有看到过任何一个流程图来讲述训练过程，大神们写书或者博客时都忽略了这一点，图1-16是一个简单的流程图。

<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/1/TrainFlow.png" />

图1-16 神经网络训练流程图

#### 前提条件

 1. 首先是我们已经有了训练数据；
 2. 我们已经根据数据的规模、领域，建立了神经网络的基本结构，比如有几层，每一层有几个神经元；
 3. 定义好损失函数来合理地计算误差。

#### 步骤

假设我们有表1-1所示的训练数据样本。

表1-1 训练样本示例

|Id|$x_1$|$x_2$|$x_3$|$Y$|
|---|---|---|---|---|
|1|0.5|1.4|2.7|3|
|2|0.4|1.3|2.5|5|
|3|0.1|1.5|2.3|9|
|4|0.5|1.7|2.9|1|

其中，$x_1,x_2,x_3$ 是每一个样本数据的三个特征值，$Y$ 是样本的真实结果值：

1. 随机初始化权重矩阵，可以根据正态分布等来初始化。这一步可以叫做“猜”，但不是瞎猜；
2. 拿一个或一批数据作为输入，带入权重矩阵中计算，再通过激活函数传入下一层，最终得到预测值。在本例中，我们先用Id-1的数据输入到矩阵中，得到一个 $A$ 值，假设 $A=5$；
3. 拿到Id-1样本的真实值 $Y=3$；
4. 计算损失，假设用均方差函数 $Loss = (A-Y)^2=(5-3)^2=4$；
5. 根据一些神奇的数学公式（反向微分），把 $Loss=4$ 这个值用大喇叭喊话，告诉在前面计算的步骤中，影响 $A=5$ 这个值的每一个权重矩阵，然后对这些权重矩阵中的值做一个微小的修改（当然是向着好的方向修改，这一点可以用数学家的名誉来保证）；
6. 用Id-2样本作为输入再次训练（Go to 2）；
7. 这样不断地迭代下去，直到以下一个或几个条件满足就停止训练：损失函数值非常小；准确度满足了要求；迭代到了指定的次数。

训练完成后，我们会把这个神经网络中的结构和权重矩阵的值导出来，形成一个计算图（就是矩阵运算加上激活函数）模型，然后嵌入到任何可以识别/调用这个模型的应用程序中，根据输入的值进行运算，输出预测值。

### 1.3.3 神经网络中的矩阵运算

图1-17是一个两层的神经网络，包含隐藏层和输出层，输入层不算做一层。

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
$$

$$
z1_2=
\begin{pmatrix}
x_1 & x_2
\end{pmatrix}
\begin{pmatrix}
w1_{1,2} \\\\
w1_{2,2}
\end{pmatrix}
+b1_2
$$

$$
z1_3=
\begin{pmatrix}
x_1 & x_2
\end{pmatrix}
\begin{pmatrix}
w1_{1,3} \\\\
w1_{2,3}
\end{pmatrix}
+b1_3
$$

再变成大矩阵：

$$
Z1 =
\begin{pmatrix}
x_1 & x_2 
\end{pmatrix}
\begin{pmatrix}
w1_{1,1}&w1_{1,2}&w1_{1,3} \\\\
w1_{2,1}&w1_{2,2}&w1_{2,3} \\\\
\end{pmatrix}
+\begin{pmatrix}
b1_1 & b1_2 & b1_3
\end{pmatrix}
$$

最后变成矩阵符号：

$$Z1 = X \cdot W1 + B1$$

然后是激活函数运算：

$$A1=a(Z1)$$

同理可得：

$$Z2 = A1 \cdot W2 + B2$$

注意：损失函数不是前向计算的一部分。


### 神经网络中的三个基本概念

#### 反向传播

##### 正向过程

1. 第1个人，输入层，随机输入第一个 $x$ 值，$x$ 的取值范围 $(1,10]$，假设第一个数是 $2$；
2. 第2个人，第一层网络计算，接收第1个人传入 $x$ 的值，计算：$a=x^2$；
3. 第3个人，第二层网络计算，接收第2个人传入 $a$ 的值，计算：$b=\ln (a)$；
4. 第4个人，第三层网络计算，接收第3个人传入 $b$ 的值，计算：$c=\sqrt{b}$；
5. 第5个人，输出层，接收第4个人传入 $c$ 的值

##### 反向过程

6. 第5个人，计算 $y$ 与 $c$ 的差值：$\Delta c = c - y$，传回给第4个人
7. 第4个人，接收第5个人传回$\Delta c$，计算 $\Delta b = \Delta c \cdot 2\sqrt{b}$
8. 第3个人，接收第4个人传回$\Delta b$，计算 $\Delta a = \Delta b \cdot a$
9. 第2个人，接收第3个人传回$\Delta a$，计算 $\Delta x = \frac{\Delta}{2x}$
10. 第1个人，接收第2个人传回$\Delta x$，更新 $x \leftarrow x - \Delta x$，回到第1步

##### 求 $w$ 的偏导

目前 $z=162$，如果想让 $z$ 变小一些，比如目标是 $z=150$，$w$ 应该如何变化呢？为了简化问题，先只考虑改变 $w$ 的值，而令 $b$ 值固定为 $4$。

如果想解决这个问题，最笨的办法是可以在输入端一点一点的试，把 $w$ 变成 $3.5$ 试试，再变成 $3$ 试试......直到满意为止。现在我们将要学习一个更好的解决办法：反向传播。

从 $z$ 开始一层一层向回看，图中各节点关于变量 $w$ 的偏导计算结果如下：

因为 $$z = x \cdot y$$，其中 $$x = 2w + 3b, y = 2b + 1$$

所以：

$$\frac{\partial{z}}{\partial{w}}=\frac{\partial{z}}{\partial{x}} \cdot \frac{\partial{x}}{\partial{w}}=y \cdot 2=18 \tag{4}$$

其中：

$$\frac{\partial{z}}{\partial{x}}=\frac{\partial{}}{\partial{x}}(x \cdot y)=y=9$$

$$\frac{\partial{x}}{\partial{w}}=\frac{\partial{}}{\partial{w}}(2w+3b)=2$$

<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/2/flow3.png" />

图2-6 对 $w$ 的偏导求解过程

图2-6其实就是链式法则的具体表现，$z$ 的误差通过中间的 $x$ 传递到 $w$。如果不是用链式法则，而是直接用 $z$ 的表达式计算对 $w$ 的偏导数，会怎么样呢？我们来试验一下。

根据公式1、2、3，我们有：

$$z=x \cdot y=(2w+3b)(2b+1)=4wb+2w+6b^2+3b \tag{5}$$

对上式求 $w$ 的偏导：

$$
\frac{\partial z}{\partial w}=4b+2=4 \cdot 4 + 2=18 \tag{6}
$$

公式4和公式6的结果完全一致！所以，请大家相信链式法则的科学性。

##### 求 $w$ 的具体变化值

公式4和公式6的含义是：当 $w$ 变化一点点时，$z$ 会产生 $w$ 的变化值18倍的变化。记住我们的目标是让 $z=150$，目前在初始状态时是 $z=162$，所以，问题转化为：当需要 $z$ 从 $162$ 变到 $150$ 时，$w$ 需要变化多少？

既然：

$$
\Delta z = 18 \cdot \Delta w
$$

则：

$$
\Delta w = {\Delta z \over 18}=\frac{162-150}{18}= 0.6667
$$

所以：

$$w = w - 0.6667=2.3333$$
$$x=2w+3b=16.6667$$
$$z=x \cdot y=16.6667 \times 9=150.0003$$

我们一下子就成功地让 $z$ 值变成了 $150.0003$，与 $150$ 的目标非常地接近，这就是偏导数的威力所在。


#### 梯度下降

用自然界的例子理解梯度下降：

1. 水受重力影响，会在当前位置，沿着最陡峭的方向流动，有时
   会形成瀑布（梯度下降）；
2. 水流下山的路径不是唯一的，在同一个地点，有可能有多个位
   置具有同样的陡峭程度，而造成了分流（可以得到多个解）；
3. 遇到坑洼地区，有可能形成湖泊，而终止下山过程（不能得到
   全局最优解，而是局部最优解）。

梯度下降的三要素：
> 1、当前点；

> 2、方向；

> 3、步长； 

梯度下降包含两层含义：
> 1、梯度：函数当前位置的最快上升点；

> 2、下降：与导数相反的方向，用数学语言描述就是那个减号；

### 2.3.3 单变量函数的梯度下降

假设一个单变量函数：

$$J(x) = x ^2$$

我们的目的是找到该函数的最小值，于是计算其微分：

$$J'(x) = 2x$$

假设初始位置为：

$$x_0=1.2$$

假设学习率：

$$\eta = 0.3$$

根据公式(1)，迭代公式：

$$x_{n+1} = x_{n} - \eta \cdot \nabla J(x)= x_{n} - \eta \cdot 2x$$

假设终止条件为 $J(x)<0.01$，迭代过程是：
```
x=0.480000, y=0.230400
x=0.192000, y=0.036864
x=0.076800, y=0.005898
x=0.030720, y=0.000944
```

上面的过程如图2-10所示。

<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/2/gd_single_variable.png" ch="500" />

图2-10 使用梯度下降法迭代的过程

### 2.3.4 双变量的梯度下降

假设一个双变量函数：

$$J(x,y) = x^2 + \sin^2(y)$$

我们的目的是找到该函数的最小值，于是计算其微分：

$${\partial{J(x,y)} \over \partial{x}} = 2x$$
$${\partial{J(x,y)} \over \partial{y}} = 2 \sin y \cos y$$

假设初始位置为：

$$(x_0,y_0)=(3,1)$$

假设学习率：

$$\eta = 0.1$$

根据公式(1)，迭代过程是的计算公式：
$$(x_{n+1},y_{n+1}) = (x_n,y_n) - \eta \cdot \nabla J(x,y)$$
$$ = (x_n,y_n) - \eta \cdot (2x,2 \cdot \sin y \cdot \cos y) \tag{1}$$

根据公式(1)，假设终止条件为 $J(x,y)<0.01$，迭代过程如表2-3所示。

表2-3 双变量梯度下降的迭代过程

|迭代次数|x|y|J(x,y)|
|---|---|---|---|
|1|3|1|9.708073|
|2|2.4|0.909070|6.382415|
|...|...|...|...|
|15|0.105553|0.063481|0.015166|
|16|0.084442|0.050819|0.009711|

迭代16次后，$J(x,y)$ 的值为 $0.009711$，满足小于 $0.01$ 的条件，停止迭代。

上面的过程如表2-4所示，由于是双变量，所以需要用三维图来解释。请注意看两张图中间那条隐隐的黑色线，表示梯度下降的过程，从红色的高地一直沿着坡度向下走，直到蓝色的洼地。

表2-4 在三维空间内的梯度下降过程

|观察角度1|观察角度2|
|--|--|
|<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images\Images\2\gd_double_variable.png">|<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images\Images\2\gd_double_variable2.png">|

### 2.3.5 学习率η的选择

在公式表达时，学习率被表示为$\eta$。在代码里，我们把学习率定义为`learning_rate`，或者`eta`。针对上面的例子，试验不同的学习率对迭代情况的影响，如表2-5所示。

表2-5 不同学习率对迭代情况的影响

|学习率|迭代路线图|说明|
|---|---|---|
|1.0|<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/2/gd100.png" width="500" height="150"/>|学习率太大，迭代的情况很糟糕，在一条水平线上跳来跳去，永远也不能下降。|
|0.8|<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/2/gd080.png" width="400"/>|学习率大，会有这种左右跳跃的情况发生，这不利于神经网络的训练。|
|0.4|<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/2/gd040.png" width="400"/>|学习率合适，损失值会从单侧下降，4步以后基本接近了理想值。|
|0.1|<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/2/gd010.png" width="400"/>|学习率较小，损失值会从单侧下降，但下降速度非常慢，10步了还没有到达理想状态。

#### 损失函数

> 损失函数的概念：在各种材料中经常看到的中英文词汇有：误差，偏差，Error，Cost，Loss，损失，代价......意思都差不多，在本书中，使用“损失函数”和“Loss Function”这两个词汇，具体的损失函数符号用 $J$ 来表示，误差值用 $loss$ 表示。
> “损失”就是所有样本的“误差”的总和，亦即（$m$ 为样本数）：
> $$损失 = \sum^m_{i=1}误差_i$$
> $$J = \sum_{i=1}^m loss_i$$

>损失函数的作用：损失函数的作用，就是计算神经网络每次迭代的前向计算结果与真实值的差距，从而指导下一步的训练向正确的方向进行。

使用损失函数的步骤：

> 1. 用随机值初始化前向计算公式的参数；

> 2. 代入样本，计算输出的预测值；

> 3. 用损失函数计算预测值和标签值（真实值）的误差；

> 4. 根据损失函数的导数，沿梯度最小方向将误差回传，修正前向计算公式中的各个权重值；

> 5. 进入第2步重复, 直到损失函数值达到一个满意的值就停止迭代。

##### 均方差函数

该函数就是最直观的一个损失函数了，计算预测值和真实值之间的欧式距离。预测值和真实值越接近，两者的均方差就越小。

均方差函数常用于线性回归(linear regression)，即函数拟合(function fitting)。公式如下：

$$
loss = {1 \over 2}(z-y)^2 \tag{单样本}
$$

$$
J=\frac{1}{2m} \sum_{i=1}^m (z_i-y_i)^2 \tag{多样本}
$$

均方差函数的工作原理：

要想得到预测值 $a$ 与真实值 $y$ 的差距，最朴素的想法就是用 $Error=a_i-y_i$。

对于单个样本来说，这样做没问题，但是多个样本累计时，$a_i-y_i$ 可能有正有负，误差求和时就会导致相互抵消，从而失去价值。所以有了绝对值差的想法，即 $Error=|a_i-y_i|$ 。这看上去很简单，并且也很理想，那为什么还要引入均方差损失函数呢？两种损失函数的比较如表3-1所示。

表3-1 绝对值损失函数与均方差损失函数的比较

|样本标签值|样本预测值|绝对值损失函数|均方差损失函数|
|------|------|------|------|
|$[1,1,1]$|$[1,2,3]$|$(1-1)+(2-1)+(3-1)=3$|$(1-1)^2+(2-1)^2+(3-1)^2=5$|
|$[1,1,1]$|$[1,3,3]$|$(1-1)+(3-1)+(3-1)=4$|$(1-1)^2+(3-1)^2+(3-1)^2=8$|
|||$4/3=1.33$|$8/5=1.6$|

可以看到5比3已经大了很多，8比4大了一倍，而8比5也放大了某个样本的局部损失对全局带来的影响，用术语说，就是“对某些偏离大的样本比较敏感”，从而引起监督训练过程的足够重视，以便回传误差。

##### 交叉熵损失函数

交叉熵（Cross Entropy）是Shannon信息论中一个重要概念，主要用于度量两个概率分布间的差异性信息。在信息论中，交叉熵是表示两个概率分布 $p,q$ 的差异，其中 $p$ 表示真实分布，$q$ 表示预测分布，那么 $H(p,q)$ 就称为交叉熵：

$$H(p,q)=\sum_i p_i \cdot \ln {1 \over q_i} = - \sum_i p_i \ln q_i \tag{1}$$

交叉熵可在神经网络中作为损失函数，$p$ 表示真实标记的分布，$q$ 则为训练后的模型的预测标记分布，交叉熵损失函数可以衡量 $p$ 与 $q$ 的相似性。

**交叉熵函数常用于逻辑回归(logistic regression)，也就是分类(classification)。**

###### 信息量

信息论中，信息量的表示方式：

$$I(x_j) = -\ln (p(x_j)) \tag{2}$$

$x_j$：表示一个事件

$p(x_j)$：表示 $x_j$ 发生的概率

$I(x_j)$：信息量，$x_j$ 越不可能发生时，它一旦发生后的信息量就越大

假设对于学习神经网络原理课程，我们有三种可能的情况发生，如表3-2所示。

表3-2 三种事件的概论和信息量

|事件编号|事件|概率 $p$|信息量 $I$|
|---|---|---|---|
|$x_1$|优秀|$p=0.7$|$I=-\ln(0.7)=0.36$|
|$x_2$|及格|$p=0.2$|$I=-\ln(0.2)=1.61$|
|$x_3$|不及格|$p=0.1$|$I=-\ln(0.1)=2.30$|

## Code Part

### Level1_BP_Linear.py

> 代码运行截图
![](./images/01.png)

### Level2_BP_NoneLinear>py

> 代码运行截图
![](./images/02.png)
> 代码运行截图
![](./images/11.png)
> 代码运行截图
![](./images/12.png)



### Level3_GDSingleVariable.py

> 代码运行截图
![](./images/03.png)

### Level4_GDDoubleVariable.py

> 代码运行截图
![](./images/13.png)


### Level5_LearningRate.py

> 代码运行截图
![](./images/04.png)

> 代码运行截图
![](./images/05.png)

> 代码运行截图
![](./images/06.png)

> 代码运行截图
![](./images/07.png)

> 代码运行截图
![](./images/08.png)

> 代码运行截图
![](./images/09.png)

> 代码运行截图
![](./images/10.png)


### 学习心得
> Pyhton是一门很重要的语言，恰当的使用numpy以及其他功能是必须的。

# Step 2 线性回归

用线性回归作为学习神经网络的起点，是一个非常好的选择，因为线性回归问题本身比较容易理解，在它的基础上，逐步的增加一些新的知识点，会形成一条比较平缓的学习曲线，或者说是迈向神经网络的第一个小台阶。

## 单入单出的单层神经网路 - 单变量线性回归

### 4.0.2 一元线性回归模型

回归分析是一种数学模型。当因变量和自变量为线性关系时，它是一种特殊的线性模型。

最简单的情形是一元线性回归，由大体上有线性关系的一个自变量和一个因变量组成，模型是：

$$Y=a+bX+\varepsilon \tag{1}$$

$X$ 是自变量，$Y$ 是因变量，$\varepsilon$ 是随机误差，$a$ 和 $b$ 是参数，在线性回归模型中，$a,b$ 是我们要通过算法学习出来的。

什么叫模型？第一次接触这个概念时，可能会有些不明觉厉。从常规概念上讲，是人们通过主观意识借助实体或者虚拟表现来构成对客观事物的描述，这种描述通常是有一定的逻辑或者数学含义的抽象表达方式。

比如对小轿车建模的话，会是这样描述：由发动机驱动的四轮铁壳子。对能量概念建模的话，那就是爱因斯坦狭义相对论的著名推论：$E=mc^2$。

对数据建模的话，就是想办法用一个或几个公式来描述这些数据的产生条件或者相互关系，比如有一组数据是大致满足 $y=3x+2$ 这个公式的，那么这个公式就是模型。为什么说是“大致”呢？因为在现实世界中，一般都有噪音（误差）存在，所以不可能非常准确地满足这个公式，只要是在这条直线两侧附近，就可以算作是满足条件。

对于线性回归模型，有如下一些概念需要了解：

- 通常假定随机误差 $\varepsilon$ 的均值为 $0$，方差为$σ^2$（$σ^2>0$，$σ^2$ 与 $X$ 的值无关）
- 若进一步假定随机误差遵从正态分布，就叫做正态线性模型
- 一般地，若有 $k$ 个自变量和 $1$ 个因变量（即公式1中的 $Y$），则因变量的值分为两部分：一部分由自变量影响，即表示为它的函数，函数形式已知且含有未知参数；另一部分由其他的未考虑因素和随机性影响，即随机误差
- 当函数为参数未知的线性函数时，称为线性回归分析模型
- 当函数为参数未知的非线性函数时，称为非线性回归分析模型
- 当自变量个数大于 $1$ 时称为多元回归
- 当因变量个数大于 $1$ 时称为多重回归

我们通过对数据的观察，可以大致认为它符合线性回归模型的条件，于是列出了公式1，不考虑随机误差的话，我们的任务就是找到合适的 $a,b$，这就是线性回归的任务。

<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/4/regression.png" />

图4-2 线性回归和非线性回归的区别

如图4-2所示，左侧为线性模型，可以看到直线穿过了一组三角形所形成的区域的中心线，并不要求这条直线穿过每一个三角形。右侧为非线性模型，一条曲线穿过了一组矩形所形成的区域的中心线。

### 解决办法

1. 最小二乘法；
2. 梯度下降法；
3. 简单的神经网络法；
4. 更通用的神经网络算法。

#### 最小二乘法

数学原理：

线性回归试图学得：

$$z_i=w \cdot x_i+b \tag{1}$$

使得：

$$z_i \simeq y_i \tag{2}$$

其中，$x_i$ 是样本特征值，$y_i$ 是样本标签值，$z_i$ 是模型预测值。

如何学得 $w$ 和 $b$ 呢？均方差(MSE - mean squared error)是回归任务中常用的手段：
$$
J = \frac{1}{2m}\sum_{i=1}^m(z_i-y_i)^2 = \frac{1}{2m}\sum_{i=1}^m(y_i-wx_i-b)^2 \tag{3}
$$

$J$ 称为损失函数。实际上就是试图找到一条直线，使所有样本到直线上的残差的平方和最小。

<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/4/mse.png" />

图4-3 均方差函数的评估原理

图4-3中，圆形点是样本点，直线是当前的拟合结果。如左图所示，我们是要计算样本点到直线的垂直距离，需要再根据直线的斜率来求垂足然后再计算距离，这样计算起来很慢；但实际上，在工程上我们通常使用的是右图的方式，即样本点到直线的竖直距离，因为这样计算很方便，用一个减法就可以了。

假设我们计算出初步的结果是虚线所示，这条直线是否合适呢？我们来计算一下图中每个点到这条直线的距离，把这些距离的值都加起来（都是正数，不存在互相抵消的问题）成为误差。

因为上图中的几个点不在一条直线上，所以不能有一条直线能同时穿过它们。所以，我们只能想办法不断改变红色直线的角度和位置，让总体误差最小（永远不可能是 $0$），就意味着整体偏差最小，那么最终的那条直线就是我们要的结果。

如果想让误差的值最小，通过对 $w$ 和 $b$ 求导，再令导数为 $0$（到达最小极值），就是 $w$ 和 $b$ 的最优解。

推导过程如下：

$$
\begin{aligned}
\frac{\partial{J}}{\partial{w}} &=\frac{\partial{(\frac{1}{2m}\sum_{i=1}^m(y_i-wx_i-b)^2)}}{\partial{w}} \\\\
&= \frac{1}{m}\sum_{i=1}^m(y_i-wx_i-b)(-x_i) 
\end{aligned}
\tag{4}
$$

令公式4为 $0$：

$$
\sum_{i=1}^m(y_i-wx_i-b)x_i=0 \tag{5}
$$

$$
\begin{aligned}
\frac{\partial{J}}{\partial{b}} &=\frac{\partial{(\frac{1}{2m}\sum_{i=1}^m(y_i-wx_i-b)^2)}}{\partial{b}} \\\\
&=\frac{1}{m}\sum_{i=1}^m(y_i-wx_i-b)(-1) 
\end{aligned}
\tag{6}
$$

令公式6为 $0$：

$$
\sum_{i=1}^m(y_i-wx_i-b)=0 \tag{7}
$$

由式7得到（假设有 $m$ 个样本）：

$$
\sum_{i=1}^m b = m \cdot b = \sum_{i=1}^m{y_i} - w\sum_{i=1}^m{x_i} \tag{8}
$$

两边除以 $m$：

$$
b = \frac{1}{m}\left(\sum_{i=1}^m{y_i} - w\sum_{i=1}^m{x_i}\right)=\bar y-w \bar x \tag{9}
$$

其中：

$$
\bar y = \frac{1}{m}\sum_{i=1}^m y_i, \bar x=\frac{1}{m}\sum_{i=1}^m x_i \tag{10}
$$

将公式10代入公式5：

$$
\sum_{i=1}^m(y_i-wx_i-\bar y + w \bar x)x_i=0
$$

$$
\sum_{i=1}^m(x_i y_i-wx^2_i-x_i \bar y + w \bar x x_i)=0
$$

$$
\sum_{i=1}^m(x_iy_i-x_i \bar y)-w\sum_{i=1}^m(x^2_i - \bar x x_i) = 0
$$

$$
w = \frac{\sum_{i=1}^m(x_iy_i-x_i \bar y)}{\sum_{i=1}^m(x^2_i - \bar x x_i)} \tag{11}
$$

将公式10代入公式11：

$$
w = \frac{\sum_{i=1}^m (x_i \cdot y_i) - \sum_{i=1}^m x_i \cdot \frac{1}{m} \sum_{i=1}^m y_i}{\sum_{i=1}^m x^2_i - \sum_{i=1}^m x_i \cdot \frac{1}{m}\sum_{i=1}^m x_i} \tag{12}
$$

分子分母都乘以 $m$：

$$
w = \frac{m\sum_{i=1}^m x_i y_i - \sum_{i=1}^m x_i \sum_{i=1}^m y_i}{m\sum_{i=1}^m x^2_i - (\sum_{i=1}^m x_i)^2} \tag{13}
$$

$$
b= \frac{1}{m} \sum_{i=1}^m(y_i-wx_i) \tag{14}
$$

而事实上，式13有很多个变种，大家会在不同的文章里看到不同版本，往往感到困惑，比如下面两个公式也是正确的解：

$$
w = \frac{\sum_{i=1}^m y_i(x_i-\bar x)}{\sum_{i=1}^m x^2_i - (\sum_{i=1}^m x_i)^2/m} \tag{15}
$$

$$
w = \frac{\sum_{i=1}^m x_i(y_i-\bar y)}{\sum_{i=1}^m x^2_i - \bar x \sum_{i=1}^m x_i} \tag{16}
$$

以上两个公式，如果把公式10代入，也应该可以得到和式13相同的答案，只不过需要一些运算技巧。比如，很多人不知道这个神奇的公式：

$$
\begin{aligned}
\sum_{i=1}^m (x_i \bar y) &= \bar y \sum_{i=1}^m x_i =\frac{1}{m}(\sum_{i=1}^m y_i) (\sum_{i=1}^m x_i) \\\\
&=\frac{1}{m}(\sum_{i=1}^m x_i) (\sum_{i=1}^m y_i)= \bar x \sum_{i=1}^m y_i \\\\
&=\sum_{i=1}^m (y_i \bar x) 
\end{aligned}
\tag{17}
$$

#### 梯度下降法

数学原理：
在下面的公式中，我们规定 $x$ 是样本特征值（单特征），$y$ 是样本标签值，$z$ 是预测值，下标 $i$ 表示其中一个样本。

#### 预设函数（Hypothesis Function）

线性函数：

$$z_i = x_i \cdot w + b \tag{1}$$

#### 损失函数（Loss Function）

均方误差：

$$loss_i(w,b) = \frac{1}{2} (z_i-y_i)^2 \tag{2}$$


与最小二乘法比较可以看到，梯度下降法和最小二乘法的模型及损失函数是相同的，都是一个线性模型加均方差损失函数，模型用于拟合，损失函数用于评估效果。

区别在于，最小二乘法从损失函数求导，直接求得数学解析解，而梯度下降以及后面的神经网络，都是利用导数传递误差，再通过迭代方式一步一步（用近似解）逼近真实解。

### 神经网络法

#### 4.3.1 定义神经网络结构

我们是首次尝试建立神经网络，先用一个最简单的单层单点神经元，如图4-4所示。

<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/4/Setup.png" ch="500" />

图4-4 单层单点神经元

下面，我们用这个最简单的线性回归的例子，来说明神经网络中最重要的反向传播和梯度下降的概念、过程以及代码实现。

##### 输入层

此神经元在输入层只接受一个输入特征，经过参数 $w,b$ 的计算后，直接输出结果。这样一个简单的“网络”，只能解决简单的一元线性回归问题，而且由于是线性的，我们不需要定义激活函数，这就大大简化了程序，而且便于大家循序渐进地理解各种知识点。

严格来说输入层在神经网络中并不能称为一个层。

##### 权重 $w,b$

因为是一元线性问题，所以 $w,b$ 都是标量。

##### 输出层

输出层 $1$ 个神经元，线性预测公式是：

$$z_i = x_i \cdot w + b$$

$z$ 是模型的预测输出，$y$ 是实际的样本标签值，下标 $i$ 为样本。

##### 损失函数

因为是线性回归问题，所以损失函数使用均方差函数。

$$loss(w,b) = \frac{1}{2} (z_i-y_i)^2$$

#### 4.3.2 反向传播

由于我们使用了和上一节中的梯度下降法同样的数学原理，所以反向传播的算法也是一样的，细节请查看4.2.2。

##### 计算 $w$ 的梯度

$$
{\partial{loss} \over \partial{w}} = \frac{\partial{loss}}{\partial{z_i}}\frac{\partial{z_i}}{\partial{w}}=(z_i-y_i)x_i
$$

##### 计算 $b$ 的梯度

$$
\frac{\partial{loss}}{\partial{b}} = \frac{\partial{loss}}{\partial{z_i}}\frac{\partial{z_i}}{\partial{b}}=z_i-y_i
$$

为了简化问题，在本小节中，反向传播使用单样本方式，在下一小节中，我们将介绍多样本方式。


### 多样本计算

单样本计算的缺点：
1. 前后两个相邻的样本很有可能会对反向传播产生相反的作用而互相抵消。假设样本1造成了误差为 $0.5$，$w$ 的梯度计算结果是 $0.1$；紧接着样本2造成的误差为 $-0.5$，$w$ 的梯度计算结果是 $-0.1$，那么前后两次更新 $w$ 就会产生互相抵消的作用。
   
2. 在样本数据量大时，逐个计算会花费很长的时间。由于我们在本例中样本量不大（200个样本），所以计算速度很快，觉察不到这一点。在实际的工程实践中，动辄10万甚至100万的数据量，轮询一次要花费很长的时间。

> 用到矩阵运算单多样本计算能节省时间，大幅提升速度。

### 梯度下降的三种形式

#### 4.5.1 单样本随机梯度下降

SGD(Stochastic Gradient Descent)

样本访问示意图如图4-7所示。
  
<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/4/SingleSample-example.png" />

图4-7 单样本访问方式

#### 特点
  
  - 训练样本：每次使用一个样本数据进行一次训练，更新一次梯度，重复以上过程。
  - 优点：训练开始时损失值下降很快，随机性大，找到最优解的可能性大。
  - 缺点：受单个样本的影响最大，损失函数值波动大，到后期徘徊不前，在最优解附近震荡。不能并行计算。

#### 4.5.2 小批量样本梯度下降

Mini-Batch Gradient Descent

样本访问示意图如图4-8所示。

<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/4/MiniBatch-example.png" />

图4-8 小批量样本访问方式

#### 特点

  - 训练样本：选择一小部分样本进行训练，更新一次梯度，然后再选取另外一小部分样本进行训练，再更新一次梯度。
  - 优点：不受单样本噪声影响，训练速度较快。
  - 缺点：batch size的数值选择很关键，会影响训练结果。

#### 4.5.3 全批量样本梯度下降 

Full Batch Gradient Descent

样本访问示意图如图4-9所示。

<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/4/FullBatch-example.png" />

图4-9  全批量样本访问方式

#### 特点

  - 训练样本：每次使用全部数据集进行一次训练，更新一次梯度，重复以上过程。
  - 优点：受单个样本的影响最小，一次计算全体样本速度快，损失函数值没有波动，到达最优点平稳。方便并行计算。
  - 缺点：数据量较大时不能实现（内存限制），训练过程变慢。初始值不同，可能导致获得局部最优解，并非全局最优解。

#### 4.5.4 三种方式的比较

表4-6 三种方式的比较

||单样本|小批量|全批量|
|---|---|---|---|
|梯度下降过程图解|<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/4/SingleSample-Trace.png"/>|<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/4/MiniBatch-Trace.png"/>|<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/4/FullBatch-Trace.png"/>|
|批大小|1|10|100|
|学习率|0.1|0.3|0.5|
|迭代次数|304|110|60|
|epoch|3|10|60|
|结果|w=2.003, b=2.990|w=2.006, b=2.997|w=1.993, b=2.998|

表4-6比较了三种方式的结果，从结果看，都接近于 $w=2,b=3$ 的原始解。最后的可视化结果图如图4-10，可以看到直线已经处于样本点比较中间的位置。

<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/4/mbgd-result.png" ch="500" />

图4-10 较理想的拟合效果图

### 多入单出单层 - 多变量线性回归

表5-3 两种方法的比较

|方法|正规方程|梯度下降|
|---|-----|-----|
|原理|几次矩阵运算|多次迭代|
|特殊要求|$X^{\top}X$ 的逆矩阵存在|需要确定学习率|
|复杂度|$O(n^3)$|$O(n^2)$|
|适用样本数|$m \lt 10000$|$m \ge 10000$|

#### 正规方程解法

英文名是 Normal Equations。

对于线性回归问题，除了前面提到的最小二乘法可以解决一元线性回归的问题外，也可以解决多元线性回归问题。

对于多元线性回归，可以用正规方程来解决，也就是得到一个数学上的解析解。它可以解决下面这个公式描述的问题：

$$y=a_0+a_1x_1+a_2x_2+\dots+a_kx_k \tag{1}$$

#### 神经网络法

与单特征值的线性回归问题类似，多变量（多特征值）的线性回归可以被看做是一种高维空间的线性拟合。以具有两个特征的情况为例，这种线性拟合不再是用直线去拟合点，而是用平面去拟合点。

### 样本特征数据标准化

理论层面上，神经网络是以样本在事件中的统计分布概率为基础进行训练和预测的，所以它对样本数据的要求比较苛刻。具体说明如下：

1. 样本的各个特征的取值要符合概率分布，即 $[0,1]$。
2. 样本的度量单位要相同。我们并没有办法去比较1米和1公斤的区别，但是，如果我们知道了1米在整个样本中的大小比例，以及1公斤在整个样本中的大小比例，比如一个处于0.2的比例位置，另一个处于0.3的比例位置，就可以说这个样本的1米比1公斤要小。
3. 神经网络假设所有的输入输出数据都是标准差为1，均值为0，包括权重值的初始化，激活函数的选择，以及优化算法的设计。

4. 数值问题

    标准化可以避免一些不必要的数值问题。因为激活函数sigmoid/tanh的非线性区间大约在 $[-1.7，1.7]$。意味着要使神经元有效，线性计算输出的值的数量级应该在1（1.7所在的数量级）左右。这时如果输入较大，就意味着权值必须较小，一个较大，一个较小，两者相乘，就引起数值问题了。
    
5. 梯度更新
    
    若果输出层的数量级很大，会引起损失函数的数量级很大，这样做反向传播时的梯度也就很大，这时会给梯度的更新带来数值问题。
    
6. 学习率
   
    如果梯度非常大，学习率就必须非常小，因此，学习率（学习率初始值）的选择需要参考输入的范围，不如直接将数据标准化，这样学习率就不必再根据数据范围作调整。对 $w_1$ 适合的学习率，可能相对于 $w_2$ 来说会太小，若果使用适合 $w_1$ 的学习率，会导致在 $w_2$ 方向上步进非常慢，从而消耗非常多的时间；而使用适合 $w_2$ 的学习率，对 $w_1$ 来说又太大，搜索不到适合 $w_1$ 的解。

### 正确的推理预测方法

我们在针对训练数据做标准化时，得到的最重要的数据是训练数据的最小值和最大值，我们只需要把这两个值记录下来，在预测时使用它们对预测数据做标准化，这就相当于把预测数据“混入”训练数据。前提是预测数据的特征值不能超出训练数据的特征值范围，否则有可能影响准确程度。

# Step3 线性分类

## 多入多出的单层神经网络
回归问题可以分为两类：线性回归和逻辑回归。在第二步中，我们学习了线性回归模型，在第三步中，我们将一起学习逻辑回归模型。

逻辑回归（Logistic Regression），回归给出的结果是事件成功或失败的概率。当因变量的类型属于二值（1/0，真/假，是/否）变量时，我们就应该使用逻辑回归。

线性回归使用一条直线拟合样本数据，而逻辑回归的目标是“拟合”0或1两个数值，而不是具体连续数值，所以称为广义线性模型。逻辑回归又称Logistic回归分析，常用于数据挖掘，疾病自动诊断，经济预测等领域。

例如，探讨引发疾病的危险因素，并根据危险因素预测疾病发生的概率等。以胃癌病情分析为例，选择两组人群，一组是胃癌组，一组是非胃癌组，两组人群必定具有不同的体征与生活方式等。因此因变量就为是否胃癌，值为“是”或“否”；自变量就可以包括很多了，如年龄、性别、饮食习惯、幽门螺杆菌感染等。

自变量既可以是连续的，也可以是分类的。然后通过Logistic回归分析，可以得到自变量的权重，从而可以大致了解到底哪些因素是胃癌的危险因素。同时根据该权值可以根据危险因素预测一个人患癌症的可能性。

逻辑回归的另外一个名字叫做分类器，分为线性分类器和非线性分类器，本章中我们学习线性分类器。而无论是线性还是非线性分类器，又分为两种：二分类问题和多分类问题，在本章中我们学习二分类问题。线性多分类问题将会在下一章讲述，非线性分类问题在后续的步骤中讲述。

综上所述，我们本章要学习的路径是：回归问题->逻辑回归问题->线性逻辑回归即分类问题->线性二分类问题。

表6-2示意说明了线性二分类和非线性二分类的区别。

表6-2 直观理解线性二分类与非线性二分类的区别

|线性二分类|非线性二分类|
|---|---|
|<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/6/linear_binary.png"/>|<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/10/non_linear_binary.png"/>|

我们先学习如何解决线性二分类为标题，在此基础上可以扩展为非线性二分类问题。

## 二分类函数

对率函数Logistic Function，即可以做为激活函数使用，又可以当作二分类函数使用。而在很多不太正规的文字材料中，把这两个概念混用了，比如下面这个说法：“我们在最后使用Sigmoid激活函数来做二分类”，这是不恰当的。在本书中，我们会根据不同的任务区分激活函数和分类函数这两个概念，在二分类任务中，叫做Logistic函数，而在作为激活函数时，叫做Sigmoid函数。

- Logistic函数公式

$$Logistic(z) = \frac{1}{1 + e^{-z}}$$

以下记 $a=Logistic(z)$。

- 导数

$$Logistic'(z) = a(1 - a)$$

具体求导过程可以参考8.1节。

- 输入值域

$$(-\infty, \infty)$$

- 输出值域

$$(0,1)$$

- 函数图像（图6-2）

<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/8/logistic.png" ch="500" />

图6-2 Logistic函数图像

- 使用方式

此函数实际上是一个概率计算，它把 $(-\infty, \infty)$ 之间的任何数字都压缩到 $(0,1)$ 之间，返回一个概率值，这个概率值接近 $1$ 时，认为是正例，否则认为是负例。

训练时，一个样本 $x$ 在经过神经网络的最后一层的矩阵运算结果作为输入 $z$，经过Logistic计算后，输出一个 $(0,1)$ 之间的预测值。我们假设这个样本的标签值为 $0$ 属于负类，如果其预测值越接近 $0$，就越接近标签值，那么误差越小，反向传播的力度就越小。

推理时，我们预先设定一个阈值比如 $0.5$，则当推理结果大于 $0.5$ 时，认为是正类；小于 $0.5$ 时认为是负类；等于 $0.5$ 时，根据情况自己定义。阈值也不一定就是 $0.5$，也可以是 $0.65$ 等等，阈值越大，准确率越高，召回率越低；阈值越小则相反，准确度越低，召回率越高。

## 线性二分类的工作原理

||线性回归|线性分类|
|---|---|---|
|相同点|需要在样本群中找到一条直线|需要在样本群中找到一条直线|
|不同点|用直线来拟合所有样本，使得各个样本到这条直线的距离尽可能最短|用直线来分割所有样本，使得正例样本和负例样本尽可能分布在直线两侧|

1. 正向计算

$$
z = x_1 w_1+ x_2 w_2 + b  \tag{1}
$$

2. 分类计算

$$
a={1 \over 1 + e^{-z}} \tag{2}
$$

3. 损失函数计算

$$
loss = -[y \ln (a)+(1-y) \ln (1-a)] \tag{3}
$$

### 6.3.3 二分类的几何原理

几何方式：让所有正例样本处于直线的一侧，所有负例样本处于直线的另一侧，直线尽可能处于两类样本的中间。

#### 二分类函数的几何作用

二分类函数的最终结果是把正例都映射到图6-6中的上半部分的曲线上，而把负类都映射到下半部分的曲线上。

## 实现逻辑与或非门

- 与门 AND
- 与非门 NAND
- 或门 OR
- 或非门 NOR
- 非门 NOT

表6-8 五种逻辑门的结果比较

|逻辑门|分类结果|参数值|
|---|---|---|
|非|<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images\Images\6\LogicNotResult.png" width="300" height="300">|W=-12.468<br/>B=6.031|
|与|<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images\Images\6\LogicAndGateResult.png" width="300" height="300">|W1=11.757<br/>W2=11.757<br/>B=-17.804|
|与非|<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images\Images\6\LogicNandGateResult.png" width="300" height="300">|W1=-11.763<br/>W2=-11.763<br/>B=17.812|
|或|<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images\Images\6\LogicOrGateResult.png" width="300" height="300">|W1=11.743<br/>W2=11.743<br/>B=-11.738|
|或非|<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images\Images\6\LogicNorGateResult.png" width="300" height="300">|W1=-11.738<br/>W2=-11.738<br/>B=5.409|

## 用双曲正切函数做二分类函数

表6-10 对比使用不同分类函数的交叉熵函数的不同 

|分类函数|交叉熵函数|
|---|---|
|<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/6/logistic_seperator.png">|<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/3/crossentropy2.png"/>|
|输出值域a在(0,1)之间，分界线为a=0.5，标签值为y=0/1|y=0为负例，y=1为正例，输入值域a在(0,1)之间，符合对率函数的输出值域|
|<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/6/tanh_seperator.png">|<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/6/modified_crossentropy.png"/>|
|输出值域a在(-1,1)之间，分界线为a=0，标签值为y=-1/1|y=-1为负例，y=1为正例，输入值域a在(-1,1)之间，符合双曲正切函数的输出值域|

## 单入多出单层神经网络

多分类问题一共有三种解法：

1. 一对一方式
   
每次先只保留两个类别的数据，训练一个分类器。如果一共有 $N$ 个类别，则需要训练 $C^2_N$ 个分类器。以 $N=3$ 时举例，需要训练 $A|B，B|C，A|C$ 三个分类器。

<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/7/one_vs_one.png" />

图7-3 一对一方式

如图7-3最左侧所示，这个二分类器只关心蓝色和绿色样本的分类，而不管红色样本的情况，也就是说在训练时，只把蓝色和绿色样本输入网络。
   
推理时，$(A|B)$ 分类器告诉你是A类时，需要到 $(A|C)$ 分类器再试一下，如果也是A类，则就是A类。如果 $(A|C)$ 告诉你是C类，则基本是C类了，不可能是B类，不信的话可以到 $(B|C)$ 分类器再去测试一下。

2. 一对多方式
   
如图7-4，处理一个类别时，暂时把其它所有类别看作是一类，这样对于三分类问题，可以得到三个分类器。

<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/7/one_vs_multiple.png" />

图7-4 一对多方式

如最左图，这种情况是在训练时，把红色样本当作一类，把蓝色和绿色样本混在一起当作另外一类。

推理时，同时调用三个分类器，再把三种结果组合起来，就是真实的结果。比如，第一个分类器告诉你是“红类”，那么它确实就是红类；如果告诉你是非红类，则需要看第二个分类器的结果，绿类或者非绿类；依此类推。

3. 多对多方式

假设有4个类别ABCD，我们可以把AB算作一类，CD算作一类，训练一个分类器1；再把AC算作一类，BD算作一类，训练一个分类器2。
    
推理时，第1个分类器告诉你是AB类，第二个分类器告诉你是BD类，则做“与”操作，就是B类。

## 多分类函数

### 7.1.2 正向传播

#### 矩阵运算

$$
z=x \cdot w + b \tag{1}
$$

#### 分类计算

$$
a_j = \frac{e^{z_j}}{\sum\limits_{i=1}^m e^{z_i}}=\frac{e^{z_j}}{e^{z_1}+e^{z_2}+\dots+e^{z_m}} \tag{2}
$$

#### 损失函数计算

计算单样本时，m是分类数：
$$
loss(w,b)=-\sum_{i=1}^m y_i \ln a_i \tag{3}
$$

计算多样本时，m是分类数，n是样本数：
$$J(w,b) =- \sum_{j=1}^n \sum_{i=1}^m y_{ij} \log a_{ij} \tag{4}$$

如图7-6示意。

<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/7/Loss-A-Z.jpg" ch="500" />

图7-6 Softmax在神经网络结构中的示意图

### 7.1.3 反向传播

#### 实例化推导

我们先用实例化的方式来做反向传播公式的推导，然后再扩展到一般性上。假设有三个类别，则：

$$
z_1 = x \cdot w+ b_1 \tag{5}
$$
$$
z_2 = x \cdot w + b_2 \tag{6}
$$
$$
z_3 = x \cdot w + b_3 \tag{7}
$$
$$
a_1=\frac{e^{z_1}}{\sum_i e^{z_i}}=\frac{e^{z_1}}{e^{z_1}+e^{z_2}+e^{z_3}}  \tag{8}
$$
$$
a_2=\frac{e^{z_2}}{\sum_i e^{z_i}}=\frac{e^{z_2}}{e^{z_1}+e^{z_2}+e^{z_3}}  \tag{9}
$$
$$
a_3=\frac{e^{z_3}}{\sum_i e^{z_i}}=\frac{e^{z_3}}{e^{z_1}+e^{z_2}+e^{z_3}}  \tag{10}
$$

为了方便书写，我们令：

$$
E ={e^{z_1}+e^{z_2}+e^{z_3}}
$$

$$
loss(w,b)=-(y_1 \ln a_1 + y_2 \ln a_2 + y_3 \ln a_3)  \tag{11}
$$

$$
\frac{\partial{loss}}{\partial{z_1}}= \frac{\partial{loss}}{\partial{a_1}}\frac{\partial{a_1}}{\partial{z_1}} + \frac{\partial{loss}}{\partial{a_2}}\frac{\partial{a_2}}{\partial{z_1}} + \frac{\partial{loss}}{\partial{a_3}}\frac{\partial{a_3}}{\partial{z_1}}  \tag{12}
$$

依次求解公式12中的各项：

$$
\frac{\partial loss}{\partial a_1}=- \frac{y_1}{a_1} \tag{13}
$$
$$
\frac{\partial loss}{\partial a_2}=- \frac{y_2}{a_2} \tag{14}
$$
$$
\frac{\partial loss}{\partial a_3}=- \frac{y_3}{a_3} \tag{15}
$$

$$
\begin{aligned}
\frac{\partial a_1}{\partial z_1}&=(\frac{\partial e^{z_1}}{\partial z_1} E -\frac{\partial E}{\partial z_1}e^{z_1})/E^2 \\\\
&=\frac{e^{z_1}E - e^{z_1}e^{z_1}}{E^2}=a_1(1-a_1)  
\end{aligned}
\tag{16}
$$

$$
\begin{aligned}
\frac{\partial a_2}{\partial z_1}&=(\frac{\partial e^{z_2}}{\partial z_1} E -\frac{\partial E}{\partial z_1}e^{z_2})/E^2 \\\\
&=\frac{0 - e^{z_1}e^{z_2}}{E^2}=-a_1 a_2 
\end{aligned}
\tag{17}
$$

$$
\begin{aligned}
\frac{\partial a_3}{\partial z_1}&=(\frac{\partial e^{z_3}}{\partial z_1} E -\frac{\partial E}{\partial z_1}e^{z_3})/E^2 \\\\
&=\frac{0 - e^{z_1}e^{z_3}}{E^2}=-a_1 a_3  
\end{aligned}
\tag{18}
$$

把公式13~18组合到12中：

$$
\begin{aligned}    
\frac{\partial loss}{\partial z_1}&=-\frac{y_1}{a_1}a_1(1-a_1)+\frac{y_2}{a_2}a_1a_2+\frac{y_3}{a_3}a_1a_3 \\\\
&=-y_1+y_1a_1+y_2a_1+y_3a_1 \\\\
&=-y_1+a_1(y_1+y_2+y_3) \\\\
&=a_1-y_1 
\end{aligned}
\tag{19}
$$

不失一般性，由公式19可得：
$$
\frac{\partial loss}{\partial z_i}=a_i-y_i \tag{20}
$$

#### 一般性推导

1. Softmax函数自身的求导

由于Softmax涉及到求和，所以有两种情况：

- 求输出项 $a_1$ 对输入项 $z_1$ 的导数，此时：$j=1, i=1, i=j$，可以扩展到 $i,j$ 为任意相等值
- 求输出项 $a_2$ 或 $a_3$ 对输入项 $z_1$ 的导数，此时：$j$ 为 $2$ 或 $3$, $i=1,i \neq j$，可以扩展到 $i,j$ 为任意不等值

Softmax函数的分子：因为是计算 $a_j$，所以分子是 $e^{z_j}$。

Softmax函数的分母：
$$
\sum\limits_{i=1}^m e^{z_i} = e^{z_1} + \dots + e^{z_j} + \dots +e^{z_m} \Rightarrow E
$$

- $i=j$时（比如输出分类值 $a_1$ 对 $z_1$ 的求导），求 $a_j$ 对 $z_i$ 的导数，此时分子上的 $e^{z_j}$ 要参与求导。参考基本数学导数公式33：

$$
\begin{aligned}
\frac{\partial{a_j}}{\partial{z_i}} &= \frac{\partial{}}{\partial{z_i}}(e^{z_j}/E) \\\\
&= \frac{\partial{}}{\partial{z_j}}(e^{z_j}/E) \quad (因为z_i==z_i)\\\\
&=\frac{e^{z_j}E-e^{z_j}e^{z_j}}{E^2} 
=\frac{e^{z_j}}{E} - \frac{(e^{z_j})^2}{E^2} \\\\
&= a_j-a^2_j=a_j(1-a_j)  \\\\
\end{aligned}
\tag{21}
$$

- $i \neq j$时（比如输出分类值 $a_1$ 对 $z_2$ 的求导，$j=1,i=2$），$a_j$对$z_i$的导数，分子上的 $z_j$ 与 $i$ 没有关系，求导为0，分母的求和项中$e^{z_i}$要参与求导。同样是公式33，因为分子$e^{z_j}$对$e^{z_i}$求导的结果是0：

$$
\frac{\partial{a_j}}{\partial{z_i}}=\frac{-(E)'e^{z_j}}{E^2}
$$
求和公式对$e^{z_i}$的导数$(E)'$，除了$e^{z_i}$项外，其它都是0：
$$
(E)' = (e^{z_1} + \dots + e^{z_i} + \dots +e^{z_m})'=e^{z_i}
$$
所以：
$$
\begin{aligned}
\frac{\partial{a_j}}{\partial{z_i}}&=\frac{-(E)'e^{z_j}}{(E)^2}=-\frac{e^{z_j}e^{z_i}}{{(E)^2}} \\\\
&=-\frac{e^{z_j}}{{E}}\frac{e^{z_j}}{{E}}=-a_{i}a_{j} 
\end{aligned}
\tag{22}
$$

2. 结合损失函数的整体反向传播公式

看上图，我们要求Loss值对Z1的偏导数。和以前的Logistic函数不同，那个函数是一个z对应一个a，所以反向关系也是一对一。而在这里，a1的计算是有z1,z2,z3参与的，a2的计算也是有z1,z2,z3参与的，即所有a的计算都与前一层的z有关，所以考虑反向时也会比较复杂。

先从Loss的公式看，$loss=-(y_1lna_1+y_2lna_2+y_3lna_3)$，a1肯定与z1有关，那么a2,a3是否与z1有关呢？

再从Softmax函数的形式来看：

无论是a1，a2，a3，都是与z1相关的，而不是一对一的关系，所以，想求Loss对Z1的偏导，必须把Loss->A1->Z1， Loss->A2->Z1，Loss->A3->Z1，这三条路的结果加起来。于是有了如下公式：

$$
\begin{aligned}    
\frac{\partial{loss}}{\partial{z_i}} &= \frac{\partial{loss}}{\partial{a_1}}\frac{\partial{a_1}}{\partial{z_i}} + \frac{\partial{loss}}{\partial{a_2}}\frac{\partial{a_2}}{\partial{z_i}} + \frac{\partial{loss}}{\partial{a_3}}\frac{\partial{a_3}}{\partial{z_i}} \\\\
&=\sum_j \frac{\partial{loss}}{\partial{a_j}}\frac{\partial{a_j}}{\partial{z_i}}
\end{aligned}
$$

当上式中$i=1,j=3$，就完全符合我们的假设了，而且不失普遍性。

前面说过了，因为Softmax涉及到各项求和，A的分类结果和Y的标签值分类是否一致，所以需要分情况讨论：

$$
\frac{\partial{a_j}}{\partial{z_i}} = \begin{cases} a_j(1-a_j), & i = j \\\\ -a_ia_j, & i \neq j \end{cases}
$$

因此，$\frac{\partial{loss}}{\partial{z_i}}$应该是 $i=j$ 和 $i \neq j$两种情况的和：

- $i = j$ 时，loss通过 $a_1$ 对 $z_1$ 求导（或者是通过 $a_2$ 对 $z_2$ 求导）：

$$
\begin{aligned}
\frac{\partial{loss}}{\partial{z_i}} &= \frac{\partial{loss}}{\partial{a_j}}\frac{\partial{a_j}}{\partial{z_i}}=-\frac{y_j}{a_j}a_j(1-a_j) \\\\
&=y_j(a_j-1)=y_i(a_i-1) 
\end{aligned}
\tag{23}
$$

- $i \neq j$，loss通过 $a_2+a_3$ 对 $z_1$ 求导：

$$
\begin{aligned}    
\frac{\partial{loss}}{\partial{z_i}} &= \frac{\partial{loss}}{\partial{a_j}}\frac{\partial{a_j}}{\partial{z_i}}=\sum_j^m(-\frac{y_j}{a_j})(-a_ja_i) \\\\
&=\sum_j^m(y_ja_i)=a_i\sum_{j \neq i}{y_j} 
\end{aligned}
\tag{24}
$$

把两种情况加起来：

$$
\begin{aligned}    
\frac{\partial{loss}}{\partial{z_i}} &= y_i(a_i-1)+a_i\sum_{j \neq i}y_j \\\\
&=-y_i+a_iy_i+a_i\sum_{j \neq i}y_j \\\\
&=-y_i+a_i(y_i+\sum_{j \neq i}y_j) \\\\
&=-y_i + a_i*1 \\\\
&=a_i-y_i 
\end{aligned}
\tag{25}$$

因为$y_j$取值$[1,0,0]$或者$[0,1,0]$或者$[0,0,1]$，这三者加起来，就是$[1,1,1]$，在矩阵乘法运算里乘以$[1,1,1]$相当于什么都不做，就等于原值。

我们惊奇地发现，最后的反向计算过程就是：$a_i-y_i$，假设当前样本的$a_i=[0.879, 0.119, 0.002]$，而$y_i=[0, 1, 0]$，则：
$$a_i - y_i = [0.879, 0.119, 0.002]-[0,1,0]=[0.879,-0.881,0.002]$$

其含义是，样本预测第一类，但实际是第二类，所以给第一类0.879的惩罚值，给第二类0.881的奖励，给第三类0.002的惩罚，并反向传播给神经网络。

后面对 $z=wx+b$ 的求导，与二分类一样，不再赘述。


## 线性多分类的工作原理

多分类过程：
1. 线性计算

$$z_1 = x_1 w_{11} + x_2 w_{21} + b_1 \tag{1}$$
$$z_2 = x_1 w_{12} + x_2 w_{22} + b_2 \tag{2}$$
$$z_3 = x_1 w_{13} + x_2 w_{23} + b_3 \tag{3}$$

2. 分类计算

$$
a_1=\frac{e^{z_1}}{\sum_i e^{z_i}}=\frac{e^{z_1}}{e^{z_1}+e^{z_2}+e^{z_3}}  \tag{4}
$$
$$
a_2=\frac{e^{z_2}}{\sum_i e^{z_i}}=\frac{e^{z_2}}{e^{z_1}+e^{z_2}+e^{z_3}}  \tag{5}
$$
$$
a_3=\frac{e^{z_3}}{\sum_i e^{z_i}}=\frac{e^{z_3}}{e^{z_1}+e^{z_2}+e^{z_3}}  \tag{6}
$$

3. 损失函数计算

单样本时，$n$表示类别数，$j$表示类别序号：

$$
\begin{aligned}
loss(w,b)&=-(y_1 \ln a_1 + y_2 \ln a_2 + y_3 \ln a_3) \\\\
&=-\sum_{j=1}^{n} y_j \ln a_j 
\end{aligned}
\tag{7}
$$

批量样本时，$m$ 表示样本数，$i$ 表示样本序号：

$$
\begin{aligned}
J(w,b) &=- \sum_{i=1}^m (y_{i1} \ln a_{i1} + y_{i2} \ln a_{i2} + y_{i3} \ln a_{i3}) \\\\
&=- \sum_{i=1}^m \sum_{j=1}^n y_{ij} \ln a_{ij}
\end{aligned}
 \tag{8}
$$

几何原理：

在前面的二分类原理中，很容易理解为我们用一条直线分开两个部分。对于多分类问题，是否可以沿用二分类原理中的几何解释呢？答案是肯定的，只不过需要单独判定每一个类别。

假设一共有三类样本，蓝色为1，红色为2，绿色为3，那么Softmax的形式应该是：

$$
a_j = \frac{e^{z_j}}{\sum\limits_{i=1}^3 e^{z_i}}=\frac{e^{z_j}}{e^{z_1}+e^{z_2}+^{z_3}}
$$

# Step 4 非线性回归

# 第八章 激活函数

## 激活函数概论

### 激活函数的基本作用
> 激活函数的作用：
> 1、给神经网络增加非线性因素，这个问题在第1章神经网络基本工作原理中已经讲过了；
> 2、把公式1的计算结果压缩到 $[0,1]$ 之间，便于后面的计算。

> 激活函数的基本性质：
> 1、非线性：线性的激活函数和没有激活函数一样；
> 2、可导性：做误差反向传播和梯度下降，必须要保证激活函数的可导性；
> 3、单调性：单一的输入会得到单一的输出，较大值的输入得到较大值的输出。

### 使用激活函数的时间

> 激活函数用在神经网络的层与层之间，但是神经网络的最后一层不用激活函数。

![](./images/IMG_0381.png)

1. 神经网络最后一层不需要激活函数
2. 激活函数只用于连接前后两层神经网络



##  挤压型激活函数

有两个常用的挤压型激活函数，一个是Lodistic函数，另一个是Tanh函数。

### Logistic函数

$$Sigmoid(z) = \frac{1}{1 + e^{-z}} \rightarrow a \tag{1}$$

![](./images/IMG_453A3EA3484B-1.jpeg)

#### 优点

从函数图像来看，Sigmoid函数的作用是将输入压缩到 $(0,1)$ 这个区间范围内，这种输出在0~1之间的函数可以用来模拟一些概率分布的情况。它还是一个连续函数，导数简单易求。  

从数学上来看，Sigmoid函数对中央区的信号增益较大，对两侧区的信号增益小，在信号的特征空间映射上，有很好的效果。 

从神经科学上来看，中央区酷似神经元的兴奋态，两侧区酷似神经元的抑制态，因而在神经网络学习方面，可以将重点特征推向中央区，
将非重点特征推向两侧区。

#### 缺点

指数计算代价大。

反向传播时梯度消失：从梯度图像中可以看到，Sigmoid的梯度在两端都会接近于0，根据链式法则，如果传回的误差是$\delta$，那么梯度传递函数是$\delta \cdot a'$，而$a'$这时接近零，也就是说整体的梯度也接近零。这就出现梯度消失的问题，并且这个问题可能导致网络收敛速度比较慢。

### Tanh函数

$$Tanh(z) = \frac{e^{z} - e^{-z}}{e^{z} + e^{-z}} = (\frac{2}{1 + e^{-2z}}-1) \rightarrow a \tag{3}$$
即
$$Tanh(z) = 2 \cdot Sigmoid(2z) - 1 \tag{4}$$

#### 优点

具有Sigmoid的所有优点。

#### 缺点

exp指数计算代价大。梯度消失问题仍然存在。

![](./images/8.1.1.png)
![](./images/8.1.2.png)
![](./images/8.1.3.png)

## 8.2 半线性激活函数

又可以叫非饱和型激活函数。

### 8.2.1 ReLU函数 

Rectified Linear Unit，修正线性单元，线性整流函数，斜坡函数。

#### 公式

$$ReLU(z) = max(0,z) = \begin{cases} 
  z, & z \geq 0 \\\\ 
  0, & z < 0 
\end{cases}$$

#### 导数

$$ReLU'(z) = \begin{cases} 1 & z \geq 0 \\\\ 0 & z < 0 \end{cases}$$

#### 值域

- 输入值域：$(-\infty, \infty)$
- 输出值域：$(0,\infty)$
- 导数值域：$\\{0,1\\}$

<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/8/relu.png"/>

图8-6 线性整流函数ReLU

#### 优点

- 反向导数恒等于1，更加有效率的反向传播梯度值，收敛速度快；
- 避免梯度消失问题；
- 计算简单，速度快；
- 活跃度的分散性使得神经网络的整体计算成本下降。

#### 缺点

无界。

### Leaky ReLU函数

LReLU，带泄露的线性整流函数。

#### 公式

$$LReLU(z) = \begin{cases} z & z \geq 0 \\\\ \alpha \cdot z & z < 0 \end{cases}$$

#### 导数

$$LReLU'(z) = \begin{cases} 1 & z \geq 0 \\\\ \alpha & z < 0 \end{cases}$$

#### 值域

输入值域：$(-\infty, \infty)$

输出值域：$(-\infty,\infty)$

导数值域：$\\{\alpha,1\\}$

#### 优点

继承了ReLU函数的优点。

Leaky ReLU同样有收敛快速和运算复杂度低的优点，而且由于给了$z<0$时一个比较小的梯度$\alpha$,使得$z<0$时依旧可以进行梯度传递和更新，可以在一定程度上避免神经元“死”掉的问题。

### Softplus函数

#### 公式

$$Softplus(z) = \ln (1 + e^z)$$

#### 导数

$$Softplus'(z) = \frac{e^z}{1 + e^z}$$

#### 

输入值域：$(-\infty, \infty)$

输出值域：$(0,\infty)$

导数值域：$(0,1)$

### ELU函数

#### 公式

$$ELU(z) = \begin{cases} z & z \geq 0 \\ \alpha (e^z-1) & z < 0 \end{cases}$$

#### 导数

$$ELU'(z) = \begin{cases} 1 & z \geq 0 \\ \alpha e^z & z < 0 \end{cases}$$

#### 值域

输入值域：$(-\infty, \infty)$

输出值域：$(-\alpha,\infty)$

导数值域：$(0,1]$

![](./images/8.2.1.png)
![](./images/8.2.2.png)
![](./images/8.2.3.png)
![](./images/8.2.4.png)
![](./images/8.2.5.png)

# 第9章 单入单出的双层神经网络 - 非线性回归

## 9.0 非线性回归问题

#### 9.0.3 回归模型的评估标准

回归问题主要是求值，评价标准主要是看求得值与实际结果的偏差有多大，所以，回归问题主要以下方法来评价模型。

#### 平均绝对误差

MAE（Mean Abolute Error）。

$$MAE=\frac{1}{m} \sum_{i=1}^m \lvert a_i-y_i \rvert \tag{1}$$

对异常值不如均方差敏感，类似中位数。

#### 绝对平均值率误差

MAPE（Mean Absolute Percentage Error）。

$$MAPE=\frac{100}{m} \sum^m_{i=1} \left\lvert {a_i - y_i \over y_i} \right\rvert \tag{2}$$

#### 和方差

SSE（Sum Squared Error）。

$$SSE=\sum_{i=1}^m (a_i-y_i)^2 \tag{3}$$

得出的值与样本数量有关系，假设有1000个测试样本，得到的值是120；如果只有100个测试样本，得到的值可能是11，我们不能说11就比120要好。

#### 均方差

MSE（Mean Squared Error）。

$$MSE = \frac{1}{m} \sum_{i=1}^m (a_i-y_i)^2 \tag{4}$$

就是实际值减去预测值的平方再求期望，没错，就是线性回归的代价函数。由于MSE计算的是误差的平方，所以它对异常值是非常敏感的，因为一旦出现异常值，MSE指标会变得非常大。MSE越小，证明误差越小。

#### 均方根误差

RMSE（Root Mean Squard Error）。

$$RMSE = \sqrt{\frac{1}{m} \sum_{i=1}^m (a_i-y_i)^2} \tag{5}$$

是均方差开根号的结果，其实质是一样的，只不过对结果有更好的解释。

例如：要做房价预测，每平方是万元，我们预测结果也是万元，那么MSE差值的平方单位应该是千万级别的。假设我们的模型预测结果与真实值相差1000元，则用MSE的计算结果是1000,000，这个值没有单位，如何描述这个差距？于是就求个平方根就好了，这样误差可以是标签值是同一个数量级的，在描述模型的时候就说，我们模型的误差是多少元。

#### R平方

R-Squared。

上面的几种衡量标准针对不同的模型会有不同的值。比如说预测房价，那么误差单位就是元，比如3000元、11000元等。如果预测身高就可能是0.1、0.2米之类的。也就是说，对于不同的场景，会有不同量纲，因而也会有不同的数值，无法用一句话说得很清楚，必须啰啰嗦嗦带一大堆条件才能表达完整。

我们通常用概率来表达一个准确率，比如89%的准确率。那么线性回归有没有这样的衡量标准呢？答案就是R-Squared。

$$R^2=1-\frac{\sum (a_i - y_i)^2}{\sum(\bar y_i-y_i)^2}=1-\frac{MSE(a,y)}{Var(y)} \tag{6}$$

R平方是多元回归中的回归平方和（分子）占总平方和（分母）的比例，它是度量多元回归方程中拟合程度的一个统计量。R平方值越接近1，表明回归平方和占总平方和的比例越大，回归线与各观测点越接近，回归的拟合程度就越好。

## 9.1 用多项式回归法拟合正弦曲线

多项式回归有几种形式：

#### 一元一次线性模型

因为只有一项，所以不能称为多项式了。它可以解决单变量的线性回归，我们在第4章学习过相关内容。其模型为：

$$z = x w + b \tag{1}$$

#### 多元一次多项式

多变量的线性回归，我们在第5章学习过相关内容。其模型为：

$$z = x_1 w_1 + x_2 w_2 + ...+ x_m w_m + b \tag{2}$$

这里的多变量，是指样本数据的特征值为多个，上式中的 $x_1,x_2,...,x_m$ 代表了m个特征值。

#### 多元多次多项式

多变量的非线性回归，其参数与特征组合繁复，但最终都可以归结为公式2和公式4的形式。

### 9.1.2 用二次多项式拟合

鉴于以上的认知，我们要考虑使用几次的多项式来拟合正弦曲线。在没有什么经验的情况下，可以先试一下二次多项式，即：

$$z = x w_1 + x^2 w_2 + b \tag{5}$$

![](./images/9.1.1.png)
![](./images/9.1.2.png)

## 9.2 用多项式回归法拟合复合函数曲线

### 9.2.1 用四次多项式拟合

![](./images/9.2png)

## 9.3 验证与测试

### 9.3.1 基本概念

#### 训练集

Training Set，用于模型训练的数据样本。

#### 验证集

Validation Set，或者叫做Dev Set，是模型训练过程中单独留出的样本集，它可以用于调整模型的超参数和用于对模型的能力进行初步评估。
  
在神经网络中，验证数据集用于：

- 寻找最优的网络深度
- 或者决定反向传播算法的停止点
- 或者在神经网络中选择隐藏层神经元的数量
- 在普通的机器学习中常用的交叉验证（Cross Validation）就是把训练数据集本身再细分成不同的验证数据集去训练模型。

#### 测试集

Test Set，用来评估最终模型的泛化能力。但不能作为调参、选择特征等算法相关的选择的依据。

三者之间的关系如图9-5所示。

<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/9/dataset.png" />

图9-5 训练集、验证集、测试集的关系


### 9.3.2 交叉验证

### 9.3.3 留出法 Hold out

使用交叉验证的方法虽然比较保险，但是非常耗时，尤其是在大数据量时，训练出一个模型都要很长时间，没有可能去训练出10个模型再去比较。

在深度学习中，有另外一种方法使用验证集，称为留出法。亦即从训练数据中保留出验证样本集，主要用于解决过拟合情况，这部分数据不用于训练。如果训练数据的准确度持续增长，但是验证数据的准确度保持不变或者反而下降，说明神经网络亦即过拟合了，此时需要停止训练，用测试集做最终测试。

## 9.4 双层神经网络实现非线性回归

### 9.4.1 万能近似定理

万能近似定理(universal approximation theorem) $^{[1]}$，是深度学习最根本的理论依据。它证明了在给定网络具有足够多的隐藏单元的条件下，配备一个线性输出层和一个带有任何“挤压”性质的激活函数（如Sigmoid激活函数）的隐藏层的前馈神经网络，能够以任何想要的误差量近似任何从一个有限维度的空间映射到另一个有限维度空间的Borel可测的函数。

前馈网络的导数也可以以任意好地程度近似函数的导数。

万能近似定理其实说明了理论上神经网络可以近似任何函数。但实践上我们不能保证学习算法一定能学习到目标函数。即使网络可以表示这个函数，学习也可能因为两个不同的原因而失败：

1. 用于训练的优化算法可能找不到用于期望函数的参数值；
2. 训练算法可能由于过拟合而选择了错误的函数。


## 9.5 曲线拟合

### 9.5.1 正弦曲线的拟合

#### 隐层只有一个神经元的情况

令`n_hidden=1`，并指定模型名称为`sin_111`，训练过程见图9-10。图9-11为拟合效果图。

<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/9/sin_loss_1n.png" />

图9-10 训练过程中损失函数值和准确率的变化

<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/9/sin_result_1n.png" ch="500" />

图9-11 一个神经元的拟合效果

### 9.5.2 复合函数的拟合

基本过程与正弦曲线相似，区别是这个例子要复杂不少，所以首先需要耐心，增大`max_epoch`的数值，多迭代几次。其次需要精心调参，找到最佳参数组合。

#### 隐层只有两个神经元的情况

<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/9/complex_result_2n.png" ch="500" />

图9-14 两个神经元的拟合效果

![](./images/9.3.1.png)
![](./images/9.3.2.png)
![](./images/9.4.1.png)
![](./images/9.4.2.png)

## 9.6 非线性回归的工作原理

### 9.6.1 多项式为何能拟合曲线

单层神经网络的多项式回归法，需要$x,x^2,x^3$三个特征值，组成如下公式来得到拟合结果：

$$
z = x \cdot w_1 + x^2 \cdot w_2 + x^3 \cdot w_3 + b \tag{1}
$$

我们可以回忆一下第5章学习的多变量线性回归问题，公式1实际上是把一维的x的特征信息，增加到了三维，然后再使用多变量线性回归来解决问题的。本来一维的特征只能得到线性的结果，但是三维的特征就可以得到非线性的结果，这就是多项式拟合的原理。

### 9.6.2 神经网络的非线性拟合工作原理

1. 隐层把x拆成不同的特征，根据问题复杂度决定神经元数量，神经元的数量相当于特征值的数量；
2. 隐层通过激活函数做一次非线性变换；
3. 输出层使用多变量线性回归，把隐层的输出当作输入特征值，再做一次线性变换，得出拟合结果。

与多项式回归不同的是，不需要指定变换参数，而是从训练中学习到参数，这样的话权重值不会大得离谱。

![](./images/9.5.1.png)
![](./images/9.5.2.png)
![](./images/9.5.3.png)
![](./images/9.5.4.png)
![](./images/9.5.5.png)

## 9.7 超参数优化的初步认识

超参数优化（Hyperparameter Optimization）主要存在两方面的困难：

1. 超参数优化是一个组合优化问题，无法像一般参数那样通过梯度下降方法来优化，也没有一种通用有效的优化方法。
2. 评估一组超参数配置（Conﬁguration）的时间代价非常高，从而导致一些优化方法（比如演化算法）在超参数优化中难以应用。

对于超参数的设置，比较简单的方法有人工搜索、网格搜索和随机搜索。 

### 9.7.1 可调的参数

我们使用表9-21所示的参数做第一次的训练。

表9-21 参数配置

|参数|缺省值|是否可调|注释|
|---|---|---|---|
|输入层神经元数|1|No|
|隐层神经元数|4|Yes|影响迭代次数|
|输出层神经元数|1|No|
|学习率|0.1|Yes|影响迭代次数|
|批样本量|10|Yes|影响迭代次数|
|最大epoch|10000|Yes|影响终止条件,建议不改动|
|损失门限值|0.001|Yes|影响终止条件,建议不改动|
|损失函数|MSE|No|
|权重矩阵初始化方法|Xavier|Yes|参看15.1|

表9-21中的参数，最终可以调节的其实只有三个：

- 隐层神经元数
- 学习率
- 批样本量

#### 避免权重矩阵初始化的影响

权重矩阵中的参数，是神经网络要学习的参数，所以不能称作超参数。

权重矩阵初始化是神经网络训练非常重要的环节之一，不同的初始化方法，甚至是相同的方法但不同的随机值，都会给结果带来或多或少的影响。

在后面的几组比较中，都是用Xavier方法初始化的。在两次参数完全相同的试验中，即使两次都使用Xavier初始化，因为权重矩阵参数的差异，也会得到不同的结果。为了避免这个随机性，我们在代码`WeightsBias.py`中使用了一个小技巧，调用下面这个函数：

### 9.7.2 手动调整参数

手动调整超参数，我们必须了解超参数、训练误差、泛化误差和计算资源（内存和运行时间）之间的关系。手动调整超参数的主要目标是调整模型的有效容量以匹配任务的复杂性。有效容量受限于3个因素：

- 模型的表示容量；
- 学习算法与代价函数的匹配程度；
- 代价函数和训练过程正则化模型的程度。

表9-22比较了几个超参数的作用。具有更多网络层、每层有更多隐藏单元的模型具有较高的表示能力，能够表示更复杂的函数。学习率是最重要的超参数。如果你只有一个超参数调整的机会，那就调整学习率。

表9-22 各种超参数的作用

|超参数|目标|作用|副作用|
|---|---|---|---|
|学习率|调至最优|低的学习率会导致收敛慢，高的学习率会导致错失最佳解|容易忽略其它参数的调整|
|隐层神经元数量|增加|增加数量会增加模型的表示能力|参数增多、训练时间增长|
|批大小|有限范围内尽量大|大批量的数据可以保持训练平稳，缩短训练时间|可能会收敛速度慢|

通常的做法是，按经验设置好隐层神经元数量和批大小，并使之相对固定，然后调整学习率。

#### 学习率的调整

我们固定其它参数，即隐层神经元`ne=4`、`batch_size=10`不变，改变学习率，来试验网络训练情况。为了节省时间，不做无限轮次的训练，而是设置`eps=0.001`为最低精度要求，一旦到达，就停止训练。

#### 批大小的调整

我们固定其它参数，即隐层神经元`ne=4`、`eta=0.5`不变，调整批大小，来试验网络训练情况，设置`eps=0.001`为精度要求。

表9-25和图9-24展示了四种批大小值的不同结果。

表9-25 四种批大小数值的比较

|批大小|迭代次数|说明|
|----|----|----|
|5|2500|批数据量小到1，收敛最快|
|10|8200|批数据量增大，收敛变慢|
|15|10000|批数据量进一步增大，收敛变慢|
|20|10000|批数据量太大，反而会降低收敛速度|

#### 隐层神经元数量的调整

我们固定其它参数，即`batch_size=10`、`eta=0.5`不变，调整隐层神经元的数量，来试验网络训练情况，设置`eps=0.001`为精度要求。

表9-26和图9-25展示了四种神经元数值的不同结果。

表9-26 四种隐层神经元数量值的比较

|隐层神经元数量|迭代次数|说明|
|---|---|---|
|2|10000|神经元数量少，拟合能力低|
|4|8000|神经元数量增加会有帮助|
|6|5500|神经元数量进一步增加，收敛更快|
|8|3500|再多一些神经元，还会继续提供收敛速度|

![](./images/9.5.1.png)
![](./images/9.5.2.png)
![](./images/9.5.3.png)
![](./images/9.5.4.png)
![](./images/9.5.5.png)

# 第五步  非线性分类

# 第10章 多入单出的双层神经网络 - 非线性二分类

## 10.0 非线性二分类问题

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

## 10.1 为什么必须用双层神经网络

## 10.1.3 非线性的可能性

我们前边学习过如何实现与、与非、或、或非，我们看看如何用已有的逻辑搭建异或门，如图10-5所示。

<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/10/xor_gate.png" />

图10-5 用基本逻辑单元搭建异或运算单元

表10-6 组合运算的过程

|样本与计算|1|2|3|4|
|----|----|----|----|----|
|$x_1$|0|0|1|1|
|$x_2$|0|1|0|1|#
|$s_1=x_1$ NAND $x_2$|1|1|1|0|
|$s_2=x_1$ OR $x_2$|0|1|1|1|
|$y=s_1$ AND $s_2$|0|1|1|0|

经过表10-6所示的组合运算后，可以看到$y$的输出与$x_1,x_2$的输入相比，就是异或逻辑了。所以，实践证明两层逻辑电路可以解决问题。另外，我们在地四步中学习了非线性回归，使用双层神经网络可以完成一些神奇的事情，比如复杂曲线的拟合，只需要6、7个参数就搞定了。我们可以模拟这个思路，用两层神经网络搭建模型，来解决非线性分类问题。

## 10.2 非线性二分类实现

### 10.2.1 定义神经网络结构

首先定义可以完成非线性二分类的神经网络结构图，如图10-6所示。

<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/10/xor_nn.png" />

图10-6 非线性二分类神经网络结构图

对于一般的用于二分类的双层神经网络可以是图10-7的样子。

<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/10/binary_classifier.png" width="600" ch="500" />

图10-7 通用的二分类神经网络结构图

输入特征值可以有很多，隐层单元也可以有很多，输出单元只有一个，且后面要接Logistic分类函数和二分类交叉熵损失函数。

### 10.2.2 前向计算

根据网络结构，我们有了前向计算过程图10-8。

<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/10/binary_forward.png" />

图10-8 前向计算过程

### 10.2.3 反向传播

图10-9展示了反向传播的过程。

<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/10/binary_backward.png" />

图10-9 反向传播过程

## 10.3 实现逻辑异或门

![](./images/10.1.1.png)

## 10.4 逻辑异或门的工作原理

### 10.4.2 更直观的可视化结果

#### 3D图

神经网络真的可以同时画两条分割线吗？这颠覆了笔者的认知，因为笔者一直认为最后一层的神经网络只是一个线性单元，它能做的事情有限，所以它的行为就是线性的行为，画一条线做拟合或分割，......，稍等，为什么只能是一条线呢？难道不可以是一个平面吗？

这让笔者想起了在第5章里，曾经用一个平面拟合了空间中的样本点，如表10-10所示。

表10-10 平面拟合的可视化结果

|正向|侧向|
|---|---|
|<img src='../Images/5/level3_result_1.png'/>|<img src='../Images/5/level3_result_2.png'/>|

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


![](./images/10.2.1.png)
![](./images/10.2.2.png)
![](./images/10.2.3.png)
![](./images/10.2.4.png)
![](./images/10.2.5.png)
![](./images/10.2.6.png)
![](./images/10.2.7.png)
![](./images/10.2.8.png)
![](./images/10.2.9.png)
![](./images/10.2.10.png)

## 10.5 实现双弧形二分类

![](./images/10.3.1.png)
![](./images/10.3.2.png)
![](./images/10.3.3.png)

# 第11章 多入多出的双层神经网络 - 非线性多分类

## 11.0 非线性多分类问题

### 11.1.1 定义神经网络结构
### 11.1.2 前向计算
### 11.1.3 反向传播

![](./images/11.1.1.png)
![](./images/11.1.2.png)

## 11.2 非线性多分类的工作原理
### 11.2.1 隐层神经元数量的影响

下面列出了隐层神经元为2、4、8、16、32、64的情况，设定好最大epoch数为10000，以比较它们各自能达到的最大精度值的区别。每个配置只测试一轮，所以测出来的数据有一定的随机性。

表11-3展示了隐层神经元数与分类结果的关系。

表11-3 神经元数与网络能力及分类结果的关系

|神经元数|损失函数|分类结果|
|---|---|---|
|2|<img src='../Images/11/loss_n2.png'/>|<img src='../Images/11/result_n2.png'/>|
||测试集准确度0.618，耗时49秒，损失函数值0.795。类似这种曲线的情况，损失函数值降不下去，准确度值升不上去，主要原因是网络能力不够。|没有完成分类任务|
|4|<img src='../Images/11/loss_n4.png'/>|<img src='../Images/11/result_n4.png'/>|
||测试准确度0.954，耗时51秒，损失函数值0.132。虽然可以基本完成分类任务，网络能力仍然不够。|基本完成，但是边缘不够清晰|
|8|<img src='../Images/11/loss_n8.png'/>|<img src='../Images/11/result_n8.png'/>|
||测试准确度0.97，耗时52秒，损失函数值0.105。可以先试试在后期衰减学习率，如果再训练5000轮没有改善的话，可以考虑增加网络能力。|基本完成，但是边缘不够清晰|
|16|<img src='../Images/11/loss_n16.png'/>|<img src='../Images/11/result_n16.png'/>|
||测试准确度0.978，耗时53秒，损失函数值0.094。同上，可以同时试着使用优化算法，看看是否能收敛更快。|较好地完成了分类任务|
|32|<img src='../Images/11/loss_n32.png'/>|<img src='../Images/11/result_n32.png'/>|
||测试准确度0.974，耗时53秒，损失函数值0.085。网络能力够了，从损失值下降趋势和准确度值上升趋势来看，可能需要更多的迭代次数。|较好地完成了分类任务|
|64|<img src='../Images/11/loss_n64.png'/>|<img src='../Images/11/result_n64.png'/>|
||测试准确度0.972，耗时64秒，损失函数值0.075。网络能力足够。|较好地完成了分类任务|

![](./images/11.2.1.png)
![](./images/11.2.2.png)
![](./images/11.2.3.png)
![](./images/11.2.4.png)

## 11.3 分类样本不平衡问题
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

## 12.1 三层神经网络的实现

### 12.1.1 定义神经网络

为了完成MNIST分类，我们需要设计一个三层神经网络结构，如图12-2所示。

<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/12/nn3.png" ch="500" />

图12-2 三层神经网络结构
### 12.1.2 前向计算

我们都是用大写符号的矩阵形式的公式来描述，在每个矩阵符号的右上角是其形状。
### 12.1.3 反向传播

和以前的两层网络没有多大区别，只不过多了一层，而且用了tanh激活函数，目的是想把更多的梯度值回传，因为tanh函数比sigmoid函数稍微好一些，比如原点对称，零点梯度值大。

![](./images/12.1.1.png)
![](./images/12.1.2.png)

## 12.2 梯度检查

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

#### 泰勒公式

泰勒公式是将一个在$x=x_0$处具有n阶导数的函数$f(x)$利用关于$(x-x_0)$的n次多项式来逼近函数的方法。若函数$f(x)$在包含$x_0$的某个闭区间$[a,b]$上具有n阶导数，且在开区间$(a,b)$上具有$n+1$阶导数，则对闭区间$[a,b]$上任意一点$x$，下式成立：

$$f(x)=\frac{f(x_0)}{0!} + \frac{f'(x_0)}{1!}(x-x_0)+\frac{f''(x_0)}{2!}(x-x_0)^2 + ...+\frac{f^{(n)}(x_0)}{n!}(x-x_0)^n+R_n(x) \tag{3}$$

其中,$f^{(n)}(x)$表示$f(x)$的$n$阶导数，等号后的多项式称为函数$f(x)$在$x_0$处的泰勒展开式，剩余的$R_n(x)$是泰勒公式的余项，是$(x-x_0)^n$的高阶无穷小。 

### 12.2.4 算法实现

在神经网络中，我们假设使用多分类的交叉熵函数，则其形式为：

$$J(w,b) =- \sum_{i=1}^m \sum_{j=1}^n y_{ij} \ln a_{ij}$$

m是样本数，n是分类数。

## 12.3 学习率与批大小

普通梯度下降法，包含三种形式：

1. 单样本
2. 全批量样本
3. 小批量样本

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

![](./images/12.3.1.png)
![](./images/12.3.2.png)
![](./images/12.3.3.png)
![](./images/12.3.4.png)
![](./images/12.3.5.png)


# 深度神经网络
## 搭建深度神经网络框架

###  功能/模式分析
前向计算中：

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

###  抽象与设计

``迷你框架设计``
<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/14/class.png" />

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


### 搭建模型

``` 完成拟合任务的抽象模型```

<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/14/ch09_net.png" />

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

1. 输入层1个神经元，因为只有一个`x`值
2. 隐层4个神经元，对于此问题来说应该是足够了，因为特征很少
3. 输出层1个神经元，因为是拟合任务
4. 学习率=0.5
5. 最大`epoch=10000`轮
6. 批量样本数=10
7. 拟合网络类型
8. Xavier初始化
9. 绝对损失停止条件=0.001


### 训练结果   

```训练过程中损失函数值和准确率的变化```
<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/14/ch09_loss.png" />

```拟合结果```
<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/14/ch09_result.png" />

#### 数据字段解读

- id：唯一id
- date：售出日期
- price：售出价格（标签值）
- bedrooms：卧室数量
- bathrooms：浴室数量
- sqft_living：居住面积
- sqft_lot：停车场面积
- floors：楼层数
- waterfront：泳池
- view：有多少次看房记录
- condition：房屋状况
- grade：评级
- sqft_above：地面上的面积
- sqft_basement：地下室的面积
- yr_built：建筑年份
- yr_renovated：翻修年份
- zipcode：邮政编码
- lat：维度
- long：经度
- sqft_living15：2015年翻修后的居住面积
- sqft_lot15：2015年翻修后的停车场面积

``完成房价预测任务的抽象模型``
<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/14/non_linear_regression.png" />

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

1. 学习率=0.1
2. 最大`epoch=1000`
3. 批大小=16
4. 拟合网络
5. 初始化方法Xavier
6. 停止条件为相对误差`1e-7`

``训练过程中损失函数值和准确率的变化``
<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/14/house_loss.png" />

## 二分类任务功能测试

``完成非线性二分类教学案例的抽象模型``
<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/14/ch10_net.png" />

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

1. 输入层神经元数为2
2. 隐层的神经元数为3，使用Sigmoid激活函数
3. 由于是二分类任务，所以输出层只有一个神经元，用Logistic做二分类函数
4. 最多训练1000轮
5. 批大小=5
6. 学习率=0.1
7. 绝对误差停止条件=0.02


``训练过程中损失函数值和准确率的变化``
<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/14/ch10_loss.png" />

``分类效果``
<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/14/ch10_result.png" ch="500" />

## 模型
``完成非线性多分类教学案例的抽象模型``
<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/14/ch11_net_sigmoid.png" />

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

1. 隐层8个神经元
2. 最大`epoch=5000`
3. 批大小=10
4. 学习率0.1
5. 绝对误差停止条件=0.08
6. 多分类网络类型
7. 初始化方法为Xavier

`net.train()`函数是一个阻塞函数，只有当训练完毕后才返回。

### 运行结果
``训练过程中损失函数值和准确率的变化``
<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/14/ch11_loss_sigmoid.png" />

``分类效果图``
<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/14/ch11_result_sigmoid.png" ch="500" />

#### 模型
``使用ReLU函数抽象模型``
<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/14/ch11_net_relu.png" />

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

1. 隐层8个神经元
2. 最大`epoch=5000`
3. 批大小=10
4. 学习率0.1
5. 绝对误差停止条件=0.08
6. 多分类网络类型
7. 初始化方法为MSRA

### 运行结果
``训练过程中损失函数值和准确率的变化``
<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/14/ch11_loss_relu.png" />

``分类效果图``
<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/14/ch11_result_relu.png" ch="500" />

使用不同的激活函数的分类结果比较

|Sigmoid|ReLU|
|---|---|
|<img src='../Images/14/ch11_result_sigmoid.png'/>|<img src='../Images/14/ch11_result_relu.png'/>|

Relu能直则直，对方形边界适用；Sigmoid能弯则弯，对圆形边界适用。


# 网络优化

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

###  零初始化
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

### 15.1.2 标准初始化

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

``标准初始化在Sigmoid激活函数上的表现``
<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/15/init_normal_sigmoid.png" ch="500" />

###  Xavier初始化方法
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

``Xavier初始化在Sigmoid激活函数上的表现``
<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/15/init_xavier_sigmoid.png" ch="500" />

随机初始化和Xavier初始化的各层激活值与反向传播梯度比较

| |各层的激活值|各层的反向传播梯度|
|---|---|---|
| 随机初始化 |<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images\Images\15\forward_activation1.png"><br/>激活值分布渐渐集中|<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images\Images\15\backward_activation1.png"><br/>反向传播力度逐层衰退|
| Xavier初始化 |<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images\Images\15\forward_activation2.png"><br/>激活值分布均匀|<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images\Images\15\backward_activation2.png"><br/>反向传播力度保持不变|

``Xavier初始化在ReLU激活函数上的表现``
<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/15/init_xavier_relu.png" ch="500" />

###  MSRA初始化方法

MSRA初始化方法$^{[2]}$，又叫做He方法，因为作者姓何

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

``MSRA初始化在ReLU激活函数上的表现``
<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/15/init_msra_relu.png" ch="500" />

几种初始化方法的应用场景

|ID|网络深度|初始化方法|激活函数|说明|
|---|---|---|---|---|
|1|单层|零初始化|无|可以|
|2|双层|零初始化|Sigmoid|错误，不能进行正确的反向传播|
|3|双层|随机初始化|Sigmoid|可以|
|4|多层|随机初始化|Sigmoid|激活值分布成凹形，不利于反向传播|
|5|多层|Xavier初始化|Tanh|正确|
|6|多层|Xavier初始化|ReLU|激活值分布偏向0，不利于反向传播|
|7|多层|MSRA初始化|ReLU|正确|

### 随机梯度下降 SGD

``随机梯度下降算法的梯度搜索轨迹示意图``
<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/15/sgd_algorithm.png" />

学习率对SGD的影响

|学习率|损失函数与准确率|
|---|---|
|0.1|<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images\Images\15\op_sgd_ch09_loss_01.png">|
|0.3|<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images\Images\15\op_sgd_ch09_loss_03.png">|


###  动量算法 Momentum

``动量算法的前进方向``
<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/15/momentum_algorithm.png" />

#### 输入和参数
- $\eta$ - 全局学习率
- $\alpha$ - 动量参数，一般取值为0.5, 0.9, 0.99
- $v_t$ - 当前时刻的动量，初值为0

#### 实际效果

  SGD和动量法的比较

|算法|损失函数和准确率|
|---|---|
|SGD|<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images\Images\15\op_sgd_ch09_loss_01.png">|
|Momentum|<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images\Images\15\op_momentum_ch09_loss_01.png">|

###  梯度加速算法 NAG
Nesterov Accelerated Gradient，或者叫做Nesterov Momentum。
在小球向下滚动的过程中，我们希望小球能够提前知道在哪些地方坡面会上升，这样在遇到上升坡面之前，小球就开始减速。这方法就是Nesterov Momentum，其在凸优化中有较强的理论保证收敛。并且，在实践中Nesterov Momentum也比单纯的Momentum 的效果好。

#### 输入和参数

- $\eta$ - 全局学习率
- $\alpha$ - 动量参数，缺省取值0.9
- $v$ - 动量，初始值为0

动量法和NAG法的比较

|算法|损失函数和准确率|
|---|---|
|Momentum|<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images\Images\15\op_momentum_ch09_loss_01.png">|
|NAG|<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images\Images\15\op_nag_ch09_loss_01.png">|


###  AdaGrad

Adaptive subgradient method.$^{[1]}$

AdaGrad是一个基于梯度的优化算法，它的主要功能是：它对不同的参数调整学习率，具体而言，对低频出现的参数进行大的更新，对高频出现的参数进行小的更新。因此，他很适合于处理稀疏数据。

在这之前，我们对于所有的参数使用相同的学习率进行更新。但 Adagrad 则不然，对不同的训练迭代次数t，AdaGrad 对每个参数都有一个不同的学习率。这里开方、除法和乘法的运算都是按元素运算的。这些按元素运算使得目标函数自变量中每个元素都分别拥有自己的学习率。

AdaGrad算法的学习率设置

|初始学习率|损失函数值变化|
|---|---|
|eta=0.3|<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images\Images\15\op_adagrad_ch09_loss_03.png">|
|eta=0.5|<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images\Images\15\op_adagrad_ch09_loss_05.png">|
|eta=0.7|<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images\Images\15\op_adagrad_ch09_loss_07.png">|

### AdaDelta

Adaptive Learning Rate Method. $^{[2]}$

AdaDelta法是AdaGrad 法的一个延伸，它旨在解决它学习率不断单调下降的问题。相比计算之前所有梯度值的平方和，AdaDelta法仅计算在一个大小为w的时间区间内梯度值的累积和。

但该方法并不会存储之前梯度的平方值，而是将梯度值累积值按如下的方式递归地定义：关于过去梯度值的衰减均值，当前时间的梯度均值是基于过去梯度均值和当前梯度值平方的加权平均，其中是类似上述动量项的权值。

#### 输入和参数

- $\epsilon$ - 用于数值稳定的小常数，建议缺省值为1e-5
- $\alpha \in [0,1)$ - 衰减速率，建议0.9
- $s$ - 累积变量，初始值0
- $r$ - 累积变量变化量，初始为0


AdaDelta法的学习率设置

|初始学习率|损失函数值|
|---|---|
|eta=0.1|<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images\Images\15\op_adadelta_ch09_loss_01.png">|
|eta=0.01|<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images\Images\15\op_adadelta_ch09_loss_001.png">|

### 均方根反向传播 RMSProp

Root Mean Square Prop。$^{[3]}$

RMSprop 是由 Geoff Hinton 在他 Coursera 课程中提出的一种适应性学习率方法，至今仍未被公开发表。RMSprop法要解决AdaGrad的学习率缩减问题。


#### 输入和参数

- $\eta$ - 全局学习率，建议设置为0.001
- $\epsilon$ - 用于数值稳定的小常数，建议缺省值为1e-8
- $\alpha$ - 衰减速率，建议缺省取值0.9
- $r$ - 累积变量矩阵，与$\theta$尺寸相同，初始化为0


RMSProp的学习率设置

|初始学习率|损失函数值|
|---|---|
|eta=0.1|<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images\Images\15\op_rmsprop_ch09_loss_01.png">|<
||迭代了10000次，损失值一直在0.005下不来，说明初始学习率太高了，需要给一个小一些的初值|
|eta=0.01|<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images\Images\15\op_rmsprop_ch09_loss_001.png">|
||合适的学习率初值设置||
|eta=0.005|<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images\Images\15\op_rmsprop_ch09_loss_0005.png">|


###  Adam - Adaptive Moment Estimation

计算每个参数的自适应学习率，相当于RMSProp + Momentum的效果，Adam$^{[4]}$算法在RMSProp算法基础上对小批量随机梯度也做了指数加权移动平均。和AdaGrad算法、RMSProp算法以及AdaDelta算法一样，目标函数自变量中每个元素都分别拥有自己的学习率。

#### 输入和参数

- $t$ - 当前迭代次数
- $\eta$ - 全局学习率，建议缺省值为0.001
- $\epsilon$ - 用于数值稳定的小常数，建议缺省值为1e-8
- $\beta_1, \beta_2$ - 矩估计的指数衰减速率，$\in[0,1)$，建议缺省值分别为0.9和0.999


Adam法的学习率设置

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


###  模拟效果比较

``不同梯度下降优化算法的模拟比较``
<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/15/Optimizers_sample.png" ch="500" />

- SGD算法，每次迭代完全受当前梯度的控制，所以会以折线方式前进。
- Momentum算法，学习率只有0.1，每次继承上一次的动量方向，所以会以比较平滑的曲线方式前进，不会出现突然的转向。
- RMSProp算法，有历史梯度值参与做指数加权平均，所以可以看到比较平缓，不会波动太大，都后期步长越来越短也是符合学习规律的。
- Adam算法，因为可以被理解为Momentum和RMSProp的组合，所以比Momentum要平缓一些，比RMSProp要平滑一些。

### 真实效果比较

用`Python`代码实现的前向计算、反向计算、损失函数计算的函数：

```Python
def ForwardCalculationBatch(W,B,batch_x):
    Z = np.dot(W, batch_x) + B
    return Z

def BackPropagationBatch(batch_x, batch_y, batch_z):
    m = batch_x.shape[1]
    dZ = batch_z - batch_y
    dB = dZ.sum(axis=1, keepdims=True)/m
    dW = np.dot(dZ, batch_x.T)/m
    return dW, dB

def CheckLoss(W, B, X, Y):
    m = X.shape[1]
    Z = np.dot(W, X) + B
    LOSS = (Z - Y)**2
    loss = LOSS.sum()/m/2
    return loss
```

各种算法的效果比较

|||
|---|---|
|<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images\Images\15\op_sgd_ch04.png">|<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images\Images\15\op_sgd2_ch04.png">|
|SGD当学习率为0.1时，需要很多次迭代才能逐渐向中心靠近|SGD当学习率为0.5时，会比较快速地向中心靠近，但是在中心的附近有较大震荡|
|<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images\Images\15\op_momentum_ch04.png">|<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images\Images\15\op_nag_ch04.png">|
|Momentum由于惯性存在，一下子越过了中心点，但是很快就会得到纠正|Nag是Momentum的改进，有预判方向功能|
|<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images\Images\15\op_adagrad_ch04.png">|<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images\Images\15\op_adadelta_ch04.png">|
|AdaGrad的学习率在开始时可以设置大一些，因为会很快衰减|AdaDelta即使把学习率设置为0，也不会影响，因为有内置的学习率策略|
|<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images\Images\15\op_rmsprop_ch04.png">|<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images\Images\15\op_adam_ch04.png">|
|RMSProp解决AdaGrad的学习率下降问题，即使学习率设置为0.1，收敛也会快|Adam到达中点的路径比较直接|

``放大后各优化器的训练轨迹``
<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/15/Optimizers_zoom.png" ch="500" />

- SGD：接近中点的过程很曲折，步伐很慢，甚至有反方向的，容易陷入局部最优。
- Momentum：快速接近中点，但中间跳跃较大。
- RMSProp：接近中点很曲折，但是没有反方向的，用的步数比SGD少，跳动较大，有可能摆脱局部最优解的。
- Adam：快速接近中点，难怪很多人喜欢用这个优化器。


##  批量归一化的原理
###  深度神经网络的挑战

在深度神经网络中，我们可以将每一层视为对输入的信号做了一次变换：

$$
Z = W \cdot X + B \tag{5}
$$

``标准正态分布的数值密度占比``
<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/15/bn2.png" ch="500" />

``偏移后的数据分布区域和Sigmoid激活函数``
<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/15/bn3.png" ch="500" />

``ReLU函数曲线``
<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/15/bn4.png" ch="500" />

###  批量归一化

``数据处理过程``
<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/15/bn6.png" ch="500" />

###  前向计算
各个参数的含义和数据形状 

|符号|数据类型|数据形状|
|:---------:|:-----------:|:---------:|
|$X$| 输入数据矩阵 | [m, n] |
|$x_i$|输入数据第i个样本| [1, n] |
|$N$| 经过归一化的数据矩阵 | [m, n] |
|$n_i$| 经过归一化的单样本 | [1, n] |
|$\mu_B$| 批数据均值 | [1, n] |
|$\sigma^2_B$| 批数据方差 | [1, n] |
|$m$|批样本数量| [1] |
|$\gamma$|线性变换参数| [1, n] |
|$\beta$|线性变换参数| [1, n] |
|$Z$|线性变换后的矩阵| [1, n] |
|$z_i$|线性变换后的单样本| [1, n] |
|$\delta$| 反向传入的误差 | [m, n] |

#  正则化
正则化的英文为Regularization，用于防止过拟合。

###  拟合程度比较
``回归任务中的欠拟合、正确的拟合、过拟合``
<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/16/fitting.png" />

``分类任务中的欠拟合、正确的拟合、过拟合``
<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/16/classification.png" />

出现过拟合的原因：

1. 训练集的数量和模型的复杂度不匹配，样本数量级小于模型的参数
2. 训练集和测试集的特征分布不一致
3. 样本噪音大，使得神经网络学习到了噪音，正常样本的行为被抑制
4. 迭代次数过多，过分拟合了训练数据，包括噪音部分和一些非重要特征

既然模型过于复杂，那么我们简化模型不就行了吗？为什么要用复杂度不匹配的模型呢？有两个原因：

1. 因为有的模型以及非常成熟了，比如VGG16，可以不调参而直接用于你自己的数据训练，此时如果你的数据数量不够多，但是又想使用现有模型，就需要给模型加正则项了。
2. 使用相对复杂的模型，可以比较快速地使得网络训练收敛，以节省时间。


### 16.1.3 偏差-方差分解

 符号含义

|符号|含义|
|---|---|
|$x$|测试样本|
|$D$|数据集|
|$y$|x的真实标记|
|$y_D$|x在数据集中标记(可能有误差)|
|$f$|从数据集D学习的模型|
|$f_{x;D}$|从数据集D学习的模型对x的预测输出|
|$f_x$|模型f对x的期望预测输出|

学习算法期望的预测：
$$f_x=E[f_{x;D}] \tag{1}$$
不同的训练集/验证集产生的预测方差：
$$var(x)=E[(f_{x;D}-f_x)^2] \tag{2}$$
噪声：
$$\epsilon^2=E[(y_D-y)^2] \tag{3}$$
期望输出与真实标记的偏差：
$$bias^2(x)=(f_x-y)^2 \tag{4}$$
算法的期望泛化误差：

$$
\begin{aligned}
E(f;D)&=E[(f_{x;D}-y_D)^2]=E[(f_{x;D}-f_x+f_x-y_D)^2] \\\\
&=E[(f_{x;D}-f_x)^2]+E[(f_x-y_D)^2]+E[2(f_{x;D}-f_x)(f_x-y_D)]=E[(f_{x;D}-f_x)^2]+E[(f_x-y_D)^2] \\\\
&=E[(f_{x;D}-f_x)^2]+E[(f_x-y+y-y_D)^2]=E[(f_{x;D}-f_x)^2]+E[(f_x-y)^2]+E(y-y_D)^2]+E[2(f_x-y)(y-y_D)] \\\\
&=E[(f_{x;D}-f_x)^2]+(f_x-y)^2+E[(y-y_D)^2]=var(x) + bias^2(x) + \epsilon^2
\end{aligned}
$$

``训练过程中的偏差和方差变化``
<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/16/error.png" width="600" ch="500" />

###  理论基础

``损失函数值的等高线图``
<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/16/regular0.png" />

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

在`TrainingTrace`类中，增加以下成员以支持早停机制：

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


##  集成学习 Ensemble Learning

``集成学习的示意图``
<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/16/ensemble.png" ch="500" />

#### Individual Learner 个体学习器

如果所有的个体学习器都是同一类型的学习器，即同质模式，比如都用神经网路，称为“基学习器”（base learner），相应的学习算法称为“基学习算法”（base learning algorithm）。

在传统的机器学习中，个体学习器可以是不同的，比如用决策树、支持向量机等，此时称为异质模式。

#### Aggregator 结合模块

个体学习器的输出，通过一定的结合策略，在结合模块中有机结合在一起，可以形成一个能力较强的学习器，所以有时称为强学习器，而相应地称个体学习器为弱学习器。

###  Bagging法集成学习的基本流程

``Bagging集成学习示意图``
<img src="https://aiedugithub4a2.blob.core.windows.net/a2-images/Images/16/bagging.png" />

1. 首先是数据集的使用，采用自助采样法（Bootstrap Sampling）。假设原始数据集Training Set中有1000个样本，我们从中随机取一个样本的拷贝放到Training Set-1中，此样本不会从原始数据集中被删除，原始数据集中还有1000个样本，而不是999个，这样下次再随机取样本时，此样本还有可能被再次选到。如此重复m次（此例m=1000），我们可以生成Training Set-1。一共重复N次（此例N=9），可以得到N个数据集。
2. 然后搭建一个神经网络模型，可以参数相同。在N个数据集上，训练出N个模型来。
3. 最后再进入Aggregator。N值不能太小，否则无法提供差异化的模型，也不能太大而带来训练模型的时间花销，一般来说取5到10就能满足要求。


###  生成数据集

```Python
def GenerateDataSet(count=9):
    mdr = MnistImageDataReader(train_image_file, train_label_file, test_image_file, test_label_file, "vector")
    mdr.ReadLessData(1000)
    
    for i in range(count):
        X = np.zeros_like(mdr.XTrainRaw)
        Y = np.zeros_like(mdr.YTrainRaw)
        list = np.random.choice(1000,1000)
        k=0
        for j in list:
            X[k] = mdr.XTrainRaw[j]
            Y[k] = mdr.YTrainRaw[j]
            k = k+1
        # end for
        np.savez("level6_" + str(i)+".npz", data=X, label=Y)
    # end for
```
在上面的代码中，我们假设只有1000个手写数据样本，用`np.random.choice(1000,1000)`函数来可重复地选取1000个数字，分别取出对应的图像数据X和标签数据Y，命名为`level6_N.npz`，`N=[1,9]`，保存到9个`npz`文件中。

假设数据集中有m个样本，这样的采样方法，某个样本在第一次被选择的概率是1/m，那么不被选择的概率就是1-1/m，则选择m次后，不被采样到的概率是$(1-\frac{1}{m})^m$，取极限值：

$$\lim_{m \rightarrow \infty} (1-\frac{1}{m})^m \simeq \frac{1}{e} = 0.368$$

即，对于一个新生成的训练数据集来说，原始数据集中的样本没有被使用的概率为36.8%。