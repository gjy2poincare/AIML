#课堂笔记

##一 环境配置

* 安装git和用git clone克隆远程库
  git clone + 网站名
  ![avatar](https://note.youdao.com/yws/api/personal/file/WEB7d6e16da6cd10aff932bfd9daf5072e8?method=download&shareKey=d58b3427d05f1021f08333e2009752bb)
  
* 安装python
  可用powershell或cmd打开
* 在vscode中安装python插件
  pip install
* 利用github提交作业
  1. 首先fork老师的库
  2. 下载Github Desktop
  3. 与网站上的Github绑定

##二 Step1及慕课学习

### 权重的确定：

数值型转换，通过训练神经网络，拟合样本，调整权重。

利用激活函数来实现从输入到输出的任意非线性映射。

激活函数又为映射函数，即f。

### 训练方法：

**批处理，先前向计算。**

输入层相当于末梢神经不处理，输入层的输入和输出相等。
![avatar](https://note.youdao.com/yws/api/personal/file/WEBf088fd30894c13385c91ea7b5541876d?method=download&shareKey=91b58bae74f54e25f98edd78303edf80)

最后y1，y2，y3为真实输出。

因为计算输出与真实输出的不同，将他们差的平方定义为损失函数或为代价函数。

将所有样本差值的平方加起来。

反馈：向后计算调整权重偏置，使误差减小。 用迭代的方法：偏导数为0，梯度下降最快的反方向。


上次权重加上负梯度乘以步长。
![avatar](https://note.youdao.com/yws/api/personal/file/WEBeadb4561a1944df958971c79f0204691?method=download&shareKey=89ac151ffb82f73393b59890fbbf259b)

整个过程为批处理训练方式。批量训练。

随机梯度下降：
随机的样本进行epoch。

批处理出现梯度消失问题。

![avatar](https://note.youdao.com/yws/api/personal/file/WEBc8afcaaa88fa2bedf37fb72cea74d0c8?method=download&shareKey=2f0830d920c8091625e47482813502ce)
纵坐标是损失函数值，横坐标是变量。不断地改变变量的值，会造成损失函数值的上升或下降。而梯度下降算法会让计算沿着损失函数值下降的方向前进。


source liner 代码
x=2*w+3*b
factor_b=2*x+3*y，z对b作偏导

##三 Step1代码理解
**1. 线性反向传播**
  * **def single_variable(w,b,t):
    print("\nsingle variable new: b ----- ")**
    单变量只将b作为变量。
  * **factor_b = 2*x+3*y
    delta_b = delta_z/factor_b**
    在第二个单变量实验中，每次计算b的贡献值改变，可以发现实验二所花步骤更少。

  * **def double_variable(w,b,t):
    print("\ndouble variable: w, b -----")**
    双变量将b和w作为变量
        ![avatar](https://note.youdao.com/yws/api/personal/file/WEB6aad9090c72e0ec1a693a33c893bc8d7?method=download&shareKey=956c894033ead794807de158c17c16af)
  * **delta_b = delta_z/factor_b/2
        delta_w = delta_z/factor_w/2**
    同理，在第二个实验中每次迭代计算贡献值的改变，注意把这个误差的一半算在w上，另外一半算在b上。
**结论：** 通过偏导计算得出了反向与正向的解，之后发现，当在每次迭代中重复计算改变变量的贡献值，迭代的次数会变少，效率更高。

**2. 非线性反向传播**
  * 与线性反向传播运用while循环不同，非线性反向传播中，定义forword，backword函数分别进行前向和反向的传播，之后利用for循环，和append函数建立了一个多维数组进行反复迭代。
  * x和d的曲线为绿色，反映出贡献值的变化。
    x和y的曲线为蓝色，橙色箭头反映出误差逐渐减小，逼近精确的1.8值
    ![avatar](https://note.youdao.com/yws/api/personal/file/WEBdd82d236ef154c958321c4be198264e4?method=download&shareKey=62b6c670d6380cf9dc88e3db3adc4ce8)
    plt.plot(X,Y,'x')
    d = 1/(x*np.sqrt(np.log(x**2)))
    plt.plot(x,d)
    
**3. 单变量梯度下降**
  * 梯度：函数当前位置的最快上升点
  * 下降：与导数相反的方向
  * 梯度下降的目的：使得x值向极值点逼近
    ![avatar](https://note.youdao.com/yws/api/personal/file/WEB670ddefff705a1a3ead4d8c6ad0d55da?method=download&shareKey=b14a427b415364d88d3ad025cb0abfa0) 
  * 图形中，可观察到橙线向左逐渐逼近极值。
    ![avatar](https://note.youdao.com/yws/api/personal/file/WEB930127ecb90ac1df2b25e66340b485f6?method=download&shareKey=742929427945dbb35f3b6bbde551b61d)
    x = x - eta * derivative_function(x)

**4. 双变量梯度下降**
  * 双变量梯度下降与单变量方法类似，利用偏导分别计算x和y，再利用梯度下降的公式，求出变化值，之后代入原式求解。
  * J = x**2 + np.sin(y)**2
    return np.array([2*x,2*np.sin(y)*np.cos(y)])
    R[i, j] = X[i, j]**2 + np.sin(Y[i, j])**2
    ![avatar](https://note.youdao.com/yws/api/personal/file/WEB61c80c7a511f83af2e77695d9c83813b?method=download&shareKey=9dec894f375f10782ae32d5798931e3e)
  * 可以观察到，在图形中，x与y沿着黑线逐渐逼近极值。

**5. 学习率**
  * 学习率是一个步长，控制好的学习率可以提高迭代的效率。
    在代码中分别将eta取多值进行比较，可以明显看出学习率的改变对实验过程造成的影响。
  * 由图可看出，学习率过大或过小都不好，导致计算结果跳跃过大和迭代的增加。
    ![avatar](https://note.youdao.com/yws/api/personal/file/WEB14c977bd7c6b372f3b8d4587cf81b4f6?method=download&shareKey=305f23edf2e6c0a70e9853d880b673b0)
    ![avatar](https://note.youdao.com/yws/api/personal/file/WEBad956ee0ea3b538839ee2e9cad21b0e8?method=download&shareKey=af1df1fb75b8a6f36aa7a990dcf65b73)


**6. 损失函数**
  * 损失：所有样本的“误差”的总和。
  * 损失函数的作用：计算神经网络每次迭代的前向计算结果与真实值的差距，从而指导下一步的训练向正确的方向进行。
     1. 均方差损失函数（MSE）：主要用于回归
        作用：对某些偏离大的样本比较敏感，从而引起监督训练过程的足够重视，以便回传误差。

     2. 交叉熵函数，主要用于分类。
        作用：用于度量两个概率分布间的差异性信息。信息论中，交叉熵是表示两个概率分布p,q的差异，其中p表示真实分布，q表示预测分布，那么H(p,q)就称为交叉熵。
        图示为二分类交叉熵：y=1 意味着当前样本标签值是1，当预测输出越接近1时，损失函数值越小，训练结果越准确。
      ![avatar](https://note.youdao.com/yws/api/personal/file/WEB750c833408a3ea9272c991e40671cbcb?method=download&shareKey=e67f998bb971dcd61cc40ae1181cf65f)

##四 心得体会
在本次学习中，我对于神经网络有了初步的理解，神经网络给我带来的高级和神秘感增加了我想探索的欲望，Step1主要告诉了我神经网络的一些基本概念，同时让我深刻领会了BP反向传播的含义，梯度下降算法，损失函数的作用，相信通过后面的学习可以让我对神经网络有更多的认识。
   

  


