# 非线性分类

## 二分类

- **评估标准**：
    
    - 准确率：$Accuracy = (正例数 + 负例数)/总数$
    - 精准率：$Precision = \frac{TP}{TP+FP}$
    - 召回率：$Recall = \frac{TP}{TP+FN}$
    - 调和平均值：$F_1 = \frac{2 \times Precision \times Recall}{recision + Recall}$
    <font color="RED">注</font>：`T,N为原正负|P,N为判正负`
    - ROC 曲线：` 曲线的积分面积越大分类性能越强`
    - Kappa值：`反应了两个被观察对象的一致程度`
      
      - $K = \frac{p_o-p_e}{1-p_e}$
    <font color="RED">注</font>： $p_o$是正确样本数量除以总样本数
    $p_e$是预期偶然一致达成的比率
- **实现过程**:

    - 准备数据
    - 定义神经元
    - 前向计算`使用矩阵计算`
    - 反向传播`使用上一层的输出(一般是a)和本层的dZ进行运算且dZ可以联合本层的dA求出`
    - 测试函数
    - 可视化`可以清楚的看到神经网络对样本空间的变换(由低到高)`
 <font color="RED">PS</font>：感觉在训练的时候反向传播的过程都像是在作假设推理并企图用线性*的方式去解答。   *(x_x; )*
 <font color="RED">注</font>：隐层神经元的个数越多，功能越强大。
 
 ## 多分类`双`

 - **评估标准**:`混淆矩阵对角线上的值越大越好,AUC会更好`
 - **样本不平衡**:`通常指数据类别的样本数量差距太大`
 
    - 大样本删除，小样本增加
    - 改进采样方法
    - 尝试不同采样比例
    - 分析采样效果
<font color="RED">注</font>:
    <img src = "https://img-blog.csdn.net/20180628173940759?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0Rhd2VpXzAx/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70"/>
    
    - 朴素贝叶斯方法：`属性联系`
        $P(A|B) = \frac{P(B|A)P(A)}{P(B)}$
    - SMOTE：
      
      - 在样本集中随机选取一个样本$x_i$
      - 在样本集中将距离$x_i$的k个近邻$x_{inn}$归入一个新的集合
      - 随机选取$x_{inn}$和一个随机数$(0,1)\varsigma_i$
      - $x_i = x_i + \varsigma_i(x_{inn}-x_i)$
      - 循环有线次便可增加样本
    - 异常点监测

        - 标准差法
        - 画箱型图
        - DBScan --基于密度的聚类
        - 孤立森林（Isolation Forest）
        - Robust Random Cut Forest`随机森林变体`
    - 变化趋势检测 ：`检测不寻常的变化趋势`
    - 随机梯度下降
    - `聚类算法`
 <font color="RED">注</font>:以上方法破坏数据分布且可以对类别不平衡进行改进
    - 集成学习
      
      - 使用原始的数据集去训练学习器L1
      - 将L1的学习成果中抽取正确和不正确的各50%给训练器L2
      - 将对L1,L2都不相同的样本给L3训练
      - 结合L1,L2,L3投票来进行输出 
  
  ## 多分类`三`
  
  `将图像输入全连接网络是要转换成向量并归一化`
  
  - **梯度检查**：
    
    - 单边逼近误差
        $f'(\theta)+O(h) = \frac{f(\theta+h)-f(\theta)}{h}$
    - 双边逼近误差
     $f'(\theta)+O(h^2) = \frac{f(\theta+h)-f(\theta-h)}{2h}$
     - 欧式距离
    $diff = \frac{||d\theta_r -d\theta_a||_2}{||d\theta_a||_2+||d\theta_r||_2}$
    <font color="RED">注</font>:
    1.不要用于训练
    2.梯度检验的过程中出错了，则需要对所有参数进行计算，来判断出错范围
    3.要正则化`若正则化为drop-out要事先用梯度检验然后在把概率调小`
    4.按梯度改变趋势来检验

  - **学习率与批大小**

    - 学习率初始化条件： $\sum\eta_k = \infty and \sum{\eta_k}^2 <\infty$
    - 初始时学习率大，过一会儿学习率下降
    
      - `step`:$Lr_n = Lr_c * Dr^\frac{GS}{DS}$
      - `fixed`：固定学习率
      - `multistep`: 不均匀迭代
      - `exp`:$$lr_{new}=lr_{base} * \gamma^{iteration}  $$
      - `inv`:$lr_{new}=lr_{base} * \frac{1}{( 1 + \gamma * iteration)^{p}}  $
      - `poly`:$lr_{new}=lr_{base} * (1 - {iteration \over iteration_{max}})^p  $
  - 大批量对应大的学习率，小批量对应小的学习率，其大小在(0,1)中。
    - 到一定程度，学习率的增加会缩小，变成批大小的$\sqrt m$倍
    - 到了比较极端的程度，无论批大小再怎么增加，也不能增加学习率了