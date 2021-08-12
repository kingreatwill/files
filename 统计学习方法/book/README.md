# 统计学习方法

[第一版](./Lihang-first_edition)

[第二版](./Lihang-second_edition)

## 第1章 统计学习及监督学习概论


统计学习的主要特点是：
1. 统计学习以计算机及网络为平台，是建立在计算机及网络之上的；
2. 统计学习以数据为研究对象，是数据驱动的学科；
3. 统计学习的目的是对数据进行预测与分析；
4. 统计学习以方法为中心，统计学习方法构建模型并应用模型进行预测与分析；
5. 统计学习是概率论、统计学、信息论、计算理论、最优化理论及计算机科学等多个领域的交叉学科，并且在发展中逐步形成独自的理论体系与方法论。

统计学习的三要素：
1. 模型的假设空间(hypothesis space)，简称：模型(model)
2. 模型选择的准则(evaluation criterion)，简称：策略(strategy)或者学习准则
2. 模型学习的算法(algorithm)，简称：算法(algorithm)

假设空间(hypothesis space)：
$$\mathcal H = \{ f(x;\theta) | \theta \in \mathbb{R}^D\}$$
其中$f(x; \theta)$是参数为$\theta$ 的函数，也称为模型（Model），$D$ 为参数的数量．

以线性回归（Linear Regression）为例：
模型： $f(x;w,b) = w^Tx +b$
策略(strategy)或者学习准则: 平方损失函数 $\mathcal L(y,\hat{y}) = (y-f(x,\theta))^2$
算法：也称为优化算法，如：梯度下降法


机器学习的定义：
```mermaid
graph LR;
    F(["未知的目标函数(理想中完美的函数)：𝑓: 𝒙⟶𝑦"])-->D["训练样本D:{(𝒙¹,𝑦¹),...,(𝒙ⁿ,𝑦ⁿ)}"];
    D-->A{{"算法"}}
    H{{"假设空间"}}-->A
    A-->G["模型 g≈f"]
```
使用训练数据来计算接近目标𝑓的假设（hypothesis ）g [^1]

[^1]:[Machine Learning Foundations,25页](https://www.csie.ntu.edu.tw/~htlin/course/mlfound17fall/doc/01_handout.pdf)