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

### 参考文献
[1] Hastie  T,Tibshirani  R,Friedman  J.  [The  Elements  of  Statistical  Learning:  DataMining,Inference,and Prediction](http://www.web.stanford.edu/~hastie/ElemStatLearn/printings/ESLII_print12_toc.pdf). Springer. 2001（中译本：统计学习基础——数据挖掘、推理与预测。范明，柴玉梅，昝红英等译。北京：电子工业出版社，2004）

[2] Bishop M. [Pattern Recognition and Machine Learning](https://www.microsoft.com/en-us/research/uploads/prod/2006/01/Bishop-Pattern-Recognition-and-Machine-Learning-2006.pdf). Springer,2006

[3] [Probabilistic Graphical Models: Principles and Techniques](https://djsaunde.github.io/read/books/pdfs/probabilistic%20graphical%20models.pdf) by Daphne Koller, Nir Friedman from The MIT Press

[4] [Deep Learning](https://raw.fastgit.org/Zhenye-Na/machine-learning-uiuc/master/docs/Deep%20Learning.pdf) (Ian Goodfellow, Yoshua Bengio, Aaron Courville)

[5] Tom M Michelle. [Machine Learning](https://www.cs.cmu.edu/afs/cs.cmu.edu/user/mitchell/ftp/mlbook.html). McGraw-Hill Companies,Inc. 1997（中译本：机器学习。北京：机械工业出版社，2003）

[6] [Bayesian Reasoning and Machine Learning by David Barber 2007–2020](http://web4.cs.ucl.ac.uk/staff/D.Barber/textbook/200620.pdf) ,[other version](http://web4.cs.ucl.ac.uk/staff/D.Barber/textbook/)

[7] [Reinforcement Learning:An Introduction (second edition 2020) by Richard S. Sutton and Andrew G. Barto](http://incompleteideas.net/book/RLbook2020trimmed.pdf) ,[other version](http://incompleteideas.net/book/)

[8] 周志华，机器学习，清华大学出版社

[9] [Lecture Notes in MACHINE LEARNING](https://news.vidyaacademy.ac.in/wp-content/uploads/2018/10/NotesOnMachineLearningForBTech-1.pdf)  Dr V N Krishnachandran

## 第1章 统计学习及监督学习概论