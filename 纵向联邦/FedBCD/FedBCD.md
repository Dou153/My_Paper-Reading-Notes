# FedBCD: A Communication-Efficient Collaborative Learning Framework for Distributed Features

当使用小批量梯度下降的时候，训练一个模型可能需要几千甚至几万轮迭代，这个时候就会产生大量的通信开销，Fedavg可以让每个客户端在本地训练多轮来减少通信开销。在纵向联邦上执行这种局部更新策略是否可行尚不清楚。



**主要贡献：**本文提出了一种用于分布式特征的协作学习框架，称为联邦随机块坐标下降（FedBCD），其中各方在每次通信期间仅共享模型参数和每个样本的原始数据的内积，并且可以连续执行局部模型更新（以并行或顺序方式），而无需每次迭代通信。





## 块坐标下降



### 坐标下降法

坐标下降法是一种非梯度的优化方法，和梯度下降方法不同，**它每次沿着单个维度方向进行搜索**，当得到一个当前维度最小值之后再循环使用不同的维度方向，最终收敛得到最优解。





### 块坐标下降

块坐标下降是坐标下降的更一般化，它通过对变量的**子集**进行同时优化，把原问题分解为多个子问题。在下降的过程中更新的次序**可以是确定或随机的**，如果是确定的次序，我们可以设计某种方式，或是周期或是贪心的方法选择更新子集。




## 分布式块坐标下降

论文：”A distributed block coordinate descent method for training l1regularized linear classifiers“

适用于**L1正则化**分类器并行训练的分布式块坐标下降（DBCD）方法，降低通信量

参数集合：$\mathbf{w}=\{w_j\}_{j=1}^m$。数据集有$n$个样本，每个样本是$m$维的，$X_{n\times m}$表示样本矩阵。

线性分类器的输出：$y_i=w^Tx_i$，标签$c_i \in\{1,-1\}$，$l()$是损失函数

我们假设$l()$满足non-negative and convex（非负凸）、continuously differentiable（连续可微）、Lipschitz continuous。最小二乘损失、逻辑损失、SVM平方铰链损失和Huber损失等损失函数满足这些假设。

我们假设使用均方误差损失函数
$$
l(y_i;c_i)=max\{0,1-c_iy_i\}^2\\
f(w)=\frac{1}{n}\sum_il(y_i;c_i)	 	\\
u(w)=\lambda\sum_j|w_j|			\\
\min_{w\in R^m}F(w)=f(w)+u(w).
$$
我们将这$m$个特征分给$P$个人，用$\{B_p\}_{p=1}^P$来表示这m个特征：$M=\{1,\cdots,m\}$



### 经典的训练流程

![image-20250513092510615](E:\论文\论文阅读笔记\My_Paper-Reading-Notes\纵向联邦\FedBCD\FedBCD.assets\image-20250513092510615.png)











## 参数定义

$K$个数据方，$N$个数据样本$D=\{\xi_i\}_{i=1}^n$，样本由特征和标签组成$\xi=(X,y)$，特征是$d$维的，分布在K个数据方中$\mathcal{D}_K \triangleq\left\{\mathbf{x}_{i, K}, y_{i, K}\right\}_{i=1}^N$,假设数据标签在第$K$个用户手中.



协作训练问题描述如下：

$$
\min _{\Theta} \mathcal{L}(\Theta ; \mathcal{D}) \triangleq \frac{1}{N} \sum_{i=1}^N f\left(\theta_1, \ldots, \theta_K ; \xi_i\right)+\lambda \sum_{k=1}^K \gamma\left(\theta_k\right)	\tag{1}
$$
其中$\theta_k \in \mathbb{R}^{d_k}$标识第$k$方的训练参数，$\gamma()$正则化。



## FedBCD算法



### 普通纵向联邦SGD算法

如果对$S$个数据点的小批量$S\subset D$进行采样，则第k个用户的随机部分梯度由下式给出：

$$
g_k(\Theta ; \mathcal{S}) \triangleq \nabla_k f(\Theta ; \mathcal{S})+\lambda \nabla \gamma\left(\theta_k\right)	\tag{2}
$$
对于线性回归、逻辑回归或SVM等模型，预测值$H_i$由各数据方的局部线性组合之和构成：

$$
H_i \triangleq \sum_{k=1}^K H_i^k \triangleq \sum_{k=1}^K\mathbf{x}_{i, k} \theta_k	\tag{3}
$$
对于线性\逻辑回归和SVM来说,

$$
\nabla_k f(\Theta ; \mathcal{S})=\frac{1}{S} \sum_{\xi_i \in \mathcal{S}} \frac{\partial f\left(H_i, y_{i, K}\right)}{\partial H_i}\left(\mathbf{x}_{i, k}\right)^T	\tag{4}
$$
梯度推导过程:

- 链式法则: 对第k方的参数计算梯度:$\frac{\partial f}{\partial \theta_k}=\frac{\partial f}{\partial H_i} \cdot \frac{\partial H_i}{\partial \theta_k}$

- 梯度分量计算: $\frac{\partial f}{\partial H_i}$和损失函数形式有关，对于线性回归、逻辑回归或SVM等模型$\frac{\partial H_i}{\partial \theta_k}=x_{i,k}$
- 小批量平均：最后求平均

要计算梯度，需要其他$K-1$个人给第$k$个人发送自己的本地预测$H_{i,k}$，由第$k$个人算出公式（4）中的求和符号中的第一项，然后发给每个人计算自己的梯度





对于一个任意的损失函数，用户$k$把计算$\nabla_k f(\Theta ; \mathcal{S})$需要的整体信息记为：
$$
I_{\mathcal{S}}^{-k} \triangleq\left\{I_{\mathcal{S}}^{q, k}\right\}_{q \neq k}	\tag{4}
$$
第$k$个用户的梯度可以写为：
$$
\begin{aligned} g_k(\Theta ; \mathcal{S}) & =\nabla_k f\left(I_{\mathcal{S}}^{-k}, \theta_k ; \mathcal{S}\right)+\lambda \nabla \gamma\left(\theta_k\right) \\ & \triangleq g_k\left(I_{\mathcal{S}}^{-k}, \theta_k ; \mathcal{S}\right)\end{aligned}	\tag{5}
$$
整体的梯度也可以表示为：
$$
g({\Theta};\mathcal{S})\triangleq[g_{1}(I_{\mathcal{S}}^{-1},\theta_{1}; \mathcal{S});\cdots;g_{K}(I_{\mathcal{S}}^{-K},\theta_{K};\mathcal{S})].
$$
然后就可以用梯度下降训练了。





### 并行FedBCD和顺序FedBCD

![image-20250512110736006](E:\论文\论文阅读笔记\My_Paper-Reading-Notes\纵向联邦\FedBCD\FedBCD.assets\image-20250512110736006.png)

![1](E:\论文\论文阅读笔记\My_Paper-Reading-Notes\纵向联邦\FedBCD\FedBCD.assets\1.png)

**在本地进行更新迭代时，由于本地无法单独进行梯度下降，因此只能使用相同的陈旧小批次进行梯度下降。**



### 本地迭代轮次Q的影响

实验结果表明FedBCD-p在本地迭代轮次**Q=15**时，效果最好。文献【1】建议在局部迭代较大时，向局部目标函数添加近端项，以减轻潜在的发散。
$$
g_{k} \left( y_{k}^{r}; \xi_{i} \right)=g_{k} \left( \left[ \Theta_{-k}^{r_{0}}, \theta_{k}^{r} \right]; \xi_{i} \right])+ \mu \left( \theta_{k}^{r}- \theta_{k}^{r_{0}} \right)
$$
$r_0$表示$r$轮的前一轮，$r$轮完成了同步并交换中间信息。$y^r_k$表示节点$k$在$r$轮用于计算局部梯度的局部向量。$\mu \left( \theta_{k}^{r}- \theta_{k}^{r_{0}} \right)$就是近端项。



[1] T. Li, A. K. Sahu, M. Zaheer, M. Sanjabi, A. Talwalkar, and V. Smith,  “Federated optimization in heterogeneous networks,” in Proc. Mach. Learn Syst., vol. 2, pp. 429–450, 2020.



### 客户端数量K的影响

影响不大



