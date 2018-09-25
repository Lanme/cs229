## 1、方差和偏差

高方差：过拟合

Variance反映的是模型每一次输出结果与模型输出期望之间的误差

高偏差：欠拟合

Bias反映的是模型在样本上的输出与真实值之间的误差

![1](https://github.com/Lanme/cs229/raw/master/add_note/img/add_notes4_1.png))

## 2、样本量与误差

$$
\hat{\epsilon} (h_i) = \frac{1}{m} \sum_{j=1} ^m Z_j \\
Z_j = 1\{h_i (x^{(j)} \neq\ y^{(j)} \}
$$

$\hat{\epsilon} (h_i)$为训练误差，表示错误分类的概率，即$Z$的期望值，$Z_j$是服从伯努利分布的独立随机变量。
$$
P（|\epsilon(h_i)- \hat{\epsilon}(h_i)>\gamma） \leq 2 \exp(-2 \gamma^2 m)
$$
对于我们给定的某个固定的$h_i$，假如训练样本的规模 m 规模很大的时候，训练误差有很接近泛化误差（generalization error）的概率是很高的。

## 3、VC维

VC维表示能被 H 打散（shatter）的最大的集合规模。

要保证使用 假设集合 H 的 机器学习的算法的学习效果良好（well），那么训练集样本规模 m 需要与 H 的 VC 维度 线性相关（linear in the VC dimension of H）。这也表明，对于绝大多数（most）假设类来说，VC 维度也大概会和参数的个数线性相关。把这些综合到一起，我们就能得出这样的一个结论：对于一个试图将训练误差最小化的学习算法来说：训练样本个数 通常都大概与假设类 H 的参数个数 线性相关。

d = VC(H)，对于所有的 h∈H，都有至少为 1−δ 的概率使下面的关系成立：
$$
|\epsilon(h)- \hat{\epsilon}(h)| \leq 
O(\sqrt{\frac{d}{m} \log \frac{m}{d} + \frac{1}{m} \log \frac{1}{\delta}})
$$
