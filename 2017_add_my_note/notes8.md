## 1、EM

- EM是利用极大似然进行参数估计的一种算法，因为存在隐变量，所以极大似然估计$\log (Z,X|\theta)$无法直接求解，所以引入Q函数。

- 坐标上升法

  根据Jensen不等式，$l(\theta) >=J(Q,\theta)$
  $$
  J(\theta) =\sum_{i=1}^m \sum_{z^{(i)}} Q_i (z^{(i)}) \log \frac{p(x^{(i)},z^{(i)};\theta)}{Q_i(z^{(i)})}
  $$

  $$
 l(\theta) = \sum_{i=1}^m \log p(x;\theta) \\
= \sum_{i=1}^m \log \sum_{z^{(i)}} p(x^{(i)},z^{(i)};\theta) \\
= \sum_{i=1}^m \log \sum_{z^{(i)}} Q_i(z^{(i)}) \frac{p(x^{(i)},z^{(i)};\theta)}{Q_i(z^{(i)})} \\
\geq J(\theta)
  $$

  EM can also be viewed a coordinate ascent on J,in which the E-step maximizes it with respect on Q,and the M-step maximizes it with respect to $\theta$.

  EM 算法也可看作是在 J 上的坐标上升（coordinate ascent），其中 E 步骤在 Q 上对 J 进行了最大化，然后 M 步骤则在 θ 上对 J 进行最大化。

  ![1](/img/add_notes8_1.png)

  此时$l(\theta)$为$p(x|\theta)$，$r(x|\theta)$即$J(\theta)$。令$r(x|\theta_1) = p(x|\theta_1)$ (E-step)，然后对$r(x|\theta)$求最大化得到新的$\theta = \theta_2$ （M-step），将$p(x|\theta)$从$\theta_1$移动到$\theta_2$。
