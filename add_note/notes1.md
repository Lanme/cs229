### 1、关于矩阵求导

#### 1.1 基本公式

$$
\frac{\partial\beta^TX}{\partial X}=\beta
$$

如Y = 3$x_1$+2$x_2$+2 即A = $\left[\begin{matrix}3&2&2\end{matrix}\right]$,X = $\left[\begin{matrix}x_1 \\ x_2 \\ x_3 \end{matrix}\right]$,$\frac{DY}{DX}$ = $\left[\begin{matrix}\partial Y/\partial x_1\\\partial Y/\partial x_2\\ \partial Y/\partial x_3\end{matrix}\right]$=$\left[\begin{matrix}3\\2\\2\end{matrix}\right]$

#### 1.2 最小二乘法的参数求解

参考：https://en.wikipedia.org/wiki/Matrix_calculus#Scalar-by-vector_identities
$$
J(\theta) = \frac{1}{2}(h_{\theta}(x)-y)\\
=\frac{1}{2}(X\theta-Y)^T(X\theta-Y)\\
=\frac{1}{2}(\theta^TX^T-Y^T)(X\theta-Y)\\
=\frac{1}{2}(\theta^TX^TX\theta-\theta^TX^TY-Y^TX\theta+Y^TY)
$$
求导
$$
\frac{\partial Y^TY}{\partial\theta} = 0
$$

$$
\frac{\partial \theta^TX^TY}{\partial \theta} = \frac{\partial (\theta^TX^TY)^T}{\partial \theta} = \frac{\partial Y^TX\theta}{\partial \theta}=X^TY
$$

因为分子是标量，标量的转置等于本身，所以对分子进行转置操作时等价的。
$$
\frac{\partial Y^TX\theta}{\partial \theta} = X^TY
$$

$$
\frac{\partial \theta^TX^TX\theta}{\partial \theta} =\frac{\partial （\theta^TX^T）^T}{\partial \theta}\cdot(X\theta)+\frac{\partial X\theta}{\partial \theta}\cdot（\theta^TX^T）^T = 2X^TX\theta
$$

即
$$
\frac{\partial J（\theta）}{\partial \theta} =\frac{1}{2}(2X^TX\theta-2X^TY) = 0
$$
解得
$$
\theta =(X^TX)^{-1}X^TY
$$

## 2、回归问题解决分类？



![1](https://github.com/Lanme/cs229/master/add_note/img/add_notes1_1.png)

一般来说，回归不用在分类问题上，因为回归是连续型模型，而且受噪声影响比较大。