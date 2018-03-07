## 学习资料1
https://muxuezi.github.io/posts/fitting-a-line-through-data.html

## 学习代码
fitting-a-line-through-data.py

## 学习心得

* 一个特征值矩阵 + 预测结果集合进行拟合，再对新数据进行处理
* 问题是这个拟合对象怎么保存，以用于后续其他数据，不能每次运行都训练一次，解决方案可以是 python的持久化数据库 persistent

## How it works...
线性回归的基本理念是找出满足 y=Xβ 的相关系数集合 β ，其中 X 是因变量数据矩阵。想找一组完全能够满足等式的相关系数很难，因此通常会增加一个误差项表示不精确程度或测量误差。因此，方程就变成了 y=Xβ+ϵ ，其中 ϵ 被认为是服从正态分布且与 X 独立的随机变量。用几何学的观点描述，就是说这个变量与 X 是正交的（perpendicular）。这超出了本书的范围，可以参考其他信息证明 E(Xϵ)=0 。

为了找到相关系数集合 β ，我们最小化误差项，这转化成了残差平方和最小化问题。

这个问题可以用解析方法解决，其解是 β=(XTX)−1XTy^

## 学习资料2
https://muxuezi.github.io/posts/2-linear-regression.html
http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/

## 学习代码
2-linear-regression.py

## 学习心得
* 梯度下降法只能保证找到的是局部最小值，并非全局最小值

