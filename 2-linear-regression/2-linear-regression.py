
# In [1]
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
font = FontProperties(fname=r"c:\windows\fonts\msyh.ttc", size=10)

def runplt():
    plt.figure()
    plt.title('匹萨价格与直径数据',fontproperties=font)
    plt.xlabel('直径（英寸）',fontproperties=font)
    plt.ylabel('价格（美元）',fontproperties=font)
    plt.axis([0, 25, 0, 25])
    plt.grid(True)
    return plt
X = [[6], [8], [10], [14], [18]]
y = [[7], [9], [13], [17.5], [18]]
from sklearn.linear_model import LinearRegression

'''
plt = runplt()
plt.plot(X, y, 'k.')
plt.show()

# In [2]
from sklearn.linear_model import LinearRegression
# 创建并拟合模型
model = LinearRegression()
model.fit(X, y)
print('预测一张12英寸匹萨价格：$%.2f' % model.predict([12])[0])

# In [3]
plt = runplt()
plt.plot(X, y, 'k.')
X2 = [[0], [10], [14], [25]]
model = LinearRegression()
model.fit(X, y)
y2 = model.predict(X2)
plt.plot(X, y, 'k-')
plt.plot(X2, y2, 'g-')
plt.show()


# In [4]

plt = runplt()
plt.plot(X, y, 'k.')
X2 = [[0], [10], [14], [25]]
model = LinearRegression()
model.fit(X, y)
y2 = model.predict(X2)
plt.plot(X, y, 'k.')
plt.plot(X2, y2, 'g-')

# 残差预测值
yr = model.predict(X)
for idx, x in enumerate(X):
    plt.plot([x, x], [y[idx], yr[idx]], 'r-')

plt.show()


# In [5]

xbar = (6 + 8 + 10 + 14 + 18) / 5
variance = ((6 - xbar)**2 + (8 - xbar)**2 + (10 - xbar)**2 + (14 - xbar)**2 + (18 - xbar)**2) / 4
print(variance)

import numpy as np

print(np.var([6, 8, 10, 14, 18], ddof=1))

print(np.cov([6, 8, 10, 14, 18], [7, 9, 13, 17.5, 18]))



# In [6]
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
X_train = [[6], [8], [10], [14], [18]]
y_train = [[7], [9], [13], [17.5], [18]]
X_test = [[6], [8], [11], [16]]
y_test = [[8], [12], [15], [18]]
regressor = LinearRegression()
regressor.fit(X_train, y_train)
xx = np.linspace(0, 26, 100)
yy = regressor.predict(xx.reshape(xx.shape[0], 1))



plt = runplt()
plt.plot(X_train, y_train, 'k.')
plt.plot(xx, yy)

#pd.DataFrame(xx).plot(kind='line')

quadratic_featurizer = PolynomialFeatures(degree=2)
X_train_quadratic = quadratic_featurizer.fit_transform(X_train)
regressor_quadratic = LinearRegression()
regressor_quadratic.fit(X_train_quadratic, y_train)

X_test_quadratic = quadratic_featurizer.transform(X_test)



xx_quadratic = quadratic_featurizer.transform(xx.reshape(xx.shape[0], 1))
plt.plot(xx, regressor_quadratic.predict(xx_quadratic), 'r-')

print('一元线性回归 r-squared', regressor.score(X_test, y_test))
print('二次回归 r-squared', regressor_quadratic.score(X_test_quadratic, y_test))

plt.show()

from mpl_toolkits.axes_grid1.inset_locator import inset_axes
plt.style.use('ggplot')
def colorbar(axm,im,vmax,vmin):
    axins1 = inset_axes(axm, width="10%", height="10%", loc=2, bbox_to_anchor=(-0.3, 0.2, 2.5, 1.01), bbox_transform=axm.transAxes,) 
    cbar=plt.colorbar(im, cax=axins1, orientation='horizontal',ticks=[vmin*0.5, 0,vmax*0.8])
    cbar.ax.set_title('Color Key',fontsize=15,y=1.02)


from sklearn.cross_validation import train_test_split
df1 = pd.read_csv('winequality-red.csv', sep=';')
df = df1.corr()

[axm_x, axm_y, axm_w, axm_h] = [0.35, 0.22, 0.5, 0.6]
fig = plt.figure(figsize=(7,6))
axm = fig.add_axes([axm_x, axm_y, axm_w, axm_h], axisbg = 'white')

im = axm.pcolormesh(df)
colorbar(axm,im,1,-1)

X = df1[list(df.columns)[:-1]]
y = df1['quality']
X_train, X_test, y_train, y_test = train_test_split(X, y)
regressor = LinearRegression()
regressor.fit(X_train, y_train)
y_predictions = regressor.predict(X_test)
print('R-squared:', regressor.score(X_test, y_test))


from sklearn.cross_validation import cross_val_score
regressor = LinearRegression()
scores = cross_val_score(regressor, X, y, cv=5)
print(scores.mean(), scores)
'''
import numpy as np
from sklearn.datasets import load_boston
from sklearn.linear_model import SGDRegressor
from sklearn.cross_validation import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import train_test_split
data = load_boston()
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target)



X_scaler = StandardScaler()
y_scaler = StandardScaler()
X_train = X_scaler.fit_transform(X_train)
y_train = y_scaler.fit_transform(y_train)
X_test = X_scaler.transform(X_test)
y_test = y_scaler.transform(y_test)


regressor = SGDRegressor(loss='squared_loss')
scores = cross_val_score(regressor, X_train, y_train, cv=5)
print('交叉验证R方值:', scores)
print('交叉验证R方均值:', np.mean(scores))
regressor.fit_transform(X_train, y_train)
print('测试集R方值:', regressor.score(X_test, y_test))
plt.show()