#In [1]
from sklearn import datasets
boston = datasets.load_boston()
import pandas as pd

df_x = pd.DataFrame(boston.data,columns=boston.feature_names)
df_y = pd.DataFrame(boston.target,columns=['price'])

#In [2]
from sklearn.linear_model import LinearRegression
lr = LinearRegression()

#In [3]
lr.fit(df_x, df_y)
##print (lr)

#In [4]
predictions = pd.DataFrame(lr.predict(df_x),columns=['price'])

#In [5]  #实际值和预测值差异
from matplotlib import pyplot as plt

fig = plt.figure(figsize=(9,4))
ax = fig.add_axes([0.1, 0.25, 0.4, 0.6], frame_on=True, axisbg = 'white')

#ax.hist(df_y['price']-predictions['price'],bins=40, label='Residuals Linear', color='b', alpha=.5);
ax.set_title("Histogram of Residuals")
#ax.legend(loc='best')


print (list(zip(boston.feature_names, lr.coef_[0])))


#In [6]  #各因素和因变量正负相关性

cor_df = pd.DataFrame(lr.coef_[0],index=boston.feature_names,columns=['cor'])

ax2 = fig.add_axes([0.55, 0.25, 0.4, 0.6], frame_on=True, axisbg = 'white')
cor_df.plot(ax=ax2, kind='bar', title ="x cor",legend=True, fontsize=12)

#In [7]  #LinearRegression对象可以自动标准正态化（normalize或scale）输入数据：

lr2 = LinearRegression(normalize=True)
lr2.fit(boston.data, boston.target)
predictions2 = lr2.predict(boston.data)
ax.hist([boston.target-predictions2,df_y['price']-predictions['price']],bins=40, label=['Residuals Linear','Residuals Linear2'], color=['b','r'], alpha=.5);

ax.legend(loc='best')

plt.show()