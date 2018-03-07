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

f, ax = plt.subplots(figsize=(7, 5))
f.tight_layout()
ax.hist(df_y['price']-predictions['price'],bins=40, label='Residuals Linear', color='b', alpha=.5);
ax.set_title("Histogram of Residuals")
ax.legend(loc='best');

plt.show()