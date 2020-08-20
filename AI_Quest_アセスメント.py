# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.5.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# !pip install pandas==0.24.0 #csv出力時にpandasのエラーが出るためダウングレード

import pandas as pd

# !yes | conda install pandas-profiling

import pandas_profiling as pdp

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
sample = pd.read_csv("sample_submit.csv", header=None)

train.head()

test.head()

sample.head()



# # 1.EDA
# - 目的変数の分布確認
# - カラム選択
# - 欠損値確認
# - 
#
#
# #### 気付き
# - アメニティは、wordに分解して対処出来そう
# - キャンセルポリシーでgroup byしたグループのyの中央値を求めてみたい。

profile = pdp.ProfileReport(train)
profile.to_file(outputfile="signate_ai_quest_train.html")

# # 2.model
# - モデルの選択
# -- GBDTを試してみる

# ! yes | conda install -c anaconda py-xgboost

# +
from sklearn.model_selection import train_test_split  
x = train[['accommodates', 'bathrooms', 'bedrooms', 'beds']]#数値データかつcorrelationが高い
y = train['y']

x_train, x_validation, y_train, y_validation = train_test_split(x, y, train_size=0.8, random_state=123)

# +
from xgboost import XGBRegressor
# 学習  
model = XGBRegressor(objective="reg:linear")  
model.fit(x_train, y_train)  

# 予測  
y_pred = model.predict(x_validation)  
# -



# # 3.validation
# - cross　validationの実装

# ### 3.1mseの計算

from sklearn.metrics import mean_squared_error 
import numpy as np
mse=mean_squared_error(y_validation, y_pred)  
rmse=np.sqrt(mse)
print(f"mse:{mse}")  
print(f"rmse:{rmse}")


# ### 3.2予想値と実測値のグラフを描写

size = y_pred.shape

shapeX = x_train.shape
shapeY = y_train.shape
print(f"ytrain shape {shapeX}")
print(f"xtrain shape {shapeY}")

y_validation_index_list = y_validation.index.to_list()

y_pred_series = pd.Series(y_pred, index=y_validation_index_list) #ndarrayをpandasSeriesに変換

#predYとyとindexのdataframeを作成
y_predy = pd.concat([y_pred_series,y_validation ], axis=1)

#index順にsort
sorted_y_predy = y_predy.sort_index()
sorted_y_predy

# +
# %matplotlib inline
import matplotlib.pyplot as plt


plt.figure(figsize=(40,15),dpi=200)
plt.title("comparison pred_y to y")
plt.xlabel("id")
plt.ylabel("price")
# Traing score と Test score をプロット
plt.plot(sorted_y_predy.index, sorted_y_predy[0], 'o-',alpha=0.3, color="b", label="y_pred")
plt.plot(sorted_y_predy.index, sorted_y_predy['y'], 'o-',alpha=0.5, color="g", label="y")
plt.legend(loc="best")
plt.show()
# -

# # 4.submit

test.head()



test_x = test[['accommodates', 'bathrooms', 'bedrooms', 'beds']]

test_x

test_y_pred = model.predict(test_x)  

test_y_pred_list = test_y_pred.tolist()
len(test_y_pred_list)

sample[1] = test_y_pred_list

sample

sample.to_csv('submit_ver.1.csv', header=False, index=False)

sub1 = pd.read_csv("submit_ver.1.csv")
sub1


