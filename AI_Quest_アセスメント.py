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

# + jupyter={"outputs_hidden": true}
# ! yes | conda install -c anaconda py-xgboost
# -

# !yes | conda install pandas-profiling

import pandas as pd

import pandas_profiling as pdp

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
sample = pd.read_csv("sample_submit.csv", header=None)

# # 1.EDA
# - 目的変数の分布確認
# - カラム選択
# - 欠損値確認
#
#
# ## next
#
# - chooseing columns
#     - numeric
#         - ver.1
#             - ['accommodates', 'bathrooms', 'bedrooms', 'beds']
#         - ver.2
#             - ['accommodates', 'bathrooms', 'bedrooms', 'beds','review_scores_rating']
#         - next
#             - latitude,longitudeを組み合わせた特徴量作成できそう。
#             - last_review
#                 - 経過日数変換したい。
#             - number_of_reviews
#             - 'host_response_rate'がうまく行かない
#     - categorical
#             - label(順序性がある場合)
#                 - cleaning_fee
#                 - instant_bookable
#                 - bed_type
#                 - cancellation_policy
#                 - city
#                     - label encodingをする(大小関係を踏まえて)
#                         - (cityごとのyの平均値、中央値を出して、それを踏まえたlabel encoding)多分有効でない。
#                 - one hot(順序性がない場合)
#                 - target
#                 - 未定
#                     - amenities:wordに分解して対処出来そう
#                     
#             - next　step
#                 -- ! host_identity_verified→labelencodingできへん、、
#         
# - 目的変数は対数正規分布
#     - 何か適切なアプローチがあるかもしれない。

# + jupyter={"outputs_hidden": true}
profile = pdp.ProfileReport(train)
profile.to_file(outputfile="signate_ai_quest_train.html")

# + jupyter={"outputs_hidden": true}
train[train['y']>750]
# -

#y: over750
profile = pdp.ProfileReport(train[train['y']>750])
profile.to_file(outputfile="signate_ai_quest_train_y_over750.html")

# # 2.model
# - モデルの選択
#     - GBDTの特徴
#         - 数値の大きさ自体に意味はなく大小関係のみが影響する
#         - 欠損値があってもそのまま扱える
#         - 決定木の分岐の繰り返しによって変数間の相互作用を反映する。
#
# ## next

test[cat_cols]

# +
from sklearn.model_selection import train_test_split  
from sklearn.preprocessing import LabelEncoder#カテゴリ変数をループしてlabelencoding

columns = ['accommodates', 'bathrooms', 'bedrooms', 'beds',
           'review_scores_rating','cleaning_fee','instant_bookable',
           'bed_type','cancellation_policy','city']

cat_cols = ['cleaning_fee','instant_bookable','bed_type',
           'cancellation_policy','city']




for c in cat_cols:#学習データに基づいて定義する
    le=LabelEncoder()
    le.fit(train[c])
    train[c]=le.transform(train[c])
    test[c]=le.transform(test[c])
    
x = train[columns]#数値データかつcorrelationが高い
y = train['y']

x_train, x_validation, y_train, y_validation = train_test_split(x, y, train_size=0.8, random_state=123)

# +
from xgboost import XGBRegressor
# 学習  
model = XGBRegressor(objective="reg:squarederror")  
model.fit(x_train, y_train)  

# 予測  
y_pred = model.predict(x_validation)  
# -
# # 3.validation

# ### 3.1mse・rmseの計算

# +
from sklearn.model_selection import cross_val_score

#交差検証
rmse = cross_val_score(model, x_train, y_train, scoring='neg_root_mean_squared_error')
print('Cross-Validation scores: {}'.format(-rmse))
# スコアの平均値
import numpy as np
print('Average score: {}'.format(np.mean(-rmse)))
# -


#フツーの検証
from sklearn.metrics import mean_squared_error 
import numpy as np
mse=mean_squared_error(y_validation, y_pred)  
rmse=np.sqrt(mse)
print(f"mse:{mse}")  
print(f"rmse:{rmse}")


# ### 3.2予想値と実測値のグラフを描写

# %matplotlib inline
#feature importance のプロット
import pandas as pd
import matplotlib.pyplot as plt
importances = pd.Series(model.feature_importances_, index = columns)
importances = importances.sort_values()
importances.plot(kind = "barh")
plt.title("imporance in the xgboost Model")
plt.show()

size = y_pred.shape

shapeX = x_train.shape
shapeY = y_train.shape
print(f"ytrain shape {shapeX}")
print(f"xtrain shape {shapeY}")

y_validation_index_list = y_validation.index.to_list()

y_pred_series = pd.Series(y_pred, index=y_validation_index_list) #ndarrayをpandasSeriesに変換

#predYとyとindexのdataframeを作成
y_predy = pd.concat([y_pred_series,y_validation ], axis=1)

# + jupyter={"outputs_hidden": true}
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

# +
#予測値(predY_array)と実測値(Y_arra)の差分を確認

predY_array=np.array(y_pred.tolist())
Y_array=np.array(y_validation.tolist())

diff = Y_array- predY_array
diff = abs(diff)#絶対値にする
# -

# %matplotlib inline
x = np.arange(len(diff))
plt.bar(x,diff)

# ###　上記図これぐらい差が出ていると言うことは、そもそも高い金額のホテルを当てることができていないと思われる。理由として対数正規分布があり、サンプル数が少ないところに対するアプローチが必要。

# # 4.submit

test.head()



test_x = test[columns]

test_x

test_y_pred = model.predict(test_x)  

test_y_pred_list = test_y_pred.tolist()
len(test_y_pred_list)

sample[1] = test_y_pred_list

sample

sample.to_csv('submit_ver.2.csv', header=False, index=False)

sub1 = pd.read_csv("submit_ver.2.csv")
sub1


