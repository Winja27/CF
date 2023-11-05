# 导入 numpy 库，用于矩阵运算
import numpy as np
import pandas as pd

import SVD

ratings = pd.read_csv('ratings.csv')
users = set(ratings['userId'])
rating_index_set = [(i[1], i[2]) for i in (
    ratings.loc[:, ['userId', 'movieId']].itertuples())]

dict_uir = {}
for r in ratings.itertuples():
    user = int(r.userId)  # 列索引
    item = int(r.movieId)  # 行索引
    rating = int(r.rating)
    if user in dict_uir:
        d = dict_uir[user]
        d[item] = rating
    else:
        d = {item: rating}
        dict_uir[user] = d
df_uir = pd.DataFrame.from_dict(dict_uir)
train = df_uir.iloc[0:700, 0:700].values
val = df_uir.iloc[700:1000, 70:1000].values
svd = SVD.SVD(20, 0.01, 10, 10)
prediction_matrix = svd.fit(ratings)
