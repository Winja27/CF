import random

import numpy as np
import pandas as pd
from numpy import diag
from numpy import zeros
from scipy.linalg import svd


# 读取文档并生成10*10评分矩阵
def data_read(ratings_address):
    ratings = pd.read_csv(ratings_address)
    user_all = ratings['userId'].value_counts(sort=True)
    users = user_all.iloc[0:10].to_frame()
    top10users = list(users.index)

    dict_uir = {}
    for r in ratings.itertuples():
        user = int(r.userId)  # 列索引
        item = int(r.movieId)  # 行索引
        rating = r.rating
        if user in dict_uir:
            d = dict_uir[user]
            d[item] = rating
        else:
            d = {item: rating}
            dict_uir[user] = d
    df_uir = pd.DataFrame.from_dict(dict_uir)

    top10users_rating = df_uir[top10users].copy().dropna().head(10)
    top10users_rating.columns = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    top10users_rating.index = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    matrix = top10users_rating.values
    ratings_rating = []
    for index, row in top10users_rating.iterrows():
        for i in range(0, 10):
            ratings_rating.append((index, i, row[i]))
    return matrix, ratings_rating


def trial(matrix, takes):
    U, s, VT = svd(matrix)
    Sigma = zeros(shape=(takes, takes))
    Sigma = diag(s[:takes])
    U2 = U[:, :takes]
    VT2 = VT[:takes, :]
    R = U2.dot(Sigma.dot(VT2))
    return R

