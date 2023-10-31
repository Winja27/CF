import pandas as pd
import random
import math
import itertools
import numpy as np


# 1.read
ratings = pd.read_csv("ratings.csv")
items = pd.read_csv("movies.csv", sep=',')
users = set(ratings['userId'])

# 2.切块分组
rating_index_set = [(i[1], i[2]) for i in (
    ratings.loc[:, ['userId', 'movieId']].itertuples())]
random.shuffle(rating_index_set)


def chunks(arr, m):
    n = int(math.ceil(len(arr) / float(m)))
    return [arr[i:i + n] for i in range(0, len(arr), n)]


folds = chunks(rating_index_set, 10)  # 分成10块
folds = folds

# 3.确定邻居数量
neighbors_require = [i for i in range(5, 100, 5)]

# 4.用户配对
pairs = [i for i in itertools.combinations(users, 2)]

# 5.用户，物品，评分索引,字典表示
# user->item->rating
dict_uir = {}
for r in ratings.itertuples():
    index_ = r.Index
    user = int(r.userId)
    item = int(r.movieId)
    rating = r.rating
    if user in dict_uir:
        d = dict_uir[user]
        d[item] = rating
    else:
        d = {item: rating}
        dict_uir[user] = d


# 6.获取某用户评分过物品的函数
def getItemsBy(dict_uir: dict, user: int):
    if user in dict_uir:
        return set(dict_uir[user].keys())
    else:
        print("获取用户评分过物品函数错误，似乎该用户没有评分过该物品")


# 7.获取user item，的评分
def getRaring(dict_uir: dict, user: int, item: int):
    if user in dict_uir:
        if item in dict_uir[user]:
            return dict_uir[user][item]
        else:
            print("获取user item评分函数错误，该用户没有给该物品评分")
    else:
        print("获取user item评分函数错误，似乎不存在该用户")


# 8.计算皮尔逊相似性函数
def cal_pccs(ratings1, ratings2):
    similarity = 0
    numerator = 0
    denominator1 = 0
    denominator2 = 0
    for i in range(len(ratings1)):
        if i >= len(ratings2):  # 确保 ratings1 和 ratings2 的长度相同
            break
        avg_rating1 = np.mean(ratings1[:i + 1])
        avg_rating2 = np.mean(ratings2[:i + 1])
        numerator += (ratings1[i] - avg_rating1) * (ratings2[i] - avg_rating2)
        denominator1 += (ratings1[i] - avg_rating1) ** 2
        denominator2 += (ratings2[i] - avg_rating2) ** 2
    if denominator1 != 0 and denominator2 != 0 and not np.isnan(numerator):
        similarity = numerator / \
            (np.sqrt(denominator1) * np.sqrt(denominator2))
        return similarity
    else:
        similarity = None
        return similarity


def getSimilarity(testing_set, dic_uir):
    similarity = {}
    for (u, v) in pairs:
        items_by_u = getItemsBy(dict_uir, u)
        items_by_v = getItemsBy(dict_uir, v)
        if len(items_by_u) > 0 and len(items_by_v) > 0:
            shared_set = set()
            intersected = items_by_u.intersection(items_by_v)
            for item in intersected:
                if (u, item) in testing_set or (v, item) in testing_set:
                    ()
                else:
                    shared_set.add(item)
            if len(intersected) > 0:
                ratings_u = [getRaring(dict_uir, u, i) for i in shared_set]
                ratings_v = [getRaring(dict_uir, v, i) for i in shared_set]
                if ratings_u != None and ratings_v != None:
                    s = cal_pccs(ratings_u, ratings_v)
                    if s == None or math.isnan(s):
                        ()
                    else:
                        if u in similarity:
                            similarity[u][v] = s
                        else:
                            d = {v: s}
                            similarity[u] = d
                        if v in similarity:
                            similarity[v][u] = s
                        else:
                            d = {u: s}
                            similarity[v] = d
                else:
                    ()
            else:
                ()
        else:
            ()
    return similarity


# 10.邻居排序
def getNeighbors(similarity):
    neighbors = {}
    for s in similarity:
        neigh = similarity[s]
        r = [(k, neigh[k]) for k in neigh]
        r = sorted(r, key=lambda i: i[1], reverse=True)
        neighbors[s] = r
    return neighbors


# 11.定义邻居数据单元
class NeighborInfo():
    def __init__(self, neighbor_id, rating_on_target, similarity):
        self.Neighbor_id = neighbor_id
        self.Rating = rating_on_target
        self.Similarity = similarity


# 12.预测公式
def predict(user_average_rating_dict: dict, target_user: int, neighbors: list):
    ave_u = 3.5
    if target_user in user_average_rating_dict:
        ave_u = user_average_rating_dict[target_user]
    numerator = 0.0
    denominator = 0.0
    for n in neighbors:
        ave_v = 3.5
        if n.Neighbor_id in user_average_rating_dict:
            ave_v = user_average_rating_dict[n.Neighbor_id]
        numerator = numerator + (n.Rating - ave_v) * n.Similarity
        denominator = denominator + math.fabs(n.Similarity)
    r = 0.0
    if denominator != 0.0:
        r = numerator / denominator
    if math.isnan(r):
        return ave_u
    else:
        return ave_u + r


# 13.计算所有用户平均评分
ave_rating = {}
for u in dict_uir:
    ir = dict_uir[u].values()
    ave = sum(ir) / len(ir)
    ave_rating[u] = ave

# 14.针对u，i对进行评分预测
prediction = {}
for fold in folds:
    testing_set = set(fold)
    similarity = getSimilarity(testing_set, dict_uir)
    neighbors = getNeighbors(similarity)
    for (user, item) in fold:
        result = []
        if user in neighbors:
            for (user2, sim) in neighbors[user]:
                if (user2 in dict_uir) and item in dict_uir[user2] and not ((user2, item) in testing_set):
                    result.append((user2, sim))
        else:
            ()

        result2 = []
        for (user2, sim) in result:
            n = NeighborInfo(user2, dict_uir[user2][item], sim)
            result2.append(n)
        for k in neighbors_require:
            predicted = predict(ave_rating, user, result2[0:k])
            if k in prediction:
                prediction[k][(user, item)] = predicted
            else:
                prediction[k] = {(user, item): predicted}
np.save('prediction.npy', prediction)
np.save('dict_uir.npy', dict_uir)
