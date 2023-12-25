# 导入包
import math
import numpy as np
import pandas as pd
import itertools
import random
import Pearson as p
import Cosine as c

# 数据导入
def dataImporting():
    # 导入用户电影评分表
    ratings = pd.read_csv("C:\\Users\l2310\PycharmProjects\\aijun\RS\\test3\MovieLens1M数据集\\ratings.csv")
    # 导入电影信息表
    items = pd.read_csv("C:\\Users\l2310\PycharmProjects\\aijun\RS\\test3\MovieLens1M数据集\movies.csv")
    # 导入所有用户序号，用集合存储
    users = set(ratings['userId'])
    return ratings, items, users

# 切块分组
def dividedIntoChunks(arr, m):
    # 计算每块的长度
    n = int(math.ceil(len(arr) / float(m)))
    # 以列表形式返回切分好的数据
    return [arr[i:i + n] for i in range(0, len(arr), n)]

# 创建切块
def crearteFolds(ratings):
    # 读取所有用户电影对
    user_to_movie_list = [(i[1], i[2]) for i in (ratings.loc[:, ['userId', 'movieId']].itertuples())]
    # 随机排序
    random.shuffle(user_to_movie_list)
    # 切块分组
    folds = dividedIntoChunks(user_to_movie_list, 10)
    # 返回切分好的电影用户对
    return folds

# 数据检索 中间结果存储
# 用户配对
def createPairs(users):
    # 用户两两配对
    pairs = [i for i in itertools.combinations(users, 2)]
    return pairs

def createUirIndex(ratings):
    dict_uir = {}
    # 遍历每一条用户评分信息
    for r in ratings.itertuples():
        user = int(r.userId)  # 记录当前用户序号
        item = int(r.movieId)  # 记录当前电影号
        rating = r.rating
        if user in dict_uir:
            d = dict_uir[user]
            d[item] = rating
        else:
            d = {item: rating}
            dict_uir[user] = d
    return dict_uir

def getMoviesBy(dict_uir: dict, user: int):
    if user in dict_uir: return set(dict_uir[user].keys())
    else: return None

def getRating(dict_uir:dict,user:int,item:int):
    if user in dict_uir:
        if item in dict_uir[user]: return dict_uir[user][item]
        else: return None
    else: return None

def cal_pcc(x,y):
    # 获取向量长度
    l1 = len(x)
    l2 = len(y)
    # 检查向量长度是否一致且不为0
    if l1 != l2 or l1 == 0 or l2 == 0: return None
    # 计算去均值化后的向量
    x_num = x - np.mean(x)
    y_num = y - np.mean(y)
    # 计算分子和分母
    num = np.dot(x_num,y_num)
    den = np.sqrt(np.dot(x_num,x_num) * np.dot(y_num, y_num))
    # 检查分母是否为0，避免除零错误
    if den == 0: return None
    # 返回皮尔逊相关系数
    return num / den

# 计算相似性
def cal_cosine(x,y):
    # 获取向量长度
    l1 = len(x)
    l2 = len(y)
    # 检查向量长度是否一致且不为0
    if l1 != l2 or l1 == 0 or l2 == 0: return None
    # 计算分子和分母
    num = np.dot(x,y)
    den = np.sqrt(np.dot(x, x) * np.dot(y, y))
    # 检查分母是否为0，避免除零错误
    if den == 0: return None
    # 返回皮尔逊相关系数
    return num / den


# 保存相似性
def saveSimilarity(testing_set, dict_uir, pairs,choice):
    similarity = {}
    # 遍历用户对
    for (u, v) in pairs:
        # 获取用户u和v评分过的电影集合
        items_by_u = getMoviesBy(dict_uir, u)
        items_by_v = getMoviesBy(dict_uir, v)
        # 如果两者都有评分过的电影
        if len(items_by_u) > 0 and len(items_by_v) > 0:
            shared_set = set()
            # 获取两者共同评价过的电影集合
            intersected = items_by_u.intersection(items_by_v)
            # 遍历共同评价过的电影
            for item in intersected:
                # 检查是否该电影不在测试集中
                if not ((u, item) in testing_set) and not ((v, item) in testing_set):
                    # 如果满足条件，加入共同评价集合
                    shared_set.add(item)
            # 如果有共同评价的电影
            if len(shared_set) > 0:
                # 获取两用户在共同评价电影上的评分
                ratings_u = [getRating(dict_uir,u,i) for i in shared_set]
                ratings_v = [getRating(dict_uir,v,i) for i in shared_set]
                # 计算相似性
                if choice == 1: s = cal_pcc(ratings_u, ratings_v)
                elif choice == 2: s = cal_cosine(ratings_u, ratings_v)
                # 如果计算结果有效，更新相似性矩阵
                if s is not None and not math.isnan(s):
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
    # 返回最终的相似性矩阵
    return similarity

# 邻居排序
def sortingNeighbors(similarity):
    # 初始化邻居字典
    neighbors = {}
    # 遍历每个用户
    for s in similarity:
        neigh = similarity[s]
        r = [(k, neigh[k]) for k in neigh]
        r = sorted(r, key=lambda i: i[1], reverse=True)
        neighbors[s] = r
    return neighbors

# 定义邻居数据单元
class NeighborInfo():
    def __init__(self, neighbor_id, rating_on_target, similarity):
        self.Neighbor_id = neighbor_id
        self.Rating = rating_on_target
        self.Similarity = similarity

# 计算所有用户平均评分
def calulateAveRating(dict_uir):
    ave_rating = {}
    for u in dict_uir:
        ir = dict_uir[u].values()
        ave = sum(ir) / len(ir)
        ave_rating[u] = ave
    # 返回计算得到的用户平均评分字典
    return ave_rating

# 预测公式
def predict(target_user, target_movie, distance, available_vertex, dict_uir):
    den = len(available_vertex)
    if den == 0: return None
    num = 0
    for vi in available_vertex:
        num += dict_uir[vi][target_movie] + distance[vi][target_user]
    return num / den

def cal_dis(x,y):
    # 获取向量长度
    l1 = len(x)
    l2 = len(y)
    # 检查向量长度是否一致且不为0
    if l1 != l2 or l1 == 0 or l2 == 0: return None
    num = sum(xi - yi for xi, yi in zip(x, y))
    # 返回皮尔逊相关系数
    return num / l1

def processDistance(user1,user2,testing_set):
    items_by_u = getMoviesBy(dict_uir, user1)
    items_by_v = getMoviesBy(dict_uir, user2)
    # 如果两者都有评分过的电影
    if len(items_by_u) > 0 and len(items_by_v) > 0:
        shared_set = set()
        # 获取两者共同评价过的电影集合
        intersected = items_by_u.intersection(items_by_v)
        # 遍历共同评价过的电影
        for item in intersected:
            # 检查是否该电影不在测试集中
            if not ((user1, item) in testing_set) and not ((user2, item) in testing_set):
                # 如果满足条件，加入共同评价集合
                shared_set.add(item)
        # 如果有共同评价的电影
        if len(shared_set) > 0:
            # 获取两用户在共同评价电影上的评分
            ratings_u = [dict_uir[user1][item] for item in shared_set]
            ratings_v = [dict_uir[user2][item] for item in shared_set]
            s = cal_dis(ratings_u, ratings_v)
    return s

def similarityToDistance(similarity,testing_set):
    num_users = 612
    distance_matrix = np.full((num_users, num_users), np.nan)
    situation_matrix = np.full((num_users, num_users), 0)
    # 阈值设定
    threshold = 0.45
    for user1 in similarity:
        for user2 in similarity[user1]:
            if situation_matrix[user1,user2] == 1 or situation_matrix[user2,user1] == 1: break
            if similarity[user1][user2] >= threshold:
                s = processDistance(user1,user2,testing_set)
                distance_matrix[user1,user2] = s
                distance_matrix[user2,user1] = -s
                situation_matrix[user1, user2] = 1
                situation_matrix[user2, user1] = 1
    distance_matrix = pd.DataFrame(distance_matrix, index=range(num_users), columns=range(num_users))
    return distance_matrix


# 预测评分
def createPrediction(folds, dict_uir, pairs):
    # 初始化预测字典
    prediction_pcc = {}
    prediction_cos = {}
    # 遍历交叉验证的每个折叠
    for fold in folds:
        # 将当前折叠转换为测试集
        testing_set = set(fold)
        # 计算相似性矩阵
        similarity_pcc = saveSimilarity(testing_set, dict_uir, pairs,1)
        similarity_cos = saveSimilarity(testing_set, dict_uir, pairs,2)
        # 将相似性矩阵转化为距离矩阵
        distance_matrix_pcc = similarityToDistance(similarity_pcc,testing_set)
        distance_matrix_cos = similarityToDistance(similarity_cos,testing_set)
        # 遍历当前折叠中的每个用户-物品对
        for (user, item) in fold:
            v_list = list()
            available_vertex = distance_matrix_pcc.columns[~np.isnan(distance_matrix_pcc.iloc[user])].tolist()
            for i in available_vertex:
                if item in dict_uir[i]: v_list.append(i)
            predicted = predict(user, item, distance_matrix_pcc, v_list, dict_uir)
            if predicted is not None: prediction_pcc[(user, item)] = predicted
        for (user, item) in fold:
            v_list = list()
            available_vertex = distance_matrix_cos.columns[~np.isnan(distance_matrix_cos.iloc[user])].tolist()
            for i in available_vertex:
                if item in dict_uir[i]: v_list.append(i)
            predicted = predict(user, item, distance_matrix_cos, v_list, dict_uir)
            if predicted is not None: prediction_cos[(user, item)] = predicted

    # 返回最终的预测字典
    return prediction_pcc,prediction_cos

def createObDict(dict_uir,prediction_pcc,prediction_cos,neighbors_require):
    observation_dict_pcc = {}
    observation_dict_cos = {}
    # 构造包含邻居数、用户电影对、预测值真实值对的结构
    for kv in prediction_pcc:
        user,movie = kv
        prediction_pcc[kv] = (prediction_pcc[kv],dict_uir[user][movie])
    for k in neighbors_require:
        observation_dict_pcc[k] = prediction_pcc
    for kv in prediction_cos:
        user,movie = kv
        prediction_cos[kv] = (prediction_cos[kv],dict_uir[user][movie])
    for k in neighbors_require:
        observation_dict_cos[k] = prediction_cos
    return observation_dict_pcc,observation_dict_cos

# 主函数
if __name__ == "__main__":

    ratings, items, users = dataImporting()
    # 确定邻居数量范围
    neighbors_require = list(range(1, 200, 20))
    # 创建交叉验证折叠
    folds = crearteFolds(ratings)
    # 创建用户-物品评分矩阵
    dict_uir = createUirIndex(ratings)
    # 创建用户对
    pairs = createPairs(users)
    # 计算用户平均评分
    ave_rating = calulateAveRating(dict_uir)
    # 进行预测
    prediction_pcc,prediction_cos = createPrediction(folds, dict_uir, pairs)
    observation_dict_pcc,observation_dict_cos = createObDict(dict_uir,prediction_pcc,prediction_cos,neighbors_require)
    # 计算MAE和RMSE
    mae_df_p = p.calculateMae(observation_dict_pcc)
    rmse_df_p = p.calculateRmse(observation_dict_pcc)
    mae_df_c = c.calculateMae(observation_dict_cos)
    rmse_df_c = c.calculateRmse(observation_dict_cos)
    # 计算NDCG和HLU
    ndcg_df_p = p.processNDCG(observation_dict_pcc,ave_rating,dict_uir)
    hlu_df_p = p.processHLU(observation_dict_pcc)
    ndcg_df_c = c.processNDCG(observation_dict_cos, ave_rating, dict_uir)
    hlu_df_c = c.processHLU(observation_dict_cos)
    # 计算precision、recall、accuracy、f1
    precision_df_p,recall_df_p,accuracy_df_p,f1_df_p = p.processOverall(observation_dict_pcc,ave_rating)
    precision_df_c,recall_df_c,accuracy_df_c,f1_df_c = c.processOverall(observation_dict_cos, ave_rating)
    # 合并
    mae_df = pd.concat([mae_df_p, mae_df_c], ignore_index=True)
    rmse_df = pd.concat([rmse_df_p, rmse_df_c], ignore_index=True)
    precision_df = pd.concat([precision_df_p, precision_df_c], ignore_index=True)
    recall_df = pd.concat([recall_df_p, recall_df_c], ignore_index=True)
    accuracy_df = pd.concat([accuracy_df_p, accuracy_df_c], ignore_index=True)
    f1_df = pd.concat([f1_df_p, f1_df_c], ignore_index=True)
    ndcg_df = pd.concat([ndcg_df_p, ndcg_df_c], ignore_index=True)
    hlu_df = pd.concat([hlu_df_p, hlu_df_c], ignore_index=True)

