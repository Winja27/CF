import itertools
import math
import random
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter

def calculateMae(observation_dict_pcc):
    # 存储不同邻居数量的MAE值
    mae_all = {}
    # 初始化MAE结果的DataFrame
    mae_df = pd.DataFrame(columns=['Neighbors', 'MAE', 'Algorithm'])
    # 遍历不同的邻居数量
    for k in observation_dict_pcc:
        # 遍历每个用户-物品对的预测结果
        for kv in observation_dict_pcc[k]:
            predicted = observation_dict_pcc[k][kv][0]
            actual = observation_dict_pcc[k][kv][1]
            # 计算预测误差
            error = math.fabs(predicted - actual)
            # 更新当前邻居数量对应的误差列表
            if k in mae_all: mae_all[k].append(error)
            else: mae_all[k] = [error]
    # 计算每个邻居数量的平均绝对误差（MAE）
    for i in mae_all:
        mae_data = mae_all[i]
        mae = sum(mae_data) / len(mae_data)
        # 将结果添加到MAE结果的DataFrame中
        df_row = pd.DataFrame({'Neighbors': [i], 'MAE': [mae], 'Algorithm': ['Pearson']})
        mae_df = pd.concat([mae_df, df_row], ignore_index=True)
    # 返回最终的MAE结果DataFrame
    return mae_df

# 计算误差
def calculateRmse(observation_dict_pcc):
    # 存储不同邻居数量的MAE值
    rmse_all = {}
    # 初始化MAE结果的DataFrame
    rmse_df = pd.DataFrame(columns=['Neighbors', 'RMSE', 'Algorithm'])
    # 遍历不同的邻居数量
    for k in observation_dict_pcc:
        # 遍历每个用户-物品对的预测结果
        for kv in observation_dict_pcc[k]:
            user1, movie1 = kv
            predicted = observation_dict_pcc[k][kv][0]
            actual = observation_dict_pcc[k][kv][1]
            # 计算预测误差
            error = math.fabs(predicted - actual)
            # 更新当前邻居数量对应的误差列表
            if k in rmse_all:
                rmse_all[k].append(error)
            else:
                rmse_all[k] = [error]
    # 计算每个邻居数量的平均绝对误差（MAE）
    for i in rmse_all:
        rmse_data = rmse_all[i]
        rmse = (sum(rmse_data) / len(rmse_data))**0.5
        # 将结果添加到MAE结果的DataFrame中
        df_row = pd.DataFrame({'Neighbors': [i], 'RMSE': [rmse], 'Algorithm': ['Pearson']})
        rmse_df = pd.concat([rmse_df, df_row], ignore_index=True)
    # 返回最终的MAE结果DataFrame
    return rmse_df

def positiveNegativeConvert(actual,predicted,ave):
    if actual >= ave and predicted >= ave: return 1 # TruePositive
    elif actual >= ave and predicted < ave: return 2 # FalseNegative
    elif actual < ave and predicted >= ave: return 3 # FalsePositive
    elif actual < ave and predicted < ave: return 4 # TrueNegative

def calculateStandard(result_cnt,choice):
    tp = result_cnt[1]
    fn = result_cnt[2]
    fp = result_cnt[3]
    tn = result_cnt[4]
    # Precision
    if choice == 1:
        if tp + fp == 0: return None
        else: return float(tp) / (tp + fp)
    # Recall
    if choice == 2:
        if tp + fn == 0: return None
        else: return float(tp) / (tp + fn)
    # Accuracy
    if choice == 3:
        if tp + fp + tn + fn == 0: return None
        else: return float(tp + tn) / (tp + fp + tn + fn)
    # F1
    if choice == 4:
        p = float(tp) / (tp + fp)
        r = float(tp) / (tp + fn)
        if p + r == 0: return None
        else: return float(2 * p * r) / (p + r)


def processOverall(observation_dict,ave_rating):
    precision_df = pd.DataFrame(columns=['Neighbors', 'Precision', 'Algorithm'])
    recall_df = pd.DataFrame(columns=['Neighbors', 'Recall', 'Algorithm'])
    accuracy_df = pd.DataFrame(columns=['Neighbors', 'Accuracy', 'Algorithm'])
    f1_df = pd.DataFrame(columns=['Neighbors', 'F1', 'Algorithm'])
    for k in observation_dict:
        result = []
        for kv in observation_dict[k]:
            user,movie = kv
            predicted = observation_dict[k][kv][0]
            actual = observation_dict[k][kv][1]
            result.append(positiveNegativeConvert(actual,predicted,ave_rating[user]))
        result_cnt = Counter(result)
        precision = calculateStandard(result_cnt,1)
        recall = calculateStandard(result_cnt, 2)
        accuracy = calculateStandard(result_cnt, 3)
        f1 = calculateStandard(result_cnt, 4) # 存在重复计算
        p_row = pd.DataFrame({'Neighbors': [k], 'Precision': [precision], 'Algorithm': ['Pearson']})
        r_row = pd.DataFrame({'Neighbors': [k], 'Recall': [recall], 'Algorithm': ['Pearson']})
        a_row = pd.DataFrame({'Neighbors': [k], 'Accuracy': [accuracy], 'Algorithm': ['Pearson']})
        f_row = pd.DataFrame({'Neighbors': [k], 'F1': [f1], 'Algorithm': ['Pearson']})
        precision_df = pd.concat([precision_df, p_row],ignore_index=True)
        recall_df = pd.concat([recall_df, r_row], ignore_index=True)
        accuracy_df = pd.concat([accuracy_df, a_row], ignore_index=True)
        f1_df = pd.concat([f1_df, f_row], ignore_index=True)
    return precision_df,recall_df,accuracy_df,f1_df

def getUserDict(k_dict,user):
    # 指定邻居数的情况下，在observation_dict中检索出指定用户的内容
    result = {k: v for k, v in k_dict.items() if k[0] == user}
    return result

def getRecommendList(user_dict):
    # 生成计算DCG时所需要的推荐列表
    sorted_dict = dict(sorted(user_dict.items(),key=lambda x: x[1][0], reverse=True))
    filtered_dict = {k: v for i, (k, v) in enumerate(sorted_dict.items()) if i < 20}
    recommend_list = [k[1] for k in filtered_dict.keys()]
    return recommend_list

def getIdealList(recommend_list,user,dict_uir):
    # 生成计算IDCG时所需要的推荐列表
    result = []
    for movie in recommend_list:
        result.append((movie,dict_uir[user][movie]))
    sorted_list = sorted(result, key=lambda x: x[1], reverse=True)
    resulting_list = [a for a, b in sorted_list[:20]]
    return resulting_list

def judgeOfLike(actual,ave_rate):
    # 判断用户是否可能喜欢电影
    if actual >= ave_rate: return 1
    else: return 0
def calculateNDCG(dict_uir,result,ideal_result,ave_rating):
    b = 2
    DCG_result = []
    IDCG_result = []
    for user in result:
        ave_rate = ave_rating[user]
        summ = 0.0
        r_list = result[user]
        for movie in r_list[:b]:
            summ = summ + judgeOfLike(dict_uir[user][movie],ave_rate)
        for index,movie in enumerate(r_list[b:], start=b):
            summ = summ + judgeOfLike(dict_uir[user][movie],ave_rate) / math.log(index+1,b)
        DCG_result.append(summ)
        for user in result:
            summ = 0.0
            r_list = ideal_result[user]
            for movie in r_list[:b]:
                summ = summ + 1
            for index, movie in enumerate(r_list[b:], start=b):
                summ = summ + 1 / math.log(index + 1, b)
            IDCG_result.append(summ)
    quotients = [(a+1) / (b+1) for a,b in zip(DCG_result,IDCG_result)]
    NDCG = sum(quotients) / len(result)
    return NDCG

def calculateHLU(observation_dict,result):
    ave_rating_threshold = 3.8
    total = 0.0
    for user in result:
        summ = 0.0
        for i,movie in enumerate(result[user]):
            summ = summ + max(observation_dict[(user, movie)][1] - ave_rating_threshold, 0) / (2 ** ((float(i) / 1.0)))
        total = total + summ
    return total / len(result)

def processNDCG(observation_dict,ave_rating,dict_uir):
    ndcg_df = pd.DataFrame(columns=['Neighbors', 'NDCG', 'Algorithm'])
    NDCG_result = {}
    for k in observation_dict:
        result = {}
        ideal_result = {}
        for user in list(set(v[0] for v in observation_dict[k].keys())):
            user_dict = getUserDict(observation_dict[k],user)
            recommend_list = getRecommendList(user_dict)
            ideal_list = getIdealList(recommend_list,user,dict_uir)
            result[user] = recommend_list
            ideal_result[user] = ideal_list
        NDCG_result[k] = calculateNDCG(dict_uir,result,ideal_result,ave_rating)
        n_row = pd.DataFrame({'Neighbors': [k], 'NDCG': [NDCG_result[k]], 'Algorithm': ['Pearson']})
        ndcg_df = pd.concat([ndcg_df, n_row], ignore_index=True)
    return ndcg_df

def processHLU(observation_dict):
    hlu_df = pd.DataFrame(columns=['Neighbors', 'HLU', 'Algorithm'])
    HLU_result = {}
    for k in observation_dict:
        result = {}
        for user in list(set(v[0] for v in observation_dict[k].keys())):
            user_dict = getUserDict(observation_dict[k],user)
            recommend_list = getRecommendList(user_dict)
            result[user] = recommend_list
        HLU_result[k] = calculateHLU(observation_dict[k],result)
        h_row = pd.DataFrame({'Neighbors': [k], 'HLU': [HLU_result[k]], 'Algorithm': ['Pearson']})
        hlu_df = pd.concat([hlu_df, h_row], ignore_index=True)
    return hlu_df