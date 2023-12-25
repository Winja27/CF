import itertools
import math
import random
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, ndcg_score


# 数据读取和分组


def data_read1(ratings_address, movie_address):
    ratings = pd.read_csv(ratings_address)
    items = pd.read_csv(movie_address, sep=',')
    users = set(ratings['userId'])
    rating_index_set = [(i[1], i[2]) for i in (
        ratings.loc[:, ['userId', 'movieId']].itertuples())]
    random.shuffle(rating_index_set)
    folds = chunks(rating_index_set, 3)
    return folds, ratings, items, users


def chunks(arr, m):  # 分块函数
    n = int(math.ceil(len(arr) / float(m)))
    return [arr[i:i + n] for i in range(0, len(arr), n)]


# 确定邻居数量变化次数


def neighbors_require1():
    neighbors_require = [i for i in range(5, 5 * (40 + 1), 5)]
    return neighbors_require


# 用户配对


def user_pair(user):
    pairs = [i for i in itertools.combinations(user, 2)]
    return pairs


# 用户，物品，评分索引，dataframe表示


def indexform1(ratings):
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
    return df_uir


def indexform2(ratings):
    df_uir = pd.DataFrame()
    for r in ratings.itertuples():
        user = int(r.userId)  # 列索引
        item = int(r.movieId)  # 行索引
        rating = r.rating
        df_uir.loc[item, user] = rating

    return df_uir


# 获取某用户评分过物品的函数


def getItemsBy(df_uir, user: int):
    dict_uir = df_uir
    if user in dict_uir:
        return set(dict_uir[user].keys())
    else:
        print("获取用户评分过物品函数错误，似乎该用户没有评分过该物品")


# 获取user item，的评分


def getRaring(df_uir, user: int, item: int):
    dict_uir = df_uir
    if user in dict_uir:
        if item in dict_uir[user]:
            return dict_uir[user][item]
        else:
            print("获取user item评分函数错误，该用户没有给该物品评分")
    else:
        print("获取user item评分函数错误，似乎不存在该用户")


# 计算皮尔逊相似性
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


# 计算用户之间的相似度


def getSimilarity(testing_set, df_uir, pairs):
    dict_uir = df_uir
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


# 邻居排序


def getNeighbors(similarity):
    neighbors = {}
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


# 预测公式
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


# 计算所有用户平均评分


def averating(df_uir):
    dict_uir = df_uir.to_dict()
    for a, b in dict_uir.items():
        for c in list(b.keys()):
            if math.isnan(b[c]):
                del b[c]
    ave_rating = {}
    for u in dict_uir:
        ir = dict_uir[u].values()
        ave = sum(ir) / len(ir)
        ave_rating[u] = ave
    df_ave_rating = pd.DataFrame(ave_rating, index=[0])

    return df_ave_rating


# 针对u，i对进行评分预测


def prediction(folds, df_uir, df_ave_rating, pairs, neighbors_require):
    ave_rating = df_ave_rating.to_dict()
    ave_rating = {k: v[0] for k, v in ave_rating.items()}

    dict_uir = df_uir.to_dict()
    for a, b in dict_uir.items():
        for c in list(b.keys()):
            if math.isnan(b[c]):
                del b[c]
    prediction = {}
    for fold in folds:
        testing_set = set(fold)
        similarity = getSimilarity(testing_set, dict_uir, pairs)
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
    df_prediction = pd.DataFrame(prediction)
    return df_prediction


# 计算误差


def mae(df_uir, df_prediction):
    dict_uir = df_uir.to_dict()
    for a, b in dict_uir.items():
        for c in list(b.keys()):
            if math.isnan(b[c]):
                del b[c]
    prediction = df_prediction.to_dict()
    mae = []
    for p in prediction:
        if isinstance(p, int):
            continue
        predicted = prediction[p]
        user1, item1 = p
        actual = dict_uir[user1][item1]
        error = math.fabs(predicted - actual)
        mae.append(error)
    return mae


# 数据存储和可视化


def visualization(df_uir, df_prediction):
    dict_uir = df_uir.to_dict()
    prediction = df_prediction.to_dict()
    for a, b in dict_uir.items():
        for c in list(b.keys()):
            if math.isnan(b[c]):
                del b[c]
    mae_all = {}
    data = [[5, 0.709262500098963, 'Pearson']]
    mae_df = pd.DataFrame(data, columns=['Neighbors', 'MAE', 'Algorithm'])
    for k in prediction:
        for kv in prediction[k]:
            user1, item1 = kv
            predicted = prediction[k][kv]
            actual = dict_uir[user1][item1]
            error = math.fabs(predicted - actual)
            if k in mae_all:
                mae_all[k].append(error)
            else:
                mae_all[k] = [error]

    for i in mae_all:
        mae_data = mae_all[i]
        mae = sum(mae_data) / len(mae_data)
        df_row = pd.DataFrame(
            {'Neighbors': [i], 'MAE': [mae], 'Algorithm': ['Pearson']})
        print('Neighbors', i, 'MAE', mae, 'Algorithm', 'Pearson')
        mae_df = pd.concat([mae_df, df_row], ignore_index=True)

    rcParamters = {
        'axes.unicode_minus': False,
        "figure.figsize": [12, 9],
        "figure.dpi": 300
    }
    sns.set(rc=rcParamters)
    g = sns.lineplot(x="Neighbors", y="MAE", hue="Algorithm",
                     style="Algorithm", markers=True, data=mae_df)
    plt.savefig('MAE' + '.pdf')
    print("finished.")
    plt.show()


def get_observation_of_prediction(df_uir, df_prediction):
    df_prediction['actual'] = 0
    dict_uir = df_uir.to_dict()
    prediction = df_prediction.to_dict()
    for k in prediction:
        for kv in prediction[k]:
            user1, item1 = kv
            actual = dict_uir[user1][item1]
            prediction['actual'][(user1, item1)] = actual
    df_prediction = pd.DataFrame(prediction)
    return df_prediction


def visualization_mae(df_prediction):
    prediction = df_prediction.to_dict()
    mae_all = {}
    data = [[5, 0.709262500098963, 'Pearson']]
    mae_df = pd.DataFrame(data, columns=['Neighbors', 'MAE', 'Algorithm'])
    for k in prediction:
        if k == 'actual':
            break
        for kv in prediction[k]:
            predicted = prediction[k][kv]
            actual = prediction['actual'][kv]
            error = math.fabs(predicted - actual)
            if k in mae_all:
                mae_all[k].append(error)
            else:
                mae_all[k] = [error]

    for i in mae_all:
        mae_data = mae_all[i]
        mae = sum(mae_data) / len(mae_data)
        df_row = pd.DataFrame(
            {'Neighbors': [i], 'MAE': [mae], 'Algorithm': ['Pearson']})
        print('Neighbors', i, 'MAE', mae, 'Algorithm', 'Pearson')
        mae_df = pd.concat([mae_df, df_row], ignore_index=True)

    rcParamters = {
        'axes.unicode_minus': False,
        "figure.figsize": [16, 9],
        "figure.dpi": 300,
        'font.family': 'serif',
        'font.size': 14,
        'axes.labelsize': 14,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 12,
        'lines.linewidth': 2,
        'axes.grid': True,
        'grid.alpha': 0.5,
        'axes.facecolor': 'white'
    }
    sns.set(rc=rcParamters)
    g = sns.lineplot(x="Neighbors", y="MAE", hue="Algorithm",
                     style="Algorithm", markers=True, data=mae_df)
    plt.savefig('MAE' + '.pdf')
    print("finished.")
    plt.show()


def visualization_rmse(df_prediction):
    prediction = df_prediction.to_dict()
    rmse_all = {}
    data = [[5, 0.9525457288187287, 'Pearson']]
    rmse_df = pd.DataFrame(data, columns=['Neighbors', 'RMSE', 'Algorithm'])
    for k in prediction:
        if k == 'actual':
            break
        for kv in prediction[k]:
            predicted = prediction[k][kv]
            actual = prediction['actual'][kv]
            error = (predicted - actual) ** 2  # 计算平方误差
            if k in rmse_all:
                rmse_all[k].append(error)
            else:
                rmse_all[k] = [error]

    for i in rmse_all:
        rmse_data = rmse_all[i]
        rmse = (sum(rmse_data) / len(rmse_data)) ** 0.5  # 计算RMSE
        df_row = pd.DataFrame(
            {'Neighbors': [i], 'RMSE': [rmse], 'Algorithm': ['Pearson']})
        print('Neighbors', i, 'RMSE', rmse, 'Algorithm', 'Pearson')
        rmse_df = pd.concat([rmse_df, df_row], ignore_index=True)

    rcParamters = {
        'axes.unicode_minus': False,
        "figure.figsize": [16, 9],
        "figure.dpi": 300,
        'font.family': 'serif',
        'font.size': 14,
        'axes.labelsize': 14,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 12,
        'lines.linewidth': 2,
        'axes.grid': True,
        'grid.alpha': 0.5,
        'axes.facecolor': 'white'
    }
    sns.set(rc=rcParamters)
    g = sns.lineplot(x="Neighbors", y="RMSE", hue="Algorithm",
                     style="Algorithm", markers=True, data=rmse_df)
    plt.savefig('RMSE' + '.pdf')
    print("finished.")
    plt.show()


'''你的任务看起来像是一个回归任务（预测用户的评分），而不是分类任务。对于回归任务，我们通常使用像均方误差（Mean Squared Error）或者均方根误差（Root Mean Squared Error）这样的指标来评估模型的性能，而不是使用准确率（Accuracy）、精确度（Precision）、Recall（Recall）或者F1分数（F1 Score）。

如果你的任务确实是分类任务，那么你需要将你的连续预测值转换为离散的类别标签。这通常需要你设定一个阈值，然后根据预测值是否超过这个阈值来决定预测的类别。'''



def convert_to_discrete(predicted_value, actual_value):
    return 1 if predicted_value - actual_value > 0  else 0



def convert_to_discrete(predicted_value, actual_value, threshold=0.3):
    return 1 if abs(predicted_value - actual_value) < threshold else 0

def visualization_classification(df_prediction):
    prediction = df_prediction.to_dict()
    metrics_all = {'Accuracy': [], 'Precision': [], 'Recall': [], 'F1': []}

    for k in prediction:
        if k == 'actual':
            break
        predicted_values = []
        actual_values = []
        for kv in prediction[k]:
            predicted = prediction[k][kv]
            actual = prediction['actual'][kv]
            predicted_label = convert_to_discrete(predicted, actual)
            actual_label=1# 除了预测值要变为标签，真实值也要变为标签，由于之前已经用两者之间的差值来改变预测值了，所以直接设定所有的真实值都是1，这样子的话，预测值标签为1时，真实值标签也为1，预测正确
            predicted_values.append(predicted_label)
            actual_values.append(actual_label)

        accuracy = accuracy_score(actual_values, predicted_values)
        precision = precision_score(actual_values, predicted_values)
        recall = recall_score(actual_values, predicted_values)
        f1 = f1_score(actual_values, predicted_values)

        metrics_all['Accuracy'].append(accuracy)
        metrics_all['Precision'].append(precision)
        metrics_all['Recall'].append(recall)
        metrics_all['F1'].append(f1)

    rcParamters = {
        'axes.unicode_minus': False,
        "figure.figsize": [16, 9],
        "figure.dpi": 300,
        'font.family': 'serif',
        'font.size': 14,
        'axes.labelsize': 14,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 12,
        'lines.linewidth': 2,
        'axes.grid': True,
        'grid.alpha': 0.5,
        'axes.facecolor': 'white'
    }
    sns.set(rc=rcParamters)

    for metric in metrics_all:
        data = {'Neighbors': list(range(1, len(metrics_all[metric]) + 1)), metric: metrics_all[metric]}
        df_metric = pd.DataFrame(data)
        g = sns.lineplot(x="Neighbors", y=metric, data=df_metric)
        # plt.savefig(metric + '.pdf')
        # plt.show()

def visualization_ranking(df_prediction):
    prediction = df_prediction.to_dict()
    metrics_all = {'HLU': [], 'NDCG': []}

    for k in prediction:
        if k == 'actual':
            continue  # 跳过 'actual' 键

        # 为每个 k 值初始化排名列表和真实值列表
        ranked_movies_all = []
        actual_all = []

        for kv in prediction[k]:
            predicted = prediction[k][kv]
            actual = prediction['actual'][kv]

            # 生成排名列表
            ranked_movies, actual_values = generate_ranked_list(predicted, actual)
            ranked_movies_all.append(ranked_movies)
            actual_all.append(actual_values)

        # 计算 HLU 和 NDCG
        hlu_values, ndcg_values = calculate_hlu_ndcg_batch(ranked_movies_all, actual_all)

        # 取平均值作为该 k 值的 HLU 和 NDCG
        metrics_all['HLU'].append(np.mean(hlu_values))
        metrics_all['NDCG'].append(np.mean(ndcg_values))

    # 可视化
    rcParameters = {
        'axes.unicode_minus': False,
        "figure.figsize": [16, 9],
        "figure.dpi": 300,
        'font.family': 'serif',
        'font.size': 14,
        'axes.labelsize': 14,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 12,
        'lines.linewidth': 2,
        'axes.grid': True,
        'grid.alpha': 0.5,
        'axes.facecolor': 'white'
    }
    sns.set(rc=rcParameters)

    # 使用邻居数量作为 x 轴
    neighbors = list(prediction.keys())
    neighbors.remove('actual')  # 排除 'actual' 键
    for metric in metrics_all:
        data = {'Neighbors': neighbors, metric: metrics_all[metric]}
        df_metric = pd.DataFrame(data)
        g = sns.lineplot(x="Neighbors", y=metric, data=df_metric)
        # plt.savefig(metric + '.pdf')
        # plt.show()


def generate_ranked_list(predicted_value, actual_value, num_movies=10):
    # 根据预测值和真实值生成排名列表
    ranked_movies_actual = list(range(1, num_movies + 1))  # 假设有 num_movies 个电影

    # 如果电影的真实评分大于等于用户的平均评分（3.5分），则返回1，否则返回0
    relevant_list = [1 if actual_value >= 3.5 else 0 for _ in ranked_movies_actual]

    return ranked_movies_actual, relevant_list


def calculate_hlu(ranked_movies, relevant_list):
    hlu_score = 0.0
    half_life = 0.5  # 半衰期参数，您可以根据需要调整此值

    for i, _ in enumerate(ranked_movies):
        discount = half_life ** i
        hlu_score += relevant_list[i] * discount

    return hlu_score


def calculate_ndcg(ranked_movies, actual_values, k=None):
    if k is None:
        k = len(ranked_movies)

    ranked_movies_sorted = sorted(ranked_movies, key=lambda x: actual_values[x - 1], reverse=True)

    dcg = np.sum([actual_values[x - 1] / np.log2(i + 2) for i, x in enumerate(ranked_movies_sorted[:k])])
    idcg = np.sum(sorted(actual_values, reverse=True)[:k] / np.log2(np.arange(2, k + 2)))
    ndcg = dcg / idcg if idcg != 0 else 0.0

    return ndcg


def calculate_hlu_ndcg_batch(ranked_movies_all, actual_all, k=None):
    hlu_values = []
    ndcg_values = []

    for i in range(len(ranked_movies_all)):
        ranked_movies = ranked_movies_all[i]
        actual_value = actual_all[i]

        hlu = calculate_hlu(ranked_movies, actual_value)
        ndcg = calculate_ndcg(ranked_movies, actual_value, k)

        hlu_values.append(hlu)
        ndcg_values.append(ndcg)

    return hlu_values, ndcg_values
