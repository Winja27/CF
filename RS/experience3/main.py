import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 读入Ratings数据并保存和索引
ratings = pd.read_csv('ratings.csv',nrows=10000)#

# 数据按折n验证分割成训练集和测试集
n = 5  # 将数据分为n折

# 随机打乱数据
ratings = ratings.sample(frac=1).reset_index(drop=True)

# 分割数据
split_point = int(len(ratings) / n)
folds = [ratings[i*split_point:(i+1)*split_point] for i in range(n)]

# 计算用户间相似性
def calculate_user_similarity(train_set):
    user_similarity = {}
    for user1 in train_set['userId'].unique():
        user_similarity[user1] = {}
        for user2 in train_set['userId'].unique():
            if user1 != user2:
                user1_ratings = train_set[train_set['userId'] == user1]
                user2_ratings = train_set[train_set['userId'] == user2]
                common_items = set(user1_ratings['movieId']).intersection(set(user2_ratings['movieId']))

                if len(common_items) > 0:
                    user1_ratings = user1_ratings[user1_ratings['movieId'].isin(common_items)]
                    user2_ratings = user2_ratings[user2_ratings['movieId'].isin(common_items)]

                    diff = user1_ratings['rating'].values - user2_ratings['rating'].values
                    similarity = len(common_items) / (np.sum(diff ** 2) + 1e-9)
                    user_similarity[user1][user2] = similarity
    return user_similarity

# 预测用户对物品的评分
def predict_rating(user, item, train_set, user_similarity):
    ratings = train_set[train_set['movieId'] == item]
    if len(ratings) == 0:
        return None

    numerator = 0
    denominator = 0
    for neighbor, similarity in user_similarity[user].items():
        neighbor_ratings = train_set[(train_set['userId'] == neighbor) & (train_set['movieId'] == item)]
        if len(neighbor_ratings) > 0:
            difference = neighbor_ratings['rating'].values[0] - np.mean(train_set[train_set['userId'] == neighbor]['rating'].values)
            numerator += similarity * difference
            denominator += similarity
    if denominator > 0:
        predicted_rating = np.mean(train_set[train_set['userId'] == user]['rating'].values) + (numerator / denominator)
    else:
        predicted_rating = None
    return predicted_rating

# 评估预测结果的平均绝对误差
def calculate_mae(predictions, test_set):
    errors = []
    for pred, rating in zip(predictions, test_set['rating'].values):
        if pred is None or rating is None:
            continue
        errors.append(np.abs(pred - rating))
    # errors = np.abs(predictions - test_set['rating'].values)
    return np.mean(errors)

# 画出k个邻居时User-based Slope One距离模型的平均绝对误差和误差直线
k_values = [i for i in range(5, 101, 5)]  # 邻居的数量

mae_values = []
for k in k_values:
    fold_maes = []
    for fold in folds:
        train_set = pd.concat([f for f in folds if f is not fold])
        test_set = fold.copy()

        # 计算用户间相似性
        user_similarity = calculate_user_similarity(train_set)

        # 预测用户对物品的评分
        predictions = []
        for index, row in test_set.iterrows():
            user = row['userId']
            item = row['movieId']
            prediction = predict_rating(user, item, train_set, user_similarity)
            predictions.append(prediction)

        # 评估预测结果的平均绝对误差
        mae = calculate_mae(predictions, test_set)
        fold_maes.append(mae)

    # 计算平均绝对误差的平均值
    average_mae = np.mean(fold_maes)
    mae_values.append(average_mae)

# 画出k个邻居时User-based Slope One距离模型的平均绝对误差和误差直线
plt.plot(k_values, mae_values, marker='o')
plt.xlabel('Number of Neighbors (k)')
plt.ylabel('Mean Absolute Error (MAE)')
plt.title('User-based Slope One Distance Model')
plt.xticks(k_values)
plt.show()
