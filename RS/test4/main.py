import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

def calculate_mae(predict_dict, true_dict):
    error = 0
    predict_nums = 0
    for user in true_dict.keys():
        for movie in true_dict[user].keys():
            if user in predict_dict and movie in predict_dict[user]:
                error += abs(predict_dict[user][movie] - true_dict[user][movie])
                predict_nums += 1
    return error / predict_nums

def calculate_rmse(predict_dict, true_dict):
    error = 0
    predict_nums = 0
    for user in true_dict.keys():
        for movie in true_dict[user].keys():
            if user in predict_dict and movie in predict_dict[user]:
                error += (predict_dict[user][movie] - true_dict[user][movie]) ** 2
                predict_nums += 1
    return np.sqrt(error / predict_nums)

def vivaldi(user_movie_pairs, user_co_dict, movie_co_dict, expect_distance):
    user_co = user_co_dict.copy()
    movie_co = movie_co_dict.copy()
    c = 0.001
    epoch = 30
    for _ in range(epoch):
        for pair in user_movie_pairs:
            user, movie = pair
            if user in expect_distance and movie in expect_distance[user]:
                p_u = np.array(user_co[user])
                p_m = np.array(movie_co[movie])
                exp_distance = expect_distance[user][movie]
                cur_distance = np.sqrt(np.sum((p_u - p_m) ** 2))
                e = exp_distance - cur_distance
                direction = p_u - p_m
                f = e * direction
                p_u_new = p_u + f * c
                p_m_new = p_m - f * c
                user_co[user] = p_u_new.tolist()
                movie_co[movie] = p_m_new.tolist()

    return user_co, movie_co


def train(data, user_coord_dict, movie_coord_dict):
    user_movie_pairs = [(row['userId'], row['movieId']) for _, row in data.iterrows()]
    expect_distance = data.set_index(['userId', 'movieId'])['rating'].apply(lambda rating: (5 - rating) / 4).to_dict()
    user_coord_dict, movie_coord_dict = vivaldi(user_movie_pairs, user_coord_dict, movie_coord_dict, expect_distance)

def predict(test_data, user_coord_dict, movie_coord_dict, test_dict):
    maxR = 5
    minR = 0.5
    predict_dict = {}
    for user, movies in test_dict.items():
        for movie in movies:
            user_coord = np.array(user_coord_dict[user])
            movie_coord = np.array(movie_coord_dict[movie])
            distance = np.sqrt(np.sum((user_coord - movie_coord) ** 2))
            predict_rating = maxR - (maxR - minR) * distance / 100
            predict_dict.setdefault(user, {})[movie] = predict_rating
    return predict_dict

# 读取CSV文件
data = pd.read_csv("ratings.csv")
# 去掉时间戳
data = data.iloc[:, :3]

# 将指定列转换为 int 类型
userId = 'userId'
movieId = 'movieId'
data[userId] = data[userId].astype(int)
data[movieId] = data[movieId].astype(int)

# 打乱数据
data = data.sample(frac=1, random_state=42)

# 转坐标
# 获取唯一的userId和movieId
unique_users = data['userId'].unique()
unique_movies = data['movieId'].unique()

# 创建userId和movieId的随机映射字典
np.random.seed(0)  # 设置随机种子以保证结果可复现

# 随机生成四十维坐标
user_mapping = {user_id: np.random.rand(40).tolist() for user_id in unique_users}
movie_mapping = {movie_id: np.random.rand(40).tolist() for movie_id in unique_movies}

# 将映射存储为字典
user_coord_dict = {user_id: user_mapping[user_id] for user_id in unique_users}  # 所有用户的坐标
movie_coord_dict = {movie_id: movie_mapping[movie_id] for movie_id in unique_movies}  # 所有电影的坐标

# 训练集测试集
split_point = int(len(data) * (1 - 0.1))
train_data = data[:split_point]
test_data = data[split_point:]

# 数据字典
def data_dict(data: list[list]) -> dict[int, dict[int, float]]:
    data_dict = {}
    data = data.values.tolist()
    for record in data:
        user = int(record[0])   # 用户名
        movie = int(record[1])  # 电影
        rating = record[2]      # 评分
        if user not in data_dict:
            data_dict[user] = {movie: rating}
        else:
            data_dict[user][movie] = rating
    return data_dict

test_dict = data_dict(test_data)
# 可视化MAE和RMSE的变化
mae_values = []
rmse_values = []
# 在这里进行 Vivaldi 算法的迭代训练
for iteration in range(1, 31):
    # 清空列表
    mae_values = []
    rmse_values = []

    train(train_data, user_coord_dict, movie_coord_dict)

    predict_dict = predict(test_data, user_coord_dict, movie_coord_dict, test_dict)

    mae = calculate_mae(predict_dict, test_dict)
    rmse = calculate_rmse(predict_dict, test_dict)

    mae_values.append(mae)
    rmse_values.append(rmse)


predict_dict = predict(test_data, user_coord_dict, movie_coord_dict, test_dict)
mae = calculate_mae(predict_dict, test_dict)
rmse = calculate_rmse(predict_dict, test_dict)


for i in range(30):
    train(train_data, user_coord_dict, movie_coord_dict)

    predict_dict = predict(test_data, user_coord_dict, movie_coord_dict, test_dict)

    mae = calculate_mae(predict_dict, test_dict)
    rmse = calculate_rmse(predict_dict, test_dict)

    mae_values.append(mae)
    rmse_values.append(rmse)



