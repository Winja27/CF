import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

def main():
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
    user_mapping = {user_id: np.random.rand(1) for user_id in unique_users}
    movie_mapping = {movie_id: np.random.rand(1) for movie_id in unique_movies}

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

    # 训练函数
    def train(data, user_coord_dict, movie_coord_dict):
        c = 0.001
        maxR = 5
        minR = 0.5
        # 遍历测试集的每一行
        for index, row in data.iterrows():
            userId = row['userId']
            movieId = row['movieId']
            rating = row['rating']
            true_dic = (abs(maxR - rating) / (maxR - minR)) * 100
            radom_dic = np.sqrt(np.sum((user_coord_dict[userId] - movie_coord_dict[movieId]) ** 2))
            e = true_dic - radom_dic
            d = user_coord_dict[userId] - movie_coord_dict[movieId]
            f = e * d
            pu = user_coord_dict[userId] + f * c
            pi = movie_coord_dict[movieId] - f * c

            user_coord_dict[userId] = pu
            movie_coord_dict[movieId] = pi

    # 预测函数
    def predict(test_data, user_coord_dict, movie_coord_dict, test_dict):
        maxR = 5
        minR = 0.5
        predict_dict = {}
        # 遍历测试集的用户
        for user in test_dict.keys():
            predict_movies = test_dict[user].keys()

            # 遍历需要预测的电影
            for movie in predict_movies:
                dictanct = np.sqrt(np.sum((user_coord_dict[user] - movie_coord_dict[movie]) ** 2))
                predict_rating = maxR - (maxR - minR) * dictanct / 100
                predict_dict.setdefault(user, {})[movie] = predict_rating
        return predict_dict

    # 评价函数
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

    mae_values = []
    rmse_values = []

    for i in range(30):
        train(train_data, user_coord_dict, movie_coord_dict)

        predict_dict = predict(test_data, user_coord_dict, movie_coord_dict, test_dict)

        mae = calculate_mae(predict_dict, test_dict)
        rmse = calculate_rmse(predict_dict, test_dict)

        mae_values.append(mae)
        rmse_values.append(rmse)

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
    # 可视化MAE和RMSE的变化
    plt.plot(range(1, 31), mae_values, label='MAE')
    plt.plot(range(1, 31), rmse_values, label='RMSE')
    plt.xlabel('Iteration')
    plt.ylabel('Error')
    plt.title('MAE and RMSE over Iterations')
    plt.legend()
    plt.savefig('error' + '.pdf')
    plt.show()


if __name__ == "__main__":
    main()