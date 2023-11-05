import functions as func
import time

start_time = time.time()
folds, ratings, items, users = func.data_read1(
    'ratings.csv', 'movies.csv')
neighbors_require = func.neighbors_require1()
pairs = func.user_pair(users)
dataframe_uir = func.indexform1(ratings)
dataframe_ave_rating = func.averating(dataframe_uir)
dataframe_prediction = func.prediction(folds, dataframe_uir, dataframe_ave_rating, pairs, neighbors_require)
mae = func.mae(dataframe_uir, dataframe_prediction)
func.visualization(dataframe_uir, dataframe_prediction)
end_time = time.time()
total_time = end_time - start_time
print('程序运行时间为：', total_time, '秒')
