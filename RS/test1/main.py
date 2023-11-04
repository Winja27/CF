import functions as func
import time

start_time = time.time()
folds, ratings, items, users = func.data_read(
    'ratings.csv', 'movies.csv')
neighbors_require = func.neighbors_require(10)
pairs = func.user_pair(users)
dataframe_uir = func.indexform(ratings)
df_ave_rating = func.averating(dataframe_uir)
df_prediction = func.prediction(folds, dataframe_uir, df_ave_rating, pairs, neighbors_require)
mae = func.mae(dataframe_uir, df_prediction)
func.visualization(dataframe_uir, df_prediction)
end_time = time.time()
total_time = end_time - start_time
print('程序运行时间为：', total_time, '秒')
