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
dataframe_observation_of_prediction = func.get_observation_of_prediction(dataframe_uir, dataframe_prediction)
func.visualization_mae(dataframe_observation_of_prediction)
func.visualization_rmse(dataframe_observation_of_prediction)
func.visualization_classification(dataframe_observation_of_prediction)
#func.visualization_ranking(dataframe_observation_of_prediction)
end_time = time.time()
total_time = end_time - start_time
print('程序运行时间为：', total_time, '秒')
