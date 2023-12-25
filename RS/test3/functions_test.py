import functions as func
import time

start_time = time.time()
folds, ratings, items, users = func.data_read1(
    'ratings.csv', 'movies.csv')
neighbors_require = func.neighbors_require1()
pairs = func.user_pair(users)
dataframe_uir = func.indexform1(ratings)
dataframe_similarity = func.similarity_info(folds, dataframe_uir, pairs)

th = 0.5  # 可以根据需要设定阈值
filtered_similarity = func.threshold_filter(dataframe_similarity.to_dict(), th)

# 当用户间PCC大于阈值th，则计算用户间距离distance并保存
distance = func.calculate_distance(filtered_similarity)

# Slope One 预测并绘制图表
func.predict_all(folds, dataframe_uir, distance, neighbors_require)

# 计算其他指标
predictions = func.slope_one_predict_all(folds, dataframe_uir, distance, neighbors_require)
ground_truth = func.get_ground_truth(folds, dataframe_uir)
func.evaluate(predictions, ground_truth)

end_time = time.time()
total_time = end_time - start_time
print('程序运行时间为：', total_time, '秒')
