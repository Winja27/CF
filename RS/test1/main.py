import functions as func
folds, ratings, items, users = func.data_read(
    'ratings.csv', 'movies.csv')
neighbors_require = func.neighbors_require(2)
pairs = func.user_pair(users)
df_uir = func.indexform(ratings)
df_ave_rating = func.averating(df_uir)
df_prediction = func.prediction(folds, df_uir, df_ave_rating, pairs)
mae = func.mae(df_uir, df_prediction)
func.visualization(df_uir)
