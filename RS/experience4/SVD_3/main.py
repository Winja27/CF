import random
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import func as fc

rating_matrix, ratings = fc.data_read('ratings.csv')
epoch = 300
round = 1
error = 100.0
R = rating_matrix.copy()

replace_count = int(rating_matrix.size * 0.3)

# 随机选择需要替换的位置
indices = random.sample(range(rating_matrix.size), replace_count)
test_ratings = {}

for index in indices:
    user = index // 10
    movie = index % 10
    rating = rating_matrix[index // 10, index % 10]
    test_ratings[(user, movie)] = rating
    rating_matrix[(user, movie)] = 0
mae_df = pd.DataFrame(columns=['epoch', 'MAE', 'RMSE'])
while (round <= epoch) and error >= 0.01:
    err = []
    sum1 = 0
    Rt = fc.trial(rating_matrix, 3)
    for u, i in test_ratings:
        r = test_ratings.get((u, i))
        mae = np.mean(np.abs(Rt[u, i] - r))
        sum1 = (Rt[u, i] - r) ** 2 + sum1
        err.append(mae)
        Rt[u, i] = r
    rmse = np.sqrt(sum1 / 100)
    error = sum(err) / len(err)
    rating_matrix = Rt
    print('epoch: ', round, '\tMAE: ', error, '\tRMSE: ', rmse)
    round = round + 1
    df_row = pd.DataFrame(
        {'epoch': [round], 'MAE': [error], 'RMSE': [rmse]})
    mae_df = pd.concat([mae_df, df_row], ignore_index=True)

rcParamters = {
        'axes.unicode_minus': False,
        "figure.figsize": [16, 9],
        "figure.dpi": 300
    }
sns.set(rc=rcParamters)
data = mae_df.melt(id_vars="epoch", var_name="method", value_name="result")
sns.lineplot(x="epoch", y="result",hue="method", markers=True, data=data)
plt.savefig('error' + '.pdf')
print("finished.")
plt.show()