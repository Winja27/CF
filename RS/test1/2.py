import math
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

prediction = np.load('prediction.npy', allow_pickle=True).item()
dict_uir = np.load('dict_uir.npy', allow_pickle=True).item()
# 15.计算误差
mae = []
for p in prediction:
    if isinstance(p, int):
        continue
    predicted = prediction[p]
    user1, item1 = p
    actual = dict_uir[user1][item1]
    error = math.fabs(predicted - actual)
    mae.append(error)

# 16.数据存储和可视化
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
    df_row = pd.DataFrame({'Neighbors': [i], 'MAE': [mae], 'Algorithm': ['Pearson']})
    print('Neighbors', i, 'MAE', mae, 'Algorithm', 'Pearson')
    mae_df = pd.concat([mae_df, df_row], ignore_index=True)

rcParamters = {
    'font.sans-serif': 'SimHei',
    'axes.unicode_minus': False,
    "figure.figsize": [12, 9],
    "figure.dpi": 300
}
sns.set(rc=rcParamters)
g = sns.lineplot(x="Neighbors", y="MAE", hue="Algorithm", style="Algorithm", markers=True, data=mae_df)
plt.savefig('MAE' + '.pdf')
plt.show()
