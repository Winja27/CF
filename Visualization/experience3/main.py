from function import *

df = get_data('sh600519')
ma, boll_up, boll_mid, boll_lower, CLOSE = get_average_boll(df, 20)
user = user()
asset = {}
date_string = list(df.index)
for i in range(5330):
    payday(user, date_string[i])
    trade(user, date_string[i], CLOSE, ma, boll_up, boll_mid, boll_lower, i)
    asset.update({f'{date_string[i].date()}': assets(user, CLOSE[i])})
df = pd.DataFrame(list(asset.items()), columns=['Date', 'Value'])
# 下面两个函数只能同时运行一个
# visual_assets(df)
visual_rate_max_drawdown(df)
