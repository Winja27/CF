from Ashare import *
from MyTT import *
import matplotlib.pyplot as plt


class user:
    salary = 500
    share = 0
    cash = 0


def assets(user, close):
    a = user.share * close + user.cash
    print("个人资产为：", a)
    return a


# 每月发放工资+利息
def add_cash(user):
    a = user.salary / 2
    b = user.cash * (1 + 8.33 / 10000)
    user.cash = b + a


# 工资每月上涨
def add_slary(user):
    user.salary = user.salary + 9500 / 268


# 获取某个股票过去20年的数据
def get_data(code):
    print("获取数据中……")
    df = get_price(code, frequency='1d', count=7305)
    df.to_csv('data.csv', index=False)
    return df


# 计算均线和布林线
def get_average_boll(df, days):
    CLOSE = df.close.values
    ma = MA(CLOSE, days)
    boll_up, boll_mid, boll_lower = BOLL(CLOSE)
    return ma, boll_up, boll_mid, boll_lower, CLOSE


def sellout(user, close):
    a = user.cash
    user.cash = user.share * close + a
    user.share = 0
    print("卖出")


def buyin(user, close):
    x = user.cash
    y = user.share
    num = x // close
    user.share = num + y
    user.cash = x - num * close
    print("买进")


def trade(user, date_string, CLOSE, ma, boll_up, boll_mid, boll_lower, i):
    now = date_string

    # 检查上次交易时间是否存在，如果不存在，说明这是第一次交易，可以直接进行
    if 'last_trade_time' not in globals():
        if CLOSE[i] <= boll_lower[i] and CLOSE[i] < ma[i]:
            buyin(user, CLOSE[i])
            globals()['last_trade_time'] = now
            return
        elif CLOSE[i] >= boll_up[i] and CLOSE[i] > ma[i]:
            sellout(user, CLOSE[i])
            globals()['last_trade_time'] = now
            return
        print("不满足交易条件")
        return

    last_trade_time = globals()['last_trade_time']

    if (now.year > last_trade_time.year) or (now.month > last_trade_time.month):
        if CLOSE[i] <= boll_lower[i] and CLOSE[i] < ma[i]:
            buyin(user, CLOSE[i])
            globals()['last_trade_time'] = now
            return
        elif CLOSE[i] >= boll_up[i] and CLOSE[i] > ma[i]:
            sellout(user, CLOSE[i])
            globals()['last_trade_time'] = now
            return
        print("不满足交易条件")
    else:
        print("满足交易条件，但是距离上次交易未满一个月")


def payday(user, date_string):
    now = date_string

    # 检查上次payday时间是否存在，如果不存在，说明这是第一次，可以直接进行
    if 'last_payday_time' not in globals():
        add_cash(user)
        add_slary(user)
        globals()['last_payday_time'] = now
        return

    last_trade_time = globals()['last_payday_time']

    if (now.year > last_trade_time.year) or (now.month > last_trade_time.month):
        add_cash(user)
        add_slary(user)
        globals()['last_payday_time'] = now


def visual_assets(df):
    df.set_index('Date', inplace=True)

    # 绘制折线图
    plt.figure(figsize=(10, 6))
    plt.plot(df.index, df['Value'])

    plt.title('Line Plot of Date vs Value')
    plt.xlabel('Date')
    plt.ylabel('Value')
    plt.xticks([])
    plt.ticklabel_format(style='plain', axis='y')
    plt.show()


def visual_rate_max_drawdown(df):
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)

    # 以周为单位重新采样数据，并计算每周的收益率
    df_weekly = df.resample('W').last()
    df_weekly['Return'] = df_weekly['Value'].pct_change()

    # 计算累积收益
    df_weekly['Cumulative Return'] = (1 + df_weekly['Return']).cumprod()

    # 计算最大回撤
    wealth_index = df_weekly['Cumulative Return']
    previous_peaks = wealth_index.cummax()
    drawdowns = (wealth_index - previous_peaks) / previous_peaks
    mdd = drawdowns.min()
    print(mdd)

    # 找到最大回撤的开始点、结束点和最低点
    end_point = drawdowns.idxmin()
    start_point = wealth_index.loc[:end_point].idxmax()
    lowest_point = wealth_index[start_point:end_point].idxmin()

    # 找到收益最高的点
    highest_point = wealth_index.idxmax()

    # 绘制累积收益图，并标出最大回撤的开始点、结束点、最低点以及收益最高的点
    fig, ax1 = plt.subplots(figsize=(12, 6))

    color = 'tab:blue'
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Cumulative Return', color=color)
    ax1.plot(df_weekly['Cumulative Return'], color=color, label='Cumulative Return')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.plot(start_point, wealth_index.loc[start_point], 'ro', label='Start Point')
    ax1.plot(lowest_point, wealth_index.loc[lowest_point], 'go', label='Lowest Point')
    ax1.plot(end_point, wealth_index.loc[end_point], 'bo', label='End Point')
    ax1.plot(highest_point, wealth_index.loc[highest_point], 'ko', label='Highest Point')

    ax2 = ax1.twinx()

    color = 'tab:red'
    ax2.set_ylabel('Return', color=color)
    ax2.plot(df_weekly['Return'], color=color, label='Return')
    ax2.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()
    plt.title('Max Drawdown and Return')
    fig.legend(loc="upper left")
    plt.show()
    '''左侧的y轴（蓝色）表示的是“累积收益率”。这是一个时间序列数据，表示从某一初始点开始，投资的价值如何随时间变化。如果累积收益率为1.5，那就意味着投资的价值增长了50%。这个数据是通过计算每个时间点的收益率，然后将这些收益率相乘（假设收益率是连续复利的）得到的。

右侧的y轴（红色）表示的是“收益率”。这是一个时间序列数据，表示在每个时间点，投资的价值相比于上一个时间点增长了多少。例如，如果在某个月份的收益率为0.1，那就意味着那个月的投资价值增长了10%。这个数据是通过计算每个时间点的投资价值相比于上一个时间点的变化率得到的。'''
