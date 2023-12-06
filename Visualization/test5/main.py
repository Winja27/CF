import pandas as pd
import matplotlib.pyplot as plt
# 假设CSV文件名为'financial_data.csv'
df = pd.read_csv('贵州茅台近十年财报.csv', index_col=1)

# 提取关键指标数据
gross_margin = df.loc['营业毛利率', '2013':'2022'].astype(float)
net_margin = df.loc['净利率 = 纯益率', '2013':'2022'].astype(float)
return_on_assets = df.loc['总资产报酬率 RoA', '2013':'2022'].astype(float)
revenue = df.loc['营业活动现金流量(百万元)', '2013':'2022'].astype(float)
net_profit = df.loc['税后净利(百万元)', '2013':'2022'].astype(float)
debt_ratio = df.loc['负债占资产比率', '2013':'2022'].astype(float)
current_ratio = df.loc['流动比率', '2013':'2022'].astype(float)
operating_activities = df.loc['营业活动现金流量(百万元)', '2013':'2022'].astype(float)
investing_activities = df.loc['投资活动现金流量(百万元)', '2013':'2022'].astype(float)
financing_activities = df.loc['筹资活动现金流量(百万元)', '2013':'2022'].astype(float)

# 准备年份标签
years = gross_margin.index.tolist()

# 设置图形大小
plt.figure(figsize=(14, 10))

# 毛利率和净利润率
plt.subplot(3, 2, 1)
plt.plot(years, gross_margin, marker='o', label='Gross Margin')
plt.plot(years, net_margin, marker='o', label='Net Margin')
plt.title('Gross Margin and Net Margin Over Years')
plt.xlabel('Year')
plt.ylabel('Percentage (%)')
plt.legend()

# 总资产报酬率
plt.subplot(3, 2, 2)
plt.plot(years, return_on_assets, marker='o', color='green', label='Return on Assets')
plt.title('Return on Assets Over Years')
plt.xlabel('Year')
plt.ylabel('Percentage (%)')
plt.legend()

# 收入和净利润
plt.subplot(3, 2, 3)
plt.bar(years, revenue, label='Revenue')
plt.bar(years, net_profit, label='Net Profit', alpha=0.7)
plt.title('Revenue and Net Profit Over Years')
plt.xlabel('Year')
plt.ylabel('Amount (Million Yuan)')
plt.legend()

# 债务比率和流动比率
plt.subplot(3, 2, 4)
plt.plot(years, debt_ratio, marker='o', label='Debt Ratio')
plt.plot(years, current_ratio, marker='o', label='Current Ratio')
plt.title('Debt Ratio and Current Ratio Over Years')
plt.xlabel('Year')
plt.ylabel('Ratio')
plt.legend()

# 经营活动、投资活动和筹资活动的现金流
plt.subplot(3, 2, 5)
plt.plot(years, operating_activities, marker='o', label='Operating Activities')
plt.plot(years, investing_activities, marker='o', label='Investing Activities')
plt.plot(years, financing_activities, marker='o', label='Financing Activities')
plt.title('Cash Flows from Operating, Investing, and Financing Activities')
plt.xlabel('Year')
plt.ylabel('Cash Flow (Million Yuan)')
plt.legend()

# 显示图形
plt.tight_layout()
plt.show()