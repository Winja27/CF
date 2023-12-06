import matplotlib.pyplot as plt
import numpy as np

# 初始参数
initial_annual_salary = 155606
monthly_salary = 12967.17
monthly_expenses = 2593.434
annual_growth_rate = 0.02
working_years = 35
# 将可投资金额分成四种情景，
# 第一个情景每年定投一次定期存款，
# 第二个情景每年定投一次黄，
# 第三个情景每年定投一次房屋，
# 第四个情景每年定投一次股票。
# 其中定期存款每年收益率3%，黄金每年收益率6%，但黄金有10%的概率当年会下跌10%，
# 房屋每年收益率9%，但房屋有20%的概率当年会下跌20%，
# 股票每年收益率12%，但股票有30%的概率当年会下跌30%。
# 投资参数
deposit_interest_rate = 0.03
gold_interest_rate = 0.06
gold_downside_risk = 0.1
gold_downside_probability = 0.1
house_interest_rate = 0.09
house_downside_risk = 0.2
house_downside_probability = 0.2
stock_interest_rate = 0.12
stock_downside_risk = 0.3
stock_downside_probability = 0.3

# 计算每年的工资、生活支出和可投资资金
annual_salaries = []
annual_expenses_list = []
annual_savings = []
cumulative_savings = 0

# 投资情景初始化
investment_scenarios = {
    'Deposit': [],
    'Gold': [],
    'House': [],
    'Stock': []
}

# 随机数种子，确保结果可复现
np.random.seed(0)

for year in range(working_years):
    # 计算这一年的工资和生活支出
    current_salary = initial_annual_salary * ((1 + annual_growth_rate) ** year)
    annual_salaries.append(current_salary)
    current_expenses = (monthly_expenses * 12) * ((1 + annual_growth_rate) ** year)
    annual_expenses_list.append(current_expenses)

    # 计算这一年的可投资资金
    current_savings = current_salary - current_expenses
    annual_savings.append(current_savings)
    cumulative_savings += current_savings

    # 投资情景计算
    for scenario in investment_scenarios:
        # 之前年份的投资累计
        previous_investment = investment_scenarios[scenario][-1] if year > 0 else 0

        # 根据不同情景计算这一年的投资收益
        if scenario == 'Deposit':
            current_investment = previous_investment * (1 + deposit_interest_rate) + current_savings
        elif scenario == 'Gold':
            rate = gold_interest_rate if np.random.rand() >= gold_downside_probability else -gold_downside_risk
            current_investment = previous_investment * (1 + rate) + current_savings
        elif scenario == 'House':
            rate = house_interest_rate if np.random.rand() >= house_downside_probability else -house_downside_risk
            current_investment = previous_investment * (1 + rate) + current_savings
        elif scenario == 'Stock':
            rate = stock_interest_rate if np.random.rand() >= stock_downside_probability else -stock_downside_risk
            current_investment = previous_investment * (1 + rate) + current_savings

        # 更新投资情景结果
        investment_scenarios[scenario].append(current_investment)

# 绘制工资和生活支出的变化趋势图
plt.figure(figsize=(14, 7))

# 绘制投资收益趋势
for scenario, investments in investment_scenarios.items():
    plt.plot(range(1, working_years + 1), investments, label=f'{scenario} Investment')

plt.title('Investment Scenarios Over 35 Years of Work')
plt.xlabel('Year')
plt.ylabel('Amount (Yuan)')
plt.legend()
plt.grid(True)
plt.show()

# 打印35年的累计可投资资金和各投资情景的最终收益
print(f"The cumulative amount available for investment over the working life is: {cumulative_savings:.2f} Yuan")
for scenario, investments in investment_scenarios.items():
    print(f"The final amount for {scenario} Investment after 35 years is: {investments[-1]:.2f} Yuan")
