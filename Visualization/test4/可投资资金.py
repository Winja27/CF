import matplotlib.pyplot as plt

# 初始参数
initial_annual_salary = 155606
monthly_salary = 12967.17
monthly_expenses = 2593.434
annual_growth_rate = 0.02
working_years = 35

# 计算每年的工资、生活支出和可投资资金
annual_salaries = []
annual_expenses_list = []
annual_savings = []
cumulative_savings = 0

for year in range(working_years):
    # 计算这一年的工资
    current_salary = initial_annual_salary * ((1 + annual_growth_rate) ** year)
    annual_salaries.append(current_salary)
    # 计算这一年的生活支出
    current_expenses = (monthly_expenses * 12) * ((1 + annual_growth_rate) ** year)
    annual_expenses_list.append(current_expenses)
    # 计算这一年的可投资资金
    current_savings = current_salary - current_expenses
    annual_savings.append(current_savings)
    # 累计可投资资金
    cumulative_savings += current_savings

# 绘制工资和生活支出的变化趋势图
plt.figure(figsize=(14, 7))

# 工资增长趋势
plt.plot(range(1, working_years + 1), annual_salaries, marker='o', label='Annual Salary')

# 生活支出增长趋势
plt.plot(range(1, working_years + 1), annual_expenses_list, marker='x', linestyle='--', label='Annual Expenses')

# 可投资资金趋势
plt.plot(range(1, working_years + 1), annual_savings, marker='s', label='Annual Savings')

plt.title('Annual Salary, Expenses, and Savings Over 35 Years of Work')
plt.xlabel('Year')
plt.ylabel('Amount (Yuan)')
plt.legend()
plt.grid(True)
plt.show()

# 打印35年的累计可投资资金
print(f"The cumulative amount available for investment over the working life is: {cumulative_savings:.2f} Yuan")
