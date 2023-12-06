import matplotlib.pyplot as plt

# 初始参数
initial_annual_salary = 155606
annual_growth_rate = 0.02
working_years = 35

# 计算每年的工资和生活支出
annual_salaries = []
cumulative_income = 0

for year in range(working_years):
    # 计算这一年的工资
    current_salary = initial_annual_salary * ((1 + annual_growth_rate) ** year)
    annual_salaries.append(current_salary)
    # 累计收入
    cumulative_income += current_salary

# 绘制工资变化趋势图
plt.figure(figsize=(10, 5))
plt.plot(range(1, working_years + 1), annual_salaries, marker='o')
plt.title('Annual Salary Growth Over 35 Years of Work')
plt.xlabel('Year')
plt.ylabel('Annual Salary (Yuan)')
plt.grid(True)
plt.show()

# 打印累计总收入
print(f"The cumulative total income over the working life is: {cumulative_income} Yuan")
