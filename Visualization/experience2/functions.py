import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# 读取数据
def read_data():
    df_equal_principal = pd.read_excel('等额本金.xlsx')
    df_equal_interest = pd.read_excel('等额本息.xlsx')

    P = df_equal_principal['贷款金额（70%，万元）'][0] * 10000  # 贷款总金
    loans_year_rate = df_equal_principal['公积金利率（%）'][0]  # 贷款年利率
    total_months = df_equal_principal['贷款期限（月）'][0]  # 贷款总月数
    loans_month_rate = loans_year_rate / 12 / 100  # 贷款月利率

    return P, loans_year_rate, total_months, loans_month_rate


# 计算每月还款
def month_money(P, loans_year_rate, total_months, loans_month_rate):
    principal_payment_equal_principal = []  # 等额本金的每月本金还款金额列表
    interest_payment_equal_principal = []  # 等额本金的每月利息还款金额列表

    for i in range(1, total_months + 1):
        principal = P / total_months
        interest = (P - (i - 1) * P / total_months) * loans_month_rate
        principal_payment_equal_principal.append(principal)
        interest_payment_equal_principal.append(interest)

    principal_payment_equal_installment = []  # 等额本息的每月本金还款金额列表
    interest_payment_equal_installment = []  # 等额本息的每月利息还款金额列表
    P_copy = P
    month_payment = (P * loans_month_rate * (1 + loans_month_rate) ** total_months) / (
            (1 + loans_month_rate) ** total_months - 1)

    for i in range(total_months):
        interest = P_copy * loans_month_rate
        principal = month_payment - interest
        principal_payment_equal_installment.append(principal)
        interest_payment_equal_installment.append(interest)
        P_copy -= principal

    total_interest_equal_principal = sum(interest_payment_equal_principal)  # 等额本金的每月利息还款金额总和
    total_principal_equal_principal = sum(principal_payment_equal_principal)  # 等额本金的每月本金还款金额总和
    monthly_decrease = P / total_months * loans_month_rate  # 等额本金的每月月供递减额
    m_total_principal_payment = [i + j for i, j in
                                 zip(principal_payment_equal_principal, interest_payment_equal_principal)]  # 等额本金每月还款总额
    total_principal = total_interest_equal_principal + total_principal_equal_principal  # 等额本金的还款总额

    total_interest_equal_installment = sum(interest_payment_equal_installment)  # 等额本息的每月利息还款金额总和
    total_principal_equal_installment = sum(principal_payment_equal_installment)  # 等额本息的每月本金还款金额总和
    m_total_principal_interest = round(interest_payment_equal_installment[0] + principal_payment_equal_installment[0])
    total_interest = total_interest_equal_installment + total_principal_equal_installment  # 等额本息的还款总额

    return principal_payment_equal_principal, interest_payment_equal_principal, principal_payment_equal_installment, interest_payment_equal_installment, total_interest_equal_principal, total_principal_equal_principal, monthly_decrease, m_total_principal_payment, total_principal, total_interest_equal_installment, total_principal_equal_installment, m_total_principal_interest, total_interest


# 比较还款总额
def compare_payment(total_principal, total_interest):
    total_payment = pd.DataFrame({
        '贷款方式': ['等额本金', '等额本息'],
        '还款总额': [total_principal / 10000, total_interest / 10000]
    })

    plt.figure(figsize=(8, 6))
    sns.barplot(x='贷款方式', y='还款总额', data=total_payment, palette="pastel")
    plt.title('等额本金与等额本息的还款总额比较')
    plt.xlabel('贷款方式')
    plt.ylabel('还款总额（万元）')
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.show()


# 比较每月还款额
def compare_payment_month(total_months, m_total_principal_payment, m_total_principal_interest):
    data = {'月份': range(1, total_months + 1),
            '等额本息': m_total_principal_interest,
            '等额本金': m_total_principal_payment}
    df = pd.DataFrame(data)

    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df, x='月份', y='等额本息', label='等额本息')
    sns.lineplot(data=df, x='月份', y='等额本金', label='等额本金')
    plt.title('月供对比')
    plt.xlabel('月份')
    plt.ylabel('月供（元）')
    plt.show()

    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df[['等额本息', '等额本金']])
    plt.title('月供对比')
    plt.xlabel('还款类型')
    plt.ylabel('月供（元）')
    plt.show()


# 两种方式月供本金与利息关系
def principal_interest(total_months, principal_payment_equal_principal, interest_payment_equal_principal,
                       principal_payment_equal_installment, interest_payment_equal_installment):
    data = pd.DataFrame({'月份': range(1, total_months + 1),  # 等额本金
                         '月供本金': principal_payment_equal_principal,
                         '月供利息': interest_payment_equal_principal})

    sns.set(style="whitegrid")
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=data[['月供本金', '月供利息']], dashes=False)

    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.title('等额本金月供本金与利息关系折线图')
    plt.xlabel('月份')
    plt.ylabel('总额')
    plt.show()

    data = pd.DataFrame({'月份': range(1, total_months + 1),  # 等额本息
                         '月供本金': principal_payment_equal_installment,
                         '月供利息': interest_payment_equal_installment})

    sns.set(style="whitegrid")
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=data[['月供本金', '月供利息']], dashes=False)

    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.title('等额本息月供本金与利息关系折线图')
    plt.xlabel('月份')
    plt.ylabel('总额')
    plt.show()
