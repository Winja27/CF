import pandas as pd
# 通过JoitQuant聚宽网提供的API导入相关财务报表的数据
from jqdatasdk import *

auth('13629036040', 'Lzy20030227')
DATE = 2022
CODE = "600519.XSHG"  # 贵州茅台

# STK_CASHFLOW_STATEMENT: 合并现金流量表
# fix_intan_other_asset_acqui_cash: 构建固定资产、无形资产和其他长期资产支付的现金
# fixed_assets_depreciation：固定资产折旧
# intangible_assets_amortization：无形资产摊销
# defferred_expense_amortization：长期待摊费用摊销

q = query(finance.STK_CASHFLOW_STATEMENT.fix_intan_other_asset_acqui_cash,
          finance.STK_CASHFLOW_STATEMENT.start_date,
          finance.STK_CASHFLOW_STATEMENT.end_date,
          finance.STK_CASHFLOW_STATEMENT.fixed_assets_depreciation,
          finance.STK_CASHFLOW_STATEMENT.intangible_assets_amortization,
          finance.STK_CASHFLOW_STATEMENT.defferred_expense_amortization
          ).filter(
    finance.STK_CASHFLOW_STATEMENT.code == CODE,
    # 查询2022-01-01~2022-12-31时段的合并现金流量表
    finance.STK_CASHFLOW_STATEMENT.end_date == str(DATE) + "-12-31"
)
cashflow = finance.run_query(q)# 现金流
# income: 利润表
# income_tax_expense: 所得税费用
# financial_expense: 财务费用

q = query(income.income_tax_expense,
          income.total_profit,
          income.financial_expense
          ).filter(
    income.code == CODE,
)
incstat = get_fundamentals(q, statDate='2022')  # 查询2022年年报

# balance: 资产负债表
# total_assets: 总资产
# total_non_current_assets: 非流动资产总计

q = query(balance.total_assets,
          balance.total_non_current_assets
          ).filter(
    balance.code == CODE,
)
bs1 = get_fundamentals(q, statDate=str(DATE))  # 基准年度数据 （2022）
bs2 = get_fundamentals(q, statDate=str(DATE - 1))  # 上一年资产负债表数据 (2021)
#bs2为空导致计算cwc代码报错，直接代入一个cwc值
# 计算 CWC (annual change in net working capital): 净营运资本增加值
# cwc = (bs1['total_assets'].iloc[0] - bs1['total_non_current_assets'].iloc[0]) - \
#       (bs2['total_assets'].iloc[0] - bs2['total_non_current_assets'].iloc[0])
cwc = 21162636701.51001
# 接下来我们需要假设一些估算所必须的比率，对最终我们得到相对准确的公司估值
# 非常重要。这里粗略使用一些其他报告里采取的估计值

earnings_growth_rate = 0.1  # 收入增长率
discount_rate = 0.08  # 贴现率
cap_ex_growth_rate = 0.1  # 资本支出增长率
perpetual_growth_rate = 0.02  # 永续增长率

# 计算 EBIT: 息税前利润，这里用(总利润+财务费用)作为近似
# tax_rate: 税率
# non_cash_charges: 折旧和摊销的总和
# cap_ex: 资本化支出

ebit = incstat['total_profit'][0] + incstat['financial_expense'][0]
tax_rate = incstat['income_tax_expense'][0] / \
           incstat['total_profit'][0]
non_cash_charges = cashflow['fixed_assets_depreciation'][0] + \
                   cashflow['intangible_assets_amortization'][0] + cashflow['defferred_expense_amortization'][0]
cap_ex = cashflow['fix_intan_other_asset_acqui_cash'][0]


def ulFCF(ebit, tax_rate, non_cash_charges, cwc, cap_ex):
    # 返回无杠杆自由现金流 (unleveraged future cash flow)
    return ebit * (1 - tax_rate) + non_cash_charges + cwc - cap_ex


# year为基准年份，period表示还可以earning_growth_rate增长的年份，之后为永续增长期

def enterprise_value(year, period, discount_rate, earnings_growth_rate, cap_ex_growth_rate, perpetual_growth_rate,
                     cwc, ebit, tax_rate, non_cash_charges, cap_ex):
    flows = []
    dfcf = ulFCF(ebit, tax_rate, non_cash_charges, cwc, cap_ex)
    output = pd.DataFrame([dfcf, ebit, non_cash_charges, cwc, cap_ex], index=["DFCF", "EBIT", "D&A", "CWC", "CAP_EX"])
    index = ["DFCF", "EBIT", "D&A", "CWC", "CAP_EX"]
    columns = [year]
    for yr in range(1, 1 + period):
        ebit = ebit * (1 + (yr * earnings_growth_rate))
        non_cash_charges = non_cash_charges * (1 + (yr * earnings_growth_rate))
        cwc = cwc * 0.7
        cap_ex = cap_ex * (1 + (yr * cap_ex_growth_rate))

        flow = ulFCF(ebit, tax_rate, non_cash_charges, cwc, cap_ex)
        # print(flow, ebit, non_cash_charges, cwc, cap_ex)

        PV_flow = flow / ((1 + discount_rate) ** yr)
        flows.append(PV_flow)
        year += 1
        columns.append(year)
        pdSeries = pd.Series([PV_flow, ebit, non_cash_charges, cwc, cap_ex], index=index)
        output = pd.concat([output, pdSeries], axis=1)

    output.columns = columns

    # DATE ~ DATE+period 期间的折现值
    NPV_FCF = sum(flows)

    # 计算永续期的折现值
    final_cashflow = flows[-1] * (1 + perpetual_growth_rate)
    TV = final_cashflow / (discount_rate - perpetual_growth_rate)
    NPV_TV = TV / (1 + discount_rate) ** (1 + period)

    return ((NPV_TV + NPV_FCF, output))


ulFCF(ebit, tax_rate, non_cash_charges, cwc, cap_ex)

var = enterprise_value(DATE, 5, discount_rate, earnings_growth_rate, cap_ex_growth_rate, perpetual_growth_rate,
                       cwc, ebit, tax_rate, non_cash_charges, cap_ex)[1]


# valuaion: 公司财务指标表
q = query(valuation.capitalization,
          valuation.market_cap
          ).filter(
    valuation.code == CODE,
)
# capitalization: 总股本(万股)
capitalization = get_fundamentals(q, statDate=str(DATE)).iloc[0, 0] * 10000
# market_cap: 总市值(亿元)
market_cap = get_fundamentals(q, statDate=str(DATE)).iloc[0, 1] * 100000000


q = query(balance.total_liability,
          balance.cash_equivalents
          ).filter(
    balance.code == CODE,
)
# total_liability: 总负债
total_liability = get_fundamentals(q, statDate=str(DATE)).iloc[0, 0]
# cash_equivalents: 现金及现金等价物
cash_equivalents = get_fundamentals(q, statDate=str(DATE)).iloc[0, 1]



def equity_value(enterprise_val):
    equity_val = enterprise_val - total_liability + cash_equivalents
    share_price = equity_val / capitalization

    # 返回公司股价的估计值
    return share_price


enterprise_val = enterprise_value(DATE, 5, discount_rate, earnings_growth_rate, cap_ex_growth_rate,
                                  perpetual_growth_rate,
                                  cwc, ebit, tax_rate, non_cash_charges, cap_ex)

# 估计的股价， 600319贵州茅台，2022年
print('按照DCF模型估算，贵州茅台价值为：',equity_value(enterprise_val[0]))
