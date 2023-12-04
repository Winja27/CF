import functions as f

P, loans_year_rate, total_months, loans_month_rate = f.read_data()
principal_payment_equal_principal, interest_payment_equal_principal, principal_payment_equal_installment, interest_payment_equal_installment, total_interest_equal_principal, total_principal_equal_principal, monthly_decrease, m_total_principal_payment, total_principal, total_interest_equal_installment, total_principal_equal_installment, m_total_principal_interest, total_interest = f.month_money(
    P, loans_year_rate, total_months, loans_month_rate)
f.compare_payment(total_principal, total_interest)
f.compare_payment_month(total_months, m_total_principal_payment, m_total_principal_interest)
f.principal_interest(total_months, principal_payment_equal_principal, interest_payment_equal_principal,
                     principal_payment_equal_installment, interest_payment_equal_installment)
