import pandas as pd
import matplotlib.pyplot as plt
import squarify

data = {'投资金额': [100, 200, 300, 100, 200, 100]}
data2 = [100, 200, 300, 100, 200, 100]
df = pd.DataFrame(data, index=['大学毕业', '创办公司', '公司扩张', '买房', '教育', '退休'])
labels = ['大学毕业', '创办公司', '公司扩张', '买房', '教育', '退休']
df.plot(kind='bar')
plt.title('关键节点的投资金额')
plt.xlabel('关键节点')
plt.ylabel('投资金额（万）')
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.show()
df.plot(kind='line')
plt.title('关键节点的投资金额')
plt.xlabel('关键节点')
plt.ylabel('投资金额（万）')
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.show()
df.plot(kind='area', stacked=True)
plt.title('关键节点的投资金额')
plt.xlabel('关键节点')
plt.ylabel('投资金额（万）')
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.show()
squarify.plot(sizes=data2, label=labels, alpha=0.6)
plt.axis('off')
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.show()