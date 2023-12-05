import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

data = {'小明': [90, 85, 88, 95, 80, 85],
        '小红': [85, 90, 92, 88, 90, 88],
        '小刚': [80, 85, 90, 85, 88, 90],
        '小花': [85, 88, 90, 92, 85, 88]}
df = pd.DataFrame(data, index=['创新', '执行力', '商业头脑', '责任感', '技术实力', '团队合作'])
labels=np.array(df.index)
dataLenth = 6
angles = np.linspace(0, 2*np.pi, dataLenth, endpoint=False)

fig = plt.figure()
ax = fig.add_subplot(111, polar=True)

for key in data.keys():
    values = df[key].values
    values = np.concatenate((values, [values[0]]))
    angles = np.concatenate((angles, [angles[0]]))
    ax.plot(angles, values, 'o-', linewidth=2, label=key)
    ax.fill(angles, values, alpha=0.25)

ax.set_thetagrids(angles * 180/np.pi, labels)
plt.legend(loc='upper right', bbox_to_anchor=(1.3,1.0),ncol=1,fancybox=True,shadow=True)
plt.show()

df.plot(kind='bar')
plt.title('朋友们的特质对比')
plt.xlabel('特质')
plt.ylabel('评分')
plt.show()

df.plot(kind='line')
plt.title('朋友们的特质对比')
plt.xlabel('特质')
plt.ylabel('评分')
plt.show()
