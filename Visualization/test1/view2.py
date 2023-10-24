# 导入所需的库
import jieba.analyse
import networkx as nx
import matplotlib.pyplot as plt
from networkx.algorithms import community

# 读取文本文件
text = open("《平凡的世界》.txt", "r", encoding="utf-8").read()

# 使用jieba.analyse.extract_tags函数，直接从文本中提取出关键词
keywords = jieba.analyse.extract_tags(text, topK=100, withWeight=True)  # 设置提取的关键词数量为100，并返回权重值
entities = [w for w, _ in keywords]  # 获取关键词列表

# 构建词图模型，利用词之间的共现关系计算词的重要性和相似度
G = nx.Graph()
G.add_nodes_from(entities, weight=1) # 添加节点，并设置默认权重为1
window = 5  # 设置窗口大小
for i in range(len(entities) - window + 1):
    for j in range(i + 1, i + window):
        if G.has_edge(entities[i], entities[j]):  # 如果边已存在，增加权重
            G[entities[i]][entities[j]]["weight"] += 1
        else:  # 如果边不存在，添加边并设置权重为1
            G.add_edge(entities[i], entities[j], weight=1)

# 使用networkx.algorithms.community模块，对网络图进行社区划分，找出文本中的不同主题或类别
communities = community.greedy_modularity_communities(G)  # 使用贪婪算法进行社区划分
community_dict = {}  # 创建一部字典，存储每个节点所属的社区编号
for i, c in enumerate(communities):
    for w in c:
        community_dict[w] = i

# 使用matplotlib.pyplot.title函数，为网络图添加一个标题，说明数据来源或目的
plt.title("Text Network Analysis")  # 设置标题

# 使用matplotlib将词图模型绘制成复杂网络图，并标注出关键词和短语
plt.figure(figsize=(12, 8))
pos = nx.spring_layout(G)  # 设置布局
nx.draw_networkx_nodes(G, pos, node_size=[G.nodes[w]['weight'] * 50 for w in G.nodes()],
                       node_color=[community_dict[w] for w in G.nodes()])  # 绘制节点，大小与权重成正比，颜色与社区编号一致
nx.draw_networkx_edges(G, pos, edge_color="red")  # 绘制边，颜色为灰色
nx.draw_networkx_labels(G, pos, font_size=14)  # 绘制标签，字号为14
plt.axis("off")  # 关闭坐标轴
plt.show()  # 显示图像
