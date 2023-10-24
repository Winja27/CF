# 导入所需的库
import jieba
import jieba.analyse
import networkx as nx
import matplotlib.pyplot as plt
from networkx.algorithms import centrality

# 读取文本文件
text = open("《平凡的世界》.txt", "r", encoding="utf-8").read()

# 使用jieba.analyse.textrank函数，从文本中提取出关键词，并按照TextRank算法给出每个词的权重值
keywords = jieba.analyse.textrank(text, topK=100, withWeight=True) # 设置提取的关键词数量为100，并返回权重值
entities = [w for w, _ in keywords] # 获取关键词列表

# 构建词图模型，利用词之间的共现关系计算词的重要性和相似度
G = nx.Graph()
G.add_nodes_from(entities, weight=1) # 添加节点，并设置默认权重为1
window = 5 # 设置窗口大小
for i in range(len(entities) - window + 1):
    for j in range(i + 1, i + window):
        if G.has_edge(entities[i], entities[j]): # 如果边已存在，增加权重
            G[entities[i]][entities[j]]["weight"] += 1
        else: # 如果边不存在，添加边并设置权重为1
            G.add_edge(entities[i], entities[j], weight=1)

# 使用networkx.algorithms.centrality模块，计算网络图中每个节点的中心性指标，如度中心性、接近中心性、介数中心性等
degree_centrality = centrality.degree_centrality(G) # 计算度中心性
closeness_centrality = centrality.closeness_centrality(G) # 计算接近中心性
betweenness_centrality = centrality.betweenness_centrality(G) # 计算介数中心性

# 使用matplotlib.pyplot.savefig函数，将网络图保存为一个图片文件，方便在其他地方使用或展示
plt.savefig("text_network.png") # 保存图片

# 使用matplotlib将词图模型绘制成复杂网络图，并标注出关键词和短语
plt.figure(figsize=(12, 8))
pos = nx.spring_layout(G) # 设置布局
nx.draw_networkx_nodes(G, pos, node_size=[G.nodes[w]['weight'] * 50 for w in G.nodes()], node_color=[degree_centrality[w] for w in G.nodes()]) # 绘制节点，大小与权重成正比，颜色与度中心性一致
nx.draw_networkx_edges(G, pos, edge_color="gray") # 绘制边，颜色为灰色
nx.draw_networkx_labels(G, pos, font_size=14) # 绘制标签，字号为14
plt.axis("off") # 关闭坐标轴
plt.show() # 显示图像

