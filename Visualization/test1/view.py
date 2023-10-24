import jieba  # 中文分词库
import networkx as nx  # 复杂网络分析库
import matplotlib.pyplot as plt  # 绘图库


# 定义文本预处理函数，进行分词，虚词过滤，同义名词预处理等操作
def preprocess(text):
    # 加载自定义词典和停用词表
    jieba.load_userdict('userdict.txt')
    stopwords = set(open('cn_stopwords.txt',encoding="utf8", errors="ignore").read().splitlines())
    # 分词并过滤停用词
    words = [w for w in jieba.cut(text) if w not in stopwords]
    # 同义名词替换
    synonyms = {'蕊生': '胡兰成', '我': '胡兰成', '兰成': '胡兰成', '胡先生': '胡兰成'}
    # 示例
    words = [synonyms.get(w, w) for w in words]
    return words


# 定义复杂网络模型构建函数，使用networkx库
def build_network(words):
    # 创建一个空的无向图
    G = nx.Graph()
    # 遍历每个词，将其作为节点添加到图中，并记录其出现次数作为节点权重
    for w in words:
        if G.has_node(w):
            G.nodes[w]['weight'] += 1
        else:
            G.add_node(w, weight=1)
    # 遍历每两个相邻的词，将其作为边添加到图中，并记录其共现次数作为边权重
    for i in range(len(words) - 1):
        w1 = words[i]
        w2 = words[i + 1]
        if G.has_edge(w1, w2):
            G.edges[w1, w2]['weight'] += 1
        else:
            G.add_edge(w1, w2, weight=1)
    return G


# 定义复杂网络模型可视化函数，使用matplotlib库
def visualize_network(G):
    # 设置节点的大小和颜色，根据节点权重（出现次数）进行映射
    node_size = [G.nodes[w]['weight'] * 50 for w in G.nodes()]
    node_color = [G.nodes[w]['weight'] for w in G.nodes()]
    # 设置边的粗细和颜色，根据边权重（共现次数）进行映射
    edge_width = [G.edges[w1, w2]['weight'] * 0.5 for w1, w2 in G.edges()]
    edge_color = [G.edges[w1, w2]['weight'] for w1, w2 in G.edges()]
    # 使用spring布局，使得图中的节点分布更加美观
    pos = nx.spring_layout(G)
    # 绘制图形，并显示节点标签和图例
    plt.figure(figsize=(12, 8))

    # 修改部分：把绘制结果赋给mappable变量，并传给plt.colorbar函数
    mappable = nx.draw_networkx(G, pos, node_size=node_size, node_color=node_color, edge_color=edge_color,
                                width=edge_width,
                                with_labels=True)

    plt.colorbar(mappable, label='Weight')

    plt.show()


text = open("《平凡的世界》.txt", "r", encoding="utf-8").read()
words = preprocess(text)
G = build_network(words)
visualize_network(G)
