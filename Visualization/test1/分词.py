import jieba
import networkx as nx
import torch
from ltp import LTP
import matplotlib.pyplot as plt  # 绘图库


def preprocess(text):
    # 加载自定义词典和停用词表
    jieba.load_userdict('userdict.txt')
    stopwords = set(open('cn_stopwords.txt', encoding="utf8", errors="ignore").read().splitlines())
    # 分词并过滤停用词
    words = [w for w in jieba.cut(text) if w not in stopwords]
    # 同义名词替换
    synonyms = {'丁博士': '丁仪', '爸爸': '丁仪', '女儿': '文文', '妈妈': '方琳', '孩子': '文文', '爱人': '方琳',
                '父亲': '丁仪', '母亲': '方琳', '\u3000': '', '\n': '', '排险者': '鲁迅', '诺贝尔物理学奖': '',
                '星云': '', '引力波': ''}
    # 示例
    words = [synonyms.get(w, w) for w in words]
    return words


def nameprocess(words):
    # 默认 huggingface 下载，可能需要代理

    ltp = LTP("LTP/small")  # 默认加载 Small 模型
    # 也可以传入模型的路径，ltp = LTP("/path/to/your/model")
    # /path/to/your/model 应当存在 config.json 和其他模型文件

    # 将模型移动到 GPU 上
    if torch.cuda.is_available():
        # ltp.cuda()
        ltp.to("cuda")

    namelist = []

    for i in words:
        if i == '':
            continue
        cp = [i]
        # 自定义词表
        #  分词 cws、词性 pos、命名实体标注 ner、语义角色标注 srl、依存句法分析 dep、语义依存分析树 sdp、语义依存分析图 sdpg
        output = ltp.pipeline(cp, tasks=["pos"])
        # 使用字典格式作为返回结果
        # print(output[0]) / print(output['cws']) # 也可以使用下标访问
        if output.pos == ['nh']:
            namelist.append(i)

    return namelist


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


if torch.cuda.is_available():
    print('cuda')
else:
    print('cuda buneng yong ')
text = open("zhaowendao.txt", "r", encoding="utf-8").read()
words = preprocess(text)
namelist = nameprocess(words)
g = build_network(namelist)
nx.write_gexf(g, "test.gexf")

