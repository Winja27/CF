import jieba
import torch
from ltp import LTP
import time


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

    print(namelist)


start_time = time.time()  # 设置起始的时间

if torch.cuda.is_available():
    print('cuda')
else:
    print('cuda buneng yong ')
text = open("zhaowendao.txt", "r", encoding="utf-8").read()
words = preprocess(text)
nameprocess(words)

elapsed_time = time.time() - start_time  # 设置截止时间
print('inference time cost: {}'.format(elapsed_time))  # 输出消耗的时间
