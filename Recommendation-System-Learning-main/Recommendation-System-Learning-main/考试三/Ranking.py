import sys
from Observation import Observation_of_prediction
from itertools import groupby
import math
class RankingList:
    """
    保存单个用户的预测排序列表和真实排序列表, 计算单个用户的HLU值和NDCG值
    """
    def __init__(self, observation_list:list[Observation_of_prediction]) -> None:
        """
        :param observation_list: 某一邻居数量时某个用户的预测结果列表
        :param h: 半衰期
        :param b: 折扣位置
        """
        self.observation_list = observation_list
        self.get_ranking_list()

    def get_ranking_list(self):
        """
        获取预测排序列表和真实排序列表
        """
        self.predict_ranking_list = sorted(self.observation_list, key=lambda x: x.predict, reverse=True)
        self.actual_ranking_list = sorted(self.observation_list, key=lambda x: x.actual, reverse=True)

    def get_HLU(self, h:int=2)->float:
        """
        计算单个用户的HLU值
        :param h: 半衰期
        :return: 单个用户的HLU值
        """
        HLU_value = 0.
        for i, observation in enumerate(self.predict_ranking_list):
            position = i + 1
            if (position - 1) / (h - 1) > sys.float_info.max_10_exp:    # 如果指数过大，则跳过
                continue
            HLU_value += max(observation.actual - observation.avg, 0.0) / (2.0 ** float((position - 1) / (h - 1))) # 计算HLU
        return HLU_value
    
    def get_NDCG(self, b:int=2)->float:
        """
        计算单个用户的NDCG值
        :param b: 折扣位置
        :return: 单个用户的NDCG值
        """
        dcg = 0.  # 预测排序的DCG
        idcg = 0.    # 真实排序的DCG
        for i, (observation_predict, observation_actual) in enumerate(zip(self.predict_ranking_list, self.actual_ranking_list)):
            if observation_predict.predict <= 3.5:   # 如果预测值小于平均值，则后面的预测值都小于平均值，直接跳出循环
                break
            like_predict = 1 if observation_predict.actual >= observation_predict.avg else 0    # 预测值是否是正例
            like_actual = 1 if observation_actual.actual >= observation_actual.avg else 0       # 真实值是否是正例
            position = i + 1    # 位置
            if position <= b:   # 如果位置小于等于折扣位置，则不折扣
                dcg += like_predict
                idcg += like_actual
            else:   # 如果位置大于折扣位置，则折扣
                dcg += like_predict / math.log(position, b)
                idcg += like_actual / math.log(position, b)
        return dcg / idcg if idcg != 0 else 0.
    
    def __str__(self) -> str:
        """打印一下就能看到单个用户的预测排序列表和真实排序列表"""
        string = f"user:{self.actual_ranking_list[0].user}\tneighbor:{self.actual_ranking_list[0].neighbors}\tavg:{self.actual_ranking_list[0].avg}\nHLU:{self.get_HLU()}\tNDCG:{self.get_NDCG()}\npred movie\tpred rating\t\t\tact rating\t\t\tact movie\tact rating\t\tpred rating\n"
        for observation_predict, observation_actual in zip(self.predict_ranking_list, self.actual_ranking_list):
            string += f"{observation_predict.item}\t\t{observation_predict.predict}\t\t{observation_predict.actual}\t\t\t\t{observation_actual.item}\t\t{observation_actual.actual}\t\t\t{observation_actual.predict}\n"
        return string


class RankingPerformance:
    """
    计算某个邻居数量时, 所有用户的HLU值和NDCG均值, 以及所有用户的预测排序列表和真实排序列表
    """
    def __init__(self, observation_list:list[Observation_of_prediction]) -> None:
        """
        :param observation_list: 某个邻居数量下所有用户的预测结果列表
        :param h: 半衰期
        :param b: 折扣位置
        """
        self.observation_list = sorted(observation_list, key=lambda x:x.user)    # 按照用户排序
        self.ranking_list_dict = {} # {user: RankingList} 保存每个用户的RankingList
        self.get_all_ranking_list()

    def HLU(self, h:int=2)->float:
        """
        计算所有用户的HLU均值
        :param h: 半衰期
        :return: 所有用户的HLU均值
        """
        HLU_list = []
        for user, group in groupby(self.observation_list, key=lambda x: x.user):
            ranking_list = RankingList(list(group))
            HLU_list.append(ranking_list.get_HLU(h=h))
        return sum(HLU_list) / len(HLU_list)

    def NDCG(self, b:int=2)->float:
        """
        计算所有用户的NDCG均值
        :param b: 折扣位置
        :return: 所有用户的NDCG均值
        """
        NDCG_list = []
        for user, group in groupby(self.observation_list, key=lambda x: x.user):
            ranking_list = RankingList(list(group))
            NDCG_list.append(ranking_list.get_NDCG(b=b))
        return sum(NDCG_list) / len(NDCG_list)

    def get_all_ranking_list(self):
        """
        记录所有用户的预测排序列表和真实排序列表
        """
        for user, group in groupby(self.observation_list, key=lambda x: x.user):
            ranking_list = RankingList(list(group))
            self.ranking_list_dict[user] = ranking_list