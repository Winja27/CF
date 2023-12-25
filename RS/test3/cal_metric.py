from enum import Enum
from Observation import Observation_of_prediction
class Positive_or_negative(Enum):
    """判断observation列表的预测值和真实值是正例还是负例"""
    POSITIVE = 1
    NEGATIVE = 0

    def is_predict_positive(observation:Observation_of_prediction)->bool:
        """判断单个observation的预测值是正例还是负例"""
        return Positive_or_negative.POSITIVE if observation.predict >= observation.avg else Positive_or_negative.NEGATIVE
    
    def is_actual_positive(observation:Observation_of_prediction)->bool:
        """判断单个observation的真实值是正例还是负例"""
        return Positive_or_negative.POSITIVE if observation.actual >= observation.avg else Positive_or_negative.NEGATIVE
    
    @staticmethod
    def get_bool_list(observation_list:list[Observation_of_prediction])->tuple[list[bool], list[bool]]:
        """
        判断observation列表的预测值和真实值是正例还是负例
        :param observation_list: 预测结果列表(某个neighbor数量下的所有预测结果)
        :return: 预测值和真实值的正例负例列表
        """
        is_predict_positive_list = list(map(Positive_or_negative.is_predict_positive, observation_list))
        is_actual_positive_list = list(map(Positive_or_negative.is_actual_positive, observation_list))
        return is_predict_positive_list, is_actual_positive_list
    

class Count(Enum):
    """统计TP、TN、FP、FN的数量"""
    TP = 0
    TN = 1
    FP = 2
    FN = 3

    @staticmethod
    def classify(actual:Positive_or_negative, predict:Positive_or_negative):
        """
        判断预测结果是TP、TN、FP还是FN
        :param observation: 单个预测结果
        :return: TP、TN、FP还是FN
        """
        if actual == Positive_or_negative.POSITIVE and predict == Positive_or_negative.POSITIVE:
            return Count.TP
        elif actual == Positive_or_negative.NEGATIVE and predict == Positive_or_negative.NEGATIVE:
            return Count.TN
        elif actual == Positive_or_negative.NEGATIVE and predict == Positive_or_negative.POSITIVE:
            return Count.FP
        elif actual == Positive_or_negative.POSITIVE and predict == Positive_or_negative.NEGATIVE:
            return Count.FN
        
    @staticmethod
    def count(observation_list:list[Observation_of_prediction])->dict:
        """
        统计TP、TN、FP、FN的数量
        :param observation_list: 预测结果列表
        :return: 包含TP、TN、FP、FN数量的字典
        """
        metric_dict = {metric: 0 for metric in Count}
        for is_predict_positive, is_actual_positive in zip(*Positive_or_negative.get_bool_list(observation_list)):
            metric_dict[Count.classify(is_actual_positive, is_predict_positive)] += 1
        return metric_dict


class Metric:
    """评价指标: 准确率、精确率、召回率、F1值"""
    def __init__(self, observation_list:list[Observation_of_prediction]) -> None:
        """
        param observation_list: 预测结果列表(某个neighbor数量下的所有预测结果)
        """
        self.observation_list = observation_list

    def accuracy(self)->float:
        """
        准确率
        :return: 准确率
        """
        metric_dict = Count.count(self.observation_list)
        return (metric_dict[Count.TP] + metric_dict[Count.TN]) / (metric_dict[Count.TP] + metric_dict[Count.TN] + metric_dict[Count.FP] + metric_dict[Count.FN])
    
    def precision(self)->float:
        """
        精确率
        :return: 精确率
        """
        metric_dict = Count.count(self.observation_list)
        return metric_dict[Count.TP] / (metric_dict[Count.TP] + metric_dict[Count.FP])
    
    def recall(self)->float:
        """
        召回率
        :return: 召回率
        """
        metric_dict = Count.count(self.observation_list)
        return metric_dict[Count.TP] / (metric_dict[Count.TP] + metric_dict[Count.FN])
    
    def f1(self)->float:
        """
        F1值
        :return: F1值
        """
        precision = self.precision()
        recall = self.recall()
        return 2 * precision * recall / (precision + recall)
    