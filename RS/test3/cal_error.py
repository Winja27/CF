import numpy as np
from Observation import Observation_of_prediction
class Error:
    """MAE和RMSE的计算"""
    def __init__(self, observation_list_all:list[Observation_of_prediction]):
        self.observation_list_all = observation_list_all

    def MAE(self)->float:
        """
        计算MAE
        param observation_list: 预测结果列表(某个neighbor数量下的所有预测结果)
        return: MAE
        """
        return np.mean([abs(observation.predict - observation.actual) for observation in self.observation_list_all])
    
    def RMSE(self)->float:
        """
        计算RMSE
        param observation_list: 预测结果列表(某个neighbor数量下的所有预测结果)
        return: RMSE
        """
        return np.sqrt(np.mean([(observation.predict - observation.actual) ** 2 for observation in self.observation_list_all]))