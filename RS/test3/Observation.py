class Observation_of_prediction:
    def __init__(self, user:int, item:int, neighbors:int,
                 predict:float, actual:float, avg:float):
        """
        保存单个预测结果
        :param user: 用户
        :param item: 电影
        :param neighbors: 相邻用户个数
        :param predict: 预测评分
        :param actual: 实际评分
        :param avg: 用户平均评分
        """
        self.user = user
        self.item = item
        self.neighbors = neighbors
        self.predict = predict
        self.actual = actual
        self.avg = avg