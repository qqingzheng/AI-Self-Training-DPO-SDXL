class Scorer:
    def __init__(self, text, images):
        """
        初始化Scorer对象。

        :param text: 输入的文本。
        :param images: 输入的图片列表。
        """
        self.text = text
        self.images = images
        self.scores = []

    def _process_input(self):
        """
        处理输入的文本和图片。
        """
        pass

    def _calculate_score(self):
        """
        计算分数。
        """
        pass

    def get_score(self):
        """
        返回最终的分数。
        """
        self._process_input()
        self._calculate_score()
        return self.scores
