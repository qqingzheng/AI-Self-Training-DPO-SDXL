class Scorer:
    def __init__(self, text, images):
        """
        Initialize the Scorer object.

        :param text: The input text.
        :param images: The input list of images.
        """
        self.text = text
        self.images = images
        self.scores = []

    def _process_input(self):
        """
        Process the input text and images.
        """
        pass

    def _calculate_score(self):
        """
        Calculate the score.
        """
        pass

    def get_score(self):
        """
        Return the final score.
        """
        self._process_input()
        self._calculate_score()
        return self.scores
