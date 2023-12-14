import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from torchvision.transforms import functional as TF
from torch.nn import functional as F
from scorer import Scorer
from scorer.simulacra_aesthetic_models.simulacra_fit_linear_model import AestheticMeanPredictionLinearModel
from transformers import CLIPImageProcessor, CLIPModel

class MLP(torch.nn.Module):
    def __init__(self, input_size, xcol='emb', ycol='avg_rating'):
        super().__init__()
        self.input_size = input_size
        self.xcol = xcol
        self.ycol = ycol
        self.layers = nn.Sequential(
            nn.Linear(self.input_size, 1024),
            nn.Dropout(0.2),
            nn.Linear(1024, 128),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.Dropout(0.1),
            nn.Linear(64, 16),
            nn.Linear(16, 1)
        )

    def forward(self, x):
        return self.layers(x)


def normalized(a, axis=-1, order=2):
    import numpy as np
    l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
    l2[l2 == 0] = 1
    return a / np.expand_dims(l2, axis)


class AestheticScorer(Scorer):
    def __init__(self, text, images, clip_model: CLIPModel, clip_processor: CLIPImageProcessor, aesthetic_model: MLP):
        """
        Initialize Scorer object.

        :param text: Input text.
        :param images: List of input images.
        """
        super().__init__(text, images)
        self.scores = []
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        # Load the CLIP model
        self.clip_model = clip_model
        self.clip_processor = clip_processor

        # Load the aesthetic mean prediction linear model
        self.model = aesthetic_model

    def _calculate_score(self):
        """
        Calculate scores for each image.
        """
        for img in self.images:
            image = torch.from_numpy(self.clip_processor(img)['pixel_values'][0]).unsqueeze(0)
            with torch.no_grad():
                image_features = self.clip_model.get_image_features(image)

            im_emb_arr = normalized(image_features.cpu().detach().numpy())
            prediction = self.model(torch.from_numpy(im_emb_arr))
            self.scores.append(prediction.item())
            
    def get_scores(self):
        """
        Return the final scores.
        """
        self._process_input()
        self._calculate_score()
        return self.scores
