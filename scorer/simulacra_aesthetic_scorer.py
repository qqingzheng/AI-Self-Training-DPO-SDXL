import torch
from PIL import Image
from torchvision import transforms
from torchvision.transforms import functional as TF
from torch.nn import functional as F
from scorer import Scorer
from scorer.simulacra_aesthetic_models.simulacra_fit_linear_model import AestheticMeanPredictionLinearModel
from transformers import CLIPProcessor, CLIPModel

class SimulacraAestheticScorer(Scorer):
    def __init__(self, text, images, clip_model: CLIPModel, clip_processor: CLIPProcessor, aesthetic_model: AestheticMeanPredictionLinearModel):
        """
        Initialize Scorer object.

        :param text: Input text.
        :param images: List of input images.
        """
        self.text = text
        self.images = images
        self.scores = []
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        # Load the CLIP model
        self.clip_model = clip_model
        self.clip_processor = clip_processor

        # Load the aesthetic mean prediction linear model
        self.model = aesthetic_model

    def _process_input(self):
        """
        Process the input text and images.
        """
        for idx, img in enumerate(self.images):
            img = TF.resize(img, 224, transforms.InterpolationMode.LANCZOS)
            img = TF.center_crop(img, (224,224))
            self.images[idx] = img

    def _calculate_score(self):
        """
        Calculate scores for each image.
        """
        for img in self.images:
            inputs = self.clip_processor(images=img, return_tensors="pt")
            inputs = {k: v for k, v in inputs.items()}

            with torch.no_grad():
                clip_image_embed = self.clip_model.get_image_features(**inputs)
            
            score = self.model(F.normalize(clip_image_embed, dim=-1))
            self.scores.append(score.item())
            
    def get_scores(self):
        """
        Return the final scores.
        """
        self._process_input()
        self._calculate_score()
        return self.scores
