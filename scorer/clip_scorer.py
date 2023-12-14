from transformers import CLIPModel, CLIPProcessor
from scorer import Scorer

class CLIPScorer(Scorer):
    def __init__(self, text, images, model: CLIPModel, processor: CLIPProcessor):
        super().__init__(text, images)
        self.model = model
        self.processor = processor

    def _process_input(self):
        self.processed_images = [
            self.processor(text=self.text, images=image, return_tensors="pt", padding=True)
            for image in self.images
        ]
        
    def _calculate_score(self):
        scores = []
        for processed_image in self.processed_images:
            inputs = {
                "input_ids": processed_image.input_ids,
                "attention_mask": processed_image.attention_mask,
                "pixel_values": processed_image.pixel_values,
            }
            outputs = self.model(**inputs)
            logits_per_image = outputs.logits_per_image
            scores.append(logits_per_image.detach().cpu().item())

        self.scores = scores