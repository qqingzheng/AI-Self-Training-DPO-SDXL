# AI Feedback-Based Self-Training Direct Preference Optimization


[![comming](https://img.shields.io/badge/Arxiv-comming-b31b1b.svg?logo=arXiv)](comming)
[![HuggingFace](https://img.shields.io/badge/🤗-Open%20In%20Spaces-blue.svg)](https://huggingface.co/chestnutlzj/ai-self-training-dpo-sdxl)
[![Hits](https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fgithub.com%2Fgjbae1212%2Fhit-counter)](https://hits.seeyoufarm.com)                    

![](assets/abstract_x1.png)
![](assets/checkpoints.png)

# Dataset Details

```
Num examples = 37180
Num Epochs = 3
```

# Compared To Human Feedback Model

![](assets/compare.png)

Our model tends to perform closer to the SDXL-Base, but with optimized image details. The model provided in the original paper exhibits better color and detail performance, more in line with human preferences.
This also reflects a characteristic of using self-training to train the original model: it can optimize according to AI preferences while ensuring the capabilities of the original model. Training based on human preference data will make the output quality closely related to the human preference dataset.

# Acknowledgement

This work is based on the [*Diffusion Model Alignment Using Direct Preference Optimization*](https://arxiv.org/abs/2311.12908) method.