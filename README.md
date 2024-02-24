# Better version

The version I have implemented does not have strict code specifications and may have issues. For a more professional version, please refer to the official diffusers repository: [train_diffusion_dpo_sdxl.py](https://github.com/huggingface/diffusers/blob/main/examples/research_projects/diffusion_dpo/train_diffusion_dpo_sdxl.py)

# AI Feedback-Based Self-Training Direct Preference Optimization

[![Hits](https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fgithub.com%2Fqqingzheng%2FAI-Self-Training-DPO-SDXL&count_bg=%2379C83D&title_bg=%23555555&icon=&icon_color=%23E7E7E7&title=hits&edge_flat=false)](https://hits.seeyoufarm.com)

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
