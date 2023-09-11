import sys
sys.path.insert(0, '../../multimodal_exp')

import torch
from PIL import Image

from lavis.models import load_model_and_preprocess
from lavis.processors import load_processor

raw_image = Image.open("../docs/_static/merlion.png").convert("RGB")
device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
caption = "fight"
model, vis_processors, text_processors = load_model_and_preprocess("blip2_image_text_matching", "pretrain", device=device, is_eval=True)
img = vis_processors["eval"](raw_image).unsqueeze(0).to(device)
txt = text_processors["eval"](caption)
itm_output = model({"image": img, "text_input": txt}, match_head="itm")
itm_scores = torch.nn.functional.softmax(itm_output, dim=1)
print(f'The image and text are matched with a probability of {itm_scores[:, 1].item():.3%}')
itc_score = model({"image": img, "text_input": txt}, match_head='itc')
print('The image feature and text feature has a cosine similarity of %.4f'%itc_score)