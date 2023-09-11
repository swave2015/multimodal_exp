import sys
sys.path.insert(0, '../../multimodal_exp')

import torch
from PIL import Image
import time
from lavis.models import load_model_and_preprocess
from lavis.processors import load_processor

raw_image = Image.open("/data/xcao/code/multimodal_exp/docs/_static/merlion.png").convert("RGB")
device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
caption = "merlion in Singapore"

model, vis_processors, text_processors = load_model_and_preprocess("blip_image_text_matching", "large", device=device, is_eval=True)

img = vis_processors["eval"](raw_image).unsqueeze(0).to(device)
txt = text_processors["eval"](caption)

for i in range(100):
    start_time = time.perf_counter()
    itm_output = model({"image": img, "text_input": txt}, match_head="itm")
    end_time = time.perf_counter()
    itm_processing_time = (end_time - start_time) * 1000
    itm_scores = torch.nn.functional.softmax(itm_output, dim=1)
    print(f"Image processing time: {itm_processing_time} ms")
    print(f'The image and text are matched with a probability of {itm_scores[:, 1].item():.3%}')
