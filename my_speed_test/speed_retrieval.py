import sys
sys.path.insert(0, '../../multimodal_exp')

import torch
from PIL import Image
from lavis.models import load_model_and_preprocess
import time

raw_image = Image.open("/data/xcao/code/multimodal_exp/docs/_static/merlion.png").convert("RGB")
caption = "a large fountain spewing water into the air"
device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
model, vis_processors, txt_processors = load_model_and_preprocess(name="blip2_feature_extractor", model_type="pretrain", is_eval=True, device=device)

for i in range(100):
    start_time = time.perf_counter()
    image = vis_processors["eval"](raw_image).unsqueeze(0).to(device)
    end_time = time.perf_counter()
    image_processing_time = (end_time - start_time) * 1000
    print(f"Image processing time: {image_processing_time} ms")


    start_time = time.perf_counter()
    text_input = txt_processors["eval"](caption)
    end_time = time.perf_counter()
    text_processing_time = (end_time - start_time) * 1000
    print(f"Text processing time: {text_processing_time} seconds")

sample = {"image": image, "text_input": [text_input]}