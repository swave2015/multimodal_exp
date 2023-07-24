import torch
from PIL import Image
from lavis.models import load_model_and_preprocess

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")
print('device ', device)
# load sample image
raw_image = Image.open("/data/xcao/code/BLIP2/slice_2733_1_0_frame_0002.jpg").convert("RGB")
width, height = raw_image.size
print('width, height', width, height)
# loads BLIP caption base model, with finetuned checkpoints on MSCOCO captioning dataset.
# this also loads the associated image processors
# model, vis_processors, _ = load_model_and_preprocess(name="blip2_t5", model_type="pretrain_flant5xxl", is_eval=True, device=device)
model, vis_processors, _ = load_model_and_preprocess(name="blip2_t5", model_type="pretrain_flant5xxl", is_eval=True, device=device)
# model, vis_processors, _ = load_model_and_preprocess(name="blip2", model_type="pretrain", is_eval=True, device=device)
# preprocess the image
# vis_processors stores image transforms for "train" and "eval" (validation / testing / inference)

image = vis_processors["eval"](raw_image).unsqueeze(0).to(device)
print('image_type: ', image.dtype)
print('begin_infer--------------------------------------------------')
answer = model.generate({"image": image, "prompt": "Question:is there is a bird in this image? Answer:"})
# answer = model.generate({"image": image, "prompt": "describe this image"})
print(answer)