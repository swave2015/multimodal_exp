import sys
sys.path.insert(0, '../../multimodal_exp')
from lavis.models import load_model_and_preprocess
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

if __name__ == '__main__':
    model, vis_processors, _ = load_model_and_preprocess(name="blip2_opt", model_type="pretrain_opt6.7b", is_eval=True, device=device)