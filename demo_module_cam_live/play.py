import sys
sys.path.insert(0, '../../multimodal_exp')
from lavis.models import load_model_and_preprocess
import torch
from ultralytics import YOLO
import cv2
import time
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from collections import deque
from TrackerManager import TrackerManager
from lavis.models import load_model_and_preprocess
import os
import shutil
from demo_module_cam_live.demo_utils import caption_multi_line
from transformers import XLMRobertaTokenizer
from torchvision import transforms
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD
from timm.models import create_model
import modeling_finetune
import utils as beit3Utils

device = "cuda" if torch.cuda.is_available() else "cpu"

prompt_list = ['person', 'cat', 'dog', 'car', 'microwave']
num_max_bpe_tokens = 64
max_len = num_max_bpe_tokens

yolo_model_path = '/home/caoxh/multimodal_exp/models_weights/yolov8n.pt'
rgb_color = (84, 198, 247)
bgr_color = rgb_color[::-1]
source_path = '../single_videos/Package_Delivery_Driver_Gone_Wrong_FAIL'
target_cls = [0]
img_save_dir_base = '../video_out_jpg/'
demo_video_save_path = 'demo_video'
caption_font = ImageFont.truetype("../miscellaneous/fonts/Arial.ttf", 20)
caption_font_inputImg = ImageFont.truetype("../miscellaneous/fonts/Arial.ttf", 30)
yolo_model = YOLO(yolo_model_path)
cv2.namedWindow('Video Feed', cv2.WINDOW_NORMAL)
# cv2.setWindowProperty('Video Feed', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

input_size = 384
tokenizer = XLMRobertaTokenizer("/home/caoxh/multimodal_exp/models_weights/beit3.spm")
bos_token_id = tokenizer.bos_token_id
eos_token_id = tokenizer.eos_token_id
pad_token_id = tokenizer.pad_token_id
input_text_tensor = []
input_padding_mask_tensor = []
for prompt in prompt_list:
    tokens = tokenizer.tokenize(prompt)
    token_ids = tokenizer.convert_tokens_to_ids(tokens)
    token_ids = [bos_token_id] + token_ids[:] + [eos_token_id]
    num_tokens = len(token_ids)
    token_ids = token_ids + [pad_token_id] * (max_len - num_tokens)
    token_ids_tensor = torch.tensor(token_ids).cuda()
    input_text_tensor.append(token_ids_tensor)
    padding_mask = [0] * num_tokens + [1] * (max_len - num_tokens)
    padding_mask_tensor = torch.tensor(padding_mask).cuda()
    input_padding_mask_tensor.append(padding_mask_tensor)

input_text_tensor = torch.stack(input_text_tensor).cuda()
input_padding_mask_tensor = torch.stack(input_padding_mask_tensor).cuda()
transform = transforms.Compose([
            transforms.Resize((input_size, input_size), interpolation=3), 
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_INCEPTION_MEAN, std=IMAGENET_INCEPTION_STD)
        ])
model_path = '/home/caoxh/multimodal_exp/models_weights/beit3_large_patch16_384_coco_retrieval.pth'
model_config = "beit3_large_patch16_384_retrieval"
drop_path=0.1
vocab_size = 64010
checkpoint_activations = None
beit3_model = create_model(
    model_config,
    pretrained=False,
    drop_path_rate=drop_path,
    vocab_size=vocab_size,
    checkpoint_activations=checkpoint_activations,
)
print('create_model_over----------------------------------')
beit3Utils.load_model_and_may_interpolate(model_path, beit3_model, 'model|module', '')
beit3_model.to(device)
beit3_model.eval()
_, language_cls = beit3_model(
            text_description=input_text_tensor, 
            padding_mask=input_padding_mask_tensor, 
            only_infer=True)
language_cls = language_cls.detach().cpu().numpy()
print('language_cls_shape: ', language_cls.shape)

def infer_video():
    cap = cv2.VideoCapture(0)
    # cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    frame_counter = 0
    trackerManager = TrackerManager()
    sample_rate = 30
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        ori_frame = frame.copy()
        pil_image = Image.fromarray(cv2.cvtColor(ori_frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_image)
        results = yolo_model.predict(frame, half=True, imgsz=640, conf=0.3)
        boxes = results[0].boxes
        target_boxes = []
        for box in boxes:
            if box.cls.cpu() in target_cls:
                x1, y1, x2, y2 = box.xyxy.cpu().int()[0]
                x1 = x1.item()
                y1 = y1.item()
                x2 = x2.item()
                y2 = y2.item()
                target_boxes.append([x1, y1, x2, y2]) 
        trackerManager.update_trackers(target_boxes, merge=True)
        if frame_counter % sample_rate == 0:
            trackerManager.updateTrackerClipImg(ori_frame)
            if len(trackerManager.trackers) > 0:
                for tracker in trackerManager.trackers:
                    if tracker.clipImg is not None:
                            tracker_image_pil = Image.fromarray(cv2.cvtColor(tracker.clipImg, cv2.COLOR_BGR2RGB))
                            input_img = transform(tracker_image_pil)
                            input_img = input_img.unsqueeze(0).cuda()
                            vision_cls, _ = beit3_model(image=input_img, only_infer=True)
                            vision_cls = vision_cls.detach().cpu().numpy()
                            scores = vision_cls @ language_cls.t()
                            print('scores')
                            prompt = None

        for tracker in trackerManager.trackers:
            x1, y1 = tracker.x1, tracker.y1
            x2, y2 = tracker.x2, tracker.y2
            draw.rectangle([(x1, y1), (x2, y2)], outline=rgb_color, width=2)

        tracker_image_pil = caption_multi_line((x1, y1), str(tracker.id), 
                                                pil_image, caption_font_inputImg, 
                                                rgb_color, (0, 0), split_len=10)

        opencv_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        cv2.imshow('Video Feed', opencv_image)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

                
        frame_counter += 1

if __name__ == '__main__':
    infer_video()