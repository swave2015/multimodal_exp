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
from utils import caption_multi_line

device = "cuda" if torch.cuda.is_available() else "cpu"
rgb_color = (0, 0, 0)
bgr_color = rgb_color[::-1]
source_path = '../single_videos/toaster'
target_cls = [792, 75, 377, 224]
# target_cls = [98, 575]
img_save_dir_base = '../video_out_jpg/'
demo_video_save_path = 'demo_video'
caption_font = ImageFont.truetype("../miscellaneous/fonts/Arial.ttf", 30)
caption_font_inputImg = ImageFont.truetype("../miscellaneous/fonts/Arial.ttf", 30)
open_blip = True
if open_blip:
    model_blip, vis_processors, _ = load_model_and_preprocess(name="blip2_opt", model_type="pretrain_opt6.7b", is_eval=True, device=device)

def infer_video(source):
    source_basename = os.path.splitext(os.path.basename(source))[0]
    img_save_dir = os.path.join(img_save_dir_base, source_basename)
    if os.path.exists(os.path.join(img_save_dir)):
        shutil.rmtree(img_save_dir)
    if not os.path.exists(img_save_dir):
        os.makedirs(img_save_dir)   
    
    cap = cv2.VideoCapture(source)
    frame_counter = 0
    sample_rate = 15
    caption_show = ''
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        ori_frame = frame.copy()
        pil_image = Image.fromarray(cv2.cvtColor(ori_frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_image)
        
        if frame_counter % sample_rate == 0:
            if open_blip:
                infer_image = vis_processors["eval"](pil_image).unsqueeze(0).to(device)
                message = "describe the image, what is the person doing with the bread machine?"
                template = "Question: {} Answer: {}."
                prompt = "Question: " + message + " Answer:"
                # if tracker.x1 > 900:
                #     prompt = "Question: " + "I see a package and a boy, what is the boy doing?" + " Answer:"
                print('prompt_input: ', prompt)
                answer = model_blip.generate({"image": infer_image, "prompt": prompt})
                if answer[0] != '':
                    caption_show = answer[0]
            # Put the text caption on the image
            print('tracker.caption_show: ', caption_show)


        draw.text((15, 15), caption_show, font=caption_font, fill=rgb_color)
        filename = f"frame{frame_counter:04d}.jpg"
        label_file_name = f"frame{frame_counter:04d}.txt"
        final_img_save_path = os.path.join(img_save_dir, demo_video_save_path)

        if not os.path.exists(final_img_save_path):
            os.makedirs(final_img_save_path)

        with open(os.path.join(final_img_save_path, label_file_name), 'w') as file:
            pass
                    
        pil_image.save(os.path.join(img_save_dir, demo_video_save_path, filename))
        frame_counter += 1

if __name__ == '__main__':
    for source_file in os.listdir(source_path):
        file_path = os.path.join(source_path, source_file)
        infer_video(file_path)