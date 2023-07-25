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
import textwrap
from utils import caption_multi_line

device = "cuda" if torch.cuda.is_available() else "cpu"
yolo_model_path = '../yolo_models/yolov8x.pt'
rgb_color = (45, 165, 210)
bgr_color = rgb_color[::-1]
source = '../input_videos/dancing.mp4'
target_cls = [792]
img_save_dir = '../video_out_jpg/'
demo_video_save_path = 'demo_video'
caption_font = ImageFont.truetype("../miscellaneous/fonts/Arial.ttf", 20)

# context = [("What is your name?", "My name is ChatGPT."), ("How are you?", "I'm an AI, so I don't have feelings, but thank you for asking!")]
context = []
message = "what is the person doing?"
template = "Question: {} Answer: {}."
prompt = " ".join([template.format(context[i][0], context[i][1]) for i in range(len(context))]) + " Question: " + message + " Answer:"

print(prompt)

if __name__ == '__main__':
    if os.path.exists(os.path.join(img_save_dir)):
        shutil.rmtree(img_save_dir)
    if not os.path.exists(img_save_dir):
        os.makedirs(img_save_dir)   
    yolo_model = YOLO(yolo_model_path)
    model_blip, vis_processors, _ = load_model_and_preprocess(name="blip2_opt", model_type="pretrain_opt6.7b", is_eval=True, device=device)
    cap = cv2.VideoCapture(source)
    frame_counter = 0
    trackerManager = TrackerManager()
    sample_rate = 15
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        ori_frame = frame.copy()
        pil_image = Image.fromarray(cv2.cvtColor(ori_frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_image)
        results = yolo_model.predict(frame, half=True, imgsz=640, conf=0.5)
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
        if len(target_boxes) > 0:
            trackerManager.update_trackers(target_boxes, merge=True)
        if frame_counter % sample_rate == 0:
            trackerManager.updateTrackerClipImg(ori_frame)
        if len(trackerManager.trackers) > 0:
            for tracker in trackerManager.trackers:
                if tracker.clipImg is not None:
                        tracker_image_dir = os.path.join(img_save_dir, str(tracker.id))
                        if not os.path.exists(tracker_image_dir):
                            os.makedirs(tracker_image_dir) 
                        tracker_image_pil = Image.fromarray(cv2.cvtColor(tracker.clipImg, cv2.COLOR_BGR2RGB))
                        infer_image = vis_processors["eval"](tracker_image_pil).unsqueeze(0).to(device)
                        answer = model_blip.generate({"image": infer_image, "prompt": prompt})
                        # answer = ['this is a image of women dancing and can you see it ha ha ha']
                        infer_image_filename = f"frame{tracker.keepCounter:04d}.jpg"
                        # tracker_img_show = tracker.clipImg.copy()
                        # Put the text caption on the image
                        tracker.caption_show = answer[0]
                        # lines = textwrap.wrap(tracker.caption_show)

                        # lines = []
                        # split_length = 6
                        # split_lines = tracker.caption_show.split(' ')
                        # for i in range(int(len(split_lines) / split_length)):
                        #     lines.append(split_lines[split_length * i: split_length * (i + 1)])
                        # y_text = 10
                        # infer_draw = ImageDraw.Draw(infer_image)
                        # for line in lines:
                        #     line_show = ' '.join(line)
                        #     infer_draw.text((20, y_text), line_show, font=caption_font, fill=rgb_color)
                        #     y_text += caption_font.getsize(line_show)[1] 

                        tracker_image_pil = caption_multi_line((10, 20), tracker.caption_show, tracker_image_pil, caption_font, rgb_color, (0, 0))
                        tracker_image_pil.save(os.path.join(tracker_image_dir, filename))
                        # for line in lines:
                        #     cv2.putText(tracker_img_show, 
                        #                 text=line, 
                        #                 org=(10, y_text), # bottom-left corner of the text string in the image
                        #                 fontFace=cv2.FONT_HERSHEY_SIMPLEX, 
                        #                 fontScale=1, 
                        #                 color=bgr_color, # color in BGR format
                        #                 thickness=2) 
                        #     y_text += 15
                        # cv2.imwrite(os.path.join(tracker_image_dir, infer_image_filename), tracker_img_show)

                x1, y1 = tracker.x1, tracker.y1
                x2, y2 = tracker.x2, tracker.y2
                draw.rectangle([x1, y1, x2, y2], outline=rgb_color)
                pil_image = caption_multi_line((x1, y1), tracker.caption_show, pil_image, caption_font, rgb_color, (0, 0), isBbox=True, split_len=3)
                # lines = []
                # split_length = 6
                # split_lines = tracker.caption_show.split(' ')
                # print('final_split_lines: ', split_lines)
                # y_text = y1 - 10
                # y_text = max(y1 - 10, 0)
                # if 
                # for i in range(int(len(split_lines) / split_length)):
                #     lines.append(split_lines[split_length * i: split_length * (i + 1)])


                # for line in lines:
                # print('lines---------: ', lines)
                # line_show = ' '.join(lines[0])
                # draw.text((x1, y_text), line_show, font=caption_font, fill=rgb_color)
                # y_text -= caption_font.getsize(line_show)[1] 
                # draw.text((x1, y1 - 10), tracker.caption_show, font=caption_font, fill=rgb_color) 
                    
        filename = f"frame{frame_counter:04d}.jpg"
        pil_image.save(os.path.join(img_save_dir, demo_video_save_path, filename))
        frame_counter += 1