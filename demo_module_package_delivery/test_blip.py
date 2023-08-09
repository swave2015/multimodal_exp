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

device = "cuda" if torch.cuda.is_available() else "cpu"
yolo_model_path = '../yolo_models/yolov8x.pt'
rgb_color = (45, 165, 210)
bgr_color = rgb_color[::-1]
source = '../input_videos/dancing.mp4'
target_cls = [792]
img_save_dir = '../video_out_jpg/'
demo_video_save_path = 'demo_video'
caption_font = ImageFont.truetype("../miscellaneous/fonts/Arial.ttf", 25)

# context = [("What is your name?", "My name is ChatGPT."), ("How are you?", "I'm an AI, so I don't have feelings, but thank you for asking!")]
context = []
message = "describe this image."
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
                        # answer = model_blip.generate({"image": tracker.clipImg, "prompt": prompt})
                        infer_image_filename = f"frame{tracker.keepCounter:04d}.jpg"
                        # tracker.caption_show = answer[0]
                        tracker_img_show = tracker.clipImg.copy()
                        # Put the text caption on the image
                        # cv2.putText(tracker_img_show, 
                        #             text=answer[0], 
                        #             org=(50, 50), # bottom-left corner of the text string in the image
                        #             fontFace=cv2.FONT_HERSHEY_SIMPLEX, 
                        #             fontScale=1, 
                        #             color=bgr_color, # color in BGR format
                        #             thickness=2) 
                        cv2.imwrite(os.path.join(tracker_image_dir, infer_image_filename), tracker_img_show)

                x1, y1 = tracker.x1, tracker.y1
                x2, y2 = tracker.x2, tracker.y2
                draw.rectangle([x1, y1, x2, y2], outline=rgb_color)
                draw.text((x1, y1 - 10), tracker.caption_show, font=caption_font, fill=rgb_color) 
                    
        filename = f"frame{frame_counter:04d}.jpg"
        pil_image.save(os.path.join(img_save_dir, filename))
        frame_counter += 1