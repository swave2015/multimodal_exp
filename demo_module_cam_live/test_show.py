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

device = "cuda" if torch.cuda.is_available() else "cpu"
yolo_model_path = '../yolo_models/yolov8x.pt'
rgb_color = (84, 198, 247)
bgr_color = rgb_color[::-1]
source = '/data/xcao/code/multimodal_exp/single_videos/bear/black_bear.MOV'
target_cls = [792, 75, 377, 224]
img_save_dir = '../video_out_jpg/'
demo_video_save_path = 'demo_video'
caption_font = ImageFont.truetype("../miscellaneous/fonts/Arial.ttf", 20)

# context = [("What is your name?", "My name is ChatGPT."), ("How are you?", "I'm an AI, so I don't have feelings, but thank you for asking!")]
# context = []
# message = "describe this image."
# template = "Question: {} Answer: {}."
# prompt = " ".join([template.format(context[i][0], context[i][1]) for i in range(len(context))]) + " Question: " + message + " Answer:"

# print(prompt)

if __name__ == '__main__':
    if os.path.exists(os.path.join(img_save_dir)):
        shutil.rmtree(img_save_dir)
    if not os.path.exists(img_save_dir):
        os.makedirs(img_save_dir)   
    yolo_model = YOLO(yolo_model_path)
    cap = cv2.VideoCapture(source)
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
                            prompt = None

                            # if len(tracker.context) < tracker.context.maxlen:
                            #     message = "describe this image."
                            #     template = "Question: {} Answer: {}."
                            #     prompt = "Question: " + message + " Answer:"
                            #     answer = model_blip.generate({"image": infer_image, "prompt": prompt})
                            #     tracker.context.append(("What happened in the past?", answer[0]))
                            # else:
                            #     context = list(tracker.context)
                            #     # context = []
                            #     # message = "The upper half of the image shows the entire scene, the lower half offers a sequence of a person's actions, reference to what has happened in the past, what action is the person performing in the lower half of the image?"
                            #     message = "reference to what has happened in the past, what is the person doing in this series of photos?"
                            #     template = "Question: {} Answer: {}."
                            #     prompt = " ".join([template.format(context[i][0], context[i][1]) for i in range(len(context))]) + " Question: " + message + " Answer:"
                            #     print('prompt: ', prompt)
                            #     answer = model_blip.generate({"image": infer_image, "prompt": prompt})
                            #     tracker.caption_show = answer[0]
                            # answer = ['this is a image of women dancing and can you see it ha ha ha']

                            tracker.caption_show = 'this is a very long test caption for my video action reck test'
                            infer_image_filename = f"frame{tracker.keepCounter:04d}.jpg"
                            # Put the text caption on the image
                            print('tracker.caption_show: ', tracker.caption_show)
                            tracker_image_pil = caption_multi_line((10, 20), tracker.caption_show, tracker_image_pil, caption_font, rgb_color, (0, 0))
                            tracker_image_pil.save(os.path.join(tracker_image_dir, filename))

        for tracker in trackerManager.trackers:
            x1, y1 = tracker.x1, tracker.y1
            x2, y2 = tracker.x2, tracker.y2
            draw.rectangle([x1, y1, x2, y2], outline=rgb_color)
            if tracker.caption_show != 'recognizing':
                pil_image = caption_multi_line((x1, y1), tracker.caption_show, pil_image, caption_font, rgb_color, (0, 0), isBbox=True, split_len=4)
                    
        filename = f"frame{frame_counter:04d}.jpg"
        final_img_save_path = os.path.join(img_save_dir, demo_video_save_path)
        if not os.path.exists(final_img_save_path):
            os.makedirs(final_img_save_path)
        pil_image.save(os.path.join(img_save_dir, demo_video_save_path, filename))
        frame_counter += 1