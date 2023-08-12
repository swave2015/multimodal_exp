import sys
sys.path.insert(0, '../../multimodal_exp')
import torch
from ultralytics import YOLO
import cv2
import time
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from TrackerManager import TrackerManager
import os
import shutil
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD
from demo_module_cam_live.demo_utils import caption_multi_line
from torchvision import transforms
import base64
import io
import requests
from multiprocessing import Process, Queue

data_queue = Queue()
device = "cuda" if torch.cuda.is_available() else "cpu"
beit3_service = "http://106.75.22.28:3000/beit3"


def worker_service_request():
    while True:
        data = data_queue.get()
        # if data is None:
        #     break  # exit if a None value is received
        # call_start_time = time.time()
        # try:
        #     response = requests.post(beit3_service, json=data)
        #     response.raise_for_status()
        #     print(response.text)
        # except requests.RequestException as e:
        #     print(f"An error occurred: {e}")
        # call_end_time = time.time()
        # elapsed_time = call_end_time - call_start_time
        # print('call_elapsed_time: ', elapsed_time)
        
service_request_process = Process(target=worker_service_request)
service_request_process.start()



prompt_list = ['person', 'cat', 'dog', 'car', 'microwave']
num_max_bpe_tokens = 64
max_len = num_max_bpe_tokens

yolo_model_path = '/home/caoxh/multimodal_exp/models_weights/yolov8l.pt'
rgb_color = (84, 198, 247)
bgr_color = rgb_color[::-1]
source_path = '../single_videos/Package_Delivery_Driver_Gone_Wrong_FAIL'
target_cls = [2]
img_save_dir_base = '../video_out_jpg/'
demo_video_save_path = 'demo_video'
caption_font = ImageFont.truetype("../miscellaneous/fonts/Arial.ttf", 20)
caption_font_inputImg = ImageFont.truetype("../miscellaneous/fonts/Arial.ttf", 30)
yolo_model = YOLO(yolo_model_path)
cv2.namedWindow('Video Feed', cv2.WINDOW_NORMAL)
# cv2.setWindowProperty('Video Feed', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

input_size = 384

transform = transforms.Compose([
            transforms.Resize((input_size, input_size), interpolation=3), 
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_INCEPTION_MEAN, std=IMAGENET_INCEPTION_STD)
        ])
model_path = '/home/caoxh/multimodal_exp/models_weights/beit3_large_patch16_384_coco_retrieval.pth'

def tensor_to_base64(tensor):
    buffer = io.BytesIO()
    torch.save(tensor, buffer)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


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
        results = yolo_model.predict(frame, half=True, imgsz=640, conf=0.3, verbose=False)
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
                        base64_string = tensor_to_base64(input_img)
                        data = {
                            "tensor": base64_string,
                            "frame_id": frame_counter,
                            "tracker_id": tracker.id
                        }
                        queue_length = data_queue.qsize()
                        if queue_length <= 10:
                            data_queue.put(data)
                       
                        # try:
                        #     response = requests.post(beit3_service, json=data)
                        #     response.raise_for_status()
                        #     print(response.text)
                        # except requests.RequestException as e:
                        #     print(f"An error occurred: {e}")
                  

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