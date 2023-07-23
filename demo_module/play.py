import cv2
from YOLOv8_TensorRT.models.engine import TRTModule
import torch
from YOLOv8_TensorRT.models.utils import blob, letterbox, path_to_list
from YOLOv8_TensorRT.models.torch_utils import det_postprocess
from clip_deploy.tensorrt_utils import TensorRTModel
from clip_deploy import transform, tokenize
from YOLOv8_TensorRT.config import CLASSES, COLORS
import onnxruntime
import time
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from collections import deque
from TrackerManager import TrackerManager
import os

def get_text_features(labels):
    txt_sess_options = onnxruntime.SessionOptions()
    txt_run_options = onnxruntime.RunOptions()
    txt_onnx_model_path="/data/clip_trt/convert_output/ViT-L_14@336px.fp16.onnx"
    txt_session = onnxruntime.InferenceSession(txt_onnx_model_path,
                                        sess_options=txt_sess_options,
                                        providers=["CUDAExecutionProvider"])
    text = tokenize(labels)
    text_features = []
    for i in range(len(text)):
        one_text = np.expand_dims(text[i].cpu().numpy(),axis=0)
        text_feature = txt_session.run(["unnorm_text_features"], {"text":one_text})[0] # 未归一化的文本特征
        text_feature = torch.tensor(text_feature)
        text_features.append(text_feature)
    text_features = torch.squeeze(torch.stack(text_features),dim=1)
    text_features = text_features / text_features.norm(dim=1, keepdim=True)
    text_features = text_features.float().cuda()

    return text_features

intput_labels = ["cat is sitting on the ground", "cat is jumping onto the windowsill", "cat is standing on the windowsill"]
# det_clssid_list = [0, 15, 16]
det_clssid_list = [792, 224, 377]
text_features = get_text_features(intput_labels)
video_path = '/data/clip_for_video_action_reck/input_videos/cat_jump_to_windowsill.mp4'
detector_engine_path = '/data/clip_trt/YOLOv8-TensorRT/engines/yolov8x.engine'
clip_engine_path="/data/clip_trt/convert_output/ViT-L_14@336px.img.fp16.trt"
device = torch.device('cuda:0')
DETEngine = TRTModule(detector_engine_path, device)
CLIPEngine = TensorRTModel(clip_engine_path)
preprocess = transform(336)
H, W = DETEngine.inp_info[0].shape[-2:]
DETEngine.set_desired(['num_dets', 'bboxes', 'scores', 'labels'])
frame_counter = 0
sample_rate = 3
caption = ''
final_img = None    
check_counter = 0
bgr_color = COLORS['person']
rgb_color = tuple(bgr_color[::-1])
cap = cv2.VideoCapture(video_path)
img_save_dir = '/data/clip_for_video_action_reck/video_out_jpg'
caption_font = ImageFont.truetype("/data/clip_for_video_action_reck/fonts/Arial.ttf", 25)
box_font = ImageFont.truetype("/data/clip_for_video_action_reck/fonts/Arial.ttf", 30)
trackerManager = TrackerManager()
while True:
    ret, frame = cap.read()
    if not ret:
        break
    print('frame_counter_print: ', frame_counter)    
    start_time = time.perf_counter()
    img_h, img_w, _ = frame.shape
    ori_frame = frame.copy()
    frame, ratio, dwdh = letterbox(frame, (W, H))
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    tensor = blob(rgb, return_seg=False)
    dwdh = torch.asarray(dwdh * 2, dtype=torch.float32, device=device)
    tensor = torch.asarray(tensor, device=device)
    # inference
    detector_start_time = time.perf_counter()
    data = DETEngine(tensor)
    detector_elapsed = round((time.perf_counter() - detector_start_time) * 1000, 2)
    bboxes, scores, labels = det_postprocess(data)
    bboxes -= dwdh
    bboxes /= ratio
    bboxes = bboxes.cpu()
    pil_image = Image.fromarray(cv2.cvtColor(ori_frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_image)
    target_boxes = []
    # min_area = float("inf")
    # min_bbox = None
    for bbox, score, label in zip(bboxes, scores, labels):
        if label in det_clssid_list:
            bbox = bbox.round().int().tolist()
            x1, y1, x2, y2 = bbox
            area = (x2 - x1) * (y2 - y1)
            target_boxes.append(bbox) 
            # if area < min_area:
            #     min_area = area
            #     min_bbox = bbox
    # if min_bbox is not None:
    #     target_boxes.append(min_bbox)
    if len(target_boxes) > 0:
        trackerManager.update_trackers(target_boxes, merge=True)

    if frame_counter % sample_rate == 0:
        trackerManager.updateTrackerClipImg(ori_frame)
    if len(trackerManager.trackers) > 0:
        for tracker in trackerManager.trackers:
            if tracker.clipImg is not None:
                clipImgPIL = Image.fromarray(cv2.cvtColor(tracker.clipImg, cv2.COLOR_BGR2RGB))
                clip_input = preprocess(clipImgPIL).unsqueeze(0).cuda()
                image_features = CLIPEngine(inputs={'image': clip_input})['unnorm_image_features'] 
                logits_per_image = 100 * image_features @ text_features.t()
                logits_per_image = logits_per_image.cpu()
                max_value, max_idx = torch.max(logits_per_image.softmax(dim=-1).squeeze(), dim=0)
                clip_cls_id = int(max_idx)
                caption = intput_labels[clip_cls_id]
                tracker.captionQueue.append(caption)
                tracker.find_max_frequency_caption()
            x1, y1 = tracker.x1, tracker.y1
            x2, y2 = tracker.x2, tracker.y2
            draw.rectangle([x1, y1, x2, y2], outline="red")
            draw.text((x1, y1), tracker.caption_show, font=caption_font, fill=rgb_color) 
    opencv_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    filename = f"frame{frame_counter:04d}.jpg"
    caption_height = 100  # You can adjust this as per your requirement
    caption_image = Image.new('RGB', (opencv_image.shape[1], caption_height), (255, 255, 255))  # White background
    # Set pixels on both sides to black
    caption_pad_w = 0  # width of the black padding on both sides, set this as a config variable
    pixels = caption_image.load()
    for y in range(caption_height):
        for x in range(caption_pad_w):
            # Left side padding
            pixels[x, y] = (0, 0, 0)
            # Right side padding
            pixels[opencv_image.shape[1] - 1 - x, y] = (0, 0, 0)
    draw_caption = ImageDraw.Draw(caption_image)
    captions = [(tracker.caption_show, tracker.x1) for tracker in trackerManager.trackers if tracker.caption_show != 'recognizing' 
                and 'in front of cars' not in tracker.caption_show and tracker.caption_show != 'delivery man']
    captions.sort(key=lambda x: x[1])
    block_width = caption_image.width // 3
    for i, (caption, position) in enumerate(captions):
        block_number = i % 3  # This will give 0 for the first block, 1 for the second, and 2 for the third
        x_position = block_number * block_width + 10  # Center within the block
        if frame_counter >= 30 and caption != 'recognizing' and not ('in front of cars' in caption):
            if caption == 'person is at the front door of the house':
                caption = 'person is at the front door\nof the house'
            draw_caption.text((x_position, 25), caption, fill="black", font=caption_font)
        
    opencv_image_pil = Image.fromarray(cv2.cvtColor(opencv_image, cv2.COLOR_BGR2RGB))
    # Concatenate the original image with the caption image
    final_image_pil = Image.new('RGB', (opencv_image_pil.width, opencv_image_pil.height + caption_height))
    final_image_pil.paste(opencv_image_pil, (0, 0))
    final_image_pil.paste(caption_image, (0, opencv_image_pil.height))
    final_image = cv2.cvtColor(np.array(final_image_pil), cv2.COLOR_RGB2BGR)

    cv2.imwrite(os.path.join(img_save_dir, filename), final_image)
    frame_counter += 1
    cv2.imshow('action_reck', final_image)
    cv2.waitKey(1)