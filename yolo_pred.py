from ultralytics import YOLO
import cv2

yolo_model_path = '/data/xcao/code/multimodal_exp/yolo_models/yolov8x.pt'
img_path = '/data/xcao/code/multimodal_exp/video_out_jpg/little_boy_steal_package/demo_video/frame0000.jpg'
yolo_model = YOLO(yolo_model_path)
frame = cv2.imread(img_path)
results = yolo_model.predict(frame, half=True, imgsz=640, conf=0.3)
res_plotted = results[0].plot()
cv2.imwrite("./result_yolo.jpg", res_plotted)