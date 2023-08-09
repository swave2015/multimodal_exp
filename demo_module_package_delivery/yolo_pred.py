from ultralytics import YOLO

yolo_model_path = '/data/xcao/code/multimodal_exp/yolo_models/yolov8x.pt'
# Load a pretrained YOLOv8n model
model = YOLO(yolo_model_path)

# Define path to the image file
source = '/data/xcao/code/multimodal_exp/test_imgs/output_0007.jpg'

# Run inference on the source
results = model(source, imgsz=640, conf=0.5)  # list of Results objects
print(results)