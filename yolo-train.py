from ultralytics import YOLO
 
# Load the model.
model = YOLO('yolov8n.pt')
 
# Training.
results = model.train(
   data='/dataset/data.yaml',
   imgsz=640,
   epochs=50,
   batch=4,
   name='yolov8n_v1'
   verbose = True
)