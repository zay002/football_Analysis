from ultralytics import YOLO

model = YOLO('models/best.pt')  # Load model

results = model.predict('input_videos/test (1).mp4', save=True)