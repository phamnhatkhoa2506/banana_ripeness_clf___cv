from ultralytics import YOLO
from ultralytics.models.yolo.classify import ClassificationTrainer
from dataset import CustomizedDataset


class CustomizedTrainer(ClassificationTrainer):
    def build_dataset(self, img_path: str, mode: str = "train", batch=None):
        return CustomizedDataset(root=img_path, args=self.args, augment=mode == "train", prefix=mode)
    

if __name__ == "__main__":
    model = YOLO("yolov8n-cls.pt")
    model.train(data="Banana-Ripeness-Classification-1/", trainer=CustomizedTrainer, epochs=10, imgsz=224, batch=64)