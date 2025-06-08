from ultralytics.models.yolo.classify import ClassificationTrainer
from dataset import CustomizedDataset

class CustomizedTrainer(ClassificationTrainer):
    def build_dataset(self, img_path: str, mode: str = "train", batch=None):
        return CustomizedDataset(root=img_path, args=self.args, augment=mode == "train", prefix=mode)