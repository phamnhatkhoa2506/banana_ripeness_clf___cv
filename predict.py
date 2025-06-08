import os
from ultralytics import YOLO

if __name__ == "__main__":
    weights_path = './runs/classify/train2/weights/best.pt'

    if not os.path.exists(weights_path): 
        raise FileNotFoundError("Can not load the weights file")

    model = YOLO(weights_path)
    model.val()

    results = model("./Banana-Ripeness-Classification-1/valid/freshripe/musa-acuminata-banana-ad7d5ed0-394a-11ec-ab3f-d8c4975e38aa_jpg.rf.7b3fe30c076620bec1af793529ee12ed.jpg")

    for result in results:
        probs = result.probs.data.tolist()  # Convert Probs object to list of floats
        classes = result.names

        highest_prob = max(probs)
        highest_prob_index = probs.index(highest_prob)

        print(f"Class: {classes[highest_prob_index]}")
        print(f"Confidence: {highest_prob:.2f}")