import sys
import json


def cal_accuracy(preds, labels):
    assert len(preds) == len(labels)
    corrects = 0
    for pred, label in zip(preds, labels):
        corrects += (pred == label)

    return corrects / len(preds)


label_path = sys.argv[1]
pred_path = sys.argv[2]

with open(label_path, "r") as fin:
    labels = [json.loads(line.strip())["label"] for line in fin]

with open(pred_path, "r") as fin:
    preds = [json.loads(line.strip())["prediction"] for line in fin]

print(cal_accuracy(preds, labels))
