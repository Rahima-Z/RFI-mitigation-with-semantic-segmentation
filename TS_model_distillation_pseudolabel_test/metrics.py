
import torch
from torchmetrics.classification import (
    MulticlassJaccardIndex,
    MulticlassF1Score,
    MulticlassPrecision,
    MulticlassRecall,
    MulticlassAccuracy,
    MulticlassConfusionMatrix,
)


class MetricsAccumulator:

    def __init__(self, num_classes=2, device="cpu"):
        self.num_classes = num_classes

        # average = None : renvoie un score par classe plutot qu'une moyenne
        self.iou = MulticlassJaccardIndex(num_classes=num_classes, average=None).to(device)
        self.f1 = MulticlassF1Score(num_classes=num_classes, average=None).to(device)
        self.precision = MulticlassPrecision(num_classes=num_classes, average=None).to(device)
        self.recall = MulticlassRecall(num_classes=num_classes, average=None).to(device)
        self.acc = MulticlassAccuracy(num_classes=num_classes, average="micro").to(device)
        self.confmat = MulticlassConfusionMatrix(num_classes=num_classes).to(device)

        self._metrics = [self.iou, self.f1, self.precision, self.recall, self.acc, self.confmat]

    def reset(self):
        for m in self._metrics:
            m.reset()

    def update(self, preds, target):

        for m in self._metrics:
            m.update(preds, target)

    def compute(self):
        iou = self.iou.compute()
        f1 = self.f1.compute()          # F1 : coefficient de Dice pour un score binaire par pixel
        precision = self.precision.compute()
        recall = self.recall.compute()
        acc = self.acc.compute()
        cm = self.confmat.compute()     # cm[i, j] = vraie classe i, prédite classe j

        result = {
            "pixel_acc": acc.item(),
            "mean_iou": iou.mean().item(),
            "mean_dice": f1.mean().item(),
        }

        for c in range(self.num_classes):
            result[f"iou_class{c}"] = iou[c].item()
            result[f"dice_class{c}"] = f1[c].item()
            result[f"precision_class{c}"] = precision[c].item()
            result[f"recall_class{c}"] = recall[c].item()

        if self.num_classes == 2:
            tn, fp = cm[0, 0].item(), cm[0, 1].item()
            fn, tp = cm[1, 0].item(), cm[1, 1].item()
            result["tp_class1"] = tp
            result["fp_class1"] = fp
            result["fn_class1"] = fn
            result["tn_class1"] = tn

        return result
