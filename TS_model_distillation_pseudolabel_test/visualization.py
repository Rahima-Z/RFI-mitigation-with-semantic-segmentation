import matplotlib.pyplot as plt
import torch
from torchmetrics.classification import BinaryPrecisionRecallCurve
import numpy as np


# ----------------------------------------------------------------------
@torch.no_grad()
def compute_precision_recall_curve(student, loader, device, positive_class=1, thresholds=41):
    student.eval()
    metric = BinaryPrecisionRecallCurve(thresholds=thresholds).to(device)

    for x, y, _meta in loader:
        x, y = x.to(device), y.to(device)
        logits = student(x)
        probs = torch.softmax(logits, dim=1)[:, positive_class]
        metric.update(probs.reshape(-1), y.reshape(-1))

    precisions, recalls, thresholds_used = metric.compute()
    return (thresholds_used.cpu().numpy(),
            precisions.cpu().numpy(),
            recalls.cpu().numpy())


def plot_precision_recall_curve(thresholds, precisions, recalls,
                                 title="Precision-Recall", save_path=None):

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].plot(recalls, precisions, marker="o", markersize=3)
    axes[0].set_xlabel("Rappel (Recall)")
    axes[0].set_ylabel("Précision")
    axes[0].set_title(f"{title} — courbe PR")
    axes[0].set_xlim(0, 1)
    axes[0].set_ylim(0, 1)
    axes[0].grid(alpha=0.3)

    n = len(thresholds)
    axes[1].plot(thresholds, precisions[:n], label="Précision", marker="o", markersize=3)
    axes[1].plot(thresholds, recalls[:n], label="Rappel", marker="o", markersize=3)
    axes[1].set_xlabel("Seuil de décision")
    axes[1].set_ylabel("Score")
    axes[1].set_title(f"{title} — vs seuil")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Sauvegarde : {save_path}")
    plt.close(fig)


# ----------------------------------------------------------------------
def plot_confusion_matrix(tp, fp, fn, tn, class_names=("background", "flagged"),
                           title="Matrice de confusion", save_path=None):

   
    cm = np.array([
        [tn, fp],
        [fn, tp],
    ])
    cm_norm = cm / cm.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(5, 4.5))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(class_names)
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Prédit")
    ax.set_ylabel("Vrai")
    ax.set_title(title)

    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]:.0f}\n({cm_norm[i, j] * 100:.1f}%)",
                     ha="center", va="center",
                     color="white" if cm_norm[i, j] > 0.5 else "black")

    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Sauvegarde : {save_path}")
    plt.close(fig)
