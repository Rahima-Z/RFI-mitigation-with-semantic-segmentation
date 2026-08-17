
import torch
import torch.nn.functional as F
from model_student import AEStudent
from losses import ce_loss
from metrics import MetricsAccumulator



# Sauvegarde / chargement de checkpoints

def save_checkpoint(path, student, metrics, model_config, run_config, epoch):

    torch.save({
        "model_state_dict": student.state_dict(),
        "model_config": model_config,
        "run_config": run_config,
        "metrics": metrics,
        "epoch": epoch,
    }, path)


def load_checkpoint(path, device):

    ckpt = torch.load(path, map_location=device)
    student = AEStudent(**ckpt["model_config"]).to(device)
    student.load_state_dict(ckpt["model_state_dict"])
    student.eval()
    return student, ckpt



def distillation_kl_loss(student_logits, teacher_logits, temperature=4.0):

    if student_logits.shape != teacher_logits.shape:
        raise ValueError(
            f"Shapes incompatibles entre student et teacher : "
            f"student={tuple(student_logits.shape)}, "
            f"teacher={tuple(teacher_logits.shape)}. "
            f"Verifiez la resolution de sortie des deux modeles "
            f"(un F.interpolate sur teacher_logits est peut-etre necessaire)."
        )

    B, C, H, W = student_logits.shape

    s = student_logits.permute(0, 2, 3, 1).reshape(-1, C)
    t = teacher_logits.permute(0, 2, 3, 1).reshape(-1, C)

    log_p_student = F.log_softmax(s / temperature, dim=1)
    p_teacher = F.softmax(t / temperature, dim=1)

    kl = F.kl_div(log_p_student, p_teacher, reduction="batchmean")
    return kl * (temperature ** 2)


def pseudo_label_loss(student_logits, teacher_logits, hard_loss_fn):

    with torch.no_grad():
        pseudo_labels = teacher_logits.argmax(dim=1)
    return hard_loss_fn(student_logits, pseudo_labels)


def compute_loss(student, teacher, x, y, alpha=0.5, temperature=4.0,
                  hard_loss_fn=None, mode="soft"):
    if hard_loss_fn is None:
        hard_loss_fn = ce_loss
    if mode not in ("soft", "hard"):
        raise ValueError(f"mode doit etre 'soft' ou 'hard', recu: {mode!r}")

    student_logits = student(x)

    with torch.no_grad():
        teacher_logits = teacher(x)

    loss_true = hard_loss_fn(student_logits, y)

    if mode == "soft":
        distill_term = distillation_kl_loss(student_logits, teacher_logits, temperature)
    else:  # mode == "hard"
        distill_term = pseudo_label_loss(student_logits, teacher_logits, hard_loss_fn)

    loss = (1 - alpha) * loss_true + alpha * distill_term
    return loss, loss_true, distill_term, student_logits


def train_one_epoch(student, teacher, loader, optimizer, device,
                     alpha=0.5, temperature=4.0, hard_loss_fn=None, mode="soft"):
    student.train()
    teacher.eval()

    total_loss = 0.0
    total_acc = 0.0

    for x, y, _meta in loader:
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        loss, loss_true, distill_term, preds = compute_loss(
            student, teacher, x, y, alpha, temperature, hard_loss_fn, mode)
        loss.backward()
        optimizer.step()

        acc = (preds.argmax(dim=1) == y).float().mean()
        total_loss += loss.item()
        total_acc += acc.item()

    n = len(loader)
    return total_loss / n, total_acc / n


@torch.no_grad()
def evaluate(student, teacher, loader, device, alpha=0.5, temperature=4.0,
             hard_loss_fn=None, num_classes=2, mode="soft"):
    student.eval()
    teacher.eval()

    total_loss = 0.0
    metrics_acc = MetricsAccumulator(num_classes=num_classes, device=device)

    for x, y, _meta in loader:
        x, y = x.to(device), y.to(device)
        loss, loss_true, distill_term, logits = compute_loss(
            student, teacher, x, y, alpha, temperature, hard_loss_fn, mode)

        preds = logits.argmax(dim=1)
        metrics_acc.update(preds, y)
        total_loss += loss.item()

    n = len(loader)
    metrics = metrics_acc.compute()
    metrics["val_loss"] = total_loss / n
    return metrics

