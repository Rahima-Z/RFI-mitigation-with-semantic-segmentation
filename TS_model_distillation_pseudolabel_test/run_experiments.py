import csv
import os
import numpy as np
import torch
from torch.utils.data import Subset, DataLoader, random_split
from model_teacher import ResNet18Segmentation
from model_student import AEStudent
from rfidataset import NenuFARDatasetH5
from datetime import datetime
from losses import build_combined_loss
from train import train_one_epoch, evaluate, save_checkpoint, load_checkpoint
from visualization import (
    compute_precision_recall_curve,
    plot_precision_recall_curve,
    plot_confusion_matrix,
)



# Configuration de base + familles d'experiences
BASE_CONFIG = dict(
    components={"ce": 1.0},   # loss par defaut : CE seule
    alpha_kd=0.5,               # poids CE/loss vs signal du teacher
    temperature_kd=4.0,         # temperature de la KL (ignoree si mode="hard")
    mode="soft",                 # "soft" = distillation KL (soft targets)
                                  # "hard" = pseudo-labeling (argmax du teacher)
)

LOSS_KWARGS = dict(
    alpha=0.3, beta=0.7, gamma_focal=2.0, gamma_ft=1.33,
)

LOSS_EXPERIMENTS = [
    ("ce_only",            dict(components={"ce": 1.0})),
    ("dice_only",          dict(components={"dice": 1.0})),
    ("focal_only",         dict(components={"focal": 1.0})),
    ("tversky_only",       dict(components={"tversky": 1.0})),
    ("focal_tversky_only", dict(components={"focal_tversky": 1.0})),
]

ALPHA_EXPERIMENTS = [
    ("alpha_0.1", dict(alpha_kd=0.1)),
    ("alpha_0.3", dict(alpha_kd=0.3)),
    ("alpha_0.5", dict(alpha_kd=0.5)),
    ("alpha_0.7", dict(alpha_kd=0.7)),
    ("alpha_0.9", dict(alpha_kd=0.9)),
]

TEMPERATURE_EXPERIMENTS = [
    #temperature ignoreé en mode "hard"
    ("T_1", dict(temperature_kd=1.0)),
    ("T_2", dict(temperature_kd=2.0)),
    ("T_3", dict(temperature_kd=3.0)),
    ("T_4", dict(temperature_kd=4.0)),
]

MODE_EXPERIMENTS = [
    # distillation ou pseudo-labeling : a alpha/temperature/loss fixes (valeurs de BASE_CONFIG)
    ("mode_soft", dict(mode="soft")),
    ("mode_hard", dict(mode="hard")),
]

#Choisir la liste a lancer
EXPERIMENTS = LOSS_EXPERIMENTS[:2] #LOSS_EXPERIMENTS
# EXPERIMENTS = ALPHA_EXPERIMENTS
# EXPERIMENTS = TEMPERATURE_EXPERIMENTS
# EXPERIMENTS = MODE_EXPERIMENTS
# EXPERIMENTS = LOSS_EXPERIMENTS + ALPHA_EXPERIMENTS + TEMPERATURE_EXPERIMENTS

# Coarse-to-fine : affine alpha_kd ou temperature_kd autour du meilleur point de la grille. Ne s'applique qu'a un paramètre continu a la fois
COARSE_TO_FINE = False           # True pour activer l'affinage
COARSE_TO_FINE_PARAM = "alpha_kd"  # "alpha_kd" ou "temperature_kd"
N_FINE_POINTS = 5                # nombre de points testés dans la passe fine
# Bornes valides pour chaque paramètre (pour ne pas proposer devvaleurs hors domaine lors de l'affinage, ex. alpha < 0 ou > 1)
PARAM_BOUNDS = {
    "alpha_kd": (0.0, 1.0),
    "temperature_kd": (0.1, None),  # T > 0
}



# Hyperparametres fixes du sweep

N_EPOCHS_PER_RUN = 2 #50
N_FOLDS = 2 #5                   # nombre de folds pour la cross-validation
LR = 1e-3
BATCH_SIZE = 4 #64
BEST_METRIC = "dice_class1"    # critère pour la meilleure epoch et le meilleur fold
TOP_K_TO_PLOT = 2 #3
TEST_FRACTION = 0.2 #0.15            # partie du dataset jamais vue pendant la CV
CV_SEED = 42

# Dossier de sortie
BASE_OUTPUT_DIR = "/TS_model_distillation_pseudolabel_test/results"
RUN_TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
RUN_DIR = os.path.join(BASE_OUTPUT_DIR, RUN_TIMESTAMP)

FIGURES_DIR = os.path.join(RUN_DIR, "figures")
CHECKPOINTS_DIR = os.path.join(RUN_DIR, "checkpoints")
CSV_PATH = os.path.join(RUN_DIR, "experiment_results.csv")

# Decoupage K-Fold 
def k_fold_indices(n_samples, k, seed=42):

    rng = np.random.default_rng(seed)
    indices = rng.permutation(n_samples)

    fold_sizes = np.full(k, n_samples // k, dtype=int)
    fold_sizes[: n_samples % k] += 1

    folds, start = [], 0
    for size in fold_sizes:
        folds.append(indices[start:start + size])
        start += size

    for i in range(k):
        val_idx = folds[i]
        train_idx = np.concatenate([folds[j] for j in range(k) if j != i])
        yield train_idx, val_idx


# Entrainement d'un seul fold

def train_one_fold(config, teacher, train_loader, val_loader, device, n_epochs):
    hard_loss_fn = build_combined_loss(config["components"], **LOSS_KWARGS)
    alpha = config["alpha_kd"]
    temperature = config["temperature_kd"]
    mode = config["mode"]

    student = AEStudent(num_classes=2, dropout_p=0.2).to(device)
    optimizer = torch.optim.Adam(student.parameters(), lr=LR)

    best_score = -1.0
    best_metrics = None
    best_state_dict = None
    best_epoch = -1

    for epoch in range(n_epochs):
        train_one_epoch(student, teacher, train_loader, optimizer, device,
                         alpha=alpha, temperature=temperature, hard_loss_fn=hard_loss_fn, mode=mode)
        val_metrics = evaluate(student, teacher, val_loader, device,
                                alpha=alpha, temperature=temperature, hard_loss_fn=hard_loss_fn, mode=mode)

        score = val_metrics[BEST_METRIC]
        if score > best_score:
            best_score = score
            best_metrics = val_metrics
            best_epoch = epoch
            best_state_dict = {k: v.detach().clone() for k, v in student.state_dict().items()}

    del student, optimizer
    return best_score, best_metrics, best_epoch, best_state_dict



# Cross-validation complete d'une config (K folds)
def run_cross_validated_experiment(name, overrides, teacher, train_val_dataset, device):
    config = {**BASE_CONFIG, **overrides}
    print(f"\n=== Experiment: {name} ({overrides}) ===")

    n_samples = len(train_val_dataset)
    fold_scores = []
    best_overall_score = -1.0
    best_overall_state_dict = None
    best_overall_metrics = None

    for fold_idx, (train_idx, val_idx) in enumerate(k_fold_indices(n_samples, N_FOLDS, seed=CV_SEED)):
        train_subset = Subset(train_val_dataset, train_idx)
        val_subset = Subset(train_val_dataset, val_idx)

        train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_subset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

        score, metrics, epoch, state_dict = train_one_fold(
            config, teacher, train_loader, val_loader, device, N_EPOCHS_PER_RUN)

        print(f"  Fold {fold_idx + 1}/{N_FOLDS} | best {BEST_METRIC}={score:.4f} (epoch {epoch + 1})")
        fold_scores.append(score)

        if score > best_overall_score:
            best_overall_score = score
            best_overall_state_dict = state_dict
            best_overall_metrics = metrics

    mean_score = float(np.mean(fold_scores))
    std_score = float(np.std(fold_scores))
    print(f"  --> {name} : {BEST_METRIC} = {mean_score:.4f} +/- {std_score:.4f} "
          f"(sur {N_FOLDS} folds)")

    # On sauvegarde uniquement le meilleur modèle individuel parmi les K folds mais le score retenu pour comparer les configs est la moyenne sur les K folds
    os.makedirs(CHECKPOINTS_DIR, exist_ok=True)
    checkpoint_path = os.path.join(CHECKPOINTS_DIR, f"{name}_best.pth")

    best_student = AEStudent(num_classes=2, dropout_p=0.2).to(device)
    best_student.load_state_dict(best_overall_state_dict)

    save_checkpoint(
        path=checkpoint_path,
        student=best_student,
        metrics=best_overall_metrics,
        model_config={"num_classes": 2, "dropout_p": 0.2},
        run_config={
            "name": name,
            "components": config["components"],
            "loss_kwargs": LOSS_KWARGS,
            "alpha_kd": config["alpha_kd"],
            "temperature_kd": config["temperature_kd"],
            "mode": config["mode"],
            "lr": LR,
            "batch_size": BATCH_SIZE,
            "best_metric_name": BEST_METRIC,
            "cv_mean": mean_score,
            "cv_std": std_score,
            "n_folds": N_FOLDS,
        },
        epoch=-1,  # non pertinent au niveau agrégé (score = moyenne sur les folds)
    )

    return dict(
        name=name,
        overrides=overrides,
        cv_mean=mean_score,
        cv_std=std_score,
        fold_scores=fold_scores,
        checkpoint_path=checkpoint_path,
    )



# Coarse-to-fine : affinage automatique autour du meilleur point

def refine_grid(best_value, coarse_values, n_points=5, min_val=None, max_val=None):
    
    sorted_vals = sorted(coarse_values)
    idx = sorted_vals.index(best_value)

    if len(sorted_vals) > 1:
        step = sorted_vals[1] - sorted_vals[0]
    else:
        step = best_value * 0.5 if best_value != 0 else 0.5

    lower = sorted_vals[idx - 1] if idx > 0 else best_value - step
    upper = sorted_vals[idx + 1] if idx < len(sorted_vals) - 1 else best_value + step

    if min_val is not None:
        lower = max(lower, min_val)
    if max_val is not None:
        upper = min(upper, max_val)

    fine_values = np.linspace(lower, upper, n_points)
    # ne pas retester une valeur deja couverte par la passe grossière
    fine_values = [v for v in fine_values if not np.isclose(v, best_value, atol=1e-6)]
    return fine_values


def run_coarse_to_fine(param_name, coarse_experiments, teacher, train_val_dataset, device,
                        n_fine_points=5, min_val=None, max_val=None):

    print(f"\n{'#' * 70}\n# PASSE GROSSIERE — {param_name}\n{'#' * 70}")
    coarse_results = []
    for name, overrides in coarse_experiments:
        coarse_results.append(
            run_cross_validated_experiment(name, overrides, teacher, train_val_dataset, device))

    coarse_sorted = sorted(coarse_results, key=lambda r: r["cv_mean"], reverse=True)
    best_coarse = coarse_sorted[0]
    best_value = best_coarse["overrides"][param_name]
    coarse_values = [ov[param_name] for _, ov in coarse_experiments]

    print(f"\nMeilleure valeur grossière : {param_name}={best_value} "
          f"(cv_mean={best_coarse['cv_mean']:.4f} +/- {best_coarse['cv_std']:.4f})")

    fine_values = refine_grid(best_value, coarse_values, n_points=n_fine_points,
                               min_val=min_val, max_val=max_val)

    if not fine_values:
        print("Aucune nouvelle valeur à tester en passe fine (fenêtre trop étroite) — "
              "on garde le résultat de la passe grossière.")
        return coarse_sorted

    fine_experiments = [
        (f"{param_name}_fine_{v:.3f}", {param_name: float(v)})
        for v in fine_values
    ]

    print(f"\n{'#' * 70}\n# PASSE FINE — {param_name} : "
          f"{[round(float(v), 3) for v in fine_values]}\n{'#' * 70}")
    fine_results = []
    for name, overrides in fine_experiments:
        fine_results.append(
            run_cross_validated_experiment(name, overrides, teacher, train_val_dataset, device))

    all_results = coarse_results + fine_results
    all_sorted = sorted(all_results, key=lambda r: r["cv_mean"], reverse=True)

    coarse_names = {r["name"] for r in coarse_results}
    print(f"\n{'=' * 80}\nRESUME COARSE-TO-FINE — {param_name}\n{'=' * 80}")
    for r in all_sorted:
        tag = "[grossier]" if r["name"] in coarse_names else "[fin]     "
        print(f"{tag} {r['name']:<28} {r['cv_mean']:.4f} +/- {r['cv_std']:.4f}   {r['overrides']}")

    return all_sorted



# Boucle principale
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #  Teacher
    ckpt_path = "TS_model_distillation_pseudolabel_test/teacher_model/model_and_data/best_model.pth"  
    teacher = ResNet18Segmentation(num_classes=2,pretrained=False,dropout_p=0.3).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    teacher.load_state_dict(ckpt["model_state_dict"])
    teacher.eval()

    for p in teacher.parameters():
        p.requires_grad = False

    # Datasets

    h5_path = "TS_model_distillation_pseudolabel_test/teacher_model/model_and_data/train_supervised_secondattempt.h5"

    full_dataset = NenuFARDatasetH5(
        h5_path=h5_path,
        transform=None,
        return_meta=True,
        )



    #Réduction du dataset pour tester
    TEST = True         # mettre a False pour l'entraînement complet
    TEST_SIZE = 40 #200       # nombre de données a garder en mode test

    if TEST:
        print("Mode test")
        n_total_full = len(full_dataset)
        rng = np.random.default_rng(seed=42)
        debug_indices = rng.choice(n_total_full, size=min(TEST_SIZE, n_total_full), replace=False)
        dataset = Subset(full_dataset, debug_indices)
    else:
        dataset = full_dataset



    #  Test set fige, jamais utilise pendant la cross-validation 
    n_total = len(dataset)
    n_test = int(TEST_FRACTION * n_total)
    n_train_val = n_total - n_test
    train_val_dataset, test_dataset = random_split(
        dataset, [n_train_val, n_test],
        generator=torch.Generator().manual_seed(CV_SEED),
    )
    print(f"Dataset : {n_total} echantillons -> "
          f"{n_train_val} pour la cross-validation ({N_FOLDS} folds), "
          f"{n_test} reserves pour l'evaluation finale")

    # Sweep avec cross-validation
    if COARSE_TO_FINE:
        if COARSE_TO_FINE_PARAM not in ("alpha_kd", "temperature_kd"):
            raise ValueError(
                "COARSE_TO_FINE_PARAM doit etre 'alpha_kd' ou 'temperature_kd' "
                "(parametre continu) — pas adapte a une loss categorielle.")
        min_val, max_val = PARAM_BOUNDS[COARSE_TO_FINE_PARAM]
        results_sorted = run_coarse_to_fine(
            COARSE_TO_FINE_PARAM, EXPERIMENTS, teacher, train_val_dataset, device,
            n_fine_points=N_FINE_POINTS, min_val=min_val, max_val=max_val,
        )
    else:
        results = []
        for name, overrides in EXPERIMENTS:
            result = run_cross_validated_experiment(
                name, overrides, teacher, train_val_dataset, device)
            results.append(result)

        # trie par score CV moyen (uniquement pour la grille normale ; en coarse-to-fine, results_sorted vient deja de run_coarse_to_fine)
        results_sorted = sorted(results, key=lambda r: r["cv_mean"], reverse=True)

    print("\n" + "=" * 80)
    print(f"RESUME (trie par {BEST_METRIC} moyen decroissant, {N_FOLDS}-fold CV)")
    print("=" * 80)
    for r in results_sorted:
        print(f"{r['name']:<20} {r['cv_mean']:.4f} +/- {r['cv_std']:.4f}   {r['overrides']}")

    with open("experiment_results.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["name", "overrides", "cv_mean", "cv_std"] +
                         [f"fold_{i + 1}" for i in range(N_FOLDS)])
        for r in results_sorted:
            writer.writerow([r["name"], r["overrides"], r["cv_mean"], r["cv_std"]] + r["fold_scores"])
    print("\nResultats sauvegardes dans experiment_results.csv")

    # Figures pour le top-K, evaluées sur le TEST SET (jamais vu)
    os.makedirs(FIGURES_DIR, exist_ok=True)
    top_configs = results_sorted[:TOP_K_TO_PLOT]
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    print(f"\nGeneration des figures pour le top {TOP_K_TO_PLOT} sur le test set : "
          f"{[r['name'] for r in top_configs]}")

    for rank, r in enumerate(top_configs, start=1):
        student, ckpt = load_checkpoint(r["checkpoint_path"], device)
        run_cfg = ckpt["run_config"]
        hard_loss_fn = build_combined_loss(run_cfg["components"], **run_cfg["loss_kwargs"])

        # Métriques sur le test set
        test_metrics = evaluate(student, teacher, test_loader, device,
                                 alpha=run_cfg["alpha_kd"], temperature=run_cfg["temperature_kd"],
                                 hard_loss_fn=hard_loss_fn, mode=run_cfg["mode"])

        plot_confusion_matrix(
            tp=test_metrics["tp_class1"], fp=test_metrics["fp_class1"],
            fn=test_metrics["fn_class1"], tn=test_metrics["tn_class1"],
            title=f"[{rank}] {r['name']} — matrice de confusion (test set)",
            save_path=os.path.join(FIGURES_DIR, f"{rank:02d}_{r['name']}_confusion.png"),
        )

        thresholds, precisions, recalls = compute_precision_recall_curve(student, test_loader, device)
        plot_precision_recall_curve(
            thresholds, precisions, recalls,
            title=f"[{rank}] {r['name']} (test set)",
            save_path=os.path.join(FIGURES_DIR, f"{rank:02d}_{r['name']}_pr_curve.png"),
        )

    print(f"\nFigures sauvegardees dans '{FIGURES_DIR}/'")


if __name__ == "__main__":
    main()
