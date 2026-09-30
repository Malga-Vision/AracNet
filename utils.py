from typing import List
import torch
import numpy as np
import torch.nn as nn
import os
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt


class Hook:
    """Forward hook that captures a layer's output tensor after each forward pass."""

    def __init__(self, module, backward=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not backward:
            try:
                self.hook = module[-1].register_forward_hook(self.hook_fn)
            except TypeError:
                self.hook = module.register_forward_hook(self.hook_fn)
        else:
            self.hook = module[-1].register_backward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):
        self.output = output

    def close(self):
        self.hook.remove()


class FlatPooling(nn.Module):
    """Adaptive average-pools a 4-D feature map down to a 2-D embedding vector."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            return x
        elif x.dim() == 4:
            return torch.flatten(self.avgpool(x), 1)
        else:
            raise ValueError(f"Unexpected input dimension {x.dim()} in FlatPooling")


def manual_avgpooling(x: torch.Tensor) -> torch.Tensor:
    return x.mean(dim=(-1, -2), keepdim=True).squeeze((-1, -2))


class CumulativeCELoss(nn.Module):
    """Averages cross-entropy loss across all parallel classifier heads."""

    def __init__(self, reduction: str = "mean", *args, **kwargs) -> None:
        assert reduction in {"none", "sum", "mean"}
        super().__init__(*args, **kwargs)
        self.reduction = reduction
        self.loss_fn = nn.CrossEntropyLoss(reduction="none")

    def forward(self, y_preds: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        loss = torch.zeros(y_true.size(0), device=y_true.device, dtype=torch.float32)
        for y_pred in y_preds.transpose(1, 0):
            loss += self.loss_fn(y_pred, y_true)
        loss = loss / y_preds.size(0)

        match self.reduction:
            case "none": return loss
            case "sum":  return loss.sum()
            case "mean": return loss.mean()


# ---------------------------------------------------------------------------
# Layer-ranking scores — used to select the most bimodal parallel head
# ---------------------------------------------------------------------------

def ranking_score_histogram(softmax, targets, target_class, **_):
    """
    Spread-based bimodality score: (p99 - p1) * median of per-class softmax values.
    Higher → the head's confidence distribution is more spread out (bimodal signal).
    """
    probs = torch.nn.functional.softmax(softmax.detach().cpu(), dim=1).clamp(max=1.0).numpy().astype(np.float64)
    class_mask = targets.cpu().numpy() == target_class
    maxp = probs[class_mask][:, target_class]

    p1   = np.percentile(maxp, 1)
    p99  = np.percentile(maxp, 99)
    p50  = np.percentile(maxp, 50)
    return (p99 - p1) * p50



# ---------------------------------------------------------------------------
# Visualisation helpers
# ---------------------------------------------------------------------------

@torch.no_grad()
def softmax_distribution_hist(network_outputs, targets, biases, target_class, epoch, wb, layer="", dataset="BFFHQ", save_dir=None):
    """
    Plots and optionally logs the softmax confidence distribution split by
    bias-alignment (aligned vs. conflicting) for a given class and layer.
    """
    import seaborn as sns

    criterion = nn.CrossEntropyLoss(reduction="none")
    try:
        values = criterion(network_outputs, targets)
    except RuntimeError:
        network_outputs = network_outputs[:, 0, :]
        values = criterion(network_outputs, targets)

    target_mask = targets == target_class

    if dataset == "bar":
        aligned_mask    = biases[target_mask] == 0
        conflicting_mask = biases[target_mask] == 1
    elif dataset == "UrbanCars":
        aligned_mask     = biases[target_mask] == 0
        conflicting_mask  = biases[target_mask] == 1
        conflicting_mask2 = biases[target_mask] == 2
        conflicting_mask3 = biases[target_mask] == 3
    else:
        aligned_mask    = targets[target_mask] == biases[target_mask]
        conflicting_mask = targets[target_mask] != biases[target_mask]

    if dataset != "UrbanCars":
        network_outputs = network_outputs[target_mask].double()
        softmax_probs   = torch.nn.functional.softmax(network_outputs, dim=1).clamp(min=1e-6, max=1.0)
        softmax_on_target = softmax_probs[:, target_class]

        aligned_probs_np    = softmax_on_target[aligned_mask].cpu().numpy()
        conflicting_probs_np = softmax_on_target[conflicting_mask].cpu().numpy()

        try:
            bins = 40
            plt.rcParams.update({
                "font.family": "serif",
                "font.size": 60,
                "axes.labelsize": 60,
                "xtick.labelsize": 60,
                "ytick.labelsize": 60,
                "legend.fontsize": 60,
                "legend.frameon": False,
            })
            plt.figure(figsize=(16, 12))
            w_a = 100 * np.ones_like(aligned_probs_np) / len(aligned_probs_np)
            w_c = 100 * np.ones_like(conflicting_probs_np) / len(conflicting_probs_np)

            plt.hist(aligned_probs_np,    bins=bins, alpha=0.65, weights=w_a, color="#0004fe", label="Aligned")
            plt.hist(conflicting_probs_np, bins=bins, alpha=0.65, weights=w_c, color="#ff2600", label="Conflicting")
            plt.xlabel(r"$c_\ell(x)$")
            plt.ylabel("samples (%)")

            if layer == 0:
                plt.legend(loc="upper center")
                plt.tight_layout(pad=0.2)

            if epoch % 10 == 0 or epoch == 49:
                filename = f"hist_epoch_{epoch}_class{target_class}_layer{layer}.pdf"
                if save_dir:
                    save_path = os.path.join(save_dir, filename)
                else:
                    save_path = filename
                plt.savefig(save_path, format="pdf", bbox_inches="tight")

            plt.close()
        except Exception:
            pass

    else:
        network_outputs = network_outputs[target_mask].double()
        softmax_probs   = torch.nn.functional.softmax(network_outputs, dim=1).clamp(min=1e-6, max=1.0)
        softmax_on_target = softmax_probs[:, target_class]

        aligned_probs_np    = softmax_on_target[aligned_mask].cpu().numpy()
        conflicting_probs_np  = softmax_on_target[conflicting_mask].cpu().numpy()
        conflicting_probs_np2 = softmax_on_target[conflicting_mask2].cpu().numpy()
        conflicting_probs_np3 = softmax_on_target[conflicting_mask3].cpu().numpy()

        plt.rcParams.update({
            "text.usetex": False,
            "font.family": "serif",
            "font.size": 18,
            "font.weight": "bold",
            "axes.labelweight": "bold",
        })
        plt.figure(figsize=(16, 10))
        plt.hist(aligned_probs_np,    bins=50, alpha=0.5, color="#37b84a", density=True, label="Aligned")
        try:
            plt.hist(conflicting_probs_np,  bins=50, alpha=0.5, color="#ff7f00",  density=True, label="Conflicting_bg")
            plt.hist(conflicting_probs_np2, bins=50, alpha=0.5, color="#00ffc8",  density=True, label="Conflicting_coobj")
            plt.hist(conflicting_probs_np3, bins=50, alpha=0.5, color="#ff2600",  density=True, label="Conflicting_both")
            sns.kdeplot(conflicting_probs_np,  color="#ff7f00", linewidth=2, warn_singular=False)
            sns.kdeplot(conflicting_probs_np2, color="#00ffc8", linewidth=2, warn_singular=False)
            sns.kdeplot(conflicting_probs_np3, color="#ff2600", linewidth=2, warn_singular=False)
        except Exception:
            pass
        sns.kdeplot(aligned_probs_np, color="#5337b8", linewidth=2)
        plt.axvline(0.5, color="r", linestyle="dashed", linewidth=2, label="Random Guess")
        plt.xlim((0.0, 1.0))
        plt.ylabel("Density (%)")
        plt.grid()
        plt.xlabel(f"Softmax Output (Epoch {epoch})")
        plt.legend()

    if wb is not None:
        img_name = (
            f"Class_{target_class}_probs_layer_{layer}_{epoch}"
            if isinstance(epoch, str)
            else f"Class_{target_class}_probs_layer_{layer}"
        )
        wb.log_output({img_name: wb.backend.Image(plt), "epoch": epoch})

    plt.close()
