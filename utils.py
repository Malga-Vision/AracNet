from typing import Any, Dict, List, Tuple
from os import path, makedirs
from scipy.stats import skew, kurtosis
import torch
import numpy as np
from sklearn.manifold import TSNE
import torch.functional as nn
from matplotlib import pyplot as plt
import torch.nn as nn
from diptest import diptest


def visualize_latent_concat(Z, y, img_name, show_plot: bool = False, figsize: tuple = (12, 8), save_fig: bool=False):    
    tsne2D = TSNE(n_components=2, init="pca", random_state=0)

    Ztsne_2D = tsne2D.fit_transform(Z.reshape(Z.shape[0], -1))
    
    fig = plt.figure(figsize=figsize)
    ax2d   = fig.add_subplot()    
    
    ax2d.scatter(Ztsne_2D[:, 0], Ztsne_2D[:, 1], c=y, cmap=plt.cm.get_cmap("jet", len(np.unique(y))))
    
    fig.tight_layout()
    plt.title(img_name.split(".")[0])
    plt.axis("off")
    if save_fig:
        plt.savefig(img_name)
    if show_plot:
        plt.show()
    plt.close()

    return img_name

def visualize_latent_space(vae_model, X, y, batch_size, img_name: str = "tsne.png"):
    import warnings
    warnings.filterwarnings("ignore")
    
    tsne2D = TSNE(n_components=2, init="pca", random_state=0)
    # tsne3D = TSNE(n_components=3, init="pca", random_state=0)    
    Z: np.ndarray = vae_model.get_latent(X, batch_size)
    Ztsne_2D = tsne2D.fit_transform(Z.reshape(Z.shape[0], -1))
    # Ztsne_3D = tsne3D.fit_transform(Z.reshape(Z.shape[0], -1))
    
    # fig = plt.figure(figsize=(12, 8))
    # ax2d   = fig.add_subplot(2, 2, 1)    
    # ax3d_1 = fig.add_subplot(2, 2, 2, projection="3d")
    # ax3d_2 = fig.add_subplot(2, 2, 3, projection="3d")
    # ax3d_3 = fig.add_subplot(2, 2, 4, projection="3d")

    # ax3d_1.view_init(30, 0)
    # ax3d_2.view_init(30, 90)
    # ax3d_3.view_init(30, 270)
    plt.figure(figsize=(12, 8))
    plt.scatter(Ztsne_2D[:, 0], Ztsne_2D[:, 1], c=y, cmap=plt.cm.get_cmap("jet", len(np.unique(y))))
    # ax3d_1.scatter(Ztsne_3D[:, 0], Ztsne_3D[:, 1], Ztsne_3D[:, 2], c=y, cmap=plt.cm.get_cmap("jet", len(np.unique(y))))
    # ax3d_2.scatter(Ztsne_3D[:, 0], Ztsne_3D[:, 1], Ztsne_3D[:, 2], c=y, cmap=plt.cm.get_cmap("jet", len(np.unique(y))))
    # ax3d_3.scatter(Ztsne_3D[:, 0], Ztsne_3D[:, 1], Ztsne_3D[:, 2], c=y, cmap=plt.cm.get_cmap("jet", len(np.unique(y))))
    
    # fig.tight_layout()
    plt.savefig(img_name)
    plt.close()
    # plt.scatter(Z_tsne[:, 0], Z_tsne[:, 1], c=y, cmap=plt.cm.get_cmap("jet", 10))
    # plt.colorbar()
    # plt.savefig("tsne.png")
    

def lognorm(sample, mean, logvar, raxis=1):
    log2pi = np.log(2. * np.pi)

    return torch.sum(
        -.5 * ((sample - mean) ** 2. * torch.exp(-logvar) + logvar + log2pi),
        dim=raxis)

class Hook:
    """Registers a hook at a specific layer of a network"""

    def __init__(self, module, backward=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # TODO: Add pooled version, and grids of feature maps with ~tv.makegrid
        if backward == False:
            try:
                self.hook = module[-1].register_forward_hook(self.hook_fn)
                self.name = module[0]
            except:
                self.hook = module.register_forward_hook(self.hook_fn)
                self.name = module.__class__.__name__

        else:
            self.hook = module[-1].register_backward_hook(self.hook_fn)
            self.name = module[0]

    def hook_fn(self, module, input, output):
        self.input = input
        self.output = output

    def close(self):
        self.hook.remove()



class GCELoss(nn.Module):
    def __init__(self, q: float = 0.7, reduction: str = "mean", *args, **kwargs) -> None:
        if reduction not in {"mean", "none", "sum"}:
            raise ValueError(f"Unsupported reduction parameter {reduction}. Use one among ['mean', 'sum', 'none'] passed as 'string' type argument.")

        super().__init__(*args, **kwargs)
        self.reduction: str = reduction
        self.q: float = q

    def forward(self, outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        outputs = outputs.double()
        softmax_probs : torch.Tensor = torch.nn.functional.softmax(outputs, dim=1)
        gathered_preds: torch.Tensor = torch.gather(softmax_probs, dim=1, index=torch.unsqueeze(targets, 1))
        unreduced_loss: torch.Tensor = (1 - gathered_preds ** self.q) / self.q
        return unreduced_loss.mean()
            # match self.reduction:
            #     case "mean": return unreduced_loss.mean()
            #     case "sum" : return unreduced_loss.sum()
            #     case "none": return unreduced_loss

    def __repr__(self) -> str:
        return f"GCELoss(q={self.q}, reduction={self.reduction})"


class FlatPooling(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # ✅ If input is already flattened (2D), skip pooling
        if x.dim() == 2:
            return x
        elif x.dim() == 4:
            x = self.avgpool(x)
            return torch.flatten(x, 1)
        else:
            raise ValueError(f"Unexpected input dimension {x.dim()} in FlatPooling")

# class FlatPooling(nn.Module):
#     def __init__(self, *args, **kwargs) -> None:
#         super().__init__(*args, **kwargs)
#         self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         x = self.avgpool(x)
#         return torch.flatten(x, 1)
from sklearn.mixture import GaussianMixture
import numpy as np


def logit(p):
    p = np.clip(p, 1e-7, 1-1e-7)


    return np.log(p / (1-p))

def ranking_score_histogram(softmax,targets, target_class, thresh,thresh_max, t_high=0.9, eps_var=1e-6):
    """
    Computes a ranking score for a histogram of softmax predictions.
    
    Args:
        softmax: np.array of shape (N, num_classes)
        t_high: threshold to consider a sample confident
        eps_var: variance threshold to detect memorization collapse
        
    Returns:
        dict with metrics and final_score for ranking
    """
    softmax = torch.nn.functional.softmax(softmax.detach().cpu(), dim=1).clamp(max=1.0).numpy().astype(np.float64)
    
    mask = (targets.cpu().numpy() == target_class)
    # max probability per sample

    maxp = softmax[mask][:,target_class]
    
        # Hartigan’s Dip Test (measure of multimodality)
   # dip, pval = diptest(maxp)
    thresh = float(thresh)
    # Tail mass: how many samples are NOT extremely confident
    

    #URBAN CARS
    # tail_mass = (maxp < 0.3).sum()/len(maxp)
    
    # max_maxx = (maxp > 0.8).sum()/len(maxp)

    # Final score:
    #   high dip  → bimodal → good → high score 0.15 0.3 for BAR
    #   high tail → has conflicting samples → good → high score
    #   memorized → dip ~ 0, tail_mass ~ 0 → low score
    #tail_mass = (maxp < thresh).sum()/len(maxp)
    tail_thr = np.percentile(maxp,1)
    max_thr = np.percentile(maxp,99)
    max_2 = np.percentile(maxp,50)

    # tail_mass = (maxp < tail_thr).sum()/len(maxp)
    # max_maxx = (maxp > max_thr).sum()/len(maxp)

  #  if max_maxx>thresh_max:     #0.8 FOR URBANCARS, and for CIFAR10. 0.5 FOR bar  (O 0.3)
    #score = (max_thr - tail_thr) * max_2

    spread = (max_thr - tail_thr)*max_2
    #bulkness = 1.0 - (max_thr - max_2) / (spread + 1e-10)   # close to 1 if top is flat

    score = spread 
#    else:
 #       score=0

    return score


def ranking_score_histogram_SARLE(softmax,targets, target_class, thresh,thresh_max, t_high=0.9, eps_var=1e-6):
    """
    Computes a ranking score for a histogram of softmax predictions.
    
    Args:
        softmax: np.array of shape (N, num_classes)
        t_high: threshold to consider a sample confident
        eps_var: variance threshold to detect memorization collapse
        
    Returns:
        dict with metrics and final_score for ranking
    """
    softmax = torch.nn.functional.softmax(softmax.detach().cpu(), dim=1).clamp(max=1.0).numpy().astype(np.float64)
    
    mask = (targets.cpu().numpy() == target_class)
    # max probability per sample

    maxp = softmax[mask][:,target_class]
    
        # Hartigan’s Dip Test (measure of multimodality)
   # dip, pval = diptest(maxp)
    thresh = float(thresh)
    # Tail mass: how many samples are NOT extremely confident
    

    #URBAN CARS
    # tail_mass = (maxp < 0.3).sum()/len(maxp)
    
    # max_maxx = (maxp > 0.8).sum()/len(maxp)

    # Final score:
    #   high dip  → bimodal → good → high score 0.15 0.3 for BAR
    #   high tail → has conflicting samples → good → high score
    #   memorized → dip ~ 0, tail_mass ~ 0 → low score
    #tail_mass = (maxp < thresh).sum()/len(maxp)
    g = skew(maxp, bias=False)
    k = kurtosis(maxp, fisher=False, bias=False)  # Pearson kurtosis, not excess kurtosis

    if np.isnan(g) or np.isnan(k) or abs(k) < 1e-12:
        score = np.nan
    else:
        score = (g ** 2 + 1.0) / k
#    else:
 #       score=0

    return score

def ranking_score_histogram_tail(softmax,targets, target_class, thresh,thresh_max, t_high=0.9, eps_var=1e-6):
    """
    Computes a ranking score for a histogram of softmax predictions.
    
    Args:
        softmax: np.array of shape (N, num_classes)
        t_high: threshold to consider a sample confident
        eps_var: variance threshold to detect memorization collapse
        
    Returns:
        dict with metrics and final_score for ranking
    """
    softmax = torch.nn.functional.softmax(softmax.detach().cpu(), dim=1).clamp(max=1.0).numpy().astype(np.float64)
    
    mask = (targets.cpu().numpy() == target_class)
    # max probability per sample

    maxp = softmax[mask][:,target_class]
    
        # Hartigan’s Dip Test (measure of multimodality)
   # dip, pval = diptest(maxp)
    thresh = float(thresh)
    # Tail mass: how many samples are NOT extremely confident
    

    #URBAN CARS
    # tail_mass = (maxp < 0.3).sum()/len(maxp)
    
    # max_maxx = (maxp > 0.8).sum()/len(maxp)

    # Final score:
    #   high dip  → bimodal → good → high score 0.15 0.3 for BAR
    #   high tail → has conflicting samples → good → high score
    #   memorized → dip ~ 0, tail_mass ~ 0 → low score
    #tail_mass = (maxp < thresh).sum()/len(maxp)
    dip, pval = diptest(maxp)
    score = dip
#    else:
 #       score=0

    return score




def ranking_score_histogram_supervised(dataset, biases, softmax,targets, target_class, thresh,thresh_max, t_high=0.9, eps_var=1e-6):
    """
    Computes a ranking score for a histogram of softmax predictions.
    
    Args:
        softmax: np.array of shape (N, num_classes)
        t_high: threshold to consider a sample confident
        eps_var: variance threshold to detect memorization collapse
        
    Returns:
        dict with metrics and final_score for ranking
    """
    softmax = torch.nn.functional.softmax(softmax.detach().cpu(), dim=1).clamp(max=1.0).numpy().astype(np.float64)
    
    target_mask = (targets.cpu().numpy() == target_class)

    if dataset=='bar':
        aligned_mask = biases[target_mask] == 0
        conflicting_mask = biases[target_mask] == 1 

    
    elif dataset=='UrbanCars':

        aligned_mask = biases[target_mask] == 0
        conflicting_mask = biases[target_mask] == 1 
        conflicting_mask2 = biases[target_mask] == 2 
        conflicting_mask3 = biases[target_mask] == 3

    
    
    
    else:
        aligned_mask = targets[target_mask] == biases[target_mask]
        conflicting_mask = targets[target_mask] != biases[target_mask]

    # max probability per sample

    
    
    
    maxp_aligned = softmax[aligned_mask][:,target_class]
    maxp_conflicting = softmax[conflicting_mask][:,target_class]



    
        # Hartigan’s Dip Test (measure of multimodality)
   # dip, pval = diptest(maxp)
    thresh = float(thresh)
    # Tail mass: how many samples are NOT extremely confident
    

    #URBAN CARS
    # tail_mass = (maxp < 0.3).sum()/len(maxp)
    
    # max_maxx = (maxp > 0.8).sum()/len(maxp)

    # Final score:
    #   high dip  → bimodal → good → high score 0.15 0.3 for BAR
    #   high tail → has conflicting samples → good → high score
    #   memorized → dip ~ 0, tail_mass ~ 0 → low score
    #tail_mass = (maxp < thresh).sum()/len(maxp)
    bottom_aligned = np.percentile(maxp_aligned,5)
    top_conflicting = np.percentile(maxp_conflicting,95)
    #max_2 = np.percentile(maxp,70)

    # tail_mass = (maxp < tail_thr).sum()/len(maxp)
    # max_maxx = (maxp > max_thr).sum()/len(maxp)

  #  if max_maxx>thresh_max:     #0.8 FOR URBANCARS, and for CIFAR10. 0.5 FOR bar  (O 0.3)
    #score = (max_thr - tail_thr) * max_2

    spread = (bottom_aligned - top_conflicting)
    #bulkness = 1.0 - (max_thr - max_2) / (spread + 1e-10)   # close to 1 if top is flat

    score = spread 
#    else:
 #       score=0

    return score
     

def manual_avgpooling(x: torch.Tensor) -> torch.Tensor:
    return x.mean(dim=(-1, -2), keepdim=True).squeeze((-1, -2))

@torch.no_grad
def softmax_distribution_hist(network_outputs: torch.Tensor, targets, biases, target_class: int, epoch: int, wb, layer="",dataset='BFFHQ'):
    import seaborn as sns
    criterion = torch.nn.CrossEntropyLoss(reduction="none")
    try:
        values = criterion(network_outputs,targets)
    except:
        network_outputs = network_outputs[:,0,:]
        values = criterion(network_outputs,targets)
    target_mask = targets == target_class
    values = values[target_mask]

    if dataset=='bar':
        aligned_mask = biases[target_mask] == 0
        conflicting_mask = biases[target_mask] == 1 

    
    elif dataset=='UrbanCars':

        aligned_mask = biases[target_mask] == 0
        conflicting_mask = biases[target_mask] == 1 
        conflicting_mask2 = biases[target_mask] == 2 
        conflicting_mask3 = biases[target_mask] == 3

    
    
    
    else:
        aligned_mask = targets[target_mask] == biases[target_mask]
        conflicting_mask = targets[target_mask] != biases[target_mask]

    

    if dataset !='UrbanCars':


        network_outputs = network_outputs[target_mask]
        network_outputs = network_outputs.double()     # numerical stability
        softmax_probs: torch.Tensor = torch.nn.functional.softmax(network_outputs, dim=1).clamp(min=1e-6, max=1.0) # logits to probs
        
        softmax_on_target = softmax_probs[:, target_class]    
        #print(f'target_class_{target_class}_layer_{layer}_variance_{torch.var(softmax_on_target)}')
        aligned_probs = softmax_on_target[aligned_mask]
        conflicting_probs = softmax_on_target[conflicting_mask]

        aligned_probs_np = aligned_probs.cpu().numpy()
        conflicting_probs_np = conflicting_probs.cpu().numpy()
        
        # plt.rcParams["text.usetex"] = False
        # plt.rcParams['font.family'] = 'serif'
        # plt.rcParams["font.size"]  = 18
        # plt.rcParams["font.weight"]  = "bold"
        # plt.rcParams["axes.labelweight"] = "bold"
        # plt.rcParams['axes.labelsize'] = 'medium'  # Label font size
        # plt.rcParams['xtick.labelsize'] = 'small'  # X-axis tick font size
        # plt.rcParams['ytick.labelsize'] = 'small'  # Y-axis tick font size
        # plt.rcParams['legend.fontsize'] = 'medium'  # Legend font size
        # plt.rcParams['lines.linewidth'] = 5.0  # Line width
        # plt.rcParams['lines.markersize'] = 8  # Marker size
       
        #plt.hist(aligned_probs_np, bins=50, alpha=0.5, color="#37b84a", density=True, label='Aligned')
        # try:
        #     bins = 50
        #     w_a = 100 * np.ones_like(aligned_probs_np) / len(aligned_probs_np)
        #     w_c = 100 * np.ones_like(conflicting_probs_np) / len(conflicting_probs_np)

        #     plt.hist(aligned_probs_np, bins=bins, alpha=0.5, weights=w_a, color="#0004fe", label="Aligned")
        #     plt.hist(conflicting_probs_np, bins=bins, alpha=0.5, weights=w_c,color = "#ff2600", label="Conflicting")
            


        # except:
        #     pass



        try:
                bins = 40  # keep your bins

                plt.rcParams.update({
                    "font.family": "serif",
                    "font.size": 60,            # base
                    "axes.labelsize": 60,       # big axis labels
                    "xtick.labelsize": 60,
                    "ytick.labelsize": 60,
                    "legend.fontsize": 60,
                    "legend.frameon": False,
                })
                plt.figure(figsize=(16,12))
                w_a = 100 * np.ones_like(aligned_probs_np) / len(aligned_probs_np)
                w_c = 100 * np.ones_like(conflicting_probs_np) / len(conflicting_probs_np)

                plt.hist(aligned_probs_np, bins=bins, alpha=0.65,
                        weights=w_a, color="#0004fe", label="Aligned")

                plt.hist(conflicting_probs_np, bins=bins, alpha=0.65,
                        weights=w_c, color="#ff2600", label="Conflicting")

                plt.xlabel(r"$c_\ell(x)$")     # short + mathematical
                plt.ylabel("samples (%)")               # compact

                if layer==0:
                    plt.legend(loc="upper center")
                    plt.tight_layout(pad=0.2)

                if epoch % 10 == 0 or epoch==49:
                    save_path = f"hist_epoch_{epoch}_class{target_class}_layer{layer}.pdf"
                    plt.savefig(save_path, format="pdf", bbox_inches="tight")
                    print(f"Saved histogram to {save_path}")

                plt.close()

        except:
            pass



        # sns.kdeplot(aligned_probs_np, color='#377eb8', linewidth=2)

        # plt.axvline(0.5, color='r', linestyle='dashed', linewidth=2, label='Random Guess')
        # plt.xlim((0.0, 1.0))
        
        # plt.grid()
        # plt.ylabel("Samples (%)")
        # plt.xlabel(f"P(target_class) (Epoch {epoch})")
        # plt.legend()
        
    else:
        network_outputs = network_outputs[target_mask]
        network_outputs = network_outputs.double()     # numerical stability
        softmax_probs: torch.Tensor = torch.nn.functional.softmax(network_outputs, dim=1).clamp(min=1e-6, max=1.0) # logits to probs
        
        softmax_on_target = softmax_probs[:, target_class]    
        #print(f'target_class_{target_class}_layer_{layer}_variance_{torch.var(softmax_on_target)}')
        aligned_probs = softmax_on_target[aligned_mask]
        conflicting_probs = softmax_on_target[conflicting_mask]
        conflicting_probs2 = softmax_on_target[conflicting_mask2]
        conflicting_probs3 = softmax_on_target[conflicting_mask3]

        aligned_probs_np = aligned_probs.cpu().numpy()
        conflicting_probs_np = conflicting_probs.cpu().numpy()
        conflicting_probs_np2 = conflicting_probs2.cpu().numpy()
        conflicting_probs_np3 = conflicting_probs3.cpu().numpy()
        
        plt.rcParams["text.usetex"] = False
        plt.rcParams['font.family'] = 'serif'
        plt.rcParams["font.size"]  = 18
        plt.rcParams["font.weight"]  = "bold"
        plt.rcParams["axes.labelweight"] = "bold"
        plt.rcParams['axes.labelsize'] = 'medium'  # Label font size
        plt.rcParams['xtick.labelsize'] = 'small'  # X-axis tick font size
        plt.rcParams['ytick.labelsize'] = 'small'  # Y-axis tick font size
        plt.rcParams['legend.fontsize'] = 'medium'  # Legend font size
        plt.rcParams['lines.linewidth'] = 5.0  # Line width
        plt.rcParams['lines.markersize'] = 8  # Marker size
        plt.figure(figsize=(16, 10))
        plt.hist(aligned_probs_np, bins=50, alpha=0.5, color="#37b84a", density=True, label='Aligned')
        try:
            plt.hist(conflicting_probs_np, bins=50, alpha=0.5, color='#ff7f00', density=True, label='Conflicting_bg')
            plt.hist(conflicting_probs_np2, bins=50, alpha=0.5, color="#00ffc8", density=True, label='Conflicting_coobj')
            plt.hist(conflicting_probs_np3, bins=50, alpha=0.5, color="#ff2600", density=True, label='Conflicting_both')

            sns.kdeplot(conflicting_probs_np, color='#ff7f00', linewidth=2, warn_singular=False)
            sns.kdeplot(conflicting_probs_np, color="#00ffc8", linewidth=2, warn_singular=False)
            sns.kdeplot(conflicting_probs_np, color="#ff2600", linewidth=2, warn_singular=False)



        except:
            pass
        sns.kdeplot(aligned_probs_np, color="#5337b8", linewidth=2)

        plt.axvline(0.5, color='r', linestyle='dashed', linewidth=2, label='Random Guess')
        plt.xlim((0.0, 1.0))
        plt.ylabel("Density (%)")
        plt.grid()
        plt.xlabel(f"Softmax Output (Epoch {epoch})")
        plt.legend()
    if wb is not None:
        img_name = f"Class_{target_class}_probs_layer_{layer}_{epoch}" if isinstance(epoch, str) \
            else f"Class_{target_class}_probs_layer_{layer}"
        
        wb.log_output({
            f"{img_name}": wb.backend.Image(plt),
            "epoch": epoch
        })
    
    # plt.savefig(f"Class_{target_class}_probs_epoch_{epoch}_layer_{layer}.pdf", format="pdf", dpi=1200)
    plt.close()
    
class EMA:

    def __init__(self, label, alpha=0.9):
        self.label = label
        self.alpha = alpha
        self.parameter = torch.zeros(label.size(0))
        self.updated = torch.zeros(label.size(0))

    def update(self, data, index):
        self.parameter[index] = self.alpha * self.parameter[index] + (1-self.alpha*self.updated[index]) * data
        self.updated[index] = 1

    def max_loss(self, label):
        label_index = np.where(self.label == label)[0]
        return self.parameter[label_index].max()
    
    
class CumulativeCELoss(nn.Module):
    def __init__(self, reduction: str = "mean", *args, **kwargs) -> None:
        assert reduction in {"none", "sum", "mean"}
        
        super().__init__(*args, **kwargs)
        self.reduction = reduction
        self.loss_fn = nn.CrossEntropyLoss(reduction="none")

    def forward(self, y_preds: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        loss: torch.Tensor = torch.full((y_true.size(0), ), 0.0).to(y_true.device).float()
        for i, y_pred in enumerate(y_preds.transpose(1, 0)):
            cur_loss = self.loss_fn(y_pred, y_true) 
            loss += cur_loss 
        
        match self.reduction:
            case "none": return (loss / y_preds.size(0))
            case "sum" : return (loss / y_preds.size(0)).sum()
            case "mean": return (loss / y_preds.size(0)).mean() 
