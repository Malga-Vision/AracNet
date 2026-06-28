from typing import Any, Callable, Dict, Iterator, List
from collections import OrderedDict
import torch
import torchvision
from torch import nn
from metrics import *
from tqdm import tqdm
import os
import torch.nn.functional as F
from resnet import resnet20
from utils import Hook, FlatPooling, CumulativeCELoss, softmax_distribution_hist, ranking_score_histogram


PATH_TO_MODELS = "./saved_models/"
os.makedirs(PATH_TO_MODELS, exist_ok=True)


# ---------------------------------------------------------------------------
# Internal epoch-level evaluation (called from training loops)
# ---------------------------------------------------------------------------

def _evaluate_epoch(model, dataset_name, val_loader, device, epoch, wb, train_set=None):
    """Prints per-epoch accuracy split by bias alignment. Called during training."""
    model.eval()

    if dataset_name in ("waterbirds", "BFFHQ"):
        with torch.no_grad():
            aligned    = torch.zeros((10, 10))
            conflicting = torch.zeros((10, 10))

            for _, (images, labels, _) in enumerate(val_loader):
                images      = images.to(device)
                class_labels = labels[0].to(device)
                bias_labels  = labels[1].to(device)
                output = model(images)

                for i in range(images.shape[0]):
                    if bias_labels[i] == class_labels[i]:
                        aligned[class_labels[i], torch.argmax(output[i])] += 1
                    else:
                        conflicting[class_labels[i], torch.argmax(output[i])] += 1

            A = aligned.numpy()
            B = conflicting.numpy()
            print(f"epoch: {epoch}")
            print(f"avg_aligned={np.trace(A) / np.sum(A):.4f}  avg_conflicting={np.trace(B) / np.sum(B):.4f}")
            print(f"aligned_cls0={A[0,0]/(A[0,0]+A[0,1]):.4f}  aligned_cls1={A[1,1]/(A[1,0]+A[1,1]):.4f}")
            print(f"conflict_cls0={B[0,0]/(B[0,0]+B[0,1]):.4f}  conflict_cls1={B[1,1]/(B[1,0]+B[1,1]):.4f}")

    elif dataset_name == "bar":
        confusion = torch.zeros((6, 6)).numpy()
        with torch.no_grad():
            model.eval()
            for _, (images, labels, _) in enumerate(val_loader):
                images       = images.to(device)
                class_labels = labels[0].to(device)
                output = model(images)
                for i in range(images.shape[0]):
                    confusion[class_labels[i], torch.argmax(output[i])] += 1

        avg_acc = np.trace(confusion) / np.sum(confusion)
        print(f"AVERAGE ACCURACY {avg_acc:.4f}")
        for i in range(6):
            print(f"  class_{i}: {confusion[i,i] / np.sum(confusion, axis=1)[i]:.4f}")

        wb.log_output({
            "epoch": epoch,
            "average_accuracy": avg_acc,
            **{f"class_{i}": confusion[i, i] / np.sum(confusion, axis=1)[i] for i in range(6)},
        })

    elif dataset_name == "UrbanCars":
        label_results = torch.zeros((2, 4), device=device)
        counts        = torch.zeros((2, 4), device=device)

        for _, (images, labels, _) in enumerate(val_loader):
            images       = images.to(device)
            class_labels = labels[0].to(device)
            bias_labels  = labels[1].to(device)
            output = model(images)

            for i in range(images.size(0)):
                pred = torch.argmax(output[i])
                if pred == class_labels[i]:
                    label_results[class_labels[i], bias_labels[i]] += 1
                counts[class_labels[i], bias_labels[i]] += 1

        per_group_acc = label_results / counts
        print(f"epoch: {epoch+1}  per_group_acc={per_group_acc}")

        _, group_counts = train_set.get_sampling_weights(classes_only=False).unique(return_counts=True)
        train_dist  = (group_counts / len(train_set)).to(per_group_acc.device)
        per_cls_avg = per_group_acc.mean(dim=0)
        in_acc      = torch.sum(per_cls_avg * train_dist)
        gaps        = torch.ones_like(per_cls_avg) * in_acc - per_cls_avg
        gaps[0]     = in_acc
        print(gaps)

    else:
        # CIFAR-10C and any other multi-class dataset
        confusion = torch.zeros((10, 10)).numpy()
        with torch.no_grad():
            model.eval()
            for _, (images, labels, _) in enumerate(val_loader):
                images       = images.to(device)
                class_labels = labels[0].to(device)
                output = model(images)
                for i in range(images.shape[0]):
                    confusion[class_labels[i], torch.argmax(output[i])] += 1

        avg_acc = np.trace(confusion) / np.sum(confusion)
        print(f"AVERAGE ACCURACY {avg_acc:.4f}")
        for i in range(10):
            print(f"  class_{i}: {confusion[i,i] / np.sum(confusion, axis=1)[i]:.4f}")

        wb.log_output({
            "epoch": epoch,
            "average_accuracy": avg_acc,
            **{f"class_{i}": confusion[i, i] / np.sum(confusion, axis=1)[i] for i in range(10)},
        })


# ---------------------------------------------------------------------------
# AracNet model
# ---------------------------------------------------------------------------

class AracNet(nn.Module):

    def __setup_monitors_resnet50(self):
        weights = torchvision.models.ResNet50_Weights.DEFAULT if self.pretrained else None
        self.base_model = torchvision.models.resnet50(weights=weights)
        self.embedding_dim = self.base_model.fc.in_features
        self.base_model.fc = nn.Linear(self.embedding_dim, self.num_classes)
        self.base_model.avgpool = FlatPooling()
        self._attach_resnet_heads(mock_input_size=(1, 3, 224, 224))

    def __setup_monitors_resnet18(self):
        weights = torchvision.models.ResNet18_Weights.DEFAULT if self.pretrained else None
        self.base_model = torchvision.models.resnet18(weights=weights)
        self.embedding_dim = self.base_model.fc.in_features
        self.base_model.fc = nn.Linear(self.embedding_dim, self.num_classes)
        self.base_model.avgpool = FlatPooling()
        self._attach_resnet_heads(mock_input_size=(1, 3, 224, 224))

    def __setup_monitors_resnet20(self):
        self.base_model = resnet20(num_classes=self.num_classes).to(self.device)
        self.embedding_dim = self.base_model.fc.in_features
        self.base_model.fc = nn.Linear(self.embedding_dim, self.num_classes).to(self.device)
        self.base_model.avgpool = FlatPooling()
        self._attach_resnet_heads(mock_input_size=(1, 3, 224, 224))

    def _attach_resnet_heads(self, mock_input_size):
        """
        Walks every named child module of the ResNet backbone, infers its output
        shape with a mock forward pass, and attaches a parallel linear classification
        head plus a forward hook. Layers that don't produce spatial feature maps
        suitable for probing (conv1, bn1, relu, avgpool, fc) are skipped.
        """
        self.named_layers = OrderedDict(self.base_model.named_children())
        model_device = next(self.base_model.parameters()).device
        mock_x = torch.randn(mock_input_size).to(model_device)

        with torch.no_grad():
            for module_name, module in self.named_layers.items():
                out = module(mock_x)
                out_shape = out.shape[1:]
                mock_x = torch.randn((1,) + out_shape).to(model_device)

                if module_name in {"fc", "avgpool", "bn1", "relu", "conv1"}:
                    continue

                setattr(
                    self.base_model,
                    module_name,
                    nn.Sequential(self.named_layers[module_name], nn.Identity()),
                )

                if out_shape[-1] > 1:
                    parallel_head = nn.Sequential(
                        FlatPooling(),
                        nn.Linear(out_shape[0], self.num_classes),
                    ).to(self.device)
                else:
                    parallel_head = nn.Linear(out_shape[0], self.num_classes).to(self.device)

                hook = Hook(getattr(self.base_model, module_name), backward=False)
                self.parallel_heads[module_name] = parallel_head
                self.hooks[module_name] = hook

    def __setup_monitors_vit_b_16(self):
        weights = torchvision.models.ViT_B_16_Weights.DEFAULT if self.pretrained else None
        self.base_model = torchvision.models.vit_b_16(weights=weights)
        self.embedding_dim = self.base_model.hidden_dim
        self.base_model.heads = nn.Linear(self.embedding_dim, self.num_classes)

        # Only probe a sparse subset of transformer blocks to keep overhead low.
        PROBED_BLOCKS = {0, 1, 2, 4, 6}

        self.named_layers = OrderedDict({
            "conv_proj": self.base_model.conv_proj,
            **{f"block_{i}": block for i, block in enumerate(self.base_model.encoder.layers)},
            "ln": self.base_model.encoder.ln,
        })

        mock_x = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            seq = self.base_model.conv_proj(mock_x)
            seq = seq.flatten(2).transpose(1, 2)
            cls = self.base_model.class_token.expand(seq.size(0), -1, -1)
            seq = torch.cat([cls, seq], dim=1)
            seq = seq + self.base_model.encoder.pos_embedding

            for name, module in self.named_layers.items():
                if name == "conv_proj":
                    continue
                seq = module(seq)

                block_idx = int(name.split("_")[1]) if name.startswith("block_") else None
                if block_idx is not None and block_idx not in PROBED_BLOCKS:
                    continue

                cls_dim = seq.shape[-1]
                head = nn.Linear(cls_dim, self.num_classes).to(self.device)
                hook = Hook(module, backward=False)
                self.parallel_heads[name] = head
                self.hooks[name] = hook

    def __init__(
        self,
        num_classes: int,
        aracne: bool = False,
        base_model_name: str = "resnet18",
        pretrained: bool = True,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)

        self.num_classes     = num_classes
        self.aracne          = aracne
        self.pretrained      = pretrained
        self.base_model_name = base_model_name
        self.device          = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

        self.parallel_heads: Dict[str, nn.Module] = {}
        self.hooks: Dict[str, Hook] = {}

        match base_model_name:
            case "resnet50":
                self.__setup_monitors_resnet50()
            case "resnet18":
                self.__setup_monitors_resnet18()
            case "resnet20":
                self.__setup_monitors_resnet20()
            case "ViTs-16":
                self.__setup_monitors_vit_b_16()
            case "vgg16":
                raise NotImplementedError("VGG-16 backbone is not yet supported.")

        self.base_model = self.base_model.to(self.device)
        self.loss_fn = nn.CrossEntropyLoss(reduction="none").to(self.device)
        self.aracne_loss_fn = CumulativeCELoss(reduction="mean")
        self.base_model.train()

    def set_aracne(self, mode: bool) -> None:
        self.aracne = mode

    def freeze_body(self, flag: bool) -> None:
        if flag:
            self.base_model.eval()
            for p in self.base_model.parameters():
                p.requires_grad = False
            try:
                self.base_model.fc.requires_grad_(False)
                self.base_model.fc.train(False)
            except AttributeError:
                # ViT uses .heads instead of .fc
                self.base_model.heads.requires_grad_(False)
                self.base_model.heads.train(False)
        else:
            self.base_model.train()
            for p in self.base_model.parameters():
                p.requires_grad = True

    def freeze_legs(self, flag: bool) -> None:
        for head in self.parallel_heads.values():
            head.eval() if flag else head.train()
            for p in head.parameters():
                p.requires_grad = not flag

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base_model(x)


# ---------------------------------------------------------------------------
# Full evaluation pass (used after training to compute and log metrics)
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_model(
    model: AracNet,
    dataloader,
    num_classes,
    num_bias_attributes,
    wb,
    make_figures=True,
    eval_name="test",
    dataset="BFFHQ",
):
    model.eval()
    groups_size = (num_classes,) * (num_bias_attributes + 1)

    loss_task_tot   = AverageMeter()
    heads_losses_tot = [AverageMeter() for _ in range(len(model.parallel_heads))]
    heads_accs_tot   = [AverageMeter() for _ in range(len(model.parallel_heads))]
    top1             = AverageMeter()
    subgroup_top1    = AverageMeterSubgroups(size=groups_size, device=model.device)

    pbar = tqdm(dataloader, total=len(dataloader), leave=True, dynamic_ncols=True)

    parallel_preds, all_outputs, all_targets, all_biases = [], [], [], []

    for _, (images, labels, _) in enumerate(pbar):
        images      : torch.Tensor = images.to(model.device)
        class_labels: torch.Tensor = labels[0].to(model.device)
        bias_labels : torch.Tensor = labels[1].to(model.device)
        output      : torch.Tensor = model(images)

        if make_figures and model.aracne:
            model.parallel_z = {key: model.hooks[key].output for key in model.hooks}
            model.parallel_y = torch.cat(
                [model.parallel_heads[key](model.parallel_z[key]).unsqueeze(1) for key in model.parallel_z],
                dim=1,
            )
            parallel_preds.append(model.parallel_y)

            for i, key in enumerate(model.parallel_heads):
                y_head = model.parallel_y.transpose(1, 0)[i]
                head_loss = model.loss_fn(y_head, class_labels).mean()
                heads_losses_tot[i].update(head_loss.item(), images.size(0))
                heads_accs_tot[i].update(accuracy(y_head, class_labels, topk=(1,))[0], images.size(0))

        if make_figures:
            all_outputs.append(output)
            all_targets.append(class_labels)
            all_biases.append(bias_labels)

        loss = model.loss_fn(output, class_labels).mean()
        loss_task_tot.update(loss.item(), images.size(0))

        acc1 = accuracy(output, class_labels, topk=(1,))
        subgroup_masks = get_subgroup_masks(labels=labels, num_classes=groups_size, device=model.device)
        subgroup_acc1  = accuracy_subgroup(output, class_labels, subgroup_masks, num_classes=num_classes)

        top1.update(acc1[0], images.size(0))
        subgroup_top1.update(subgroup_acc1, subgroup_masks)

        acc_a = round(regroup_by(subgroup_top1, ("aligned",))[0].item(), 4)
        acc_m = round(regroup_by(subgroup_top1, ("misaligned",))[0].item(), 4)
        pbar.set_postfix(acc1=top1.avg, acc_a=acc_a, acc_m=acc_m)
        wb.log_output({"acc_1": top1.avg, "acc_a": acc_a, "acc_m": acc_m})

    if make_figures:
        all_outputs = torch.cat(all_outputs, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        all_biases  = torch.cat(all_biases,  dim=0)

        if model.aracne:
            parallel_preds = torch.cat(parallel_preds, dim=0)
            for l in range(len(model.parallel_heads)):
                softmax_distribution_hist(parallel_preds[:, l], all_targets, all_biases, target_class=0, epoch=eval_name, wb=wb, layer=l, dataset=dataset)
                softmax_distribution_hist(parallel_preds[:, l], all_targets, all_biases, target_class=1, epoch=eval_name, wb=wb, layer=l, dataset=dataset)

        softmax_distribution_hist(all_outputs, all_targets, all_biases, target_class=0, epoch=eval_name, wb=wb, layer=-1, dataset=dataset)
        softmax_distribution_hist(all_outputs, all_targets, all_biases, target_class=1, epoch=eval_name, wb=wb, layer=-1, dataset=dataset)


# ---------------------------------------------------------------------------
# Phase 1: train the backbone (body) with frozen parallel heads
# ---------------------------------------------------------------------------

def train_body(
    model: AracNet,
    train_loader,
    device,
    optimizer,
    num_classes,
    val_loader,
    base_model="resnet18",
    epochs=10,
    wb=None,
    make_figures=True,
    dataset="BFFHQ",
    bias_amount=99.5,
):
    cur_model_name = f"aracnet-{model.aracne}_{base_model}_{dataset}_{bias_amount}-biased-init.pt"
    torch.save(model.state_dict(), os.path.join(PATH_TO_MODELS, cur_model_name))

    for epoch in range(epochs):
        model.train(True)
        loss_task_tot = AverageMeter()
        top1          = AverageMeter()
        subgroup_top1 = AverageMeterSubgroups((num_classes,) * 2, device=device)

        pbar = tqdm(train_loader, total=len(train_loader), leave=True, dynamic_ncols=True)
        epoch_outputs, epoch_targets, epoch_biases = [], [], []

        with torch.enable_grad():
            for _, (images, labels, _) in enumerate(pbar):
                images       = images.to(device)
                class_labels = labels[0].to(device)
                bias_labels  = labels[1].to(device)
                output       = model(images)

                if make_figures:
                    epoch_outputs.append(output)
                    epoch_targets.append(class_labels)
                    epoch_biases.append(bias_labels)

                optimizer.zero_grad()
                loss_task: torch.Tensor = model.loss_fn(output, class_labels).mean()
                loss_task.backward()
                optimizer.step()

                loss_task_tot.update(loss_task.item(), images.size(0))

                acc1 = accuracy(output, class_labels, topk=(1,))
                subgroup_masks = get_subgroup_masks(labels, num_classes=(num_classes,) * 2, device=device)
                subgroup_acc1  = accuracy_subgroup(output, class_labels, subgroup_masks, num_classes=num_classes)

                top1.update(acc1[0], images.size(0))
                subgroup_top1.update(subgroup_acc1, subgroup_masks)

                acc_a = regroup_by(subgroup_top1, ("aligned",))
                acc_m = regroup_by(subgroup_top1, ("misaligned",))
                pbar.set_postfix(
                    epoch=epoch,
                    acc1=top1.avg,
                    acc_a=acc_a[0].item(),
                    acc_m=acc_m[0].item(),
                    loss=loss_task_tot.avg,
                )

            torch.save(model.state_dict(), os.path.join(PATH_TO_MODELS, cur_model_name))

            if epoch % 5 == 0:
                _evaluate_epoch(model, dataset, val_loader, device, epoch, wb)

        wb.log_output({
            "epoch": epoch,
            "loss": loss_task_tot.avg,
            "acc_1": acc1,
            "acc_a": acc_a[0].item(),
            "acc_m": acc_m[0].item(),
        })

        if make_figures:
            epoch_outputs = torch.cat(epoch_outputs, dim=0)
            epoch_targets = torch.cat(epoch_targets, dim=0)
            epoch_biases  = torch.cat(epoch_biases,  dim=0)
            softmax_distribution_hist(epoch_outputs, epoch_targets, epoch_biases, target_class=0, epoch=epoch, wb=wb, layer=-1, dataset=dataset)
            softmax_distribution_hist(epoch_outputs, epoch_targets, epoch_biases, target_class=1, epoch=epoch, wb=wb, layer=-1, dataset=dataset)

    final_name = f"aracnet-{model.base_model_name}-biased-final.pt"
    torch.save(model.state_dict(), os.path.join(PATH_TO_MODELS, f"{final_name}_{dataset}_{bias_amount}"))
    return f"aracnet-{model.base_model_name}-{model.aracne}"


# ---------------------------------------------------------------------------
# Phase 2: train parallel heads, then use their failure signal to debias
# ---------------------------------------------------------------------------

def learning_from_legs_failure(
    model: AracNet,
    train_loader,
    val_loader,
    device,
    optimizer: Callable[[Iterator], torch.optim.Optimizer],
    num_classes,
    epochs=10,
    wb=None,
    make_figures=True,
    dataset="waterbirds",
    train_set=None,
):
    # Build a fresh debiasing model whose training signal comes from the parallel heads
    if dataset in ("waterbirds", "UrbanCars"):
        debiasing_model = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.DEFAULT)
    elif dataset == "cifar10c":
        debiasing_model = resnet20(num_classes)
    else:
        debiasing_model = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT)

    debiasing_model.fc = nn.Linear(debiasing_model.fc.in_features, num_classes)

    # Dataset-specific hyper-parameters for the debiasing phase
    if dataset == "cifar10c":
        monitor_head_idx = 2
        confidence_threshold = 0.05
        lr_debias    = 0.005
        scratch_flag = 1     # train debiasing model from scratch during warmup
        warmup       = 70
    elif dataset == "bar":
        monitor_head_idx = 3
        confidence_threshold = 0.15
        lr_debias    = 0.00005
        scratch_flag = 0
        warmup       = 20
    elif dataset == "BFFHQ":
        monitor_head_idx = 3
        confidence_threshold = 0.3
        lr_debias    = 0.00005
        scratch_flag = 0
        warmup       = 20
    elif dataset == "waterbirds":
        monitor_head_idx = 2
        confidence_threshold = 0.3
        lr_debias    = 0.00005
        scratch_flag = 0
        warmup       = 30
    elif dataset == "UrbanCars":
        monitor_head_idx = 4
        confidence_threshold = 0.3
        lr_debias    = 0.00001
        scratch_flag = 0
        warmup       = 5

    debiasing_model = debiasing_model.to("cuda")
    debiasing_model.loss_fn = nn.CrossEntropyLoss(reduction="none")

    if dataset == "UrbanCars":
        _evaluate_epoch(debiasing_model, dataset, val_loader, device, -1, wb, train_set)
    else:
        _evaluate_epoch(debiasing_model, dataset, val_loader, device, -1, wb)

    debias_optimizer = torch.optim.AdamW(debiasing_model.parameters(), lr=lr_debias)

    cur_model_name = f"aracnet-{model.aracne}-biased-init.pt"
    torch.save(model.state_dict(), os.path.join(PATH_TO_MODELS, cur_model_name))

    # One dedicated SGD optimizer per parallel head
    head_optimizers = {
        key: optimizer(model.parallel_heads[key].parameters())
        for key in model.parallel_heads
    }
    new_criterion = nn.CrossEntropyLoss(reduction="none")

    ranking = torch.zeros(len(model.parallel_heads))

    for epoch in range(epochs):
        model.train(True)
        debiasing_model.train(True)

        heads_losses_tot = [AverageMeter() for _ in range(len(model.parallel_heads))]
        heads_accs_tot   = [AverageMeter() for _ in range(len(model.parallel_heads))]
        top1          = AverageMeter()
        subgroup_top1 = AverageMeterSubgroups((num_classes,) * 2, device=device)

        pbar = tqdm(train_loader, total=len(train_loader), leave=True, dynamic_ncols=True)
        epoch_parallel = []
        epoch_outputs, epoch_targets, epoch_biases = [], [], []

        # Cumulative loss accumulators for diagnostic printing after warmup
        loss_b_aligned   = loss_b_conflict   = 0.0
        loss_u_aligned   = loss_u_conflict   = 0.0
        correct_aligned  = incorrect_aligned = 0
        correct_conflict = incorrect_conflict = 0

        with torch.enable_grad():
            for _, (images, labels, _) in enumerate(pbar):
                images       = images.to(device)
                class_labels = labels[0].to(device)
                bias_labels  = labels[1].to(device)
                output       = model.forward(images)

                if epoch > warmup:
                    debiased_output = debiasing_model(images)
                    debiased_loss   = debiasing_model.loss_fn(debiased_output, class_labels)

                # Collect intermediate activations via hooks and compute parallel head predictions
                model.parallel_z = {key: model.hooks[key].output for key in model.hooks}
                model.parallel_y = torch.cat(
                    [model.parallel_heads[key](model.parallel_z[key]).unsqueeze(1) for key in model.parallel_z],
                    dim=1,
                )

                # Train each parallel head independently
                monitor_logits = None
                for i, key in enumerate(model.parallel_heads):
                    head_optimizers[key].zero_grad()
                    y_head: torch.Tensor = model.parallel_y.transpose(1, 0)[i]

                    if len(y_head.shape) > 2:
                        # ViT outputs sequence; use CLS token only
                        y_head = y_head[:, 0, :]

                    head_loss: torch.Tensor = model.loss_fn(y_head, class_labels).mean()
                    head_loss.backward(retain_graph=True)
                    head_optimizers[key].step()
                    heads_losses_tot[i].update(head_loss.item(), images.size(0))
                    heads_accs_tot[i].update(accuracy(y_head, class_labels, topk=(1,))[0], images.size(0))

                    if epoch > warmup and i == monitor_head_idx:
                        with torch.no_grad():
                            monitor_logits = y_head.clone()
                            biased_loss    = new_criterion(y_head, class_labels).detach()

                    elif epoch <= warmup and scratch_flag == 1:
                        # During warmup with scratch training, also pre-train the debiasing model on plain CE
                        debiased_output = debiasing_model(images)
                        debiased_loss   = debiasing_model.loss_fn(debiased_output, class_labels)
                        debias_optimizer.zero_grad()
                        debiased_loss.mean().backward()
                        debias_optimizer.step()

                # After warmup, weight the debiasing model's CE loss by the monitor head's confidence
                if epoch > warmup:
                    probs = torch.softmax(monitor_logits, dim=1)
                    # Confidence of the biased head on the true class
                    target_conf = torch.gather(probs, 1, class_labels.unsqueeze(1)).squeeze(1)

                    eps = 1e-8
                    sample_weights = (-torch.log(target_conf.clamp_min(1e-3) + eps)) ** 2.0
                    sample_weights = sample_weights / (sample_weights.mean().detach() + eps)

                    w_monitor = biased_loss / (debiased_loss.detach() + biased_loss + eps)

                    loss_task: torch.Tensor = sample_weights.detach() * w_monitor.detach() * debiased_loss

                    low_conf_mask = target_conf <= confidence_threshold
                    aligned_mask  = class_labels[low_conf_mask] == bias_labels[low_conf_mask]
                    conflict_mask = class_labels[low_conf_mask] != bias_labels[low_conf_mask]
                    preds         = torch.argmax(probs, dim=1)

                    correct_aligned   += ((preds[low_conf_mask] == class_labels[low_conf_mask]) & aligned_mask).sum()
                    incorrect_aligned += ((preds[low_conf_mask] != class_labels[low_conf_mask]) & aligned_mask).sum()
                    correct_conflict   += ((preds[low_conf_mask] == class_labels[low_conf_mask]) & conflict_mask).sum()
                    incorrect_conflict += ((preds[low_conf_mask] != class_labels[low_conf_mask]) & conflict_mask).sum()

                    mask_aligned  = bias_labels == class_labels
                    mask_conflict = bias_labels != class_labels
                    loss_b_aligned += biased_loss[mask_aligned].mean()
                    loss_u_aligned += debiased_loss[mask_aligned].mean()
                    if mask_conflict.any():
                        loss_b_conflict += biased_loss[mask_conflict].mean()
                        loss_u_conflict += debiased_loss[mask_conflict].mean()

                    loss_task = loss_task.mean()
                    debias_optimizer.zero_grad()
                    loss_task.backward()
                    debias_optimizer.step()

                if make_figures:
                    epoch_outputs.append(output)
                    epoch_targets.append(class_labels)
                    epoch_biases.append(bias_labels)

                epoch_parallel.append(model.parallel_y)

                acc1  = accuracy(output, class_labels, topk=(1,))
                subgroup_masks = get_subgroup_masks(labels, num_classes=(num_classes,) * 2, device=device)
                subgroup_acc1  = accuracy_subgroup(output, class_labels, subgroup_masks, num_classes=num_classes)
                top1.update(acc1[0], images.size(0))
                subgroup_top1.update(subgroup_acc1, subgroup_masks)

                acc_a = regroup_by(subgroup_top1, ("aligned",))
                acc_m = regroup_by(subgroup_top1, ("misaligned",))
                pbar.set_postfix(
                    epoch=epoch,
                    acc1=top1.avg,
                    acc_a=acc_a[0].item(),
                    acc_m=acc_m[0].item(),
                    heads_losses=[f"{heads_losses_tot[i].avg:.3f}" for i in range(len(model.parallel_heads))],
                    heads_accs=[f"{heads_accs_tot[i].avg:.3f}" for i in range(len(model.parallel_heads))],
                )

        if epoch > warmup:
            n = len(train_loader)
            print(f"aligned_loss_b={loss_b_aligned/n:.4f}  conflict_loss_b={loss_b_conflict/n:.4f}")
            print(f"aligned_loss_u={loss_u_aligned/n:.4f}  conflict_loss_u={loss_u_conflict/n:.4f}")
            print(f"correct_aligned={correct_aligned}  incorrect_aligned={incorrect_aligned}")
            print(f"correct_conflict={correct_conflict}  incorrect_conflict={incorrect_conflict}")
            _evaluate_epoch(debiasing_model, dataset, val_loader, device, epoch=epoch, wb=wb, train_set=train_set)

        if make_figures:
            epoch_outputs = torch.cat(epoch_outputs, dim=0)
            epoch_targets = torch.cat(epoch_targets, dim=0)
            epoch_biases  = torch.cat(epoch_biases,  dim=0)

            ranking = torch.zeros(len(model.parallel_heads))

            if model.aracne:
                epoch_parallel = torch.cat(epoch_parallel, dim=0)
                for l in range(1, len(model.parallel_heads)):
                    for cls_idx in range(num_classes):
                        softmax_distribution_hist(epoch_parallel[:, l], epoch_targets, epoch_biases, target_class=cls_idx, epoch=epoch, wb=wb, layer=l, dataset=dataset)
                        ranking[l] += ranking_score_histogram(epoch_parallel[:, l], epoch_targets, cls_idx, thresh=confidence_threshold, thresh_max=0.7) / num_classes

                for cls_idx in range(num_classes):
                    softmax_distribution_hist(epoch_outputs, epoch_targets, epoch_biases, target_class=cls_idx, epoch=epoch, wb=wb, layer=-1, dataset=dataset)

        # At the end of warmup, select the best monitor head using the three ranking metrics
        if epoch == warmup - 1:
            monitor_head_idx = int(torch.argmax(ranking))
            print(f"Selected monitor head: {monitor_head_idx}  ranking={ranking}")
            continue

        print(f"ranking={ranking}  monitor={monitor_head_idx}")

    torch.save(model.state_dict(),
               os.path.join(PATH_TO_MODELS, f"aracnet-{model.base_model_name}-biased-final.pt"))
    torch.save(debiasing_model.state_dict(),
               os.path.join(PATH_TO_MODELS, f"debiased-{model.base_model_name}.pt"))
    return debiasing_model
