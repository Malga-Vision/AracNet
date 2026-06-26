from typing import Iterable, Tuple, Union
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import VisionTransformer, ViT_B_16_Weights, vit_b_16
from torch.utils.data import DataLoader, TensorDataset
from wandb_wrapper import WandbWrapper
from loss_memory import LossMemory


import matplotlib
matplotlib.use("agg")
from matplotlib import pyplot as plt


def pairwise_kl_divergence(predictions):
    """
    Compute pairwise KL divergence between predictions of different classifiers.
    
    Args:
    predictions (torch.Tensor): A tensor of shape (N, batch_size, num_classes)
                                representing the predicted probabilities from N classifiers.

    Returns:
    torch.Tensor: A tensor of shape (batch_size, N, N) containing the pairwise KL divergence
                  between classifiers for each sample.
    """
    N, batch_size, num_classes = predictions.shape
    kl_divergences = torch.zeros(batch_size, N, N, device=predictions.device)

    # Ensure the predictions are valid probabilities
    predictions = predictions.clamp(min=1e-10)  # To avoid log(0)

    for i in range(len(predictions)):
        for j in range(i + 1, len(predictions)):
            # KL divergence P_i || P_j
            kl_ij = torch.sum(predictions[i] * (torch.log(predictions[i]) - torch.log(predictions[j])), dim=-1)
            # KL divergence P_j || P_i
            kl_ji = torch.sum(predictions[j] * (torch.log(predictions[j]) - torch.log(predictions[i])), dim=-1)
            
            # Store both divergences
            kl_divergences[:, i, j] = kl_ij
            kl_divergences[:, j, i] = kl_ji

    return kl_divergences

def pairwise_dot_product(predictions):
    """
    Compute pairwise dot product between predictions of different classifiers.
    
    Args:
    predictions (torch.Tensor): A tensor of shape (N, batch_size, num_classes)
                                representing the predicted probabilities or vectors
                                from N classifiers.

    Returns:
    torch.Tensor: A tensor of shape (batch_size, N, N) containing the pairwise dot product
                  between classifiers for each sample.
    """
    N, batch_size, num_classes = predictions.shape
    predictions = predictions / predictions.norm(dim=-1, keepdim=True)
    dot_products = torch.zeros(batch_size, N, N, device=predictions.device)

    for i in range(N):
        for j in range(i, N):
            # Dot product P_i . P_j
            dot_ij = torch.sum(predictions[i] * predictions[j], dim=-1)
            
            # Store dot products
            dot_products[:, i, j] = dot_ij 
            dot_products[:, j, i] = dot_ij  # Symmetric

    return dot_products


class PatchEmbedding(nn.Module):
    def __init__(self, img_size: int, patch_size: int, in_channels: int, embedding_dim: int, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.n_patches:     int = (img_size // patch_size) ** 2
        self.patch_size:    int = patch_size
        self.in_channels:   int = in_channels
        self.embedding_dim: int = embedding_dim
        
        self.conv_patcher: nn.Conv2d = nn.Conv2d(
            in_channels=self.in_channels, 
            out_channels=embedding_dim,
            kernel_size=patch_size,
            stride=patch_size
        )              

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, _, H, W = x.size()
        assert H == W == self.patch_size * (H // self.patch_size)
        x = self.conv_patcher(x)
        n_h = H // self.patch_size
        n_w = W // self.patch_size
        
        x = x.reshape((B, self.embedding_dim, n_h * n_w))
        x = x.permute(0, 2, 1)

        return x
    

class EncoderBlock(nn.Module):
    def __init__(self, embedding_dim: int, num_heads: int, mlp_dim: int, dropout: float = 0.1, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.embedding_dim: int = embedding_dim
        self.num_heads:     int = num_heads
        self.mlp_dim:       int = mlp_dim
        self.dropout:     float = dropout
        self.ln1 = nn.LayerNorm(self.embedding_dim)
        self.ln2 = nn.LayerNorm(self.embedding_dim)
        self.attn = nn.MultiheadAttention(self.embedding_dim, num_heads, dropout, batch_first=True)

        self.mlp: nn.Sequential = nn.Sequential(
            nn.Linear(self.embedding_dim, self.mlp_dim),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.mlp_dim, self.embedding_dim),
            nn.Dropout(self.dropout)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x), self.ln1(x), self.ln1(x))[0]
        x = x + self.mlp(self.ln2(x))

        return x

class SimpleViT(nn.Module):
    def __init__(
            self, 
            num_layers: int, 
            num_classes: int, 
            image_size: int, 
            patch_size: int = 16, 
            embedding_dim: int = 128, 
            num_heads: int = 8, 
            mlp_dim: int = 256, 
            in_channels: int = 3,
            dropout = 0.1,
            *args, 
            **kwargs
        ) -> None:
        
        super().__init__(*args, **kwargs)

        self.num_layers:        int = num_layers
        self.num_classes:       int = num_classes
        self.image_size:        int = image_size
        self.patch_size:        int = patch_size
        self.embedding_dim:     int = embedding_dim
        self.num_heads:         int = num_heads
        self.mlp_dim:           int = mlp_dim
        self.in_channels:       int = in_channels
        self.dropout:           float = dropout
        
        self.patch_embedding = PatchEmbedding(
            self.image_size,
            self.patch_size,
            in_channels=self.in_channels,
            embedding_dim=self.embedding_dim
        )

        self.cls_token = nn.Parameter(
            torch.zeros(1, 1, self.embedding_dim)
        )
        self.pos_embed = nn.Parameter(
            torch.zeros(1, 1 + self.patch_embedding.n_patches, self.embedding_dim)
        )
        self.pos_drop = nn.Dropout(p=self.dropout)

        self.transformer = nn.ModuleList((
            EncoderBlock(self.embedding_dim, self.num_heads, self.mlp_dim, self.dropout) \
            for _ in range(self.num_layers)
        ))

        self.ln = nn.LayerNorm(self.embedding_dim)
        self.head = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim),
            nn.Tanh(), 
            nn.Linear(self.embedding_dim, self.num_classes)
        )

        self.erm_loss_fn = CumulativeCELoss(reduction="none") if isinstance(self, BoostViT) else nn.CrossEntropyLoss(reduction="none")

        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.to(self.device)

    def put_on_device(self, *tensors: Iterable[torch.Tensor]) -> Iterable[torch.Tensor]:
        return (tensor.to(self.device) for tensor in tensors)

    def _preprocess(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embedding(x)
        cls_tokens = self.cls_token.expand(x.size(0), -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed[:, :x.size(1), :]
        x = self.pos_drop(x)
        return x
    
    def backbone(self, x: torch.Tensor) -> torch.Tensor:        

        for encoder_block in self.transformer:
            x = encoder_block(x)

        return self.ln(x)

    def predict(self, x: torch.Tensor) -> torch.Tensor:        
        return self.head(x[:, 0])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._preprocess(x)
        z = self.backbone(x)
        return self.predict(z)    

    @torch.no_grad()    
    def extract_features(self, loader: DataLoader) -> torch.Tensor:
        self.eval()
        features        = []
        class_labels    = []
        bias_labels     = []

        for batch in loader:
            match len(batch):
                case 3: inputs, labels, blabels = batch
                case 4: inputs, labels, blabels, _ = batch
                     
            inputs, labels, blabels = self.put_on_device(inputs, labels, blabels)
            features.append(self.backbone(inputs))
            class_labels.append(labels)
            bias_labels.append(blabels)

        features = torch.cat(features, dim=0).cpu()
        class_labels = torch.cat(class_labels, dim=0).cpu()
        bias_labels  = torch.cat(bias_labels, dim=0).cpu()

        return features, class_labels, bias_labels
    
    def misclassified_statistics(self, dataloader: DataLoader, save_results_to: Union[str, None] = None) -> TensorDataset:        
        self.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        correct_biased = 0
        correct_unbiased = 0
        total_biased = 0
        total_unbiased = 0
        correct_per_class = torch.zeros(self.num_classes)
        total_per_class   = torch.zeros(self.num_classes)
        
        perclass_bias_preds:  list[list] = [list() for _ in range(self.num_classes)]
        perclass_bias_labels: list[list] = [list() for _ in range(self.num_classes)]
        bias_preds:           list = []       
        bias_targs:           list = []      

        logits = []
        rel_blabels = []

        with torch.no_grad():
            for inputs, labels, bias_labels, _ in dataloader:
                inputs, labels, bias_labels = self.put_on_device(inputs, labels, bias_labels)
                labels = labels.type(torch.LongTensor)
                labels = labels.to(self.device)

                outputs = self(inputs)
                loss = self.erm_loss_fn(outputs, labels)
                total_loss += loss.item()

                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                bias_preds.append(torch.where(predicted == labels, 1, -1))
                bias_targs.append(bias_labels)

                logits.append(self(inputs[predicted != labels]))
                rel_blabels.append(bias_labels[predicted != labels])
                
                for _class in torch.arange(self.num_classes):
                    _class = _class.to(self.device)
                    inclass_labels: torch.Tensor = labels[labels == _class].to(self.device)
                    inclass_preds: torch.Tensor  = predicted[labels == _class].to(self.device)
                    total_per_class[_class] += inclass_labels.size(0)
                    correct_per_class[_class] += (inclass_preds == inclass_labels).sum().item()
                    perclass_bias_labels[_class].append(bias_labels[labels == _class])
                    perclass_bias_preds[_class].append(torch.where(inclass_preds == inclass_labels, 1, -1))
                 
                bias_predicted = predicted[bias_labels != -1]
                total_biased += len(bias_predicted)
                correct_biased += (bias_predicted == labels[bias_labels != -1]).sum().item()

                unbias_predicted = predicted[bias_labels == -1]
                total_unbiased += len(unbias_predicted)
                correct_unbiased += (unbias_predicted == labels[bias_labels == -1]).sum().item()

            average_loss = total_loss / len(dataloader)
            accuracy = 100 * correct / total
            num_misclassified_samples: int = total - correct
            misclassified_per_class = total_per_class - correct_per_class
            accuracy_biased = 100 * correct_biased / total_biased
            accuracy_unbiased = 100 * correct_unbiased / (total_unbiased+0.0000001)            
        
        print(f"\t -- Loss: {average_loss:.4f}, Accuracy: {accuracy:.2f}%")
        print(f"\t\t Accuracy Biased: {accuracy_biased:.4f}%, Accuracy Unbiased: {accuracy_unbiased:.4f}%")
        print(f"# Misclassified Samples: {num_misclassified_samples}")
        print(f"# Misclassified per class:", misclassified_per_class)

        logits = torch.cat(logits, dim=0).cpu()
        rel_blabels = torch.cat(rel_blabels, dim=0).cpu()

        perclass_bias_preds  = [torch.cat(p, dim=0).cpu() for p in perclass_bias_preds]
        perclass_bias_labels = [torch.cat(p, dim=0).cpu() for p in perclass_bias_labels]

        # for _class in torch.arange(self.num_classes):
        #     if save_results_to:
        #         with open(f"{save_results_to}/mistakes_preds.txt", mode="a+") as f:
        #             f.write(f"Class {_class}\n")
        #             try:
        #                 f.write(classification_report(perclass_bias_labels[_class], perclass_bias_preds[_class], target_names=["Unbiased", "Biased"]))
        #             except ValueError:
        #                 print("Error during classification report")
        #                 pass
        #     else:
        #         print(f"Class {_class}\n")
        #         try:
        #             print(classification_report(perclass_bias_labels[_class], perclass_bias_preds[_class], target_names=["Unbiased", "Biased"]))
        #         except ValueError:
        #             print("Error during classification report")
        #             pass
        # print("Mistakes CF:")
        # print(confusion_matrix(torch.cat(bias_targs, dim=0).cpu().numpy(), torch.cat(bias_preds, dim=0).cpu().numpy()))

        return torch.cat(bias_preds, dim=0).cpu(), num_misclassified_samples, total_per_class, misclassified_per_class
    
    def train_model_erm(
            self,
            train_loader,
            val_loader,
            test_loader,
            learning_rate=0.001,
            num_epochs=50,
            accumulate=1,
            wb: Union[WandbWrapper, None] = None,
            train_len: Union[None, int] = None
        ):
        
        optimizer = torch.optim.AdamW(self.parameters(), lr=learning_rate, amsgrad=False)

        validation_accuracies_avg = []
        validation_accuracies_b   = []
        validation_accuracies_u   = []

        test_accuracies_avg       = []
        test_accuracies_b         = []
        test_accuracies_u         = []

        if isinstance(self, BoostViT):
            if train_len is not None:
                self.loss_memory: LossMemory = LossMemory(train_len, alpha=0.5)
                self.blabels_memory: torch.Tensor = torch.zeros((train_len, )).to(self.device).long()      
        
        for epoch in range(num_epochs):
            train_loss = 0.0
            tr_biased_loss = 0.0
            tr_unbiased_loss = 0.0
            correct_train = 0
            total_train = 0
            total_biased = 0
            correct_biased = 0
            total_unbiased = 0
            correct_unbiased = 0
            
            self.train()
            optimizer.zero_grad()
            with torch.enable_grad():
                for batch_idx, (inputs, labels, bias_labels, dataset_idx) in enumerate(train_loader):
                    labels = labels.type(torch.LongTensor)
                    inputs, labels, bias_labels, dataset_idx = self.put_on_device(inputs, labels, bias_labels, dataset_idx)

                    if isinstance(self, BoostViT):
                        if self.just_expanded:
                            self.loss_memory.step()
                    
                    outputs: torch.Tensor = self(inputs)                    

                    loss: torch.Tensor = self.erm_loss_fn(outputs, labels) / accumulate

                    if isinstance(self, BoostViT):
                        self.batch_memory: torch.Tensor = self.loss_memory[dataset_idx].detach().clone()

                    tr_biased_loss += loss[bias_labels == 1].mean().item()      if torch.sum(bias_labels == 1) > 0 else 0.0                    
                    tr_unbiased_loss += loss[bias_labels == -1].mean().item()   if torch.sum(bias_labels == -1) > 0 else 0.0                                           


                    if isinstance(self, BoostViT):
                        if train_len is not None:
                            self.loss_memory.store(dataset_idx, loss)
                            self.blabels_memory[dataset_idx] = bias_labels
                            
                        if self.current_depth > 1:
                            weights: torch.Tensor = torch.exp(- torch.pow(torch.abs(self.batch_memory), 2) / self.batch_memory.max())
                        else:
                            weights: torch.Tensor = 1. / self.batch_memory

                        
                        loss = (weights * loss).mean()                    
                    else:
                        loss = loss.mean()
                    
                    loss.backward()                    
                    train_loss += loss.item()                         
                    
                    if isinstance(self, BoostViT):
                        _, predicted = torch.max(outputs.sum(dim=0), 1)
                    else:
                        _, predicted = torch.max(outputs, dim=1)
                    
                    total_train += labels.size(0)
                    correct_train += (predicted == labels).sum().item()

                    bias_predicted = predicted[bias_labels != -1]
                    total_biased += len(bias_predicted)
                    correct_biased += (bias_predicted == labels[bias_labels != -1]).sum().item()

                    unbias_predicted = predicted[bias_labels == -1]
                    total_unbiased += len(unbias_predicted)
                    correct_unbiased += (unbias_predicted == labels[bias_labels == -1]).sum().item()
                    if ((batch_idx + 1) % accumulate == 0) or (batch_idx + 1 == len(train_loader)):                         
                        optimizer.step()
                        optimizer.zero_grad()

                
            average_loss_train = train_loss / len(train_loader)
            tr_biased_loss = tr_biased_loss / len(train_loader)
            tr_unbiased_loss = tr_unbiased_loss / len(train_loader)
            accuracy_train = 100 * correct_train / total_train
            accuracy_train_biased = 100 * correct_biased/total_biased
            accuracy_train_unbiased = 100 * correct_unbiased/(total_unbiased+0.000001)
            tr_tot_unbiased = total_unbiased

            self.eval()
            val_loss = 0.0
            vl_biased_loss = 0.0
            vl_unbiased_loss = 0.0
            correct_val = 0
            total_val = 0
            correct_biased = 0
            correct_unbiased = 0
            total_biased = 0
            total_unbiased = 0
            
            
            with torch.no_grad():
                for inputs, labels, bias_labels in val_loader:
                    labels = labels.type(torch.LongTensor)
                    inputs, labels = self.put_on_device(inputs, labels)

                    outputs = self(inputs)
                    loss = self.erm_loss_fn(outputs, labels)
                    vl_biased_loss += loss[bias_labels == 1].mean().item()    if torch.sum(bias_labels == 1) > 0 else 0.0                    
                    vl_unbiased_loss += loss[bias_labels == -1].mean().item() if torch.sum(bias_labels == -1) > 0 else 0.0                                   
                    loss = loss.mean()
                    val_loss += loss.item()

                    if isinstance(self, BoostViT):
                        _, predicted = torch.max(outputs.sum(dim=0), 1)
                    else:
                        _, predicted = torch.max(outputs, dim=1)
                    total_val += labels.size(0)
                    correct_val += (predicted == labels).sum().item()

                    bias_predicted = predicted[bias_labels != -1]
                    total_biased += len(bias_predicted)
                    correct_biased += (bias_predicted == labels[bias_labels != -1]).sum().item()

                    unbias_predicted = predicted[bias_labels == -1]
                    total_unbiased += len(unbias_predicted)
                    correct_unbiased += (unbias_predicted == labels[bias_labels == -1]).sum().item()

            average_loss_val = val_loss / len(val_loader) 
            vl_biased_loss = vl_biased_loss / len(val_loader)
            vl_unbiased_loss = vl_unbiased_loss / len(val_loader)
            accuracy_val_biased = 100*correct_biased/total_biased
            accuracy_val_unbiased = 100*correct_unbiased/(total_unbiased+0.000001)
            accuracy_val = 100 * correct_val / total_val
            val_tot_unbiased = total_unbiased

            validation_accuracies_avg.append(accuracy_val)
            validation_accuracies_b.append(accuracy_val_biased)
            validation_accuracies_u.append(accuracy_val_unbiased)

            self.eval()
            test_loss = 0.0
            te_biased_loss = 0.0
            te_unbiased_loss = 0.0
            correct = 0
            total = 0
            correct_biased=0
            correct_unbiased=0
            total_biased=0
            total_unbiased=0

            with torch.no_grad():
                for inputs, labels, bias_labels in test_loader:
                    bias_labels = bias_labels.to(self.device)
                    labels = labels.type(torch.LongTensor)
                    inputs, labels = self.put_on_device(inputs, labels)

                    outputs = self(inputs)
                    loss = self.erm_loss_fn(outputs, labels)
                    te_biased_loss += loss[bias_labels == 1].mean().item()     if torch.sum(bias_labels == 1) > 0 else 0.0                    
                    te_unbiased_loss += loss[bias_labels == -1].mean().item()  if torch.sum(bias_labels == -1) > 0 else 0.0                                      
                    loss = loss.mean()
                    test_loss += loss.item()

                    if isinstance(self, BoostViT):
                        _, predicted = torch.max(outputs.sum(dim=0), 1)
                    else:
                        _, predicted = torch.max(outputs, dim=1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
                    
                    bias_predicted = predicted[bias_labels != -1]
                    total_biased += len(bias_predicted)
                    correct_biased += (bias_predicted == labels[bias_labels != -1]).sum().item()

                    unbias_predicted = predicted[bias_labels == -1]
                    total_unbiased += len(unbias_predicted)
                    correct_unbiased += (unbias_predicted == labels[bias_labels == -1]).sum().item()

            average_loss_test = test_loss / len(test_loader)
            te_biased_loss = te_biased_loss / len(test_loader)
            te_unbiased_loss = te_unbiased_loss / len(test_loader)
            accuracy_test = 100 * correct / total
            accuracy_test_biased = 100 * correct_biased / total_biased
            accuracy_test_unbiased = 100 * correct_unbiased / (total_unbiased+0.000001) 
            te_tot_unbiased = total_unbiased

            test_accuracies_avg.append(accuracy_test)
            test_accuracies_b.append(accuracy_test_biased)
            test_accuracies_u.append(accuracy_test_unbiased)   
                     
            
            output_str = "" + \
                f"Epoch {epoch + 1}/{num_epochs}\n\t -- Loss: {average_loss_train:.4f}, Train Accuracy: {accuracy_train:.2f}%\n" + \
                f"\t\t Train Accuracy Biased: {accuracy_train_biased:.4f} %, Train Accuracy Unbiased: {accuracy_train_unbiased:.4f}% ({tr_tot_unbiased})\n" + \
                f"\t -- Validation Loss: {average_loss_val:.4f}, Validation Accuracy: {accuracy_val:.4f} %\n" + \
                f"\t\t Valid Accuracy Biased: {accuracy_val_biased:.4f} %, Valid Accuracy Unbiased: {accuracy_val_unbiased:.4f}% ({val_tot_unbiased})\n" + \
                f"\t -- Test Loss: {average_loss_test:.4f}, Test Accuracy: {accuracy_test:.2f}%\n" + \
                f"\t\t Test Accuracy Biased: {accuracy_test_biased:.4f}%, Test Accuracy Unbiased: {accuracy_test_unbiased:.4f}% ({te_tot_unbiased})\n"
            
            if wb is not None:
                if isinstance(self, BoostViT):
                    tr_biased_losses = self.loss_memory[self.blabels_memory == 1] / torch.sum(self.blabels_memory == 1)
                    tr_unbiased_losses = self.loss_memory[self.blabels_memory == -1] / torch.sum(self.blabels_memory == -1)

                    plt.figure()
                    plt.hist([tr_biased_losses.detach().cpu().numpy(), tr_unbiased_losses.detach().cpu().numpy()], bins=100, label=["Bias-Aligned", "Bias-Conflicting"], alpha=0.7, color=["blue", "orange"], log=True)
                    plt.legend()
                    plt.tight_layout()
                else:
                    plt.figure()
                    plt.tight_layout()
                
                wb.log_output({
                    "epoch": epoch+1,
                    "avg_loss_tr": average_loss_train,
                    "avg_loss_tr_ba": tr_biased_loss,
                    "avg_loss_tr_bc": tr_unbiased_loss,
                    "acc_tr": accuracy_train,
                    "acc_ba_tr": accuracy_train_biased,
                    "acc_bc_tr": accuracy_train_unbiased,
                    "avg_loss_vl": average_loss_val,
                    "avg_loss_vl_ba": vl_biased_loss,
                    "avg_loss_vl_bc": vl_unbiased_loss,
                    "acc_vl": accuracy_val,
                    "acc_ba_vl": accuracy_val_biased,
                    "acc_bc_vl": accuracy_val_unbiased,
                    "avg_loss_te": average_loss_test,
                    "avg_loss_te_ba": te_biased_loss,
                    "avg_loss_te_bc": te_unbiased_loss,
                    "acc_te": accuracy_test,
                    "acc_ba_te": accuracy_test_biased,
                    "acc_bc_te": accuracy_test_unbiased,
                    "loss_hist": wb.backend.Image(plt)
                })
                plt.close()
                wb.log_model(self, f"SimpleViT_{wb.run_name}")
            
            print(output_str)

            if isinstance(self, BoostViT):
                self.update(epoch, train_loss)           

        torch.save(self.state_dict(), "vit.pth")
        return (
            validation_accuracies_avg, validation_accuracies_b, validation_accuracies_u, 
            test_accuracies_avg, test_accuracies_b, test_accuracies_u
        )
    

class CumulativeCELoss(nn.Module):
    def __init__(self, reduction: str = "mean", *args, **kwargs) -> None:
        assert reduction in {"none", "sum", "mean"}
        
        super().__init__(*args, **kwargs)
        self.reduction = reduction
        self.loss_fn = nn.CrossEntropyLoss(reduction="none")

    def forward(self, y_preds: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        loss: torch.Tensor = torch.full((y_true.size(0), ), 0.0).to(y_true.device).float()
        for i, y_pred in enumerate(y_preds):
            cur_loss = self.loss_fn(y_pred, y_true) 
            loss += cur_loss #* (y_preds.size(0) - i / y_preds.size(0))
        
        match self.reduction:
            case "none": return (loss / y_preds.size(0))
            case "sum" : return (loss / y_preds.size(0)).sum()
            case "mean": return (loss / y_preds.size(0)).mean()        
        

class BoostViT(SimpleViT):
    def __init__(self, num_layers: int, num_classes: int, image_size: int, patch_size: int = 16, embedding_dim: int = 128, num_heads: int = 8, mlp_dim: int = 256, in_channels: int = 3, dropout=0.1, *args, **kwargs) -> None:
        super().__init__(num_layers, num_classes, image_size, patch_size, embedding_dim, num_heads, mlp_dim, in_channels, dropout, *args, **kwargs)

        self.spawn_layer = lambda : SimpleViT(1, num_classes, image_size, patch_size, embedding_dim, num_heads, mlp_dim, in_channels, dropout)
        self.transformer: nn.ModuleList[SimpleViT] = nn.ModuleList((self.spawn_layer() for _ in range(self.num_layers)))
        self.current_depth: int = self.num_layers

        self.to(self.device)
        self.cur_loss = torch.inf
        self.just_expanded: bool = False

    def backbone(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        layer: SimpleViT #type-attr
        y = torch.zeros((self.current_depth, x.size(0), self.num_classes)).to(self.device)
        
        layer = self.transformer[0]
        x = layer._preprocess(x)
        xs = torch.zeros((self.current_depth, x.size(0), self.embedding_dim)).to(self.device)        
        for i, layer in enumerate(self.transformer):
            x = x + layer.backbone(x)
            xs[i] = x[:, 0]
            y[i] = layer.predict(x)

        return xs, y

    @torch.no_grad()    
    def extract_features(self, loader: DataLoader) -> torch.Tensor:
        self.eval()
        features        = []
        predictions     = []
        class_labels    = []
        bias_labels     = []

        for batch in loader:
            match len(batch):
                case 3: inputs, labels, blabels = batch
                case 4: inputs, labels, blabels, _ = batch
                     
            inputs, labels, blabels = self.put_on_device(inputs, labels, blabels)
            phi_x, y = self.backbone(inputs)
            features.append(phi_x)
            predictions.append(y)
            class_labels.append(labels)
            bias_labels.append(blabels)


        features = torch.cat(features, dim=1).cpu()
        predictions = torch.cat(predictions, dim=1).cpu() 
        class_labels = torch.cat(class_labels, dim=0).cpu()
        bias_labels  = torch.cat(bias_labels, dim=0).cpu()

        return features, predictions, class_labels, bias_labels
         

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        layer: SimpleViT #type-attr        
        y = torch.zeros((self.current_depth, x.size(0), self.num_classes)).to(self.device)
        
        layer = self.transformer[0]
        x = layer._preprocess(x)
        for i, layer in enumerate(self.transformer):
            x = x + layer.backbone(x)
            y[i] = layer.predict(x)

        return y

    def boost(self) -> None:
        self.transformer.append(self.spawn_layer())
        self.current_depth += 1

    def update(self, epoch, loss) -> None:        
        # if torch.abs(loss - self.cur_loss) > torch.Tensor([1e-4]):
        if epoch in range(20, 400, 20):
            cur_dep = self.current_depth
            self.boost()
            assert self.current_depth == (cur_dep + 1)
            print("--------------------------------------")
            print("\t Added Transformer Layer: ")
            print(f"\t\t Current Depth: {self.current_depth}")
            print("--------------------------------------")
            self.just_expanded = True
        else:
            self.just_expanded = False



class OriginalViT(nn.Module):
    def __init__(self, num_classes: int, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.num_classes = num_classes
        self.model = vit_b_16(weights=None)
        self.model.heads = nn.Linear(self.model.hidden_dim, self.num_classes)
        self.model.device = torch.cuda.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.device = "cuda"
        self.to("cuda")
        self.erm_loss_fn = nn.CrossEntropyLoss()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    
    def train_model_erm(self, *args, **kwargs):
        return SimpleViT.train_model_erm(self, *args, **kwargs)
    
    def put_on_device(self, *args, **kwargs):
        return SimpleViT.put_on_device(self, *args, **kwargs)
    