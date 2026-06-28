from typing import Iterable, Tuple, Union
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset
from wandb_wrapper import WandbWrapper
from utils import CumulativeCELoss

import matplotlib
matplotlib.use("agg")
from matplotlib import pyplot as plt


class PatchEmbedding(nn.Module):
    def __init__(self, img_size: int, patch_size: int, in_channels: int, embedding_dim: int, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.n_patches:     int = (img_size // patch_size) ** 2
        self.patch_size:    int = patch_size
        self.in_channels:   int = in_channels
        self.embedding_dim: int = embedding_dim

        self.conv_patcher = nn.Conv2d(
            in_channels=self.in_channels,
            out_channels=embedding_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, _, H, W = x.size()
        assert H == W == self.patch_size * (H // self.patch_size)
        x = self.conv_patcher(x)
        n_h = H // self.patch_size
        n_w = W // self.patch_size
        x = x.reshape((B, self.embedding_dim, n_h * n_w))
        return x.permute(0, 2, 1)


class EncoderBlock(nn.Module):
    def __init__(self, embedding_dim: int, num_heads: int, mlp_dim: int, dropout: float = 0.1, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.embedding_dim = embedding_dim
        self.num_heads     = num_heads
        self.mlp_dim       = mlp_dim
        self.dropout       = dropout

        self.ln1  = nn.LayerNorm(self.embedding_dim)
        self.ln2  = nn.LayerNorm(self.embedding_dim)
        self.attn = nn.MultiheadAttention(self.embedding_dim, num_heads, dropout, batch_first=True)

        self.mlp = nn.Sequential(
            nn.Linear(self.embedding_dim, self.mlp_dim),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.mlp_dim, self.embedding_dim),
            nn.Dropout(self.dropout),
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
        dropout: float = 0.1,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)

        self.num_layers    = num_layers
        self.num_classes   = num_classes
        self.image_size    = image_size
        self.patch_size    = patch_size
        self.embedding_dim = embedding_dim
        self.num_heads     = num_heads
        self.mlp_dim       = mlp_dim
        self.in_channels   = in_channels
        self.dropout       = dropout

        self.patch_embedding = PatchEmbedding(
            image_size, patch_size, in_channels=in_channels, embedding_dim=embedding_dim
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embedding_dim))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, 1 + self.patch_embedding.n_patches, self.embedding_dim)
        )
        self.pos_drop = nn.Dropout(p=self.dropout)

        self.transformer = nn.ModuleList(
            EncoderBlock(self.embedding_dim, self.num_heads, self.mlp_dim, self.dropout)
            for _ in range(self.num_layers)
        )
        self.ln = nn.LayerNorm(self.embedding_dim)
        self.head = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim),
            nn.Tanh(),
            nn.Linear(self.embedding_dim, self.num_classes),
        )
        self.erm_loss_fn = nn.CrossEntropyLoss(reduction="none")

        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.to(self.device)

    def put_on_device(self, *tensors: Iterable[torch.Tensor]) -> Iterable[torch.Tensor]:
        return (t.to(self.device) for t in tensors)

    def _preprocess(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embedding(x)
        cls_tokens = self.cls_token.expand(x.size(0), -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed[:, : x.size(1), :]
        return self.pos_drop(x)

    def backbone(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.transformer:
            x = block(x)
        return self.ln(x)

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x[:, 0])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.predict(self.backbone(self._preprocess(x)))

    @torch.no_grad()
    def extract_features(self, loader: DataLoader) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        self.eval()
        features, class_labels, bias_labels = [], [], []

        for batch in loader:
            match len(batch):
                case 3: inputs, labels, blabels = batch
                case 4: inputs, labels, blabels, _ = batch

            inputs, labels, blabels = self.put_on_device(inputs, labels, blabels)
            features.append(self.backbone(self._preprocess(inputs)))
            class_labels.append(labels)
            bias_labels.append(blabels)

        return (
            torch.cat(features, dim=0).cpu(),
            torch.cat(class_labels, dim=0).cpu(),
            torch.cat(bias_labels, dim=0).cpu(),
        )

    def train_model_erm(
        self,
        train_loader,
        val_loader,
        test_loader,
        learning_rate: float = 0.001,
        num_epochs: int = 50,
        accumulate: int = 1,
        wb: Union[WandbWrapper, None] = None,
    ):
        optimizer = torch.optim.AdamW(self.parameters(), lr=learning_rate, amsgrad=False)

        val_accuracies_avg, val_accuracies_b, val_accuracies_u = [], [], []
        test_accuracies_avg, test_accuracies_b, test_accuracies_u = [], [], []

        for epoch in range(num_epochs):
            train_loss = 0.0
            correct_train = total_train = 0
            correct_biased = total_biased = 0
            correct_unbiased = total_unbiased = 0

            self.train()
            optimizer.zero_grad()
            with torch.enable_grad():
                for batch_idx, (inputs, labels, bias_labels, _) in enumerate(train_loader):
                    labels = labels.type(torch.LongTensor)
                    inputs, labels, bias_labels, _ = self.put_on_device(inputs, labels, bias_labels, _)

                    outputs: torch.Tensor = self(inputs)
                    loss: torch.Tensor = self.erm_loss_fn(outputs, labels).mean() / accumulate
                    loss.backward()
                    train_loss += loss.item()

                    _, predicted = torch.max(outputs, dim=1)
                    total_train  += labels.size(0)
                    correct_train += (predicted == labels).sum().item()

                    total_biased    += (bias_labels != -1).sum().item()
                    correct_biased  += (predicted[bias_labels != -1] == labels[bias_labels != -1]).sum().item()
                    total_unbiased   += (bias_labels == -1).sum().item()
                    correct_unbiased += (predicted[bias_labels == -1] == labels[bias_labels == -1]).sum().item()

                    if (batch_idx + 1) % accumulate == 0 or (batch_idx + 1 == len(train_loader)):
                        optimizer.step()
                        optimizer.zero_grad()

            avg_loss_train        = train_loss / len(train_loader)
            accuracy_train        = 100 * correct_train / total_train
            accuracy_train_biased = 100 * correct_biased / max(total_biased, 1)
            accuracy_train_unbiased = 100 * correct_unbiased / max(total_unbiased, 1)

            # --- validation ---
            self.eval()
            val_loss = correct_val = total_val = 0
            correct_biased = total_biased = correct_unbiased = total_unbiased = 0

            with torch.no_grad():
                for inputs, labels, bias_labels in val_loader:
                    labels = labels.type(torch.LongTensor)
                    inputs, labels = self.put_on_device(inputs, labels)
                    outputs = self(inputs)
                    loss = self.erm_loss_fn(outputs, labels).mean()
                    val_loss += loss.item()
                    _, predicted = torch.max(outputs, dim=1)
                    total_val  += labels.size(0)
                    correct_val += (predicted == labels).sum().item()
                    total_biased    += (bias_labels != -1).sum().item()
                    correct_biased  += (predicted[bias_labels != -1] == labels[bias_labels != -1]).sum().item()
                    total_unbiased   += (bias_labels == -1).sum().item()
                    correct_unbiased += (predicted[bias_labels == -1] == labels[bias_labels == -1]).sum().item()

            avg_loss_val        = val_loss / len(val_loader)
            accuracy_val        = 100 * correct_val / total_val
            accuracy_val_biased = 100 * correct_biased / max(total_biased, 1)
            accuracy_val_unbiased = 100 * correct_unbiased / max(total_unbiased, 1)
            val_accuracies_avg.append(accuracy_val)
            val_accuracies_b.append(accuracy_val_biased)
            val_accuracies_u.append(accuracy_val_unbiased)

            # --- test ---
            self.eval()
            test_loss = correct = total = 0
            correct_biased = total_biased = correct_unbiased = total_unbiased = 0

            with torch.no_grad():
                for inputs, labels, bias_labels in test_loader:
                    bias_labels = bias_labels.to(self.device)
                    labels = labels.type(torch.LongTensor)
                    inputs, labels = self.put_on_device(inputs, labels)
                    outputs = self(inputs)
                    loss = self.erm_loss_fn(outputs, labels).mean()
                    test_loss += loss.item()
                    _, predicted = torch.max(outputs, dim=1)
                    total  += labels.size(0)
                    correct += (predicted == labels).sum().item()
                    total_biased    += (bias_labels != -1).sum().item()
                    correct_biased  += (predicted[bias_labels != -1] == labels[bias_labels != -1]).sum().item()
                    total_unbiased   += (bias_labels == -1).sum().item()
                    correct_unbiased += (predicted[bias_labels == -1] == labels[bias_labels == -1]).sum().item()

            avg_loss_test        = test_loss / len(test_loader)
            accuracy_test        = 100 * correct / total
            accuracy_test_biased = 100 * correct_biased / max(total_biased, 1)
            accuracy_test_unbiased = 100 * correct_unbiased / max(total_unbiased, 1)
            test_accuracies_avg.append(accuracy_test)
            test_accuracies_b.append(accuracy_test_biased)
            test_accuracies_u.append(accuracy_test_unbiased)

            print(
                f"Epoch {epoch+1}/{num_epochs}\n"
                f"  Train  loss={avg_loss_train:.4f} acc={accuracy_train:.2f}% "
                f"(aligned={accuracy_train_biased:.2f}% conflict={accuracy_train_unbiased:.2f}%)\n"
                f"  Val    loss={avg_loss_val:.4f}   acc={accuracy_val:.2f}% "
                f"(aligned={accuracy_val_biased:.2f}% conflict={accuracy_val_unbiased:.2f}%)\n"
                f"  Test   loss={avg_loss_test:.4f}  acc={accuracy_test:.2f}% "
                f"(aligned={accuracy_test_biased:.2f}% conflict={accuracy_test_unbiased:.2f}%)"
            )

            if wb is not None:
                plt.figure()
                plt.tight_layout()
                wb.log_output({
                    "epoch": epoch + 1,
                    "avg_loss_tr": avg_loss_train, "acc_tr": accuracy_train,
                    "acc_ba_tr": accuracy_train_biased, "acc_bc_tr": accuracy_train_unbiased,
                    "avg_loss_vl": avg_loss_val, "acc_vl": accuracy_val,
                    "acc_ba_vl": accuracy_val_biased, "acc_bc_vl": accuracy_val_unbiased,
                    "avg_loss_te": avg_loss_test, "acc_te": accuracy_test,
                    "acc_ba_te": accuracy_test_biased, "acc_bc_te": accuracy_test_unbiased,
                    "loss_hist": wb.backend.Image(plt),
                })
                plt.close()
                wb.log_model(self, f"SimpleViT_{wb.run_name}")

        torch.save(self.state_dict(), "vit.pth")
        return (
            val_accuracies_avg, val_accuracies_b, val_accuracies_u,
            test_accuracies_avg, test_accuracies_b, test_accuracies_u,
        )
