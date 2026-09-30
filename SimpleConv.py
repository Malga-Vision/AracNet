#!/usr/bin/env python3

import torch
import numpy as np
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
from typing import Iterable, Tuple, Union
from wandb_wrapper import WandbWrapper


def _lognorm(sample, mean, logvar, raxis=1):
    """Log probability under a diagonal Gaussian N(mean, exp(logvar))."""
    log2pi = np.log(2.0 * np.pi)
    return torch.sum(
        -0.5 * ((sample - mean) ** 2.0 * torch.exp(-logvar) + logvar + log2pi),
        dim=raxis,
    )


class VAE(nn.Module):
    DTYPE = torch.float32
    def __raise_not_implemented(self):
        raise NotImplementedError(
            "Any subclass that inherits from 'VAE' must implement concrete methods for building required layers\n" + \
            "Namely: 'build_mean_layer', 'build_logvar_layer', 'build_encoder', 'build_decoder'"
        )
    
    def __init__(self, latent_space_dim: int) -> None:
        super().__init__()

        self.latent_space_dim = latent_space_dim
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.encoder = self.build_encoder()
        self.mean    = self.build_mean_layer()
        self.logvar  = self.build_logvar_layer()
        self.decoder = self.build_decoder()

        # self.init_weights()
        self._reconstruction_loss_fn = nn.MSELoss(reduction="sum").to(self.device)
        self._lognorm_fn = _lognorm
        self.reconstruction_loss = torch.inf
        self.KL_divergence = torch.inf
        self.Alpha: float = 1.0
        self.Beta:  float = 1.0        

        self.to(self.device)

    def build_mean_layer(self, *args, **kwargs):
        self.__raise_not_implemented()

    def build_logvar_layer(self, *args, **kwargs):
        self.__raise_not_implemented()

    def build_encoder(self, *args, **kwargs):        
        self.__raise_not_implemented()

    def build_decoder(self, *args, **kwargs):
        self.__raise_not_implemented()

    def _total_loss(self, x: torch.Tensor, x_prime: torch.Tensor, z: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor):
        self.reconstruction_loss = -self._reconstruction_loss_fn(x, x_prime)
        zero_mean   = torch.Tensor([0.0]).to(self.device)
        zero_logvar = torch.Tensor([0.0]).to(self.device)
        log_pz = self._lognorm_fn(z, zero_mean, zero_logvar)
        log_qz_xcond = self._lognorm_fn(z, mu, logvar)
        self.KL_divergence = torch.sum(log_pz - log_qz_xcond)
        return -torch.sum((self.Alpha * self.reconstruction_loss) + (self.Beta * self.KL_divergence)) 

    def get_latent(self, X, batch_size=128):
        Z = []
        with torch.no_grad():
            for batch in DataLoader(X, batch_size=batch_size, shuffle=False):
                encoding = self.encoder(batch.to(self.device))
                z, _, _ = self.latent_space_sample(encoding)
                Z.append(z)
        Z = torch.cat(Z, dim=0)
        return Z.cpu().numpy()

    def latent_space_sample(self, x_full):
        mu = self.mean(x_full)
        logvar = self.logvar(x_full) 
        z = mu + torch.exp(0.5 * logvar) * torch.randn(mu.shape).to(self.device)
        return z, mu, logvar

    def forward(self, x):
        x_full = self.encoder(x)
        z, mu, logvar = self.latent_space_sample(x_full)
        x_prime = self.decoder(z)

        return x_prime, mu, logvar, z

class DenseVAE(VAE):
    def __init__(self, latent_space_dim: int, input_dim: int, distribution_neurons: int = 64) -> None:
        self.distribution_neurons = distribution_neurons # 64 or 512
        self.embedding_neurons = input_dim
        super().__init__(latent_space_dim)

    def build_encoder(self, *args, **kwargs):
        return nn.Sequential(
            nn.Linear(self.embedding_neurons, self.distribution_neurons, dtype=VAE.DTYPE),
            nn.GELU()
        )

    def build_decoder(self, *args, **kwargs):
        return nn.Sequential(
            nn.Linear(self.latent_space_dim, self.distribution_neurons, dtype=VAE.DTYPE),
            nn.GELU(),
            nn.Linear(self.distribution_neurons, self.embedding_neurons, dtype=VAE.DTYPE),
        )

    def build_mean_layer(self, *args, **kwargs):
        return nn.Linear(self.distribution_neurons, self.latent_space_dim, dtype=VAE.DTYPE)
    
    def build_logvar_layer(self, *args, **kwargs):
        return nn.Linear(self.distribution_neurons, self.latent_space_dim, dtype=VAE.DTYPE)

class SimpleConv(nn.Module):
    def __init__(self, num_classes, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.num_classes: int = num_classes
        self.device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        
        self.conv_layer:    nn.Conv2d = nn.Conv2d(3, 128, kernel_size=9, stride=4)
        self.relu_conv:     nn.ReLU   = nn.ReLU(inplace=True)
        self.max_pool:      nn.AdaptiveMaxPool2d = nn.AdaptiveMaxPool2d(1)
        self.linear:        nn.Linear = nn.Linear(128, num_classes)

        self.erm_loss_fn = nn.CrossEntropyLoss()
        self.vae = DenseVAE(32, input_dim=128)

        self.vae.to(self.device)
        self.to(self.device)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_layer(x)
        x = self.relu_conv(x)
        x = self.max_pool(x)
        x = torch.flatten(x, 1)

        return x

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)
    
    def sample_and_reconstruct(self, x: torch.Tensor) -> torch.Tensor:
        x_prime, mu, logvar, z = self.vae(x)
        self.vae_loss = self.vae._total_loss(x, x_prime, z, mu, logvar)

        return x_prime

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        x = self.conv_layer(x)
        x = self.relu_conv(x)
        x = self.max_pool(x)
        x = torch.flatten(x, 1)

        return self.linear(x)
    
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

        return torch.cat(bias_preds, dim=0).cpu(), num_misclassified_samples, total_per_class, misclassified_per_class
    
    def put_on_device(self, *tensors: Iterable[torch.Tensor]) -> Iterable[torch.Tensor]:
        return (tensor.to(self.device) for tensor in tensors)

    def test_model(self, test_loader):
        self.eval()
        total_loss_test = 0.0
        correct_test = 0
        total_test = 0
        correct_biased_test = 0
        correct_unbiased_test = 0
        total_biased_test = 0
        total_unbiased_test = 0

        with torch.no_grad():
            for inputs, labels, bias_labels in test_loader:
                inputs, labels, bias_labels = self.put_on_device(inputs, labels, bias_labels)
                labels = labels.type(torch.LongTensor)
                labels = labels.to(self.device)

                outputs = self(inputs)
                loss = self.erm_loss_fn(outputs, labels)
                total_loss_test += loss.item()

                _, predicted = torch.max(outputs.data, 1)
                total_test += labels.size(0)
                correct_test += (predicted == labels).sum().item()
                
                bias_predicted = predicted[bias_labels != -1]
                total_biased_test += len(bias_predicted)
                correct_biased_test += (bias_predicted == labels[bias_labels != -1]).sum().item()

                unbias_predicted = predicted[bias_labels == -1]
                total_unbiased_test += len(unbias_predicted)
                correct_unbiased_test += (unbias_predicted == labels[bias_labels == -1]).sum().item()

            average_loss_test = total_loss_test / len(test_loader)
            accuracy_test = 100 * correct_test / total_test

            accuracy_test_biased = 100 * correct_biased_test / total_biased_test
            accuracy_test_unbiased = 100 * correct_unbiased_test / (total_unbiased_test+0.0000001)            
        
        print(f"\t -- Test Loss: {average_loss_test:.4f}, Test Accuracy: {accuracy_test:.2f}%")
        print(f"\t\t Test Accuracy Biased: {accuracy_test_biased:.4f}%, Test Accuracy Unbiased: {accuracy_test_unbiased:.4f}%")

    def train_model_erm(
            self,
            train_loader,
            val_loader,
            test_loader,
            learning_rate=0.001,
            num_epochs=50,
            accumulate=1,
            wb: Union[WandbWrapper, None] = None
        ):
        
        optimizer = optim.AdamW(self.parameters(), lr=learning_rate, amsgrad=False)

        validation_accuracies_avg = []
        validation_accuracies_b   = []
        validation_accuracies_u   = []

        test_accuracies_avg       = []
        test_accuracies_b         = []
        test_accuracies_u         = []
        
        for epoch in range(num_epochs):
            total_loss = 0.0
            correct_train = 0
            total_train = 0
            total_biased = 0
            correct_biased = 0
            total_unbiased = 0
            correct_unbiased = 0
            
            self.train()
            optimizer.zero_grad()
            with torch.enable_grad():
                for batch_idx, (inputs, labels, bias_labels, _) in enumerate(train_loader):
                    labels = labels.type(torch.LongTensor)
                    inputs, labels = self.put_on_device(inputs, labels)

                    outputs: torch.Tensor = self(inputs)                    

                    loss: torch.Tensor = self.erm_loss_fn(outputs, labels) / accumulate
                    loss.backward(retain_graph=True)                    
                    total_loss += loss.item() # + self.vae_loss

                    _, predicted = torch.max(outputs.data, 1)
                    total_train += labels.size(0)
                    correct_train += (predicted == labels).sum().item()

                    bias_predicted = predicted[bias_labels != -1]
                    total_biased += len(bias_predicted)
                    correct_biased += (bias_predicted == labels[bias_labels != -1]).sum().item()

                    unbias_predicted = predicted[bias_labels == -1]
                    total_unbiased += len(unbias_predicted)
                    correct_unbiased += (unbias_predicted == labels[bias_labels == -1]).sum().item()

                    # self.eval()
                    # gradients: torch.Tensor = per_sample_gradient(self, self.erm_loss_fn, inputs, labels)
                    # self.train()

                    if ((batch_idx + 1) % accumulate == 0) or (batch_idx + 1 == len(train_loader)):                         
                        optimizer.step()
                        optimizer.zero_grad()

                
            average_loss_train = total_loss / len(train_loader)
            accuracy_train = 100 * correct_train / total_train
            accuracy_train_biased = 100 * correct_biased/total_biased
            accuracy_train_unbiased = 100 * correct_unbiased/(total_unbiased+0.000001)
            tr_tot_unbiased = total_unbiased

            self.eval()
            val_loss = 0.0
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
                    val_loss += loss.item()

                    _, predicted = torch.max(outputs.data, 1)
                    total_val += labels.size(0)
                    correct_val += (predicted == labels).sum().item()

                    bias_predicted = predicted[bias_labels != -1]
                    total_biased += len(bias_predicted)
                    correct_biased += (bias_predicted == labels[bias_labels != -1]).sum().item()

                    unbias_predicted = predicted[bias_labels == -1]
                    total_unbiased += len(unbias_predicted)
                    correct_unbiased += (unbias_predicted == labels[bias_labels == -1]).sum().item()

            average_loss_val = val_loss / len(val_loader)
            accuracy_val_biased = 100*correct_biased/total_biased
            accuracy_val_unbiased = 100*correct_unbiased/(total_unbiased+0.000001)
            accuracy_val = 100 * correct_val / total_val
            val_tot_unbiased = total_unbiased

            validation_accuracies_avg.append(accuracy_val)
            validation_accuracies_b.append(accuracy_val_biased)
            validation_accuracies_u.append(accuracy_val_unbiased)

            self.eval()
            test_loss = 0.0
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
                    test_loss += loss.item()

                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
                    
                    bias_predicted = predicted[bias_labels != -1]
                    total_biased += len(bias_predicted)
                    correct_biased += (bias_predicted == labels[bias_labels != -1]).sum().item()

                    unbias_predicted = predicted[bias_labels == -1]
                    total_unbiased += len(unbias_predicted)
                    correct_unbiased += (unbias_predicted == labels[bias_labels == -1]).sum().item()

            average_loss_test = test_loss / len(test_loader)
            accuracy_test = 100 * correct / total
            accuracy_test_biased = 100 * correct_biased / total_biased
            accuracy_test_unbiased = 100 * correct_unbiased / (total_unbiased+0.000001) 
            te_tot_unbiased = total_unbiased

            test_accuracies_avg.append(accuracy_test)
            test_accuracies_b.append(accuracy_test_biased)
            test_accuracies_u.append(accuracy_test_unbiased)   
                     
            output_str = "" + \
            f"Epoch {epoch + 1}/{num_epochs}\n\t -- Loss: {average_loss_train:.4f}, Train Accuracy: {accuracy_train:.2f}%" + \
            f"\t\t Train Accuracy Biased: {accuracy_train_biased:.4f} %, Train Accuracy Unbiased: {accuracy_train_unbiased:.4f}% ({tr_tot_unbiased})" + \
            f"\t -- Validation Loss: {average_loss_val:.4f}, Validation Accuracy: {accuracy_val:.4f} %" + \
            f"\t\t Valid Accuracy Biased: {accuracy_val_biased:.4f} %, Valid Accuracy Unbiased: {accuracy_val_unbiased:.4f}% ({val_tot_unbiased})" + \
            f"\t -- Test Loss: {average_loss_test:.4f}, Test Accuracy: {accuracy_test:.2f}%" + \
            f"\t\t Test Accuracy Biased: {accuracy_test_biased:.4f}%, Test Accuracy Unbiased: {accuracy_test_unbiased:.4f}% ({te_tot_unbiased})" 
            wb.log_output({
                "epoch": epoch+1,
                "avg_loss_tr": average_loss_train,
                "acc_tr": accuracy_train,
                "acc_ba_tr": accuracy_train_biased,
                "acc_bc_tr": accuracy_train_unbiased,
                "avg_loss_vl": average_loss_val,
                "acc_vl": accuracy_val,
                "acc_ba_vl": accuracy_val_biased,
                "acc_bc_vl": accuracy_val_unbiased,
                "avg_loss_te": average_loss_test,
                "acc_te": accuracy_test,
                "acc_ba_te": accuracy_test_biased,
                "acc_bc_te": accuracy_test_unbiased,
            }, step=(epoch+1)*len(train_loader))

        return (
            validation_accuracies_avg, validation_accuracies_b, validation_accuracies_u,
            test_accuracies_avg, test_accuracies_b, test_accuracies_u
        )


if __name__ == "__main__":
    x = torch.randn((1, 3, 224, 224))
    y = SimpleConv(num_classes=10)(x)
    print(y.size())
    