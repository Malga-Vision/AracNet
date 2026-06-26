from typing import Any, Callable, Dict, Generator, Iterator, List
from collections import OrderedDict
import torch
import torchvision
from torch import nn
from matplotlib import pyplot as plt
from metrics import *
from tqdm import tqdm
import os
import torch.nn.functional as F
from resnet import resnet20
from utils import Hook, GCELoss, FlatPooling, manual_avgpooling, CumulativeCELoss, softmax_distribution_hist, EMA, ranking_score_histogram_supervised, ranking_score_histogram,ranking_score_histogram_tail, ranking_score_histogram_SARLE



PATH_TO_MODELS = "./saved_models/"

def evaluate(model_debias, dataset, val_laoder, device, epoch, wandb, train_set=None):
    model_debias.eval()
    print(dataset)
    # if dataset == "waterbirds" or dataset == "BFFHQ":
    #     # ==== Evaluation (same as your code but ignoring bias labels where needed) ====
    #     label_results_aligned = torch.zeros((10, 10), device=device)
    #     label_results_conflicting = torch.zeros((10, 10), device=device)
    #     loss_per_class_aligned = {c: [] for c in range(10)}
    #     loss_per_class_conflict = {c: [] for c in range(10)}
    #     label_results_aligned2 = torch.zeros((10, 10), device=device)
    #     label_results_conflicting2 = torch.zeros((10, 10), device=device)
    #     loss_per_class_aligned2 = {c: [] for c in range(10)}
    #     loss_per_class_conflict2 = {c: [] for c in range(10)}
    #     label_results_aligned3 = torch.zeros((10, 10), device=device)
    #     label_results_conflicting3 = torch.zeros((10, 10), device=device)
    #     loss_per_class_aligned3 = {c: [] for c in range(10)}
    #     loss_per_class_conflict3 = {c: [] for c in range(10)}

    #     for batch, (dat, labels, _) in enumerate(val_loader):
    #         dat = dat.to(device)
    #         target = labels[0].to(device)
    #         bias_label = labels[1].to(device)

    #         output = model(dat)
            
    #         for i in range(dat.size(0)):
    #             # Here you can keep aligned/conflicting if bias labels are still present in test
    #             if bias_label[i] == target[i]:
    #                 label_results_aligned[target[i], torch.argmax(output[i])] += 1
    #                 loss_per_class_aligned[target[i].item()].append(
    #                     loss_sample[i].item()
    #                 )
    #                 label_results_aligned2[target[i], torch.argmax(output2[i])] += 1
    #                 loss_per_class_aligned2[target[i].item()].append(
    #                     loss_sample[i].item()
    #                 )
    #                 label_results_aligned3[target[i], torch.argmax(output3[i])] += 1
    #                 loss_per_class_aligned3[target[i].item()].append(
    #                     loss_sample[i].item()
    #                 )
    #             else:
    #                 label_results_conflicting[
    #                     target[i], torch.argmax(output[i])
    #                 ] += 1
    #                 loss_per_class_conflict[target[i].item()].append(
    #                     loss_sample[i].item()
    #                 )
    #                 label_results_conflicting2[
    #                     target[i], torch.argmax(output2[i])
    #                 ] += 1
    #                 loss_per_class_conflict2[target[i].item()].append(
    #                     loss_sample[i].item()
    #                 )
    #                 label_results_conflicting3[
    #                     target[i], torch.argmax(output3[i])
    #                 ] += 1
    #                 loss_per_class_conflict3[target[i].item()].append(
    #                     loss_sample[i].item()
    #                 )

    #     A = label_results_aligned.cpu().numpy()
    #     B = label_results_conflicting.cpu().numpy()
    #     A2 = label_results_aligned2.cpu().numpy()
    #     B2 = label_results_conflicting2.cpu().numpy()
    #     A3 = label_results_aligned3.cpu().numpy()
    #     B3 = label_results_conflicting3.cpu().numpy()

    #     print(f"epoch: {epoch + 1}")
    #     print(f"average_aligned {np.trace(A) / np.sum(A):.4f}")
    #     print(f"average_conflicting {np.trace(B) / np.sum(B):.4f}")
    #     print(f"ALIGNED_class_0 {A[0, 0] / (A[0, 0] + A[0, 1]):.4f}")
    #     print(f"ALIGNED_class_1 {A[1, 1] / (A[1, 0] + A[1, 1]):.4f}")
    #     print(f"CONFLICTING_class_0 {B[0, 0] / (B[0, 0] + B[0, 1]):.4f}")
    #     print(f"CONFLICTING_class_1 {B[1, 1] / (B[1, 0] + B[1, 1]):.4f}")
        
        
    #     wandb.log(
    #         {
    #             f"epoch": epoch,
    #             f"average_aligned": np.trace(A) / np.sum(A),
    #             f"average_conflicting": np.trace(B) / np.sum(B),
    #             f"aligned_class_0": A[0, 0] / (A[0, 0] + A[0, 1]),
    #             f"aligned_class_1": A[1, 1] / (A[1, 0] + A[1, 1]),
    #             f"conflicting_class_0": B[0, 0] / (B[0, 0] + B[0, 1]),
    #             f"conflicting_class_1": B[1, 1] / (B[1, 0] + B[1, 1]),
    #         },
    #         step=epoch,
    #     )
    # elif dataset == "urbancars":
    #         label_results = torch.zeros((2, 4), device=device)
    #         counts = torch.zeros((2, 4), device=device)

    #         for batch, (dat, labels, _) in enumerate(val_loader):
    #             dat = dat.to(device)
    #             target = labels[0].to(device)
    #             bias_label = labels[1].to(device)

    #             output, output2 = model(dat)

    #             for i in range(dat.size(0)):
    #                 # Here you can keep aligned/conflicting if bias labels are still present in test
    #                 pred = torch.argmax(output[i])
    #                 if pred == target[i]:
    #                     label_results[target[i], bias_label[i]] += 1
    #                 counts[target[i], bias_label[i]] += 1

    #         A = label_results / counts

    #         print(f"epoch: {epoch + 1}")
    #         print(f"average_aligned {A}")

    #         _, group_counts = train_set.get_sampling_weights(classes_only=False).unique(
    #             return_counts=True
    #         )
    #         train_distribution = (group_counts / len(train_set)).to(A.device)
    #         per_class_avg = A.mean(dim=0)
    #         in_acc = torch.sum(per_class_avg * train_distribution).to(A.device)
    #         gaps = torch.ones(len(per_class_avg)).to(A.device) * in_acc - per_class_avg

    #         gaps[0] = in_acc

    #         wandb.log(
    #             {
    #                 f"epoch": epoch,
    #                 f"group_0": A[:, 0].mean(),
    #                 f"group_1": A[:, 1].mean(),
    #                 f"group_2": A[:, 2].mean(),
    #                 f"group_3": A[:, 3].mean(),
    #                 f"in_acc": gaps[0],
    #                 f"bg_gap": gaps[1],
    #                 f"coobj_gap": gaps[2],
    #                 f"bg_coobj_gap": gaps[3],
    #             },
    #             step=epoch,
    #         )

    #         print(gaps)
    # elif dataset == "bar":
        # label_results_total = torch.zeros((6, 6)).cpu().numpy()
        # label_results_total2 = torch.zeros((6, 6)).cpu().numpy()
        # for batch, (dat, labels, _) in enumerate(val_loader):
        #     dat = dat.to(device)
        #     target = labels[0].to(device)
        #     bias_label = labels[1].to(device)

        #     output, output2 = model(dat)
        #     # output3 = (output2+output)/2.0
        #     output3 = output2
        #     loss_sample = torch.nn.CrossEntropyLoss(reduction="none")(
        #         output, target
        #     )
        #     for i in range(0, dat.shape[0]):
        #         label_results_total[target[i], torch.argmax(output[i])] += 1
        #         label_results_total2[target[i], torch.argmax(output2[i])] += 1

        # average_accuracy = np.trace(label_results_total) / np.sum(
        #     label_results_total
        # )
        # average_accuracy2 = np.trace(label_results_total2) / np.sum(
        #     label_results_total2
        # )

        # print(
        #     f"AVERAGE ACCURACY {np.trace(label_results_total)/np.sum(label_results_total)}"
        # )
        # print("biased_model")
        # print(
        #     f"AVERAGE ACCURACY_biased {np.trace(label_results_total2)/np.sum(label_results_total2)}"
        # )
        # for i in range(0, 6):
        #     print(
        #         label_results_total[i, i] / np.sum(label_results_total, axis=1)[i]
        #     )

        # wandb.log(
        #     {
        #         f"epoch": epoch,
        #         f"average_accuracy": np.trace(label_results_total)
        #         / np.sum(label_results_total),
        #         f"class_0": label_results_total[0, 0]
        #         / np.sum(label_results_total, axis=1)[0],
        #         f"class_1": label_results_total[1, 1]
        #         / np.sum(label_results_total, axis=1)[1],
        #         f"class_2": label_results_total[2, 2]
        #         / np.sum(label_results_total, axis=1)[2],
        #         f"class_3": label_results_total[3, 3]
        #         / np.sum(label_results_total, axis=1)[3],
        #         f"class_4": label_results_total[4, 4]
        #         / np.sum(label_results_total, axis=1)[4],
        #         f"class_5": label_results_total[5, 5]
        #         / np.sum(label_results_total, axis=1)[5],
        #     },
        #     step=epoch,
        # )
    
    if dataset=='waterbirds' or dataset=='BFFHQ':
                with torch.no_grad():

                    
        
                    model_debias.eval()
                    label_results_aligned = torch.zeros((10,10))
                    label_results_conflicting = torch.zeros((10,10))

                    for batch, (dat,labels,_) in enumerate(val_laoder):
                # if np.mod(batch,10)!=0:
                #     continue
                        dat = dat.to(device)
                        target = labels[0].to(device)
                        bias_l = labels[1].to(device)
                        output = model_debias(dat)
                        for i in range(0,dat.shape[0]):
                            if bias_l[i]==target[i]:
                                label_results_aligned[target[i],torch.argmax(output[i])]+=1
                                # loss_per_class_aligned[target[i].item()].append(loss_sample[i].item())
                            else:
                                label_results_conflicting[target[i],torch.argmax(output[i])]+=1
                                # loss_per_class_conflict[target[i].item()].append(loss_sample[i].item())
                    print(f'epoch: {epoch}')
                    A = label_results_aligned.cpu().numpy()
                    B = label_results_conflicting.cpu().numpy()
                    print(f'average_aligned {np.trace(A)/np.sum(A)}')
                    print(f'average_conflicting {np.trace(B)/np.sum(B)}')
                    

                    print(f'ALIGNED_class_0 {A[0,0]/(A[0,0]+A[0,1])}')
                    print(f'ALIGNED_class_1 {A[1,1]/(A[1,0]+A[1,1])}')
                    print(f'CONFLICTING_class_0 {B[0,0]/(B[0,0]+B[0,1])}')
                    print(f'CONFLICTING_class_1 {B[1,1]/(B[1,0]+B[1,1])}')


                    # wandb.log_output(
                    #             {
                    #                 f"epoch": epoch,
                    #                 f"average_aligned": np.trace(A) / np.sum(A),
                    #                 f"average_conflicting": np.trace(B) / np.sum(B),
                    #                 f"aligned_class_0": A[0, 0] / (A[0, 0] + A[0, 1]),
                    #                 f"aligned_class_1": A[1, 1] / (A[1, 0] + A[1, 1]),
                    #                 f"conflicting_class_0": B[0, 0] / (B[0, 0] + B[0, 1]),
                    #                 f"conflicting_class_1": B[1, 1] / (B[1, 0] + B[1, 1]),
                    #             },
                    #             step=epoch,
                    #         )
    
    elif dataset == "imagenet9A":
        label_results_total = torch.zeros((9, 9))
        for batch, (dat, labels, _) in enumerate(val_loader):
            dat = dat.to(device)
            target = labels[0].to(device)
            bias_label = labels[1].to(device)

            output, _ = model(dat)
            loss_sample = torch.nn.CrossEntropyLoss(reduction="none")(
                output, target
            )
            for i in range(0, dat.shape[0]):
                label_results_total[target[i], torch.argmax(output[i])] += 1

        average_accuracy = torch.trace(label_results_total) / torch.sum(
            label_results_total
        )

        print(
            f"AVERAGE ACCURACY {average_accuracy}"
        )
        for i in range(0, 9):
            print(
                label_results_total[i, i] / torch.sum(label_results_total, dim=1)[i]
            )

        wandb.log_output(
            {
                f"epoch": epoch,
                f"average_accuracy": average_accuracy
            },
            step=epoch,
        )
    
    elif dataset=='bar':
                    label_results_total = torch.zeros((6,6)).cpu().numpy()
                    label_results_total2 = torch.zeros((6,6)).cpu().numpy()
                    with torch.no_grad():
                        model_debias.eval()
                        for batch, (dat, labels, _) in enumerate(val_laoder):
                            dat = dat.to(device)
                            target = labels[0].to(device)
                            bias_label = labels[1].to(device)

                            output = model_debias(dat)
                            #output3 = (output2+output)/2.0
                            
                            loss_sample = torch.nn.CrossEntropyLoss(reduction="none")(output, target)
                            for i in range(0,dat.shape[0]):
                                label_results_total[target[i],torch.argmax(output[i])]+=1

                    

                        print(f'AVERAGE ACCURACY {np.trace(label_results_total)/np.sum(label_results_total)}')
                        print("biased_model")
                        for i in range(0,6):
                            print(label_results_total[i,i]/np.sum(label_results_total,axis=1)[i])
                        wandb.log_output(
                                {
                                    f"epoch": epoch,
                                    f"average_accuracy": np.trace(label_results_total)
                                    / np.sum(label_results_total),
                                    f"class_0": label_results_total[0, 0]
                                    / np.sum(label_results_total, axis=1)[0],
                                    f"class_1": label_results_total[1, 1]
                                    / np.sum(label_results_total, axis=1)[1],
                                    f"class_2": label_results_total[2, 2]
                                    / np.sum(label_results_total, axis=1)[2],
                                    f"class_3": label_results_total[3, 3]
                                    / np.sum(label_results_total, axis=1)[3],
                                    f"class_4": label_results_total[4, 4]
                                    / np.sum(label_results_total, axis=1)[4],
                                    f"class_5": label_results_total[5, 5]
                                    / np.sum(label_results_total, axis=1)[5],
                                },
                                step=epoch,
                            )
    elif dataset == "UrbanCars":
            label_results = torch.zeros((2, 4), device=device)
            counts = torch.zeros((2, 4), device=device)

            for batch, (dat, labels, _) in enumerate(val_laoder):
                dat = dat.to(device)
                target = labels[0].to(device)
                bias_label = labels[1].to(device)

                output  = model_debias(dat)
                for i in range(dat.size(0)):
                    # Here you can keep aligned/conflicting if bias labels are still present in test
                    pred = torch.argmax(output[i])
                    if pred == target[i]:
                        label_results[target[i], bias_label[i]] += 1
                    counts[target[i], bias_label[i]] += 1

            A = label_results / counts

            print(f"epoch: {epoch + 1}")
            print(f"average_aligned {A}")

            _, group_counts = train_set.get_sampling_weights(classes_only=False).unique(
                return_counts=True
            )
            train_distribution = (group_counts / len(train_set)).to(A.device)
            per_class_avg = A.mean(dim=0)
            in_acc = torch.sum(per_class_avg * train_distribution).to(A.device)
            gaps = torch.ones(len(per_class_avg)).to(A.device) * in_acc - per_class_avg

            gaps[0] = in_acc
            print(gaps)      
    else:

        
                label_results_total = torch.zeros((10,10)).cpu().numpy()
                label_results_total2 = torch.zeros((10,10)).cpu().numpy()

                with torch.no_grad():
                            model_debias.eval()
                            for batch, (dat, labels, _) in enumerate(val_laoder):
                                dat = dat.to(device)
                                target = labels[0].to(device)
                                
                                bias_label = labels[1].to(device)
                            
                                
                                output = model_debias(dat)
                                #output3 = (output2+output)/2.0
                                for i in range(0,dat.shape[0]):
                                
                                    label_results_total[target[i],torch.argmax(output[i])]+=1

                        

                            print(f'AVERAGE ACCURACY {np.trace(label_results_total)/np.sum(label_results_total)}')
                            print("biased_model")
                            for i in range(0,10):
                                print(label_results_total[i,i]/np.sum(label_results_total,axis=1)[i])

                            wandb.log_output(
                                {
                                    f"epoch": epoch,
                                    f"average_accuracy": np.trace(label_results_total)
                                    / np.sum(label_results_total),
                                    f"class_0": label_results_total[0, 0]
                                    / np.sum(label_results_total, axis=1)[0],
                                    f"class_1": label_results_total[1, 1]
                                    / np.sum(label_results_total, axis=1)[1],
                                    f"class_2": label_results_total[2, 2]
                                    / np.sum(label_results_total, axis=1)[2],
                                    f"class_3": label_results_total[3, 3]
                                    / np.sum(label_results_total, axis=1)[3],
                                    f"class_4": label_results_total[4, 4]
                                    / np.sum(label_results_total, axis=1)[4],
                                    f"class_5": label_results_total[5, 5]
                                    / np.sum(label_results_total, axis=1)[5],

                                    f"class_6": label_results_total[6, 6]
                                    / np.sum(label_results_total, axis=1)[6],
                                    f"class_7": label_results_total[7, 7]
                                    / np.sum(label_results_total, axis=1)[7],
                                    f"class_8": label_results_total[8, 8]
                                    / np.sum(label_results_total, axis=1)[8],
                                    f"class_9": label_results_total[9, 9]
                                    / np.sum(label_results_total, axis=1)[9],
                                },
                                step=epoch,
                            )
                           

   

class AracNet(nn.Module):
    





    
    def __setup_monitors_resnet50(self):

        weights = torchvision.models.ResNet50_Weights.DEFAULT if self.pretrained else None
        print(weights)

        self.base_model = torchvision.models.resnet50(weights=weights)

        #put back ImageNet!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        print(self.base_model_name)
        self.embedding_dim = self.base_model.fc.in_features
        self.base_model.fc = nn.Linear(self.embedding_dim, self.num_classes)
        self.base_model.avgpool = FlatPooling()

        self.named_layers = OrderedDict(self.base_model.named_children())

        mock_x = torch.randn((1, 3, 224, 224))
        with torch.no_grad():
            for module_name in self.named_layers.keys():    
                module = getattr(self.base_model, module_name)
                module_out_features = module(mock_x).shape[1:]
                mock_x = torch.randn((1, ) + module_out_features)
                
                if module_name in {"fc", "avgpool", "bn1", "relu", "conv1"}:
                    continue    
                    
                setattr(self.base_model, module_name, nn.Sequential(self.named_layers[module_name], nn.Identity()))                                
                if module_out_features[-1] > 1:
                    parallel_head = nn.Sequential(
                        FlatPooling(),
                        nn.Linear(module_out_features[0], self.num_classes)
                    ).to(self.device)
                else:
                    parallel_head = nn.Linear(module_out_features[0], self.num_classes).to(self.device)
                
                hook = Hook(getattr(self.base_model, module_name), backward=False)
                self.parallel_heads[module_name] = parallel_head
                self.hooks[module_name] = hook 

    def __setup_monitors_vit_b_16(self):

        # Load official ViT-B/16
        if self.pretrained:
            weights = torchvision.models.ViT_B_16_Weights.DEFAULT
        else:
            weights = None

        self.base_model = torchvision.models.vit_b_16(weights=weights)

        # ViT hidden dim (768 for ViT-B/16)
        self.embedding_dim = self.base_model.hidden_dim

        # Replace classifier head
        self.base_model.heads = nn.Linear(self.embedding_dim, self.num_classes)

        # Define layers we want to hook
        self.named_layers = OrderedDict({
            "conv_proj": self.base_model.conv_proj,
            **{f"block_{i}": block for i, block in enumerate(self.base_model.encoder.layers)},
            "ln": self.base_model.encoder.ln,
        })

        # Mock image
        mock_x = torch.randn(1, 3, 224, 224)

        with torch.no_grad():

            # ----- PATCH EMBEDDING -----
            seq = self.base_model.conv_proj(mock_x)               # (B, 768, 14, 14)
            seq = seq.flatten(2).transpose(1, 2)                  # (B, 196, 768)

            # ----- CLS TOKEN -----
            cls = self.base_model.class_token.expand(seq.size(0), -1, -1)
            seq = torch.cat([cls, seq], dim=1)                    # (B, 197, 768)

            # ----- ADD POSITIONAL EMBEDDING -----
            seq = seq + self.base_model.encoder.pos_embedding

            # ----- ITERATE LAYERS -----
            for name, module in self.named_layers.items():
                if name =='conv2D':
                    continue

                if name == "conv_proj":
                    # Already done
                    continue

                elif name.startswith("block_"):
                    out = module(seq)

                elif name == "ln":
                    out = module(seq)


                seq = out

                # CLS token only → shape (B, hidden_dim)
                cls_dim = seq.shape[-1]
                if name=='block_3' or name=='block_11' or name=='block_5' or name=='block_10' or name=='block_9' or name=='block_7' or name=='block_8':
                    print("yes")
                    continue
                # Parallel classifier head
                print(name)
                head = nn.Linear(cls_dim, self.num_classes).to(self.device)

                # Hook on this module
                hook = Hook(module, backward=False)

                self.parallel_heads[name] = head
                self.hooks[name] = hook
    def __setup_monitors_resnet18(self):
        weights = torchvision.models.ResNet18_Weights.DEFAULT if self.pretrained else None
        print("printing_now")
        print(self.pretrained)
        self.base_model = torchvision.models.resnet18(weights=weights)
        self.embedding_dim = self.base_model.fc.in_features
        self.base_model.fc = nn.Linear(self.embedding_dim, self.num_classes)
        self.base_model.avgpool = FlatPooling()

        self.named_layers = OrderedDict(self.base_model.named_children())

        mock_x = torch.randn((1, 3, 224, 224))
        with torch.no_grad():
            for module_name in self.named_layers.keys():    
                module = getattr(self.base_model, module_name)
                module_out_features = module(mock_x).shape[1:]
                mock_x = torch.randn((1, ) + module_out_features)
                
                if module_name in {"fc", "avgpool", "bn1", "relu", "conv1"}:
                    continue    
                    
                setattr(self.base_model, module_name, nn.Sequential(self.named_layers[module_name], nn.Identity()))                                
                if module_out_features[-1] > 1:
                    parallel_head = nn.Sequential(
                        FlatPooling(),
                        nn.Linear(module_out_features[0], self.num_classes)
                    ).to(self.device)
                else:
                    parallel_head = nn.Linear(module_out_features[0], self.num_classes).to(self.device)
                
                hook = Hook(getattr(self.base_model, module_name), backward=False)
                self.parallel_heads[module_name] = parallel_head
                self.hooks[module_name] = hook
    def __setup_monitors_resnet20(self):
        print(f"Using pretrained: {getattr(self, 'pretrained', False)}")

        # --- base model ---
        self.base_model = resnet20(num_classes=self.num_classes).to(self.device)
        self.embedding_dim = self.base_model.fc.in_features
        self.base_model.fc = nn.Linear(self.embedding_dim, self.num_classes).to(self.device)
        self.base_model.avgpool = FlatPooling()

        self.named_layers = OrderedDict(self.base_model.named_children())

        # CIFAR input
        mock_x = torch.randn((1, 3, 224, 224)).to(self.device)

        with torch.no_grad():
            for module_name, module in self.named_layers.items():

                # ----- special case for FC -----
                # if module_name == "fc":
                #     # flatten before passing through the linear layer
                #     mock_x = F.avg_pool2d(mock_x, mock_x.size()[3])
                #     mock_x = mock_x.view(mock_x.size(0), -1)
                #     out = module(mock_x)
                #     mock_x = out
                #     continue

                # ----- normal layers -----
                out = module(mock_x)
                module_out_features = out.shape[1:]
                mock_x = out  # pass sequentially

                # skip trivial layers
                if module_name in {"bn1", "conv1","avgpool"}:
                    continue

                # wrap monitored layer
                setattr(
                    self.base_model,
                    module_name,
                    nn.Sequential(self.named_layers[module_name], nn.Identity())
                )
                
                # If feature map still has H×W > 1, flatten via pooling
                if len(module_out_features) == 3 and module_out_features[-1] > 1:
                    parallel_head = nn.Sequential(
                        FlatPooling(),
                        nn.Linear(module_out_features[0], self.num_classes)
                    ).to(self.device)
                else:
                    parallel_head = nn.Linear(module_out_features[0], self.num_classes).to(self.device)

                # attach hook + head
                hook = Hook(getattr(self.base_model, module_name), backward=False)
                self.parallel_heads[module_name] = parallel_head
                self.hooks[module_name] = hook
    
    
    def __init__(self, num_classes: int, aracne: bool = False, base_model_name: str = "resnet18", pretrained: bool = True, *args, **kwargs) -> None:
            super().__init__(*args, **kwargs)

            self.num_classes: int = num_classes
            self.aracne: bool = aracne

            self.device: torch.device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

            self.pretrained         = pretrained
            self.base_model_name    = base_model_name

            self.parallel_heads: Dict[str, nn.Sequential] = {}
            self.hooks: Dict[str, Hook] = {}
            
            match base_model_name:
                case "resnet50":
                    self.__setup_monitors_resnet50()
                case "vgg16":
                    raise NotImplementedError
                    # self.__setup_monitors_vgg16()
                case "densenet201":
                    self.__setup_monitors_densenet201()
                case "resnet18":
                    self.__setup_monitors_resnet18()
                    print("yes_18_Massi")
                case "resnet20":
                    self.__setup_monitors_resnet20()
                    print("yes")
                case "ViTs-16":
                    self.__setup_monitors_vit_b_16()
                    print("vit")
                     
            self.base_model = self.base_model.to(self.device)
            self.loss_fn = nn.CrossEntropyLoss(reduction="none").to(self.device)
           #self.loss_fn = GCELoss(reduction="mean")
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
            except:
                print(self.base_model)
                self.base_model.heads.requires_grad_(False)
                self.base_model.heads.train(False)
        else:
            self.base_model.train()
            for p in self.base_model.parameters():
                p.requires_grad = True

    def freeze_legs(self, flag: bool) -> None:
        if flag:
            for key in self.parallel_heads.keys():
                head = self.parallel_heads[key]
                head.eval()
                for p in head.parameters():
                    p.requires_grad = False
        else:
            for key in self.parallel_heads.keys():
                head = self.parallel_heads[key]
                head.train()
                for p in head.parameters():
                    p.requires_grad = True

    def _forward_aracne(self, x: torch.Tensor) -> torch.Tensor:
        _ = self.base_model(x)
        self.parallel_z = { key : self.hooks[key].output for key in self.hooks.keys() }
        self.parallel_y = torch.cat(
            [self.parallel_heads[key](self.parallel_z[key]).unsqueeze(1) for key in self.parallel_z.keys()],
            dim=1,
        )
        
        return self.parallel_y
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
            # if not self.aracne:
            #     return self.base_model(x)
            # else:
            #     return self._forward_aracne(x)
            return self.base_model(x)
            
@torch.no_grad()
def evaluate_model(model: AracNet, dataloader, num_classes, num_bias_attributes, wb, make_figures=True, eval_name="test",dataset='BFFHQ'):
    model.eval()
    groups_size = (num_classes, ) * (num_bias_attributes + 1) 
    loss_task_tot   : AverageMeter = AverageMeter()
    heads_losses_tot = [AverageMeter() for _ in range(len(model.parallel_heads))]
    heads_accs_tot   = [AverageMeter() for _ in range(len(model.parallel_heads))]
    top1            : AverageMeter = AverageMeter()
    subgroup_top1   : AverageMeterSubgroups = AverageMeterSubgroups(
        size   = groups_size, 
        device = model.device
    )
    
    tk0 = tqdm(
        dataloader, total=int(len(dataloader)), leave=True, dynamic_ncols=True
    )
    
    test_parallel = []
    test_outputs = []
    test_targets = []
    test_biases  = []
    for batch, (dat, labels, _) in enumerate(tk0):
        dat     : torch.Tensor = dat.to(model.device)
        target  : torch.Tensor = labels[0].to(model.device)
        bias_l  : torch.Tensor = labels[1].to(model.device)
        output  : torch.Tensor = model(dat)        

        if make_figures:
            if model.aracne:
                model.parallel_z = { key : model.hooks[key].output for key in model.hooks.keys() }
                model.parallel_y = torch.cat(
                    [model.parallel_heads[key](model.parallel_z[key]).unsqueeze(1) for key in model.parallel_z.keys()],
                    dim=1,
                )

                heads_losses = {}
                heads_accs   = {}
                test_parallel.append(model.parallel_y)
                
                for i, key in enumerate(model.parallel_heads.keys()):                        
                    y: torch.Tensor = model.parallel_y.transpose(1, 0)[i]
                    head_loss: torch.Tensor = model.loss_fn(y, target).mean()
                    heads_losses[i] = heads_losses_tot[i].update(head_loss.item(), dat.size(0))
                    heads_accs[i] = heads_accs_tot[i].update(accuracy(y, target, topk=(1,))[0], dat.size(0)) 
                    
            test_outputs.append(output)
            test_targets.append(target)
            test_biases.append(bias_l)
        
        loss    : torch.Tensor = model.loss_fn(output, target).mean()      
        loss_task_tot.update(loss.item(), dat.size(0))
        
        acc1 = accuracy(output, target, topk=(1, ))
        subgroup_masks = get_subgroup_masks(
            labels = labels, 
            num_classes = groups_size, 
            device = model.device
        )
        subgroup_acc1 = accuracy_subgroup(output, target, subgroup_masks, num_classes=num_classes)
        
        top1.update(acc1[0], dat.size(0))
        subgroup_top1.update(subgroup_acc1, subgroup_masks)

        acc1  = top1.avg
        acc_a = round(regroup_by(subgroup_top1, ("aligned", ))[0].item(), 4)
        acc_m = round(regroup_by(subgroup_top1, ("misaligned",))[0].item(), 4)

        tk0.set_postfix(
            acc1 = acc1,
            acc_a = acc_a,
            acc_m = acc_m,
        )

        wb.log_output({"acc_1": acc1, "acc_a": acc_a, "acc_m": acc_m})
    
    if make_figures:
        test_outputs = torch.cat(test_outputs, dim=0)
        test_targets = torch.cat(test_targets, dim=0)
        test_biases  = torch.cat(test_biases, dim=0)
        if model.aracne:
            test_parallel = torch.cat(test_parallel, dim=0)
            for l in range(len(model.parallel_heads)):
                softmax_distribution_hist(test_parallel[:, l], test_targets, test_biases, target_class=0, epoch=eval_name, wb=wb, layer=l,dataset=dataset)
                softmax_distribution_hist(test_parallel[:, l], test_targets, test_biases, target_class=1, epoch=eval_name, wb=wb, layer=l,dataset=dataset)

        softmax_distribution_hist(test_outputs, test_targets, test_biases, target_class=0, epoch=eval_name, wb=wb, layer=-1,dataset=dataset)
        softmax_distribution_hist(test_outputs, test_targets, test_biases, target_class=1, epoch=eval_name, wb=wb, layer=-1,dataset=dataset)

@torch.no_grad()
def misclassified_distance_from_target(network_outputs: torch.Tensor, y_true: torch.Tensor):
    network_outputs = network_outputs.double()     # numerical stability
    _, preds = torch.max(network_outputs, dim=1)   # extract predictions
    misclassified_mask = preds != y_true           # exclude correctly classified samples
    network_outputs = network_outputs[misclassified_mask] 
    
    softmax_probs: torch.Tensor = torch.nn.functional.softmax(network_outputs, dim=1).clamp(min=1e-6, max=1.0) # logits to probs
    gathered_on_target: torch.Tensor = torch.gather(
        softmax_probs, dim=1, 
        index=torch.unsqueeze(y_true[misclassified_mask], dim=1)
    ) # onehot(y')_ij for i in N, j in C, j == y  
    
    gathered_on_pred: torch.Tensor = torch.gather(
        softmax_probs, dim=1, 
        index=torch.unsqueeze(preds[misclassified_mask], dim=1)
    ) # onehot(y')_ij for i in N, j in C, j == y  
    
    dist_from_target = ((
        gathered_on_pred.clamp(min=1e-6, max=1.0) - \
        gathered_on_target.clamp(min=1e-6, max=1.0)
    ) / 2).clamp(min=1e-6, max=1.0)
    
    return dist_from_target.mean(), dist_from_target.size(0) # Average, Size

def train_body(model: AracNet, train_loader, device, optimizer, num_classes,val_loader,base_model = 'resnet18', epochs=10, wb=None, make_figures=True,dataset='BFFHQ',bias_amount=99.5):
    cur_model_name = f"aracnet-{model.aracne}_{base_model}_{dataset}_{bias_amount}-biased-init.pt"
    print(f"Saving {cur_model_name}")
    torch.save(model.state_dict(), os.path.join(PATH_TO_MODELS,cur_model_name))

    for epoch in range(epochs):
        model.train(True)


        if dataset!='UrbanCars':
            loss_task_tot = AverageMeter()

            top1 = AverageMeter()
            
            subgroup_top1 = AverageMeterSubgroups((num_classes, )*(2), device=device)
        tk0 = tqdm(
            train_loader, total=int(len(train_loader)), leave=True, dynamic_ncols=True
        )
        print(train_loader)
        epoch_outputs = []
        epoch_targets = []
        epoch_biases  = []
        model.train()
        with torch.enable_grad():
            for batch, (dat, labels, _) in enumerate(tk0):
                dat = dat.to(device)
                target = labels[0].to(device)
                bias_l = labels[1].to(device)
                output = model(dat)
                
                if make_figures:                    
                    epoch_outputs.append(output)
                    epoch_targets.append(target)
                    epoch_biases.append(bias_l)
                
                
                optimizer.zero_grad()

                loss_task: torch.Tensor = model.loss_fn(output, target).mean()
                loss_task.backward()

                if dataset!='UrbanCars':
                    loss_task_tot.update(loss_task.item(), dat.size(0))
                optimizer.step()



                if dataset!='UrbanCars':

                    acc1  = accuracy(output, target, topk=(1,))
                    acc_a = regroup_by(subgroup_top1, ("aligned",))
                    acc_m = regroup_by(subgroup_top1, ("misaligned",))

                    subgroup_masks = get_subgroup_masks(labels, num_classes=(num_classes,)*(2),device=device)
                    subgroup_acc1 = accuracy_subgroup(output, target, subgroup_masks, num_classes=num_classes)
                    
                    top1.update(acc1[0], dat.size(0))
                    subgroup_top1.update(subgroup_acc1, subgroup_masks)
                    
                    tk0.set_postfix(
                        epoch=epoch,
                        acc1=top1.avg, 
                        acc_a = acc_a[0].item(), 
                        acc_m = acc_m[0].item(),
                        loss = loss_task_tot.avg
                    )                
                
                torch.save(model.state_dict(), os.path.join(PATH_TO_MODELS, cur_model_name))
            if epoch%5==0:
                    if dataset!='UrbanCars':
                        evaluate(model,dataset,val_loader,device,epoch,wb)    


        if dataset!='UrbanCars':
            wb.log_output({
                "epoch": epoch, 
                "loss": loss_task_tot.avg,
                "acc_1": acc1, 
                "acc_a": acc_a[0].item(), 
                "acc_m": acc_m[0].item()
            })
        
            
        if make_figures:
            epoch_outputs = torch.cat(epoch_outputs, dim=0)
            epoch_targets = torch.cat(epoch_targets, dim=0)
            epoch_biases  = torch.cat(epoch_biases, dim=0)
            if model.aracne:
                epoch_parallel = torch.cat(epoch_parallel, dim=0)
                for l in range(len(model.parallel_heads)):
                    softmax_distribution_hist(epoch_parallel[:, l], epoch_targets, epoch_biases, target_class=0, epoch=epoch, wb=wb, layer=l,dataset=dataset)
                    softmax_distribution_hist(epoch_parallel[:, l], epoch_targets, epoch_biases, target_class=1, epoch=epoch, wb=wb, layer=l,dataset=dataset)
            
            softmax_distribution_hist(epoch_outputs, epoch_targets, epoch_biases, target_class=0, epoch=epoch, wb=wb, layer=-1,dataset=dataset)
            softmax_distribution_hist(epoch_outputs, epoch_targets, epoch_biases, target_class=1, epoch=epoch, wb=wb, layer=-1,dataset=dataset)

    cur_model_name = f"aracnet-{model.base_model_name}-biased-final.pt"
    print(f"Saving {cur_model_name}")
    torch.save(model.state_dict(), os.path.join(PATH_TO_MODELS, f'{cur_model_name}_{dataset}_{bias_amount}'))
    return f"aracnet-{model.base_model_name}-{model.aracne}"



import re
def learning_from_legs_failure(model: AracNet, train_loader, val_laoder, device, optimizer: Callable[[Iterator, ], torch.optim.Optimizer], num_classes, epochs=10, wb=None, make_figures=True,dataset='waterbirds',train_set=None):
    #weights=torchvision.models.ResNet18_Weights.DEFAULT


    if dataset=='waterbirds' or dataset=='UrbanCars':
        
        model_debias = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.DEFAULT)


    elif dataset=='cifar10c':

        #model_debias = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT)
        model_debias =resnet20(10)
    

    else:

        model_debias = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT)
        #model_debias = torchvision.models.resnet18(weights=None)

    #model_debias = torchvision.models.resnet18(weights=None)
    #model_debias = resnet20(num_classes)
    print(num_classes)
    model_debias.fc = torch.nn.Linear(model_debias.fc.in_features, num_classes)
    #model_debias.load_state_dict(torch.load(os.path.join(PATH_TO_MODELS, 'model_debias_CIFAR_95.pth')))
    if dataset =='cifar10c':
            monitor = 2
            conf_thr = 0.05
            lr_debias = 0.005
            scratch_flag = 1
            thresh_max = 0.7,
            warmup_ = 70

    elif dataset =='bar':
        conf_thr = 0.15
        lr_debias = 0.00005
        scratch_flag = 0
        thresh_max = 0.3

        monitor = 3
        warmup_ = 20

    elif dataset =='BFFHQ':
            monitor = 3
            lr_debias = 0.00005
            conf_thr = 0.3
            thresh_max = 0.7
            scratch_flag = 0
            warmup_ = 20

    elif dataset =='waterbirds':
        monitor = 2
        lr_debias = 0.00005
        conf_thr = 0.3
        scratch_flag =0
        warmup_ = 30
        thresh_max = 0.7


        
    elif dataset =='UrbanCars':
            monitor = 4
            lr_debias = 0.00001
            conf_thr = 0.3   
            thresh_max = 0.7
 
            scratch_flag =0
            warmup_ = 5
            
    # Load source weights

    print(lr_debias)
    # Fix nested layer indices
    # new_state = OrderedDict()
    # for k, v in source_state.items():
    #     # Convert patterns like "layerX.0.Y." → "layerX.Y."
    #     fixed_key = re.sub(r'(layer\d+)\.0\.(\d+)\.', r'\1.\2.', k)
    #     new_state[fixed_key] = v

    # # Keep only compatible keys (matching shape and name)
    # target_state = model_debias.state_dict()
    # compatible = {k: v for k, v in new_state.items() if k in target_state and v.shape == target_state[k].shape}

    # # Load the partial state dict
    # #missing, unexpected = model_debias.load_state_dict(compatible, strict=False)

    # print(f"✅ Loaded {len(compatible)} matching layers.")
    # print(f"⚠️ Missing keys: {len(missing)} | Unexpected keys: {len(unexpected)}")
    #model_debias.fc = torch.nn.Linear(model_debias.fc.in_features, num_classes)

   

    model_debias = model_debias.to("cuda")
    model_debias.loss_fn = torch.nn.CrossEntropyLoss(reduction='none')
    if dataset=='UrbanCars':
        evaluate(model_debias,dataset,val_laoder,device,-1,wb,train_set)
    else:
        evaluate(model_debias,dataset,val_laoder,device,-1,wb)


    optimizer2 = torch.optim.AdamW(model_debias.parameters(), lr=lr_debias)


    scheduler2 = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer2,
    T_max=len(train_loader) * epochs,
    eta_min=1e-4
        )
    warmup=warmup_
    #warmup = 10 for BFFHQ
    cur_model_name = f"aracnet-{model.aracne}-biased-init.pt"
    print(f"Saving {cur_model_name}")
    torch.save(model.state_dict(), os.path.join(PATH_TO_MODELS,cur_model_name))

    optims = { # Dedicated optimizer for each parallel classification head
        key: optimizer( 
            model.parallel_heads[key].parameters(), 
        ) for key in model.parallel_heads.keys()
    } 
    new_criterion  = torch.nn.CrossEntropyLoss(reduction="none")
    


    # if dataset=='cifar10c':
    #     model_debias.load_state_dict(torch.load(os.path.join(PATH_TO_MODELS, 'model_debias_CIFAR_95.pth')))
    #     # for epoch in range(30):
    

        #     for batch, (dat, labels, _) in enumerate(train_loader):                
        #         dat = dat.to(device)
        #         target = labels[0].to(device)
        #         bias_l = labels[1].to(device)
        #         output = model_debias(dat)
        #         optimizer2.zero_grad()
        #         loss = model_debias.loss_fn(output,target).mean()
        #         loss.backward()
        #         optimizer2.step()
        #     evaluate(model_debias,dataset,val_laoder,device,epoch,wb,train_set)
        #     torch.save(model_debias.state_dict(), os.path.join(PATH_TO_MODELS, 'model_debias_CIFAR_95.pth'))    
    




    for epoch in range(epochs):


        # if epoch==15:
        #     for g in optimizer2.param_groups:
        #         g['lr'] = 5e-5  # new learning rate
       #     print("🔄 Learning rate changed to 1e-4")
        model.train(True)
        model_debias.train(True)
        heads_losses_tot = [AverageMeter() for _ in range(len(model.parallel_heads))]
        heads_accs_tot   = [AverageMeter() for _ in range(len(model.parallel_heads))]

        if dataset!='UrbanCars':
            top1 = AverageMeter()
            
            subgroup_top1 = AverageMeterSubgroups((num_classes, )*(2), device=device)
        tk0 = tqdm(
            train_loader, total=int(len(train_loader)), leave=True, dynamic_ncols=True
        )
        epoch_parallel = []
        loss_b_cumulative_aligned = 0
        loss_b_cumulative_conflict = 0
        filter_acc_aligned=0
        filter_acc_conflict=0
        loss_u_cumulative_aligned = 0
        loss_u_cumulative_conflict = 0
        correct_aligned=0
        incorrect_aligned =0

        correct_conflict =0
        incorrect_conflict =0

        epoch_outputs = []
        epoch_targets = []
        epoch_biases  = []
        model.train()
        model_debias.train()
        with torch.enable_grad():

            
            for batch, (dat, labels, _) in enumerate(tk0):                
                dat = dat.to(device)
                target = labels[0].to(device)
                bias_l = labels[1].to(device)
                output = model.forward(dat)
                if epoch>warmup:
                    output_debias = model_debias(dat)
                    loss_u = model_debias.loss_fn(output_debias, target)


                model.parallel_z = { key : model.hooks[key].output for key in model.hooks.keys() }
                model.parallel_y = torch.cat(
                    [model.parallel_heads[key](model.parallel_z[key]).unsqueeze(1) for key in model.parallel_z.keys()],
                    dim=1,
                )

                

                heads_losses = {}
                heads_accs   = {}
            

                for i, key in enumerate(model.parallel_heads.keys()):                        
                    optims[key].zero_grad()
                    y: torch.Tensor = model.parallel_y.transpose(1, 0)[i]
                    
                    if len(y.shape) > 2:
                        #i added this for ViT
                        y = y[:, 0, :]
                        print("yes")


                    g= torch.argmax(y,dim=1)
                    

                    head_loss: torch.Tensor = model.loss_fn(y, target).mean()
                    
                    
                   
                    head_loss.backward(retain_graph=True)
                    optims[key].step()
                    heads_losses[i] = heads_losses_tot[i].update(head_loss.item(), dat.size(0))
                    heads_accs[i] = heads_accs_tot[i].update(accuracy(y, target, topk=(1,))[0], dat.size(0))                     
                    
                    
                    if epoch>warmup:

                        with torch.no_grad():                 
                            # if i==0:
                            #     loss_b = new_criterion(y, target).detach()
                            # if i==1:
                            #     loss_b += 0.25 * new_criterion(y, target).detach()
                            # if i==2:
                            #     loss_b += 0.1 * new_criterion(y, target).detach()
                            if i==monitor:

                                t = y.clone()
                                loss_b =  new_criterion(y, target).detach()


                                # if np.isnan(loss_b.mean().item()):
                                #     raise NameError('loss_b_ema')
                                # if np.isnan(loss_u.mean().item()):
                                #     raise NameError('loss_d_ema')
                

                    ### ONLY IF YOU START FROM SCRATCH, TO ALIGN THE LOSSES
                    else:

                        if scratch_flag==1:
                        
                            output_debias = model_debias(dat)
                            loss_u = model_debias.loss_fn(output_debias, target)
                            optimizer2.zero_grad()
                            loss_u.mean().backward()
                            optimizer2.step()
                            
                #if epoch==50:
                    #torch.save(model_debias.state_dict(), os.path.join(PATH_TO_MODELS, 'model_debias_CIFAR_95.pth'))    
                
                if epoch>warmup:
           
                    
                   # loss_task: torch.Tensor = (loss_b/(loss_b+loss_u + 1e-8)) * model_debias.loss_fn(output_debias, target)

                    probs = torch.softmax(t, dim=1)
                    conf = torch.gather(probs, 1, target.unsqueeze(1)).squeeze(1) 
                    #loss_task: torch.Tensor = -torch.log((conf+1e-8)) *(loss_b/(loss_b+loss_u + 1e-8))* model_debias.loss_fn(output_debias, target)
                   
                    eps = 1e-8
                    cmin = 1e-3
                    p = 2.0
                    wmax = 30.0
                    wmin=1e-5

                    c = conf.clamp_min(cmin)
                    w = (-torch.log(c + eps))**p
                    #w = w.clamp(max=wmax,min=wmin)
                    w = w / (w.mean().detach() + eps) 
                    #rimosso w* ABLATION  (loss_b/(loss_b+loss_u + 1e-8)).detach()
                    #loss_task: torch.Tensor = w.detach() * * model_debias.loss_fn(output_debias, target) #* model_debias.loss_fn(output_debias, target)
                    #loss_task: torch.Tensor = (loss_b / (loss_b + loss_u + eps)).detach()  * model_debias.loss_fn(output_debias, target)
                          #model.parallel_y = model.parallel_y.detach()  (loss_b.detach()/(loss_b.detach()+loss_u.detach() + 1e-8)) 
                    loss_task: torch.Tensor =  w.detach() *   model_debias.loss_fn(output_debias, target)
                    # (loss_b / (loss_b + loss_u + eps)).detach()  * *loss_task: torch.Tensor = (loss_b/(loss_b+loss_u.detach() + 1e-8))* model_debias.loss_fn(output_debias, target)

                    #
                    conf_thr = conf_thr
                    conf_threshold = conf_thr
                    mask = (conf <= conf_threshold)
                  
                

                    # Define alignment masks
                    aligned_mask = target[mask] == bias_l[mask]
                    conflict_mask = target[mask] != bias_l[mask]
                    preds = torch.argmax(probs, dim=1)

                    # Correct and incorrect predictions within aligned/conflict
                    correct_aligned += ((preds[mask] == target[mask]) & aligned_mask).sum()
                    incorrect_aligned += ((preds[mask] != target[mask]) & aligned_mask).sum()

                    correct_conflict += ((preds[mask] == target[mask]) & conflict_mask).sum()
                    incorrect_conflict += ((preds[mask] != target[mask]) & conflict_mask).sum()

                    #mask=(target!=bias_l)
                    if epoch>1:
                        loss_task[mask] *= 1#int((len(conf)-mask.sum())/len(conf))
                        #loss_task[~mask] *=int((mask.sum())/len(conf))


                    mask_aligned = (bias_l==target)
                    mask_conflict = (bias_l!=target)
                
                    loss_b_cumulative_aligned+=loss_b[mask_aligned].mean()
                    if mask_conflict.any():
                        loss_b_cumulative_conflict+=loss_b[mask_conflict].mean()
                        loss_u_cumulative_conflict+=loss_u[mask_conflict].mean()
                    else:
                        loss_b_cumulative_conflict+=0
                        loss_u_cumulative_conflict+=0


                    loss_u_cumulative_aligned+=loss_u[mask_aligned].mean()
                    
                    loss_task = loss_task.mean()
                    optimizer2.zero_grad()

                    loss_task.backward()
                    optimizer2.step()
                    #scheduler2.step()


                if make_figures:                    
                    epoch_outputs.append(output)
                    epoch_targets.append(target)
                    epoch_biases.append(bias_l)



                epoch_parallel.append(model.parallel_y)
                
                
                if dataset!='UrbanCars':

                    acc1  = accuracy(output, target, topk=(1,))
                    acc_a = regroup_by(subgroup_top1, ("aligned",))
                    acc_m = regroup_by(subgroup_top1, ("misaligned",))



                if dataset!='UrbanCars':
                    subgroup_masks = get_subgroup_masks(labels, num_classes=(num_classes,)*(2),device=device)
                    subgroup_acc1 = accuracy_subgroup(output, target, subgroup_masks, num_classes=num_classes)
                    
                    top1.update(acc1[0], dat.size(0))
                    subgroup_top1.update(subgroup_acc1, subgroup_masks)
                
                    tk0.set_postfix(
                        epoch=epoch,
                        acc1=top1.avg, 
                        acc_a = acc_a[0].item(), 
                        acc_m = acc_m[0].item(),
                        heads_losses = [f"{heads_losses_tot[i].avg:.3f}" for i in range(len(model.parallel_heads))],
                        heads_accs   = [f"{heads_accs_tot[i].avg:.3f}" for i in range(len(model.parallel_heads))]
                    )                
                    
                #torch.save(model.state_dict(), os.path.join(PATH_TO_MODELS, cur_model_name))
        
        
        # if dataset=='cifar10c' and epoch<warmup:
        #     evaluate(model_debias,dataset,val_laoder,device,epoch,wb)
        
        if epoch>warmup:
            print(f'aligned_loss_b={loss_b_cumulative_aligned/len(train_loader)}')
            print(f'conflict_loss_b={loss_b_cumulative_conflict/len(train_loader)}')
            print(f'aligned_loss_u={loss_u_cumulative_aligned/len(train_loader)}')
            print(f'conflict_loss_u={loss_u_cumulative_conflict/len(train_loader)}')
            print(f'target_acc_aligned={filter_acc_aligned}')
            print(f'target_acc_conflict={filter_acc_conflict}')
            print(f"Correct aligned: {correct_aligned}")
            print(f"Incorrect aligned: {incorrect_aligned}")
            print(f"Correct conflict: {correct_conflict}")
            print(f"Incorrect conflict: {incorrect_conflict}")
            evaluate(model_debias,dataset,val_laoder,device,epoch=epoch,wandb=wb,train_set=train_set)
            


        # if dataset =='cifar10c' and epoch>=warmup:                    
        #     wb.log_output(
        #                             {
        #                                 f"epoch": epoch,
        #                                 f"average_accuracy": np.trace(label_results_total)/np.sum(label_results_total),
        #                             },
        #                             step=epoch)
        # if wb is not None:       
        #     wb.log_output({
        #         "epoch": epoch, 
        #         "loss": 0,
        #         "acc_1": acc1, 
        #         "acc_a": acc_a[0].item(), 
        #         "acc_m": acc_m[0].item(),
        #         "heads_losses": {i: heads_losses_tot[i].avg for i in range(len(model.parallel_heads))},
        #         "heads_accs":   {i: heads_accs_tot[i].avg for i in range(len(model.parallel_heads))}
        #     })
        
        if make_figures:
            epoch_outputs = torch.cat(epoch_outputs, dim=0)
            epoch_targets = torch.cat(epoch_targets, dim=0)
            epoch_biases  = torch.cat(epoch_biases, dim=0)

            ranking = torch.zeros(len(model.parallel_heads))
            ranking_ablation = torch.zeros(len(model.parallel_heads))
            ranking_ablation2 = torch.zeros(len(model.parallel_heads))


            if model.aracne:

                    epoch_parallel = torch.cat(epoch_parallel, dim=0)
                    for l in range(1,len(model.parallel_heads)):
                        for index in range(0,num_classes):
                            softmax_distribution_hist(epoch_parallel[:, l], epoch_targets, epoch_biases, target_class=index, epoch=epoch, wb=wb, layer=l,dataset=dataset)
                            
                            ranking[l] +=  ranking_score_histogram(epoch_parallel[:, l],epoch_targets,index,thresh=conf_thr,thresh_max=thresh_max)/num_classes
                            ranking_ablation[l] +=  ranking_score_histogram_tail(epoch_parallel[:, l],epoch_targets,index,thresh=conf_thr,thresh_max=thresh_max)/num_classes
                            ranking_ablation2[l] +=  ranking_score_histogram_SARLE(epoch_parallel[:, l],epoch_targets,index,thresh=conf_thr,thresh_max=thresh_max)/num_classes

                    for index in range(0,num_classes):
                        softmax_distribution_hist(epoch_outputs, epoch_targets, epoch_biases, target_class=index, epoch=epoch, wb=wb, layer=-1,dataset=dataset)


            
        if epoch==warmup-1:
            monitor = int(torch.argmax(ranking))
            monitor2 = int(torch.argmax(ranking_ablation))
            monitor3 = int(torch.argmax(ranking_ablation2))

            #if monitor==monitor2:
            print(monitor)
            print(ranking)
            print(ranking_ablation)
            print(ranking_ablation2)
            print("TEST_BIMODALITY")
            return
        print(ranking)
        print(ranking_ablation)
        print(ranking_ablation2)
        print(monitor)

    cur_model_name = f"aracnet-{model.base_model_name}-biased-final.pt"
    print(f"Saving {cur_model_name}")
    torch.save(model.state_dict(), os.path.join(PATH_TO_MODELS, cur_model_name))
    return f"aracnet-{model.base_model_name}-{model.aracne}"




def train_legs(model: AracNet, train_loader, device, optimizer: Callable[[Iterator, ], torch.optim.Optimizer], num_classes, epochs=10, wb=None, make_figures=True,dataset='BFFHQ'):
    cur_model_name = f"aracnet-{model.aracne}-biased-init.pt"
    print(f"Saving {cur_model_name}")
    torch.save(model.state_dict(), os.path.join(PATH_TO_MODELS,cur_model_name))

    optims = { # Dedicated optimizer for each parallel classification head
        key: optimizer(
            model.parallel_heads[key].parameters(), 
        ) for key in model.parallel_heads.keys()
    } 
    
    for epoch in range(epochs):
        model.train(True)
        heads_losses_tot = [AverageMeter() for _ in range(len(model.parallel_heads))]
        heads_accs_tot   = [AverageMeter() for _ in range(len(model.parallel_heads))]
        top1 = AverageMeter()
        
        subgroup_top1 = AverageMeterSubgroups((num_classes, )*(2), device=device)
        tk0 = tqdm(
            train_loader, total=int(len(train_loader)), leave=True, dynamic_ncols=True
        )
        epoch_parallel = []
        epoch_outputs = []
        epoch_targets = []
        epoch_biases  = []
        model.train()
        with torch.enable_grad():
            for batch, (dat, labels, _) in enumerate(tk0):                
                dat = dat.to(device)
                target = labels[0].to(device)
                bias_l = labels[1].to(device)
                output = model.forward(dat)

                model.parallel_z = { key : model.hooks[key].output for key in model.hooks.keys() }
                model.parallel_y = torch.cat(
                    [model.parallel_heads[key](model.parallel_z[key]).unsqueeze(1) for key in model.parallel_z.keys()],
                    dim=1,
                )

                heads_losses = {}
                heads_accs   = {}
            
                for i, key in enumerate(model.parallel_heads.keys()):                        
                    optims[key].zero_grad()
                    y: torch.Tensor = model.parallel_y.transpose(1, 0)[i]
                    head_loss: torch.Tensor = model.loss_fn(y, target)
                    head_loss.backward(retain_graph=True)
                    optims[key].step()
                    heads_losses[i] = heads_losses_tot[i].update(head_loss.item(), dat.size(0))
                    heads_accs[i] = heads_accs_tot[i].update(accuracy(y, target, topk=(1,))[0], dat.size(0))                     
                
                if make_figures:                    
                    epoch_outputs.append(output)
                    epoch_targets.append(target)
                    epoch_biases.append(bias_l)

                epoch_parallel.append(model.parallel_y)
                acc1  = accuracy(output, target, topk=(1,))
                acc_a = regroup_by(subgroup_top1, ("aligned",))
                acc_m = regroup_by(subgroup_top1, ("misaligned",))

                subgroup_masks = get_subgroup_masks(labels, num_classes=(num_classes,)*(2),device=device)
                subgroup_acc1 = accuracy_subgroup(output, target, subgroup_masks, num_classes=num_classes)
                
                top1.update(acc1[0], dat.size(0))
                subgroup_top1.update(subgroup_acc1, subgroup_masks)
                
                tk0.set_postfix(
                    epoch=epoch,
                    acc1=top1.avg, 
                    acc_a = acc_a[0].item(), 
                    acc_m = acc_m[0].item(),
                    heads_losses = [f"{heads_losses_tot[i].avg:.3f}" for i in range(len(model.parallel_heads))],
                    heads_accs   = [f"{heads_accs_tot[i].avg:.3f}" for i in range(len(model.parallel_heads))]
                )                
                
                torch.save(model.state_dict(), os.path.join(PATH_TO_MODELS, cur_model_name))
        if wb is not None:       
            wb.log_output({
                "epoch": epoch, 
                "loss": 0,
                "acc_1": acc1, 
                "acc_a": acc_a[0].item(), 
                "acc_m": acc_m[0].item(),
                "heads_losses": {i: heads_losses_tot[i].avg for i in range(len(model.parallel_heads))},
                "heads_accs":   {i: heads_accs_tot[i].avg for i in range(len(model.parallel_heads))}
            })
        
        if make_figures:
            epoch_outputs = torch.cat(epoch_outputs, dim=0)
            epoch_targets = torch.cat(epoch_targets, dim=0)
            epoch_biases  = torch.cat(epoch_biases, dim=0)
            if model.aracne:
                epoch_parallel = torch.cat(epoch_parallel, dim=0)
                for l in range(len(model.parallel_heads)):
                    softmax_distribution_hist(epoch_parallel[:, l], epoch_targets, epoch_biases, target_class=0, epoch=epoch, wb=wb, layer=l,dataset=dataset)
                    softmax_distribution_hist(epoch_parallel[:, l], epoch_targets, epoch_biases, target_class=1, epoch=epoch, wb=wb, layer=l,dataset=dataset)
            
            softmax_distribution_hist(epoch_outputs, epoch_targets, epoch_biases, target_class=0, epoch=epoch, wb=wb, layer=-1,dataset=dataset)
            softmax_distribution_hist(epoch_outputs, epoch_targets, epoch_biases, target_class=1, epoch=epoch, wb=wb, layer=-1, dataset=dataset)

    cur_model_name = f"aracnet-{model.base_model_name}-biased-final.pt"
    print(f"Saving {cur_model_name}")
    torch.save(model.state_dict(), os.path.join(PATH_TO_MODELS, cur_model_name))
    return f"aracnet-{model.base_model_name}-{model.aracne}"

def train_model_dist_from_target(model: AracNet, train_loader, device, optimizer, num_classes, epochs=10, check_dist=10, warmup=50, wb=None, make_figures=True,dataset='BFFHQ'):
    max_dist=0
    cur_model_name = f"aracnet-{model.aracne}-biased-init.pt"
    print(f"Saving {cur_model_name}")
    torch.save(model.state_dict(), os.path.join(PATH_TO_MODELS,cur_model_name))

    for epoch in range(epochs):
        model.train(True)
        loss_task_tot = AverageMeter()
        top1 = AverageMeter()
        dist_avg = AverageMeter()
        subgroup_top1 = AverageMeterSubgroups((num_classes, )*(2), device=device)
        tk0 = tqdm(
            train_loader, total=int(len(train_loader)), leave=True, dynamic_ncols=True
        )
        epoch_parallel = []
        epoch_outputs = []
        epoch_targets = []
        epoch_biases  = []
        model.train()
        with torch.enable_grad():
            for batch, (dat, labels, _) in enumerate(tk0):
                dat = dat.to(device)
                target = labels[0].to(device)
                bias_l = labels[1].to(device)
                output = model(dat)
                
                if make_figures:
                    if model.aracne:
                        epoch_parallel.append(model.parallel_y)
                    
                    epoch_outputs.append(output)
                    epoch_targets.append(target)
                    epoch_biases.append(bias_l)

                dist, num_elems = misclassified_distance_from_target(network_outputs=output, y_true=target)
                dist_avg.update(dist.clamp(min=1e-6, max=1.0), num_elems)
            
                loss_task: torch.Tensor = model.loss_fn(output, target)
                if model.aracne:
                    aracne_loss: torch.Tensor = model.aracne_loss_fn(model.parallel_y, target)
                    loss_task  += aracne_loss
                
                loss_task.backward()
                loss_task_tot.update(loss_task.item(), dat.size(0))
                optimizer.step()
                optimizer.zero_grad()
                
                acc1  = accuracy(output, target, topk=(1,))
                acc_a = regroup_by(subgroup_top1, ("aligned",))
                acc_m = regroup_by(subgroup_top1, ("misaligned",))

                subgroup_masks = get_subgroup_masks(labels, num_classes=(num_classes,)*(2),device=device)
                subgroup_acc1 = accuracy_subgroup(output, target, subgroup_masks, num_classes=num_classes)
                
                top1.update(acc1[0], dat.size(0))
                subgroup_top1.update(subgroup_acc1, subgroup_masks)
                
                tk0.set_postfix(
                    epoch=epoch,
                    acc1=top1.avg, 
                    acc_a = acc_a[0].item(), 
                    acc_m = acc_m[0].item(),
                    dist = torch.as_tensor(dist_avg.avg).item(),
                    loss = loss_task_tot.avg
                )
                
                if batch % check_dist == 0 :
                    if max_dist < torch.as_tensor(dist_avg.avg).item() and ((epoch + 1) * (batch + 1)) >= warmup:
                        max_dist = torch.as_tensor(dist_avg.avg).item()
                        print("New max")
                    cur_model_name = f"aracnet-{model.aracne}-biased-updated.pt"
                    # print(f"Saving {cur_model_name}")
                    torch.save(model.state_dict(), os.path.join(PATH_TO_MODELS, cur_model_name))
                
        wb.log_output({"epoch": epoch, "loss": loss_task_tot.avg,"acc_1": acc1, "acc_a": acc_a[0].item(), "acc_m": acc_m[0].item()})
        if make_figures:
            epoch_outputs = torch.cat(epoch_outputs, dim=0)
            epoch_targets = torch.cat(epoch_targets, dim=0)
            epoch_biases  = torch.cat(epoch_biases, dim=0)
            if model.aracne:
                epoch_parallel = torch.cat(epoch_parallel, dim=0)
                for l in range(len(model.parallel_heads)):
                    softmax_distribution_hist(epoch_parallel[:, l], epoch_targets, epoch_biases, target_class=0, epoch=epoch, wb=wb, layer=l,dataset=dataset)
                    softmax_distribution_hist(epoch_parallel[:, l], epoch_targets, epoch_biases, target_class=1, epoch=epoch, wb=wb, layer=l,dataset=dataset)
            
            softmax_distribution_hist(epoch_outputs, epoch_targets, epoch_biases, target_class=0, epoch=epoch, wb=wb, layer=-1,dataset=dataset)
            softmax_distribution_hist(epoch_outputs, epoch_targets, epoch_biases, target_class=1, epoch=epoch, wb=wb, layer=-1,dataset=dataset)

    cur_model_name = f"aracnet-{model.aracne}-biased-final.pt"
    print(f"Saving {cur_model_name}")
    torch.save(model.state_dict(), os.path.join(PATH_TO_MODELS, cur_model_name))
    return f"aracnet-{model.aracne}"
        
        
if __name__ == "__main__":
    net = AracNet(num_classes=2)
    # net(torch.randn((8, 3, 224, 224)).to("cuda"), inference=True)        



