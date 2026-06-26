#!/usr/bin/env python3

from typing import Iterable, List, Literal, Set, Union
from cifar10c import CIFAR10C
from BFFHQ import BFFHQ
from waterbirds import Waterbirds
from bar import BAR
from torchvision import transforms
import joblib
import numpy as np
from training_utils import GCELoss
from matplotlib import pyplot as plt
import os
from urbancars import UrbanCars
import random
import sys
print("Current working directory:", os.getcwd())
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report
from SimpleConv import SimpleConv
from SimpleViT import SimpleViT, BoostViT, OriginalViT
from wandb_wrapper import WandbWrapper
from AracNet import AracNet, train_model_dist_from_target, evaluate_model, train_body, train_legs,learning_from_legs_failure
from metrics import set_seed

import torch
from torch.utils.data import DataLoader, TensorDataset, StackDataset, WeightedRandomSampler, random_split

datasets_configs = {
    "cifar10c": lambda bias_amount: {
        "dataset_constructor": CIFAR10C,
        "dataset_kwargs": {"bias_amount": bias_amount},
        "erm_sampler_replacement": False,
        "image_size": 32,
        "batch_size": 256,
        "accumulate": 1,
        "model_base_name": "resnet20",
    },
    
    "waterbirds": lambda _: {
        "dataset_constructor": Waterbirds,
        "dataset_kwargs": {},
        "erm_sampler_replacement": True,
        "image_size": 224,              
        "batch_size": 64,
        "accumulate": 1,
        "model_base_name": "resnet50",
    },
    
    "bar": lambda bias_amount: {
        "dataset_constructor": BAR,
        "dataset_kwargs": {"bias_amount": bias_amount},
        "erm_sampler_replacement": True,
        "image_size": 224,
        "batch_size": 64,
        "accumulate": 1,
        "model_base_name": "resnet18"
    },
        "BFFHQ": lambda bias_amount: {
        "dataset_constructor": BFFHQ,
        "dataset_kwargs": {"bias_amount": bias_amount},
        "erm_sampler_replacement": True,
        "image_size": 224,
        "batch_size": 64,
        "accumulate": 1,
        "model_base_name": "resnet18"
    },
        "UrbanCars": lambda bias_amount: {
        "dataset_constructor": UrbanCars,
        "dataset_kwargs": {"bias_amount": bias_amount},
        "erm_sampler_replacement": True,
        "image_size": 224,
        "batch_size": 64,
        "accumulate": 1,
        "model_base_name": "resnet50"
    }
}


from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument(
    "--dataset",
    type=str,
    choices=["bar", "BFFHQ", "waterbirds","cifar10c","UrbanCars"],
    required=True,
)


parser.add_argument("--epochs_base_model", type=int, default=100)
parser.add_argument("--epochs_debiasing", type=int, default=50)
parser.add_argument("--train_body_aracnet", action="store_true")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--rho", type=float, default=95)














if __name__ == "__main__":


    args = parser.parse_args()
    # args = parser.parse_args([
    # "--dataset", "cifar10c",
    # "--epochs_base_model", "50",
    # "--epochs_debiasing", "100",
    # #"--train_body_aracnet", 
    # "--seed", "1",
    # "--rho", "99"
    #     ])
    epochs_base_model = args.epochs_base_model
    dataset = args.dataset

    epochs_debiasing = args.epochs_debiasing
    train_body_aracnet = args.train_body_aracnet
    seed = args.seed
    print(args.rho)
    rho = args.rho

    train_set:              Union[CIFAR10C, Waterbirds, BAR, BFFHQ, UrbanCars] #type-annot
    model_constructor:      Union[SimpleConv, SimpleViT, AracNet] #type-annot    
    model:                  Union[SimpleConv, SimpleViT, AracNet] #type-annot    
    bias_amount = rho
    print(rho)
    #dataset = "BFFHQ"

    
    print("BIAS_AMOUNT", bias_amount)
    config = datasets_configs[dataset](bias_amount)
    model_name = config["model_base_name"]
    dataset_constructor = config["dataset_constructor"]
    dataset_kwargs = config["dataset_kwargs"]
    erm_sampler_replacement = config["erm_sampler_replacement"]
    config["base_model"]=model_name



    if dataset=='bar':


        train_set = dataset_constructor(
        env="train",
        **dataset_kwargs
        )

        val_set = dataset_constructor(
            env="val",
            **dataset_kwargs
        )

        test_set = dataset_constructor(
            env="test",
            **dataset_kwargs
    )
        num_classes=6
        a = BAR(
            root="./data",
            bias_amount=95,
            env="train",
            transform=None,
            return_index=True,
        )
        class_sample_count = a.perclass_populations()
        weight = 1.0 / class_sample_count
        targets = list()
        print(train_set)
        for i in range(0, len(train_set)):
            targets.append(train_set[i][1][0])
        targets = np.array(targets)
        samples_weight = np.array([weight[t] for t in targets])
        samples_weight = torch.from_numpy(samples_weight)
        sampler = torch.utils.data.WeightedRandomSampler(
            weights=samples_weight, num_samples=len(samples_weight), replacement=True
        )
        class_weights = weight


    elif dataset=='waterbirds':
        train_set = dataset_constructor(
        env="train",
        **dataset_kwargs
        )

        val_set = dataset_constructor(
            env="val",
            **dataset_kwargs
        )

        test_set = dataset_constructor(
            env="test",
            **dataset_kwargs
    )
        num_classes=2
        class_pops, class_labels = train_set.perclass_populations(return_labels=True)
        class_weights: torch.Tensor = torch.as_tensor([1.0 / class_pops[y] for y in class_labels])
        targets = list()

        for i in range(0,len(train_set)):
            targets.append(train_set.samples[i]['class_label'])
        targets = torch.asarray(targets)
        samples_weight = torch.asarray([class_weights[t] for t in targets])
        


    elif dataset=='BFFHQ':
        train_set = dataset_constructor(
        env="train",
        **dataset_kwargs
        )

        val_set = dataset_constructor(
            env="val",
            **dataset_kwargs
        )

        test_set = dataset_constructor(
            env="test",
            **dataset_kwargs
    )
        num_classes=2


    elif dataset == "UrbanCars":
        train_set = UrbanCars(env="train", group_label="both")
        val_set = UrbanCars(env="val", group_label="both")
        test_set = UrbanCars(env="test", group_label="both")

        num_classes =2
    elif dataset=='cifar10c':
        num_classes=10
        train_transform = transforms.Compose([
        #transforms.RandomCrop(
        # ),
        #transforms.RandomResizedCrop(32, scale=(0.8,1.0)),
       # transforms.Resize((64, 64), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.RandomCrop(32,padding=4),
        transforms.RandomHorizontalFlip(0.5),
        # transforms.ToTensor(),
        
        #transforms.RandomVerticalFlip (0.5),
        
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),

        #transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC),
        #transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                                            ])
      
        eval_transform = transforms.Compose([
          #  transforms.Resize((64,64), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            #transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                                            ])

        train_set = CIFAR10C(root="./data", split="train", bias_amount=bias_amount, transform=train_transform)
        val_set   = CIFAR10C(root="./data", split="val", bias_amount=bias_amount, transform=eval_transform)
        test_set  = CIFAR10C(root="./data", split="test", bias_amount=bias_amount, transform=eval_transform)

    print("\nSizes of Sets:")
    print(f"Training set: {len(train_set)} samples")
    print(f"Validation set: {len(val_set)} samples")
    print(f"Test set: {len(test_set)} samples")

    if dataset =='waterbirds' or dataset=='bar':
        classweighted_sampler: WeightedRandomSampler = WeightedRandomSampler(
            weights=samples_weight,
            num_samples=len(train_set),
            replacement=True
                                                                        )
        print(class_weights.unique(return_counts=True))
    
    batch_size =    config["batch_size"]
    accumulate =    config["accumulate"]
    image_size =    config["image_size"]
    

    config["seed"]          = seed
    config["pretrained"]    = True
    config["train_samples"] = len(train_set)
    config["val_samples"]   = len(val_set)
    config["test_samples"]  = len(test_set)

    set_seed(seed)
    print(seed)
    val_loader  : DataLoader = DataLoader(val_set,   batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=4)
    test_loader : DataLoader = DataLoader(test_set,  batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=4)
    print(batch_size)

    if dataset=='waterbirds':
        print("passa_data_loader")
        train_loader: DataLoader = DataLoader(train_set, batch_size=batch_size, sampler=classweighted_sampler, pin_memory=True, num_workers=4)   


    elif dataset =='bar':
        train_loader: DataLoader = DataLoader(train_set, batch_size=batch_size, sampler=sampler, pin_memory=True, num_workers=4)    
    elif dataset=='cifar10c':
        
        # class_sample_count = train_set.perclass_populations()
        # print(class_sample_count)
        # weight = 1.0 / class_sample_count
        # targets = list()
        # for i in range(0, len(train_set)):
        #     targets.append(train_set[i][1][0])
        # targets = np.array(targets)
        # samples_weight = np.array([weight[t] for t in targets])
        # samples_weight = torch.from_numpy(samples_weight)
        # sampler = torch.utils.data.WeightedRandomSampler(
        #     weights=samples_weight, num_samples=len(samples_weight), replacement=True
        # )

        train_loader = DataLoader(train_set, batch_size=batch_size, drop_last=False, shuffle=True, pin_memory=True, num_workers=4)
        val_loader   = DataLoader(val_set, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=4)
        test_loader  = DataLoader(test_set, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=4)
    elif dataset=='UrbanCars':
        train_loader = DataLoader(
            train_set, batch_size=batch_size, num_workers=4, shuffle=True
        )
        val_loader = DataLoader(val_set, batch_size=batch_size, num_workers=4)
        test_loader = DataLoader(test_set, batch_size=batch_size, num_workers=4)
    else:
        train_loader: DataLoader = DataLoader(train_set, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=4)    
        val_loader = DataLoader(val_set, batch_size=batch_size, shuffle = False,num_workers=4)
        test_loader = DataLoader(test_set, batch_size=batch_size, shuffle = False, num_workers=4)
    wb = WandbWrapper(project_name="AracNet", config=config) 

    for key in config.keys():
        print(f"<{key}>: ", config[key])

    model = AracNet(num_classes=num_classes, aracne=False, base_model_name=config["base_model"], pretrained=config["pretrained"])

    save_results_to = f"./hist/SIMPLE/{dataset}/{str(bias_amount).replace('.', '')}/{model_name}"
    
    
    print(train_body_aracnet)
    if train_body_aracnet: 
    
        try:    
            
            save_results_to = f"./hist/SIMPLE/{dataset}/{str(bias_amount).replace('.', '')}/{model_name}"
            os.makedirs(save_results_to, exist_ok=True)

            model.freeze_body(False)
            model.freeze_legs(True)
            model.set_aracne(False)

            base_model_name = train_body(
                model, 
                train_loader, 
                model.device, 
                #0.001
                optimizer=torch.optim.SGD(model.parameters(), lr=0.001, weight_decay=0),
                base_model = config["base_model"],
                num_classes=num_classes,
                epochs=epochs_base_model,
                val_loader=test_loader,
                dataset=dataset,
                bias_amount = bias_amount,
                wb=wb
            )
        finally:
            torch.save(model.state_dict(), os.path.join(save_results_to, f"{base_model_name}_{dataset}_{bias_amount}-biased-final.pth"))    
            if dataset!='UrbanCars':
                evaluate_model(model, train_loader, num_classes, num_bias_attributes=1, wb=wb, make_figures=True, eval_name="train",dataset=dataset)
                evaluate_model(model, test_loader, num_classes, num_bias_attributes=1, wb=wb, make_figures=True, eval_name="test",dataset=dataset)
    
        print(os.getcwd())

        print(num_classes)
        if dataset!='UrbanCars':
            evaluate_model(model, train_loader, num_classes, num_bias_attributes=1, wb=wb, make_figures=True, eval_name="train",dataset=dataset)
            evaluate_model(model, test_loader, num_classes, num_bias_attributes=1, wb=wb, make_figures=True, eval_name="test",dataset=dataset)
        
        
    try:
        model.load_state_dict(torch.load(os.path.join(save_results_to, f"aracnet-{model.base_model_name}-{model.aracne}_{dataset}_{bias_amount}-biased-final.pth")))
    except:
        save_results_to = f"./hist/SIMPLE/{dataset}/{str(bias_amount).replace('.', '')}/{'aracnet'}"

        model.load_state_dict(torch.load(os.path.join(save_results_to, f"aracnet-{model.base_model_name}-{model.aracne}_{dataset}_{bias_amount}-biased-final.pth")))

    model.freeze_body(True)
    model.freeze_legs(False)
    model.set_aracne(True)

    base_model_name = learning_from_legs_failure(
        model, 
        train_loader, 
        test_loader,
        model.device, 
        optimizer=lambda params : torch.optim.SGD(params, lr=0.05, weight_decay=0.0),
        num_classes=num_classes,
        epochs=epochs_debiasing,
        wb=wb,
        dataset=dataset,
        train_set = train_set
    )
    
    
    torch.save(model.state_dict(), os.path.join(save_results_to, f"{base_model_name}_{dataset}_{bias_amount}.pth"))      
    model.load_state_dict(torch.load(f"./saved_models/aracnet-{config['base_model']}-biased-final.pt"))
    if dataset!='UrbanCars':
        evaluate_model(model, train_loader, num_classes, num_bias_attributes=1, wb=wb, make_figures=True, eval_name="train",dataset=dataset)
        evaluate_model(model, test_loader, num_classes, num_bias_attributes=1, wb=wb, make_figures=True, eval_name="test",dataset=dataset)         
    
    if wb is not None:
        wb.finish()



