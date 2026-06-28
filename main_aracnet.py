#!/usr/bin/env python3

from argparse import ArgumentParser
import os
import random
import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import transforms


def _seed_worker(worker_id):
    """Seeds numpy and random in each DataLoader worker from PyTorch's per-worker seed."""
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

from cifar10c import CIFAR10C
from BFFHQ import BFFHQ
from waterbirds import Waterbirds
from bar import BAR
from urbancars import UrbanCars
from wandb_wrapper import WandbWrapper
from AracNet import AracNet, evaluate_model, train_body, learning_from_legs_failure
from metrics import set_seed


datasets_configs = {
    "cifar10c": lambda bias_amount: {
        "dataset_constructor":  CIFAR10C,
        "dataset_kwargs":       {"bias_amount": bias_amount},
        "erm_sampler_replacement": False,
        "image_size":           32,
        "batch_size":           256,
        "accumulate":           1,
        "model_base_name":      "resnet20",
        "base_model":           "resnet20",
    },
    "waterbirds": lambda _: {
        "dataset_constructor":  Waterbirds,
        "dataset_kwargs":       {},
        "erm_sampler_replacement": True,
        "image_size":           224,
        "batch_size":           64,
        "accumulate":           1,
        "model_base_name":      "resnet50",
        "base_model":           "resnet50",
    },
    "bar": lambda bias_amount: {
        "dataset_constructor":  BAR,
        "dataset_kwargs":       {"bias_amount": bias_amount},
        "erm_sampler_replacement": True,
        "image_size":           224,
        "batch_size":           64,
        "accumulate":           1,
        "model_base_name":      "resnet18",
        "base_model":           "resnet18",
    },
    "BFFHQ": lambda bias_amount: {
        "dataset_constructor":  BFFHQ,
        "dataset_kwargs":       {"bias_amount": bias_amount},
        "erm_sampler_replacement": True,
        "image_size":           224,
        "batch_size":           64,
        "accumulate":           1,
        "model_base_name":      "resnet18",
        "base_model":           "resnet18",
    },
    "UrbanCars": lambda bias_amount: {
        "dataset_constructor":  UrbanCars,
        "dataset_kwargs":       {"bias_amount": bias_amount},
        "erm_sampler_replacement": True,
        "image_size":           224,
        "batch_size":           64,
        "accumulate":           1,
        "model_base_name":      "resnet50",
        "base_model":           "resnet50",
    },
}


parser = ArgumentParser()
parser.add_argument(
    "--dataset",
    type=str,
    choices=["bar", "BFFHQ", "waterbirds", "cifar10c", "UrbanCars"],
    required=True,
)
parser.add_argument("--epochs_base_model", type=int, default=100)
parser.add_argument("--epochs_debiasing",  type=int, default=50)
parser.add_argument("--train_body_aracnet", action="store_true")
parser.add_argument("--seed",  type=int,   default=0)
parser.add_argument("--rho",   type=float, default=95)


if __name__ == "__main__":
    args = parser.parse_args()

    os.makedirs("./data", exist_ok=True)

    dataset             = args.dataset
    epochs_base_model   = args.epochs_base_model
    epochs_debiasing    = args.epochs_debiasing
    train_body_aracnet  = args.train_body_aracnet
    seed                = args.seed
    bias_amount         = args.rho

    print(f"dataset={dataset}  bias_amount={bias_amount}  seed={seed}")

    config = datasets_configs[dataset](bias_amount)
    dataset_constructor     = config["dataset_constructor"]
    dataset_kwargs          = config["dataset_kwargs"]

    # --- Build dataset splits ---
    if dataset == "bar":
        train_set = dataset_constructor(env="train", **dataset_kwargs)
        val_set   = dataset_constructor(env="val",   **dataset_kwargs)
        test_set  = dataset_constructor(env="test",  **dataset_kwargs)
        num_classes = 6

        a = BAR(root="./data", bias_amount=bias_amount, env="train", transform=None, return_index=True)
        class_sample_count = a.perclass_populations()
        weight = 1.0 / class_sample_count
        targets = np.array([train_set[i][1][0] for i in range(len(train_set))])
        samples_weight = torch.from_numpy(np.array([weight[t] for t in targets]))

    elif dataset == "waterbirds":
        train_set = dataset_constructor(env="train", **dataset_kwargs)
        val_set   = dataset_constructor(env="val",   **dataset_kwargs)
        test_set  = dataset_constructor(env="test",  **dataset_kwargs)
        num_classes = 2

        class_pops, class_labels_list = train_set.perclass_populations(return_labels=True)
        class_weights  = torch.as_tensor([1.0 / class_pops[y] for y in class_labels_list])
        targets        = torch.asarray([train_set.samples[i]["class_label"] for i in range(len(train_set))])
        samples_weight = torch.asarray([class_weights[t] for t in targets])

    elif dataset == "BFFHQ":
        train_set = dataset_constructor(env="train", **dataset_kwargs)
        val_set   = dataset_constructor(env="val",   **dataset_kwargs)
        test_set  = dataset_constructor(env="test",  **dataset_kwargs)
        num_classes = 2

    elif dataset == "UrbanCars":
        train_set   = UrbanCars(env="train", group_label="both")
        val_set     = UrbanCars(env="val",   group_label="both")
        test_set    = UrbanCars(env="test",  group_label="both")
        num_classes = 2

    elif dataset == "cifar10c":
        num_classes = 10
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(0.5),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        eval_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        train_set = CIFAR10C(root="./data", split="train", bias_amount=bias_amount, transform=train_transform)
        val_set   = CIFAR10C(root="./data", split="val",   bias_amount=bias_amount, transform=eval_transform)
        test_set  = CIFAR10C(root="./data", split="test",  bias_amount=bias_amount, transform=eval_transform)

    print(f"\nDataset sizes — train: {len(train_set)}  val: {len(val_set)}  test: {len(test_set)}")

    batch_size = config["batch_size"]
    accumulate = config["accumulate"]

    config.update({
        "seed":          seed,
        "pretrained":    True,
        "train_samples": len(train_set),
        "val_samples":   len(val_set),
        "test_samples":  len(test_set),
    })

    set_seed(seed)

    _g_val  = torch.Generator().manual_seed(seed)
    _g_test = torch.Generator().manual_seed(seed)
    val_loader  = DataLoader(val_set,  batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=4, generator=_g_val,  worker_init_fn=_seed_worker)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=4, generator=_g_test, worker_init_fn=_seed_worker)

    # --- Build training DataLoader (sampler varies by dataset) ---
    if dataset == "waterbirds":
        _g_sampler = torch.Generator().manual_seed(seed)
        classweighted_sampler = WeightedRandomSampler(weights=samples_weight, num_samples=len(train_set), replacement=True, generator=_g_sampler)
        _g_train = torch.Generator().manual_seed(seed)
        train_loader = DataLoader(train_set, batch_size=batch_size, sampler=classweighted_sampler, pin_memory=True, num_workers=4, generator=_g_train, worker_init_fn=_seed_worker)

    elif dataset == "bar":
        _g_sampler = torch.Generator().manual_seed(seed)
        bar_sampler  = WeightedRandomSampler(weights=samples_weight, num_samples=len(samples_weight), replacement=True, generator=_g_sampler)
        _g_train = torch.Generator().manual_seed(seed)
        train_loader = DataLoader(train_set, batch_size=batch_size, sampler=bar_sampler, pin_memory=True, num_workers=4, generator=_g_train, worker_init_fn=_seed_worker)

    elif dataset == "cifar10c":
        _g_train = torch.Generator().manual_seed(seed)
        _g_val2  = torch.Generator().manual_seed(seed)
        _g_test2 = torch.Generator().manual_seed(seed)
        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,  drop_last=False, pin_memory=True, num_workers=4, generator=_g_train, worker_init_fn=_seed_worker)
        val_loader   = DataLoader(val_set,   batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=4, generator=_g_val2,  worker_init_fn=_seed_worker)
        test_loader  = DataLoader(test_set,  batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=4, generator=_g_test2, worker_init_fn=_seed_worker)

    elif dataset == "UrbanCars":
        _g_train = torch.Generator().manual_seed(seed)
        _g_val2  = torch.Generator().manual_seed(seed)
        _g_test2 = torch.Generator().manual_seed(seed)
        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,  num_workers=4, generator=_g_train, worker_init_fn=_seed_worker)
        val_loader   = DataLoader(val_set,   batch_size=batch_size, num_workers=4, generator=_g_val2,  worker_init_fn=_seed_worker)
        test_loader  = DataLoader(test_set,  batch_size=batch_size, num_workers=4, generator=_g_test2, worker_init_fn=_seed_worker)

    else:
        _g_train = torch.Generator().manual_seed(seed)
        _g_val2  = torch.Generator().manual_seed(seed)
        _g_test2 = torch.Generator().manual_seed(seed)
        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,  pin_memory=True, num_workers=4, generator=_g_train, worker_init_fn=_seed_worker)
        val_loader   = DataLoader(val_set,   batch_size=batch_size, shuffle=False, num_workers=4, generator=_g_val2,  worker_init_fn=_seed_worker)
        test_loader  = DataLoader(test_set,  batch_size=batch_size, shuffle=False, num_workers=4, generator=_g_test2, worker_init_fn=_seed_worker)

    wb = WandbWrapper(project_name="AracNet", config=config)
    for key, val in config.items():
        print(f"  <{key}>: {val}")

    model = AracNet(
        num_classes=num_classes,
        aracne=False,
        base_model_name=config["base_model"],
        pretrained=config["pretrained"],
    )

    save_results_to = f"./hist/SIMPLE/{dataset}/{str(bias_amount).replace('.', '')}/{config['model_base_name']}"
    os.makedirs(save_results_to, exist_ok=True)

    # --- Phase 1: train the backbone ---
    if train_body_aracnet:
        try:
            model.freeze_body(False)
            model.freeze_legs(True)
            model.set_aracne(False)

            base_model_name = train_body(
                model,
                train_loader,
                model.device,
                optimizer=torch.optim.SGD(model.parameters(), lr=0.001, weight_decay=0),
                base_model=config["base_model"],
                num_classes=num_classes,
                epochs=epochs_base_model,
                val_loader=test_loader,
                dataset=dataset,
                bias_amount=bias_amount,
                wb=wb,
                save_dir=save_results_to,
            )
        finally:
            torch.save(model.state_dict(), os.path.join(save_results_to, f"{base_model_name}_{dataset}_{bias_amount}-biased-final.pth"))
            if dataset != "UrbanCars":
                evaluate_model(model, train_loader, num_classes, num_bias_attributes=1, wb=wb, make_figures=True, eval_name="train", dataset=dataset, save_dir=save_results_to)
                evaluate_model(model, test_loader,  num_classes, num_bias_attributes=1, wb=wb, make_figures=True, eval_name="test",  dataset=dataset, save_dir=save_results_to)

        if dataset != "UrbanCars":
            evaluate_model(model, train_loader, num_classes, num_bias_attributes=1, wb=wb, make_figures=True, eval_name="train", dataset=dataset, save_dir=save_results_to)
            evaluate_model(model, test_loader,  num_classes, num_bias_attributes=1, wb=wb, make_figures=True, eval_name="test",  dataset=dataset, save_dir=save_results_to)

    # --- Load biased backbone checkpoint ---
    try:
        model.load_state_dict(torch.load(os.path.join(
            save_results_to,
            f"aracnet-{model.base_model_name}-{model.aracne}_{dataset}_{bias_amount}-biased-final.pth",
        )))
    except FileNotFoundError:
        save_results_to = f"./hist/SIMPLE/{dataset}/{str(bias_amount).replace('.', '')}/aracnet"
        model.load_state_dict(torch.load(os.path.join(
            save_results_to,
            f"aracnet-{model.base_model_name}-{model.aracne}_{dataset}_{bias_amount}-biased-final.pth",
        )))

    model.freeze_body(True)
    model.freeze_legs(False)
    model.set_aracne(True)

    # --- Phase 2: learn to debias using parallel head failures ---
    debiasing_model = learning_from_legs_failure(
        model,
        train_loader,
        test_loader,
        model.device,
        optimizer=lambda params: torch.optim.SGD(params, lr=0.05, weight_decay=0.0),
        num_classes=num_classes,
        epochs=epochs_debiasing,
        wb=wb,
        dataset=dataset,
        train_set=train_set,
        save_dir=save_results_to,
    )

    torch.save(debiasing_model.state_dict(),
               os.path.join(save_results_to, f"debiased_{dataset}_{bias_amount}.pth"))

    if wb is not None:
        wb.finish()
