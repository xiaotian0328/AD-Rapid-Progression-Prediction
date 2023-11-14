import os
import math
import random
import pickle

import numpy as np
import pandas as pd

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

from dataset import set_binary_labels, BaselineDataset
from model import BaselineModel
from trainer import train_baseline, evaluate_baseline
from utils import (
    read_yaml, 
    save_baseline_plots, 
    save_test_results
)

# read configurations
config, gpu_ids, _ = read_yaml()

# set up configurations
os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_ids[0])

torch.manual_seed(config["seed"])
torch.cuda.manual_seed(config["seed"])
random.seed(config["seed"])
np.random.seed(config["seed"])
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

if config["data_split"] == "internal":
    flag = "in"
elif config["data_split"] == "external":
    flag = "ex"

data_dir = f"{config['root_dir']}/{config['data_dir']}/{config['data_version']}"
if "data_imputation" in config:
    data_dir = f"{data_dir}/{config['data_imputation']}"

# load data
with open(f"{data_dir}/{config['data_split']}_cv_folds_data.pkl", "rb") as f:
    data_dict = pickle.load(f)

# get binary labels
label_dict = {}
for fold in data_dict:
    label_fold = data_dict[fold]
    label_dict[fold] = {}
    for split in label_fold:
        label_data = data_dict[fold][split]["labels"]
        label_df = pd.DataFrame(
            label_data,
            columns=['ADAS_ADASTS14', 'ADL_ADLOVRALS', 'CDR_CDRTS', 'MMSE_MMSETS']
        )
        binary_label_dict = set_binary_labels(label_df)
        label_dict[fold][split] = {
            exp_name: binary_label_dict[exp_name]
            for exp_name in config["exp_names"]
        }

model_save_dir = f"{config['root_dir']}/results/models/{config['data_version']}/{config['data_mode']}/{config['data_split']}/{config['exp_branch']}"
if "data_imputation" in config:
    model_save_dir = f"{model_save_dir}/{config['data_imputation']}"

if not os.path.exists(model_save_dir):
    os.makedirs(model_save_dir)

# train
for fold in data_dict:
    print()
    print('*****************')
    print(f"FOLD_{fold}")
    print('*****************')
    print()

    data_fold = data_dict[fold]
    label_fold = label_dict[fold]
    model_save_dir_fold = f"{model_save_dir}/fold_{fold}"
    if not os.path.exists(model_save_dir_fold):
        os.makedirs(model_save_dir_fold)

    for exp_name in config["exp_names"]:    
        print('**********')
        print(exp_name)
        print('**********')

        train_loss = np.zeros(config["num_epochs"])
        train_auroc = np.zeros(config["num_epochs"])
        train_auprc = np.zeros(config["num_epochs"])
        val_loss = np.zeros(config["num_epochs"])
        val_auroc = np.zeros(config["num_epochs"])
        val_auprc = np.zeros(config["num_epochs"])

        # build datasets
        train_set = BaselineDataset(data_fold["train"]["static"], label_fold["train"][exp_name])
        val_set = BaselineDataset(data_fold["val"]["static"], label_fold["val"][exp_name])
        test_set = BaselineDataset(data_fold["test"]["static"], label_fold["test"][exp_name])
        print(f'Train: {len(train_set)}  Val: {len(val_set)}  Test: {len(test_set)}')

        # build data loaders
        train_loader = DataLoader(
            train_set, batch_size=config["batch_size"], 
            drop_last=True, shuffle=True
        )
        val_loader = DataLoader(
            val_set, batch_size=config["batch_size"], shuffle=False
        )
        test_loader = DataLoader(
            test_set, batch_size=1, shuffle=False
        )

        # model, optimizer, loss
        model = BaselineModel(
            config["hidden_size"], 
            config["num_layers"], 
            config["num_classes"], 
            config["dropout"]
        ).cuda()
        optimizer = torch.optim.Adam(
            model.parameters(), 
            lr=config["learning_rate"]
        )
        criterion = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([config["pos_weight"]])
        ).cuda()

        # training epochs
        best_metric = float("inf") if config["best_metric"] == "loss" else 0.
        for epoch in range(config["num_epochs"]):
            if (epoch + 1) % 50 == 0:
                print('**********')
                print(f'Epoch {epoch}')
                print('**********')
                print('Train:')

            # train step
            model.train()
            loss, auroc, auprc = train_baseline(
                model, train_loader, criterion, optimizer
            )
            if (epoch + 1) % 50 == 0:
                print(f'Loss: {loss:.4f} | AUROC: {auroc:.4f} | AUPRC: {auprc:.4f}')

            train_loss[epoch] = loss
            train_auroc[epoch] = auroc
            train_auprc[epoch] = auprc

            if (epoch + 1) % 50 == 0:
                print('**********')
                print('Validation:')

            # validation step
            model.eval()
            loss, auroc, auprc, _, _ = evaluate_baseline(
                model, val_loader, criterion
            )

            if (epoch + 1) % 50 == 0:
                print(f'Loss: {loss:.4f} | AUROC: {auroc:.4f} | AUPRC: {auprc:.4f}')

            val_loss[epoch] = loss
            val_auroc[epoch] = auroc
            val_auprc[epoch] = auprc

            # save best model
            if config["best_metric"] == "loss" and loss < best_metric:
                best_metric = loss
                torch.save(
                    model.state_dict(), 
                    f'{model_save_dir_fold}/best_model_{exp_name}.pth'
                )
            if config["best_metric"] == "auroc" and auroc > best_metric:
                best_metric = auroc
                torch.save(
                    model.state_dict(), 
                    f'{model_save_dir_fold}/best_model_{exp_name}.pth'
                )
            if config["best_metric"] == "auprc" and auprc > best_metric:
                best_metric = auprc
                torch.save(
                    model.state_dict(), 
                    f'{model_save_dir_fold}/best_model_{exp_name}.pth'
                )

        # release memory
        del model

        # test after every epoch
        print('**********')
        print('Test:')

        # load the best model
        best_model = BaselineModel(
            config["hidden_size"], 
            config["num_layers"], 
            config["num_classes"], 
            config["dropout"]
        ).cuda()
        path = f'{model_save_dir_fold}/best_model_{exp_name}.pth'
        state_dict = torch.load(path)
        best_model.load_state_dict(state_dict)

        # test on the best model
        best_model.eval()
        test_loss, test_auroc, test_auprc, _, _ = evaluate_baseline(
            best_model, test_loader, criterion
        )

        print(f'Loss: {test_loss:.4f} | AUROC: {test_auroc:.4f} | AUPRC: {test_auprc:.4f}')

        # release memory
        del best_model

        # visualize and save evaluating results
        save_baseline_plots(train_loss, val_loss, 'Loss', config, exp_name, fold=f"fold_{fold}")
        save_baseline_plots(train_auroc, val_auroc, 'AUROC', config, exp_name, fold=f"fold_{fold}")
        save_baseline_plots(train_auprc, val_auprc, 'AUPRC', config, exp_name, fold=f"fold_{fold}")
        save_test_results(test_auroc, test_auprc, config, exp_name, fold=f"fold_{fold}")