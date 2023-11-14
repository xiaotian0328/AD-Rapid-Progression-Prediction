import os
import argparse
import yaml

import numpy as np
import matplotlib.pyplot as plt

from trainer import evaluate


def read_yaml():
    # read in yaml
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path for the config file")
    parser.add_argument("--exp_ids", type=int, nargs='+', default=[0], help="Path for the config file")
    parser.add_argument("--gpus", type=int, nargs='+', default=[0], help="Path for the config file")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)
    return config, args.gpus, args.exp_ids


def plot(metrics, title, weeks):
    for i, week in enumerate(weeks):
        plt.plot(metrics[i], label=f'{week} weeks')
    plt.title(title)
    plt.xlabel('Epoch')
    plt.legend()


def plot_baseline(metrics, title):
    plt.plot(metrics, label=f'baseline')
    plt.title(title)
    plt.xlabel('Epoch')
    plt.legend()


def plot_test(test_metric, metric_name, weeks):
    plt.plot(test_metric, 'x-')
    plt.title(f'Test {metric_name}')
    plt.xlabel('Week')
    plt.ylabel(metric_name)
    plt.xticks(np.arange(len(weeks)), weeks)
    plt.grid(True)


def save_plots(train_metric, eval_metric, metric_name, weeks, config, exp_name, fold=None):
    fig = plt.figure(figsize=(12, 4))
    plt.subplot(121)
    plot(train_metric, f'Train {metric_name}', weeks)
    plt.subplot(122)
    plot(eval_metric, f'Evaluation {metric_name}', weeks)

    save_dir = f"{config['root_dir']}/results/plots/{config['data_version']}/{config['data_mode']}/{config['data_split']}/{config['exp_branch']}/plots_{exp_name}"
    if "data_imputation" in config:
        save_dir = f"{save_dir}/{config['data_imputation']}"
    if fold is not None:
        save_dir = f"{save_dir}/{fold}"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    plt.savefig(f'{save_dir}/{metric_name}.svg', format='svg', bbox_inches="tight")
    plt.close(fig)


def save_baseline_plots(train_metric, eval_metric, metric_name, config, exp_name, fold=None):
    fig = plt.figure(figsize=(12, 4))
    plt.subplot(121)
    plot_baseline(train_metric, f'Train {metric_name}')
    plt.subplot(122)
    plot_baseline(eval_metric, f'Evaluation {metric_name}')

    save_dir = f"{config['root_dir']}/results/plots/{config['data_version']}/{config['data_mode']}/{config['data_split']}/{config['exp_branch']}/plots_{exp_name}"
    if "data_imputation" in config:
        save_dir = f"{save_dir}/{config['data_imputation']}"
    if fold is not None:
        save_dir = f"{save_dir}/{fold}"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    plt.savefig(f'{save_dir}/{metric_name}.svg', format='svg', bbox_inches="tight")
    plt.close(fig)


def save_test_plots(test_auroc, test_auprc, weeks, config, exp_name, fold=None):
    fig = plt.figure(figsize=(12, 4))
    plt.subplot(121)
    plot_test(test_auroc, 'AUROC', weeks)
    plt.subplot(122)
    plot_test(test_auprc, 'AUPRC', weeks)

    save_dir = f"{config['root_dir']}/results/plots/{config['data_version']}/{config['data_mode']}/{config['data_split']}/{config['exp_branch']}/plots_{exp_name}"
    if "data_imputation" in config:
        save_dir = f"{save_dir}/{config['data_imputation']}"
    if fold is not None:
        save_dir = f"{save_dir}/{fold}"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    plt.savefig(f'{save_dir}/Test_AUROC_AUPRC.svg', format='svg', bbox_inches="tight")
    plt.close(fig)


def save_test_results(test_auroc, test_auprc, config, exp_name, fold=None):
    save_dir = f"{config['root_dir']}/results/test_results/{config['data_version']}/{config['data_mode']}/{config['data_split']}/{config['exp_branch']}"
    if "data_imputation" in config:
        save_dir = f"{save_dir}/{config['data_imputation']}"
    if fold is not None:
        save_dir = f"{save_dir}/{fold}"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    np.save(f"{save_dir}/test_results_{exp_name}", np.array([test_auroc, test_auprc]))


def save_classification(
    label_name, 
    model, 
    data_loader, 
    criterion, 
    threshold
):
    classification = {'tp': [], 'tn': [], 'fp': [], 'fn': []}
    _, _, _, y_list, logit_list = evaluate(model, data_loader, criterion, return_logit=True)
    for idx, (y, logit) in enumerate(zip(y_list, logit_list)):
        if y == 1 and logit > threshold:
            classification['tp'].append(idx)
        elif y == 0 and logit <= threshold:
            classification['tn'].append(idx)
        elif y == 0 and logit > threshold:
            classification['fp'].append(idx)
        elif y == 1 and logit < threshold:
            classification['fn'].append(idx)
    return classification