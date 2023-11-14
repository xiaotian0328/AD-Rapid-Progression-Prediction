import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score

import torch

def train(
    model, 
    train_loader, 
    criterion, 
    optimizer, 
):
    total_loss = 0
    total_sample = 0
    y_list = []
    prob_list = []
    
    for static, dynamic, y in train_loader:
        static = static.cuda()
        dynamic = dynamic.cuda()
        y = y.cuda()
    
        logit = model(static, dynamic)
        loss = criterion(logit, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        with torch.no_grad():
            total_loss += loss.cpu().item() * y.size(0)
            total_sample += y.size(0)
            y_list.extend(y.cpu().tolist())
            prob_list.extend(logit.cpu().sigmoid().tolist())
            
    loss = total_loss / total_sample
    auroc = roc_auc_score(y_list, prob_list)
    auprc = average_precision_score(y_list, prob_list)
    return loss, auroc, auprc

def evaluate(
    model, 
    data_loader, 
    criterion, 
    return_logit=False
):
    total_loss = 0
    total_sample = 0
    y_list = []
    prob_list = []
    logit_list = []
    
    with torch.no_grad():
        for static, dynamic, y in data_loader:
            static = static.cuda()
            dynamic = dynamic.cuda()
            y = y.cuda()
    
            logit = model(static, dynamic)
            loss = criterion(logit, y)
            
            total_loss += loss.cpu().item() * y.size(0)
            total_sample += y.size(0)
            y_list.extend(y.cpu().tolist())
            prob_list.extend(logit.cpu().sigmoid().tolist())
            logit_list.extend(logit.cpu().tolist())
            
    loss = total_loss / total_sample
    auroc = roc_auc_score(y_list, prob_list)
    auprc = average_precision_score(y_list, prob_list)
    
    if return_logit:
        return loss, auroc, auprc, y_list, logit_list
    else:
        return loss, auroc, auprc, y_list, prob_list

def train_baseline(
    model, 
    train_loader, 
    criterion, 
    optimizer, 
):
    total_loss = 0
    total_sample = 0
    y_list = []
    prob_list = []
    
    for x, y in train_loader:
        x = x.cuda()
        y = y.cuda()
    
        logit = model(x)
        loss = criterion(logit, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        with torch.no_grad():
            total_loss += loss.cpu().item() * y.size(0)
            total_sample += y.size(0)
            y_list.extend(y.cpu().tolist())
            prob_list.extend(logit.cpu().sigmoid().tolist())
            
    loss = total_loss / total_sample
    auroc = roc_auc_score(y_list, prob_list)
    auprc = average_precision_score(y_list, prob_list)
    return loss, auroc, auprc

def evaluate_baseline(
    model, 
    data_loader, 
    criterion, 
    return_logit=False
):
    total_loss = 0
    total_sample = 0
    y_list = []
    prob_list = []
    logit_list = []
    
    with torch.no_grad():
        for x, y in data_loader:
            x = x.cuda()
            y = y.cuda()
    
            logit = model(x)
            loss = criterion(logit, y)
            
            total_loss += loss.cpu().item() * y.size(0)
            total_sample += y.size(0)
            y_list.extend(y.cpu().tolist())
            prob_list.extend(logit.cpu().sigmoid().tolist())
            logit_list.extend(logit.cpu().tolist())
            
    loss = total_loss / total_sample
    auroc = roc_auc_score(y_list, prob_list)
    auprc = average_precision_score(y_list, prob_list)
    
    if return_logit:
        return loss, auroc, auprc, y_list, logit_list
    else:
        return loss, auroc, auprc, y_list, prob_list