import math
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from preprocessing import get_scaler, Battery_Dataset


def loss_profile(trn_loss, val_loss):
    """
    plot loss v.s. epoch curve
    """
    plt.plot(np.arange(len(trn_loss)), trn_loss, c='blue', label='trn_loss', ls='--')
    plt.plot(np.arange(len(val_loss)), val_loss, c='red', label='val_loss', ls='--')
    plt.xlabel('epoch', fontsize=14)
    plt.ylabel('loss', fontsize=14)
    plt.legend()
    plt.savefig('loss_profile.png')
    plt.close()


def adjust_learning_rate(optimizer, full_ep, epoch, warmup_ep, base_lr, min_lr):
    """Decay the learning rate with half-cycle cosine after warmup"""
    if epoch < warmup_ep:
        lr = base_lr * epoch / warmup_ep
    else:
        lr = min_lr + (base_lr - min_lr) * 0.5 * \
            (1. + math.cos(math.pi * (epoch - warmup_ep) / (full_ep - warmup_ep)))
    for param_group in optimizer.param_groups:
        if "lr_scale" in param_group:
            param_group["lr"] = lr * param_group["lr_scale"]
        else:
            param_group["lr"] = lr
    return lr


def model_evaluation(model, eval_length):
    scaler_rul = get_scaler()

    model.eval()
    trn_rmse, val_rmse = [], []
    for cycles in eval_length:
        trn_set = Battery_Dataset(train=True, last_padding=False, fix_length=cycles)
        val_set = Battery_Dataset(train=False, last_padding=False, fix_length=cycles)
        trn_loader = DataLoader(trn_set, batch_size=86, num_workers=0, drop_last=False, shuffle=False)
        val_loader = DataLoader(val_set, batch_size=22, num_workers=0, drop_last=False, shuffle=False)
        with torch.no_grad():
            for inputs, targets in trn_loader:
                outputs = model(inputs.cuda().float()).reshape(-1, 1)
                pred = outputs.detach().cpu().numpy()
                gt = targets.reshape(-1, 1)
                gt, pred = scaler_rul.inverse_transform(gt), scaler_rul.inverse_transform(pred)
                n = len(gt)
                rmse = np.sqrt(np.sum(np.square(gt[:, 0]-pred[:, 0]))/n)
                # mape = np.sum(np.abs(pred-gt)/gt)/n
                trn_rmse.append(rmse)

            for inputs, targets in val_loader:
                outputs = model(inputs.cuda().float()).reshape(-1, 1)
                pred = outputs.detach().cpu().numpy()
                gt = targets.reshape(-1, 1)
                gt, pred = scaler_rul.inverse_transform(gt), scaler_rul.inverse_transform(pred)
                n = len(gt)
                rmse = np.sqrt(np.sum(np.square(gt[:, 0]-pred[:, 0]))/n)
                val_rmse.append(rmse)

    return trn_rmse, val_rmse