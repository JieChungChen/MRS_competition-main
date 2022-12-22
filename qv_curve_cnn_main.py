import argparse
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import torch
from torch import nn, optim
from torchsummary import summary
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from preprocessing import *
from discharge_model import Predictor
from utils import loss_profile, model_evaluation, adjust_learning_rate
torch.manual_seed(2022)


def get_args_parser():
    parser = argparse.ArgumentParser('Discharge model(QV-curve) training', add_help=False)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--epochs', default=1000, type=int)
    parser.add_argument('--split_seed', default=0, type=int)
    parser.add_argument('--gpu_id', default=0, type=int)
    parser.add_argument('--detail_epoch', default=1, type=int)

    # Model parameters
    parser.add_argument('--model', default='Predictor_1', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--checkpoint', default='checkpoint/pretrain_predictor.pth', type=str)
    parser.add_argument('--drop', default=0.20, type=float)
    parser.add_argument('--qv_ch', default=64, type=int)
    parser.add_argument('--savepath', default='checkpoint/best_predictor.pth', type=str)
    parser.add_argument('--finetune', default=True, type=bool)
    parser.add_argument('--pretrained_model', default='checkpoint/best_38p55.pth', type=str)

    # Optimizer parameters
    parser.add_argument('--weight_decay', type=float, default=1e-2)
    parser.add_argument('--lr', type=float, default=1e-5, metavar='LR')
    parser.add_argument('--lr_schedule', type=bool, default=False, metavar='LR')
    parser.add_argument('--min_lr', type=float, default=1e-5, metavar='LR')
    parser.add_argument('--warmup_epochs', type=int, default=10)

    return parser


def main(args):
    if torch.cuda.is_available():
        print(" -- GPU is available -- ")

    trn_set = Battery_Dataset(train=True, last_padding=False)
    val_set = Battery_Dataset(train=False, last_padding=False)
    trn_loader = DataLoader(trn_set, batch_size=86, num_workers=1, drop_last=False, shuffle=False)
    val_loader = DataLoader(val_set, batch_size=22, num_workers=1, drop_last=False, shuffle=False)

    if args.finetune: # load pretrained weight
        model = torch.load(args.pretrained_model)
    else:
        model = Predictor(args.qv_ch, 1, drop=args.drop)
    model.cuda()
    summary(model, (args.qv_ch, 50))

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, amsgrad=True, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=10, eta_min=0)
    criterion = nn.HuberLoss(delta=1)

    best_rmse = 50
    trn_loss_record, val_loss_record = [], []
    for epoch in range(args.epochs):
        epoch += 1
        trn_random_set = Battery_Dataset(train=True, last_padding=True)
        trn_random_loader = DataLoader(trn_random_set, batch_size=args.batch_size, num_workers=1, drop_last=False, shuffle=True)
        model.train()
        batch = 0
        n_minibatch = (len(trn_random_set)//args.batch_size)
        for inputs, targets in trn_random_loader:
            batch += 1
            optimizer.zero_grad()
            output = model(inputs.cuda().float())
            loss = criterion(output , targets.reshape(-1, 1).cuda().float())
            loss.backward()
            optimizer.step()
            if batch%30==0:
                print('epoch:[%d / %d] batch:[%d / %d] loss= %.3f' % 
                    (epoch, args.epochs, batch, n_minibatch, loss.mean()))

        if args.lr_schedule:
            scheduler.step()

        # model evaluation per epoch
        model.eval()
        with torch.no_grad():
            trn_loss, val_loss = 0, 0
            for inputs, targets in trn_loader:
                output = model(inputs.cuda().float())
                loss = criterion(output , targets.reshape(-1, 1).cuda().float())
                trn_loss += loss.mean()
            for inputs, targets in val_loader:
                output = model(inputs.cuda().float())
                loss = criterion(output , targets.reshape(-1, 1).cuda().float())
                val_loss += loss.mean()
            trn_loss_record.append(trn_loss.cpu())
            val_loss_record.append(val_loss.cpu())
            if epoch%args.detail_epoch==0:
                for g in optimizer.param_groups:
                    current_lr = g['lr']
                print('100 cycles trn_loss: %.3f, val_loss: %.3f, lr=%e' % (trn_loss, val_loss, current_lr))

        
        trn_rmse, test_rmse = model_evaluation(model, eval_length=[0, 9, 49])
        if epoch%args.detail_epoch==0:
            print('training set RMSE 1 cycle: %.3f, 10 cycle: %.3f, 100 cycle: %.3f' %
                (trn_rmse[0], trn_rmse[1], trn_rmse[2]))
            print('testing set RMSE 1 cycle: %.3f, 10 cycle: %.3f, 100 cycle: %.3f' %
                (test_rmse[0], test_rmse[1], test_rmse[2]))

        if test_rmse[2]<best_rmse:
            best_rmse = test_rmse[2]
            torch.save(model, 'checkpoint/checkpoint'+str(test_rmse[2])+'.pth')
        
    # training finished 
    loss_profile(trn_loss_record, val_loss_record)


if __name__=='__main__':
    args = get_args_parser()
    args = args.parse_args()
    main(args) 