import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from discharge_model import Predictor
from preprocessing import *
from utils import *


def full_dataset_evaluation(model_path='checkpoint/best_41p82.pth'):
    trn_input, trn_target = np.load('dataset/trn_features.npy'), np.load('dataset/trn_targets.npy')
    val_input, val_target = np.load('dataset/val_features.npy'), np.load('dataset/val_targets.npy')
    trn_input, val_input = trn_input/1.1, val_input/1.1 # nominal capacity

    model = torch.load(model_path).cuda()
    model.eval()
    scaler_y = get_scaler()
    trn_rmse, trn_mape, val_rmse, val_mape = [], [], [], []
    for n_cycle in range(50):
        trn_input_copy, val_input_copy = trn_input.copy(), val_input.copy()
        for i in range(len(trn_input)):
            trn_input_copy[i, :, n_cycle:] = trn_input_copy[i, :, n_cycle].reshape(-1, 1).repeat(50-n_cycle, axis=1)
        for i in range(len(val_input)):
            val_input_copy[i, :, n_cycle:] = val_input_copy[i, :, n_cycle].reshape(-1, 1).repeat(50-n_cycle, axis=1)
        with torch.no_grad():
            trn_pred = scaler_y.inverse_transform(model(torch.tensor(trn_input_copy).cuda().float()).detach().cpu().numpy())
            val_pred = scaler_y.inverse_transform(model(torch.tensor(val_input_copy).cuda().float()).detach().cpu().numpy())
        trn_rmse.append(np.sqrt(np.sum(np.square(trn_pred[:, 0]-trn_target))/len(trn_pred)))
        val_rmse.append(np.sqrt(np.sum(np.square(val_pred[:, 0]-val_target))/len(val_pred)))
        # trn_mape.append(np.sum(np.abs(trn_pred-trn_target)/trn_target)/len(trn_pred))
        # val_mape.append(np.sum(np.abs(val_pred-val_target)/val_target)/len(val_pred))
        if n_cycle in [0, 9, 49]:
            plt.scatter(trn_target, trn_pred, c='blue', s=25, label='training')
            plt.plot([0, 600], [0, 600], ls='--', c='black')
            plt.scatter(val_target, val_pred, c='red', s=25, label='validation', marker='^')
            plt.xlabel('ground truth', fontsize=14)
            plt.ylabel('prediction', fontsize=14)
            plt.legend()
            plt.title('Qd decrease ratio evaluation result', fontsize=16)
            plt.show()
            plt.close()

    print(trn_rmse[0], val_rmse[0])
    print(trn_rmse[9], val_rmse[9])
    print(trn_rmse[49], val_rmse[49])
    plt.plot(np.arange(50), trn_rmse, c='blue', label='training MAPE')
    plt.plot(np.arange(50), val_rmse, c='red', label='testing MAPE')
    plt.legend()
    plt.xlabel('input length', fontsize=14)
    plt.ylabel('MAPE(%)', fontsize=14)
    plt.show()
    plt.close()


def testing_output(model_path='checkpoint/seed1.pth'):
    features = np.load('dataset/test_features.npy')
    testing_set = []
    for cell_id in range(len(features)):
        curve = []
        for cycle in range(50):
            v = features[cell_id, cycle, :, 0]
            v = np.flip(np.sort(v))
            qd = features[cell_id, cycle, :, 1]
            qd = qd[np.flip(np.argsort(v))]
            interp_v, interp_qd = np.linspace(3.55, 2.05, 64), []
            v_pointer = 0
            for i in range(len(qd)):
                if v[i]<=interp_v[v_pointer]:
                    interp_qd.append(qd[i])
                    v_pointer += 1
                    if v_pointer == 64:
                        break
            curve.append(np.expand_dims(np.array(interp_qd), axis=1))
        testing_set.append(np.concatenate(curve, axis=1)) # (qv_ch, n_cycles=50)
        # print(np.concatenate(curve, axis=1).shape)
    testing_set = np.stack(testing_set, axis=0)/1.1

    model = torch.load(model_path).cuda()
    model.eval()
    scaler = get_scaler()
    test_pred = scaler.inverse_transform(model(torch.tensor(testing_set).cuda()).detach().cpu().numpy()).reshape(-1,)
    df = pd.DataFrame({'Cell ID': np.linspace(108, 126, 19).astype(np.int32),
                       'Remaining Useful Life': test_pred})
    df.to_csv('test_result.csv', index=False)



# full_dataset_evaluation()
testing_output()