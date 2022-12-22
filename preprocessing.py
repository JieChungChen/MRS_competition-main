import json
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler


def load_json_file(filepath=["dataset/timeseries_data.json", "dataset/test_data.json"]):
    training_set = json.load(open(filepath[0],'r'))
    testing_set = json.load(open(filepath[1],'r'))
    for cell, values in training_set.items():
        #Convert timeseries data JSON into a pandas dataframe for ease of use
        values["timeseries data"] = pd.read_json(values["timeseries data"])

    for cell, values in testing_set.items():
        #Convert timeseries data JSON into a pandas dataframe for ease of use
        values["timeseries data"] = pd.read_json(values["timeseries data"])

    features, targets = [], []
    for key in training_set.keys():
        # extract qv curve
        features.append(training_set[key]['timeseries data'].to_numpy()[:, [0, 2]].reshape(50, 1000, 2))
        targets.append(training_set[key]['Remaining Useful Life'])

    features = np.stack(features, axis=0).astype(np.float32)
    targets = np.array(targets).astype(np.float32)
    print(features.shape, targets.shape)
    np.save('dataset/full_features.npy', features)
    np.save('dataset/full_targets.npy', targets)

    features = []
    for key in testing_set.keys():
        # extract qv curve
        features.append(testing_set[key]['timeseries data'].to_numpy()[:, [0, 2]].reshape(50, 1000, 2))

    features = np.stack(features, axis=0).astype(np.float32)
    print(features.shape)
    np.save('dataset/test_features.npy', features)


def feature_preprocessing(qv_ch=64):
    features = np.load('dataset/full_features.npy')
    full_dataset = []
    for cell_id in range(len(features)):
        curve = []
        for cycle in range(50):
            v = features[cell_id, cycle, :, 0]
            v = np.flip(np.sort(v))
            qd = features[cell_id, cycle, :, 1]
            qd = qd[np.flip(np.argsort(v))]
            interp_v, interp_qd = np.linspace(3.55, 2.05, qv_ch), []
            v_pointer = 0
            for i in range(len(qd)):
                if v[i]<=interp_v[v_pointer]:
                    interp_qd.append(qd[i])
                    v_pointer += 1
                    if v_pointer == qv_ch:
                        break
            curve.append(np.expand_dims(np.array(interp_qd), axis=1))
        full_dataset.append(np.concatenate(curve, axis=1)) # (qv_ch, n_cycles=50)
        # print(np.concatenate(curve, axis=1).shape)
    full_dataset = np.stack(full_dataset, axis=0)
    print(full_dataset.shape)
    np.save('dataset/full_features_c'+str(qv_ch), full_dataset)


def trn_val_split(trn_ratio=0.8, seed=1):
    full_features = np.load('dataset/full_features_c64.npy')
    full_targets = np.load('dataset/full_targets.npy')
    high_rul, low_rul = [], []
    for i, rul in enumerate(full_targets):
        if rul<300:
            low_rul.append(i)
        elif rul>=300:
            high_rul.append(i)
    
    # split_point = int(len(full_features)*trn_ratio)
    low_split, high_split = int(len(low_rul)*trn_ratio), int(len(high_rul)*trn_ratio)
    random.seed(seed)
    random.shuffle(low_rul), random.shuffle(high_rul)
    trn_features, val_features = full_features[low_rul[:low_split]+high_rul[:high_split]],\
                                 full_features[low_rul[low_split:]+high_rul[high_split:]]
    trn_targets, val_targets = full_targets[low_rul[:low_split]+high_rul[:high_split]],\
                               full_targets[low_rul[low_split:]+high_rul[high_split:]]
    # trn_features, val_features = full_features[:split_point], full_features[split_point:]
    # trn_targets, val_targets = full_targets[:split_point], full_targets[split_point:]
    print(trn_features.shape, val_features.shape)
    np.save('dataset/trn_features', trn_features)
    np.save('dataset/val_features', val_features)
    np.save('dataset/trn_targets', trn_targets)
    np.save('dataset/val_targets', val_targets)


def qv_curve_visualizer(cell_id, cycle_id):
    features = np.load('dataset/full_features.npy')
    voltage = features[cell_id, cycle_id, :, 0]
    capacity = features[cell_id, cycle_id, :, 1]
    plt.scatter(capacity, voltage)
    plt.show()
    plt.close()
    

def rul_distribution():
    full_targets = np.load('dataset/full_targets.npy')
    print(np.sum(full_targets>300), np.sum(full_targets<=300))
    plt.hist(full_targets)
    plt.show()
    plt.close()


def target_normalizer(arr):
    scaler_rul = StandardScaler()
    trn_target = np.load('dataset/trn_targets.npy')
    scaler_rul.fit(trn_target.reshape(-1, 1))
    arr = scaler_rul.transform(arr.reshape(-1, 1))
    return arr


def get_scaler():
    scaler_rul = StandardScaler()
    trn_target = np.load('dataset/trn_targets.npy')
    scaler_rul.fit(trn_target.reshape(-1, 1))
    return scaler_rul


class Battery_Dataset(Dataset):
    def __init__(self, train=True, last_padding=True, fix_length=-1):
        self.train = train
        self.trn_input, self.trn_target = np.load('dataset/trn_features.npy'), np.load('dataset/trn_targets.npy')
        self.val_input, self.val_target = np.load('dataset/val_features.npy'), np.load('dataset/val_targets.npy')
        self.trn_input, self.val_input = self.trn_input/1.1, self.val_input/1.1 # nominal capacity
        self.trn_target, self.val_target = target_normalizer(self.trn_target), target_normalizer(self.val_target)
        trn_size, val_size = len(self.trn_input), len(self.val_input)

        if last_padding: # full last padding
            aug_trn_input, aug_trn_target = [], []
            for i in range(trn_size):
                for cycle_length in range(50):
                    after_padding = self.trn_input[i].copy()
                    after_padding[:, cycle_length:] = after_padding[:, cycle_length].reshape(-1, 1).repeat(50-cycle_length, axis=1)
                    aug_trn_input.append(after_padding)
                    aug_trn_target.append(self.trn_target[i, :])
            self.trn_input, self.trn_target = np.stack(aug_trn_input, axis=0), np.stack(aug_trn_target, axis=0)
            
        if fix_length>-1:
            for i in range(trn_size):
                self.trn_input[i, :, fix_length:] = self.trn_input[i, :, fix_length].reshape(-1, 1).repeat(50-fix_length, axis=1)
            for i in range(val_size):
                self.val_input[i, :, fix_length:] = self.val_input[i, :, fix_length].reshape(-1, 1).repeat(50-fix_length, axis=1)

    def __getitem__(self, index):
        if self.train:
            feature, target = self.trn_input[index], self.trn_target[index]
            return feature, target.reshape(-1, 1)
        feature, target = self.val_input[index], self.val_target[index]
        return feature, target.reshape(-1, 1)

    def __len__(self):
        if self.train:
            return len(self.trn_input)
        return len(self.val_input)


# trn_val_split()
# rul_distribution()
# feature_preprocessing()
# trn_val_split()
