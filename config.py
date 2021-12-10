import numpy as np
import torch
import os.path as pa

class GlobalConfig:
    root_dir = '/home/qlc/dataset/BraTs2020'
    train_root_dir = '/home/qlc/dataset/BraTs2020/MICCAI_BraTS2020_TrainingData'
    test_root_dir = '/home/qlc/dataset/BraTs2020/MICCAI_BraTS2020_ValidationData'
    path_to_csv = '/home/qlc/Model/BraTs/log/train_data.csv'
    pretrained_model_path = '/home/qlc/Model/BraTs/log/pretrained.pth'
    train_logs_path = '/home/qlc/Model/BraTs/log/train_log.csv'
    ae_pretrained_model_path = '/home/qlc/Model/BraTs/log/best_model.pth'
    tab_data = '/home/qlc/Model/BraTs/log/df_with_voxel_stats_and_latent_features.csv'
    seed = 55


'''
class GlobalConfig:
    root_dir = 'D:/BraTS/Data'
    train_root_dir = 'D:/BraTS/Data/MICCAI_BraTS2020_TrainingData'
    test_root_dir = 'D:/BraTS/Data/MICCAI_BraTS2020_ValidationData'
    path_to_csv = 'log/train_data.csv'
    pretrained_model_path = 'log/last_epoch_model.pth'
    train_logs_path = 'log/train_log.csv'
    ae_pretrained_model_path = 'log/best_model.pth'
    tab_data = 'log/df_with_voxel_stats_and_latent_features.csv'
    seed = 55
'''

def seed_everything(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


config = GlobalConfig()
seed_everything(config.seed)




















