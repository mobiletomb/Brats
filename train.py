import trainer
import SSL.modellib as modelib
import config
import metircs_losses
import pandas as pd
from data import BratsDataset
import argparse

# parser = argparse.ArgumentParser(description='Brats')
# parser.add_argument('-m', '--Model', default='UNet3d', type=str,
#                     help='model name')
# parser.add_argument('--PairModel', default=False, type=bool,
#                     help='whether use paired data structure')
# parser.add_argument('--log_path', default=config.config.train_logs_path, type=str,
#                     help='')
# parser.add_argument('--model_path')
# parser.add_argument('--best_model_path')
#
#
# args = parser.parse_args()
#
# model_name = args.Model
# if model_name == 'UNet3d':
#     model = modelib.UNet3d(in_channels=4, n_classes=3, n_channels=24).to('cuda')
# elif model_name == 'Double_path_Unet3D':
#     model = modelib.Double_Path_UNet3D(in_channels=4, n_classes=3, n_channels=24).to('cuda')
#     args.PairModel = True
model = modelib.UNet3d(in_channels=4, n_classes=3, n_channels=24).to('cuda')

trainer = trainer.Trainer(net=model,
                          dataset=BratsDataset,
                          criterion=metircs_losses.BCEDiceLoss(),
                          lr=5e-4,
                          accumulation_steps=4,
                          batch_size=1,
                          fold=0,
                          num_epochs=50,
                          path_to_csv=config.config.path_to_csv,
                          pair_model=False)


if __name__ == '__main__':
    trainer.run()

    if config.config.pretrained_model_path is not None:
        # trainer.load_predtrain_model(config.config.pretrained_model_path)

        # if need - load the logs.
        train_logs = pd.read_csv(config.config.train_logs_path)
        trainer.losses["train"] = train_logs.loc[:, "train_loss"].to_list()
        trainer.losses["val"] = train_logs.loc[:, "val_loss"].to_list()
        trainer.dice_scores["train"] = train_logs.loc[:, "train_dice"].to_list()
        trainer.dice_scores["val"] = train_logs.loc[:, "val_dice"].to_list()
        trainer.jaccard_scores["train"] = train_logs.loc[:, "train_jaccard"].to_list()

