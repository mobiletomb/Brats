import trainer
import SSL.modellib as modelib
import config
import metircs_losses
import pandas as pd
from data import BratsDataset
import argparse

parser = argparse.ArgumentParser(description='Brats')
parser.add_argument('-m', '--Model', default='UNet3d', type=str,
                    help='model name')
parser.add_argument('--epoch', default=50, type=int, help='')
parser.add_argument('--paired', default=False, type=bool, help='' )


args = parser.parse_args()

model_name = args.Model
if model_name == 'UNet3d':
    model = modelib.UNet3d(in_channels=4, n_classes=3, n_channels=24).to('cuda')
elif model_name == 'Double_path_UNet3D':
    model = modelib.Double_Path_UNet3D(in_channels=4, n_classes=3, n_channels=24).to('cuda')
    args.paired = True

# model = modelib.Double_Path_UNet3D(in_channels=4, n_classes=3, n_channels=24).to('cuda')
trainer = trainer.Trainer(net=model,
                          model_name=model_name,
                          dataset=BratsDataset,
                          criterion=metircs_losses.BCEDiceLoss(),
                          lr=5e-4,
                          accumulation_steps=4,
                          batch_size=1,
                          fold=0,
                          num_epochs=args.epoch,
                          path_to_csv=config.config.path_to_csv,
                          pair_model=args.paired
                          )


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

