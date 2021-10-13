import trainer
import SSL.modellib as modelib
import config
import metircs_losses
import pandas as pd
from data import BratsDataset

# model = modellib.UNet3d(in_channels=4, n_classes=3, n_channels=24).to('cuda')
model = modelib.Double_Path_UNet3D(in_channels=4, n_classes=3, n_channels=24).to('cuda')

trainer = trainer.Trainer(net=model,
                          dataset=BratsDataset,
                          criterion=metircs_losses.BCEDiceLoss(),
                          lr=5e-4,
                          accumulation_steps=4,
                          batch_size=1,
                          fold=0,
                          num_epochs=50,
                          path_to_csv=config.config.path_to_csv)


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

