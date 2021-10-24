import time

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.nn import MSELoss

from data import get_dataloader
from metircs_losses import Meter

from IPython.display import clear_output

import matplotlib.pyplot as plt
import pandas as pd

from torch.utils.tensorboard import SummaryWriter


class Trainer:
    """
    Factory for training proccess.
    Args:
        display_plot: if True - plot train history after each epoch.
        net: neural network for mask prediction.
        criterion: factory for calculating objective loss.
        optimizer: optimizer for weights updating.
        phases: list with train and validation phases.
        dataloaders: dict with data loaders for train and val phases.
        path_to_csv: path to csv file.
        meter: factory for storing and updating metrics.
        batch_size: data batch size for one step weights updating.
        num_epochs: num weights updation for all data.
        accumulation_steps: the number of steps after which the optimization step can be taken
                    (https://www.kaggle.com/c/understanding_cloud_organization/discussion/105614).
        lr: learning rate for optimizer.
        scheduler: scheduler for control learning rate.
        losses: dict for storing lists with losses for each phase.
        jaccard_scores: dict for storing lists with jaccard scores for each phase.
        dice_scores: dict for storing lists with dice scores for each phase.
    """

    def __init__(self,
                 net: nn.Module,
                 dataset: torch.utils.data.Dataset,
                 criterion: nn.Module,
                 lr: float,
                 accumulation_steps: int,
                 batch_size: int,
                 fold: int,
                 num_epochs: int,
                 path_to_csv: str,
                 display_plot: bool = False,
                 pair_model: bool = False,
                 model_name: str = 'Unet3d'
                 ):

        """Initialization."""
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.pair_model = pair_model
        self.display_plot = display_plot
        self.net = net.to(self.device)
        self.criterion = criterion
        self.optimizer = Adam(self.net.parameters(), lr=lr)
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode="min",
                                           patience=2, verbose=True)
        self.accumulation_steps = accumulation_steps // batch_size
        self.phases = ["train", "val"]
        self.num_epochs = num_epochs

        self.dataloaders = {
            phase: get_dataloader(
                dataset=dataset,
                path_to_csv=path_to_csv,
                phase=phase,
                fold=fold,
                batch_size=batch_size,
                num_workers=8,
                all_sequence=(not self.pair_model)
            )
            for phase in self.phases
        }
        self.best_loss = float("inf")
        self.losses = {phase: [] for phase in self.phases}
        self.dice_scores = {phase: [] for phase in self.phases}
        self.jaccard_scores = {phase: [] for phase in self.phases}
        self.summaryWriter = SummaryWriter(f'./tensorboard__logdir/{model_name}/')
        self.best_model_pth = f"best_model_{model_name}.pth"
        self.last_epoch_model_pth = f"last_epoch_model_{model_name}_{time.strftime('%m_%d_%H_%M')}.pth"
        self.log_pth = f"log/train_log_{model_name}_{time.strftime('%m_%d_%H_%M')}.csv"

    def _compute_loss_and_outputs(self,
                                  images: torch.Tensor,
                                  targets: torch.Tensor):
        if self.pair_model:
            imagesT1 = images['image_t1'].to(self.device)
            imagesT2 = images['image_t2'].to(self.device)
            targets = targets.to(self.device)
            logits = self.net(imagesT1, imagesT2)
        else:
            images = images.to(self.device)
            targets = targets.to(self.device)
            logits = self.net(images)
        loss = self.criterion(logits, targets)
        return loss, logits

    def _do_epoch(self, epoch: int, phase: str):
        print(f"{phase} epoch: {epoch} | time: {time.strftime('%H:%M:%S')}")

        self.net.train() if phase == "train" else self.net.eval()
        meter = Meter()
        dataloader = self.dataloaders[phase]
        total_batches = len(dataloader)
        running_loss = 0.0
        self.optimizer.zero_grad()
        for itr, data_batch in enumerate(dataloader):
            # print(f"start load data | time: {time.strftime('%H:%M:%S')}")
            if self.pair_model:
                # images_t1, images_t2, targets = data_batch['image_t1'], data_batch['image_t2'], data_batch['mask']
                images, targets = data_batch, data_batch['mask']
            else :
                images, targets = data_batch['image'], data_batch['mask']

            # print(f"start cal loss | time: {time.strftime('%H:%M:%S')}")
            loss, logits = self._compute_loss_and_outputs(images, targets)
            loss = loss / self.accumulation_steps
            # print(f"start backward | time: {time.strftime('%H:%M:%S')}")
            if phase == "train":
                loss.backward()
                if (itr + 1) % self.accumulation_steps == 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad()
            # print(f"end on step | time: {time.strftime('%H:%M:%S')}")
            running_loss += loss.item()
            meter.update(logits.detach().cpu(),
                         targets.detach().cpu()
                         )

        epoch_loss = (running_loss * self.accumulation_steps) / total_batches
        epoch_dice, epoch_iou = meter.get_metrics()

        self.losses[phase].append(epoch_loss)
        self.dice_scores[phase].append(epoch_dice)
        self.jaccard_scores[phase].append(epoch_iou)

        self.summaryWriter.add_scalar(f'loss{[phase]}', epoch_loss, epoch)
        self.summaryWriter.add_scalar(f'dice_scores{[phase]}', epoch_dice, epoch)
        self.summaryWriter.add_scalar(f'jaccard_scores{[phase]}', epoch_iou, epoch)

        print(
            f"{phase} | losses:{epoch_loss} | dice_scores: {epoch_dice} | jaccard_scores: {epoch_iou}")
        return epoch_loss

    def run(self):

        for epoch in range(self.num_epochs):
            self._do_epoch(epoch, "train")
            with torch.no_grad():
                val_loss = self._do_epoch(epoch, "val")
                self.scheduler.step(val_loss)
            if self.display_plot:
                self._plot_train_history()

            if val_loss < self.best_loss:
                print(f"\n{'#' * 20}\nSaved new checkpoint\n{'#' * 20}\n")
                self.best_loss = val_loss
                torch.save(self.net.state_dict(), self.best_model_pth)
            print()
        self._save_train_history()

    def _plot_train_history(self):
        data = [self.losses, self.dice_scores, self.jaccard_scores]
        colors = ['deepskyblue', "crimson"]
        labels = [
            f"""
            train loss {self.losses['train'][-1]}
            val loss {self.losses['val'][-1]}
            """,

            f"""
            train dice score {self.dice_scores['train'][-1]}
            val dice score {self.dice_scores['val'][-1]} 
            """,

            f"""
            train jaccard score {self.jaccard_scores['train'][-1]}
            val jaccard score {self.jaccard_scores['val'][-1]}
            """,
        ]

        clear_output(True)
        with plt.style.context("seaborn-dark-palette"):
            fig, axes = plt.subplots(3, 1, figsize=(8, 10))
            for i, ax in enumerate(axes):
                ax.plot(data[i]['val'], c=colors[0], label="val")
                ax.plot(data[i]['train'], c=colors[-1], label="train")
                ax.set_title(labels[i])
                ax.legend(loc="upper right")

            plt.tight_layout()
            plt.show()

    def load_predtrain_model(self,
                             state_path: str):
        self.net.load_state_dict(torch.load(state_path))
        print("Predtrain model loaded")


    def _save_train_history(self):
        """writing model weights and training logs to files."""
        torch.save(self.net.state_dict(),
                   self.last_epoch_model_pth)

        logs_ = [self.losses, self.dice_scores, self.jaccard_scores]

        log_names_ = ["_loss", "_dice", "_jaccard"]
        logs = [logs_[i][key] for i in list(range(len(logs_)))
                for key in logs_[i]]
        log_names = [key + log_names_[i]
                     for i in list(range(len(logs_)))
                     for key in logs_[i]
                     ]
        pd.DataFrame(
            dict(zip(log_names, logs))
        ).to_csv(self.log_pth, index=False)


