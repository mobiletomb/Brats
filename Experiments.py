import torch
import metircs_losses
import data
import SSL.modellib as modellib
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import trainer
import config
from data import BratsDataset


def compute_scores_per_classes(model,
                               dataloader,
                               classes):
    """
    Compute Dice and Jaccard coefficients for each class
    :param model: Neural net for make predictions
    :param dataloader: Dataset object to load data from
    :param classes: List with classes
    :return: Dictionaries with dice and jaccard coefficients for each
    class for each slice
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dice_socres_per_classes = {key: list() for key in classes}
    iou_socres_per_classes = {key: list() for key in classes}

    with torch.no_grad():
        for i, data in enumerate(dataloader):
            imgs, targets = data['image'], data['mask']
            imgs, targets = imgs.to(device), targets.to(device)
            logits = model(imgs)
            logits = logits.detach().cpu().numpy()
            targets = targets.detach().cpu().numpy()

            dice_scores = metircs_losses.dice_coef_metric_per_classes(logits, targets)
            iou_scores = metircs_losses.jaccard_coef_metric_per_classes(logits, targets)

            for key in dice_scores.keys():
                dice_socres_per_classes[key].extend(dice_scores[key])

            for key in iou_scores.keys():
                iou_socres_per_classes[key].extend(iou_scores[key])

    return dice_socres_per_classes, iou_socres_per_classes


def compute_results(model,
                    dataloader,
                    treshold=0.33):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    results = {'Id':[], 'image':[], 'GT':[], 'Prediction':[]}

    with torch.no_grad():
        for i, data in enumerate(dataloader):
            id_, imgs, targets = data['Id'], data['image'], data['mask']
            imgs, targets = imgs.to(device), targets.to(device)
            logits = model(imgs)
            probs = torch.sigmoid(logits)

            predictions = (probs >= treshold).float()
            predictions = predictions.cpu()
            targets = targets.cpu()

            results['Id'].append(id_)
            results['Id'].append(id_)
            results['GT'].append(targets)
            results['Prediction'].append(predictions)

            if (i>5):
                return results
    return results


if __name__ == '__main__':
    val_dataloader = data.get_dataloader(data.BratsDataset,
                                         'train_data.csv',
                                         phase='valid',
                                         fold=0)

    model = modellib.UNet3d(in_channels=4, n_classes=3, n_channels=24).to('cpu')

    # trainer = trainer.Trainer(net=model,
    #                           dataset=BratsDataset,
    #                           criterion=metircs_losses.BCEDiceLoss(),
    #                           lr=5e-4,
    #                           accumulation_steps=4,
    #                           batch_size=2,
    #                           fold=0,
    #                           num_epochs=50,
    #                           path_to_csv=config.config.path_to_csv)
    #
    # trainer.load_predtrain_model(config.config.pretrained_model_path)
    print('start load')
    model.load_state_dict(torch.load('/home/qlc/Model/BraTs/log/best_model.pth', map_location=torch.device('cpu')))
    model.eval()
    print('load completed')
    dice_scores_per_classes, iou_scores_per_classes = compute_scores_per_classes(
        model, val_dataloader, ['WT', 'TC', 'ET']
    )
    print('reference completed')
    dice_df = pd.DataFrame(dice_scores_per_classes)
    dice_df.columns = ['WT dice', 'TC dice', 'ET dice']

    iou_df = pd.DataFrame(iou_scores_per_classes)
    iou_df.columns = ['WT jaccard', 'TC jaccard', 'ET jaccard']
    val_metrics_df = pd.concat([dice_df, iou_df], axis=1, sort=True)
    val_metrics_df = val_metrics_df.loc[:, ['WT dice', 'WT jaccard',
                                            'TC dice', 'TC jaccard',
                                            'ET dice', 'ET jaccard']]
    val_metrics_df.sample(5)

    ###############    Bar      ###############
    # colors = ['#35FCFF', '#FF355A', '#96C503', '#C5035B', '#28B463', '#35FFAF']
    # palette = sns.color_palette(colors, 6)
    #
    # fig, ax = plt.subplots(figsize=(12, 6))
    # sns.barplot(x=val_metrics_df.mean().index, y=val_metrics_df.mean(),
    #             palette=palette, ax=ax)
    # ax.set_xticklabels(val_metrics_df.columns, fontsize=14, rotation=15)
    # ax.set_title('Dice and Jaccard Coefficients from Validation', fontsize=20)
    #
    # for idx, p in enumerate(ax.patches):
    #     percentage = '{:.1f}'.format(100 * val_metrics_df.mean().values[idx])
    #     x = p.get_x() + p.get_width() / 2 - 0.15
    #     y = p.get_y() + p.get_height()
    #     ax.annotate(percentage, (x, y), fontsize=15, fontweight='bold')
    #
    # fig.savefig('result1.png', format='png', pad_inches=0.2, transparent=False,
    #             bbox_inches='tight')

    def compute_results(model,
                    dataloader,
                    treshold=0.33):

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    results = {"Id": [], "image": [], "GT": [], "Prediction": []}

    with torch.no_grad():
        for i, data in enumerate(dataloader):
            id_, imgs, targets = data['Id'], data['image'], data['mask']
            imgs, targets = imgs.to(device), targets.to(device)
            logits = model(imgs)
            probs = torch.sigmoid(logits)

            predictions = (probs >= treshold).float()
            predictions = predictions.cpu()
            targets = targets.cpu()

            results["Id"].append(id_)
            results["image"].append(imgs.cpu())
            results["GT"].append(targets)
            results["Prediction"].append(predictions)

            # only 5 pars
            if (i > 5):
                return results
        return results

    % % time
    results = compute_results(
        nodel, val_dataloader, 0.33)

    for id_, img, gt, prediction in zip(results['Id'][4:],
                                        results['image'][4:],
                                        results['GT'][4:],
                                        results['Prediction'][4:]
                                        ):
        print(id_)
        break

        convert
        3
        d
        to
        2
        d
        ground
        truth and prediction

        show_result = ShowResult()
        show_result.plot(img, gt, prediction)

        3
        d
        binary
        mask
        projection
        for ground truth and prediction




