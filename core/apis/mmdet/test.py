import mmcv
import torch


def sigle_gpu_test_fomo(model, dataloader):
    model.eval()
    datasets = dataloader.dataset
    prog_bar = mmcv.ProgressBar(len(datasets))

    preds = []
    targets = []
    for idx, data in enumerate(dataloader):
        with torch.no_grad():
            pred, target = model(return_loss=False, fomo=True, **data)
            preds.append(pred)
            targets.append(target)

        prog_bar.update()
    return dict(preds=preds, targets=targets)
