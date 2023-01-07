import mmcv
import torch


def sigle_gpu_test_fomo(model, dataloader):
    model.eval()
    datasets = dataloader.dataset
    prog_bar = mmcv.ProgressBar(len(datasets))

    results = []
    for idx, data in enumerate(dataloader):
        with torch.no_grad():
            result = model(return_loss=False, fomo=True, **data)
            results.append(result)

        batch_size = len(result)
        for _ in range(batch_size):
            prog_bar.update()
    return results
