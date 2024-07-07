import datetime as dt
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

def predict(model: nn.Module, dataloader: DataLoader) -> pd.DataFrame:
    model.eval()

    filenames = []
    predictions = []
    with torch.no_grad():
        for images, fnames in dataloader:
            images = images.to(next(model.parameters()).device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            filenames.extend(fnames)
            predictions.extend(predicted.cpu().numpy())

    prediction_table = pd.DataFrame(
        data={'image': filenames, 'label': predictions}
    )
    prediction_table.to_csv(
        f'{dt.datetime.now().strftime(r"%Y%m%d%H%M%S")}.csv', 
        header=False, 
        index=False
    )
    return prediction_table

if __name__ == '__main__':

    from typing import Dict
    import argparse
    from datasets import DogHearUnlabeledDataset
    from models import NeuralNet
    test_dataset = DogHearUnlabeledDataset(data_root='Dog_heart/Test')
    test_dataloader = DataLoader(
        dataset=test_dataset, batch_size=16, shuffle=False
    )

    parser = argparse.ArgumentParser(description='Prediction Call')
    parser.add_argument('--checkpoint')
    args: Dict[str, str] = vars(parser.parse_args())
    checkpoint: str = args['checkpoint']
    predict(model=torch.load(checkpoint), dataloader=test_dataloader)
    print('Prediction completed')
