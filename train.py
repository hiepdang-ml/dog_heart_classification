from typing import List, Tuple, Optional

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Optimizer, Adam
from torch.utils.data import DataLoader

from utils import Accumulator, EarlyStopping, Logger, CheckPointSaver
from datasets import DogHeartLabeledDataset, DogHearUnlabeledDataset
from models import NeuralNet


def loss_function(
    scores: torch.Tensor,
    gt_labels: torch.Tensor,
):
    return F.cross_entropy(input=scores, target=gt_labels, reduction='mean')


def train(
    model: nn.Module,
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
    optimizer: Optimizer,
    n_epochs: int,
    patience: int,
    tolerance: float,
    checkpoint_dir: Optional[str] = None,
) -> nn.Module:

    model.train()
    train_metrics = Accumulator()
    early_stopping = EarlyStopping(patience, tolerance)
    logger = Logger()
    checkpoint_saver = CheckPointSaver(dirpath=checkpoint_dir)

    # loop through each epoch
    for epoch in range(1, n_epochs + 1):
        # Loop through each batch
        for batch, (batch_images, gt_labels, filenames) in enumerate(train_dataloader, start=1):
            batch_images = batch_images.to(device)
            gt_labels = gt_labels.to(device)
            optimizer.zero_grad()
            scores: torch.Tensor = model(batch_images)
            pred_labels: torch.Tensor = scores.max(dim=1).indices
            # print(pred_labels.detach().cpu().numpy())
            # print(gt_labels.detach().cpu().numpy())
            n_corrects: int = (pred_labels == gt_labels).sum().item()
            n_predictions: int = pred_labels.numel()
            loss: torch.Tensor = loss_function(scores, gt_labels).mean()
            loss.backward()
            optimizer.step()
            
            # Accumulate the metrics
            train_metrics.add(n_correct=n_corrects, n_predictions=n_predictions, loss=loss.item())
            train_accuracy: float = train_metrics['n_correct'] / train_metrics['n_predictions']
            train_loss = train_metrics['loss'] / batch
            logger.log(
                epoch=epoch, n_epochs=n_epochs, batch=batch, n_batches=len(train_dataloader),
                train_accuracy=train_accuracy, train_loss=train_loss
            )

        # Save checkpoint
        if checkpoint_dir:
            checkpoint_saver.save(model, filename=f'epoch{epoch}.pt')

        # Reset metric records for next epoch
        train_metrics.reset()

        # Evaluate
        val_accuracy, val_loss = evaluate(model=model, dataloader=val_dataloader)
        logger.log(epoch=epoch, n_epochs=n_epochs, val_accuracy=val_accuracy, val_loss=val_loss)
        print('='*20)

        early_stopping(val_loss)
        if early_stopping:
            print('Early Stopped')
            break
    
    return model


def evaluate(model: nn.Module, dataloader: DataLoader) -> Tuple[float, float]:
    model.eval()
    metrics = Accumulator()

    # Loop through each batch
    for batch, (batch_images, gt_labels, filenames) in enumerate(dataloader, start=1):
        batch_images = batch_images.to(device)
        gt_labels = gt_labels.to(device)
        scores: torch.Tensor = model(batch_images)
        pred_labels = scores.max(dim=1).indices
        n_corrects = (pred_labels == gt_labels).sum().item()
        n_predictions = pred_labels.numel()
        loss = loss_function(scores, gt_labels).mean()

        # Accumulate the metrics
        metrics.add(n_corrects=n_corrects, n_predictions=n_predictions, loss=loss.item())

    # Compute the aggregate metrics
    accuracy: float = metrics['n_corrects'] / metrics['n_predictions']
    loss: float = metrics['loss'] / batch
    return accuracy, loss


if __name__ == '__main__':
    train_dataset = DogHeartLabeledDataset(data_root='Dog_heart/Train')
    valid_dataset = DogHeartLabeledDataset(data_root='Dog_heart/Valid')

    device = torch.device('cuda')

    net = NeuralNet(
        n_hiddens=[
            512, 512, 512, 
            1024, 1024, 1024, 
            2048, 2048, 2048,
        ], 
        poolings=[
            True, False, False, 
            True, False, False, 
            True, False, False,
        ],
        n_classes=3,
    ).to(device)
    optimizer = Adam(params=net.parameters(), lr=0.00001)

    net = train(
        model=net,
        train_dataloader=DataLoader(dataset=train_dataset, batch_size=16, shuffle=True),
        val_dataloader=DataLoader(dataset=valid_dataset, batch_size=4, shuffle=False),
        optimizer=optimizer,
        n_epochs=100,
        patience=10,
        tolerance=0.,
        checkpoint_dir='.checkpoints',
    )


