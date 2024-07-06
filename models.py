from typing import List

import torch
import torch.nn as nn


class NeuralNet(nn.Module):

    def __init__(
        self, 
        n_hiddens: List[int], 
        poolings: List[bool], 
        n_classes: int
    ):
        assert len(n_hiddens) == len(poolings)

        super().__init__()
        self.n_hiddens: List[int] = n_hiddens
        self.poolings: List[bool] = poolings
        self.n_classes: int = n_classes

        feature_extractor_modules: List[nn.Module] = []
        for n_hidden, pooling in zip(n_hiddens, poolings):
            feature_extractor_modules.extend([
                nn.LazyConv2d(out_channels=n_hidden, kernel_size=3, padding=1),
                nn.BatchNorm2d(num_features=n_hidden),
                nn.ReLU(),
            ])
            if pooling:
                feature_extractor_modules.append(nn.MaxPool2d(kernel_size=2, stride=2))
        
        self.feature_extractor = nn.Sequential(*feature_extractor_modules)

        # Classifier
        self.classifier = nn.Sequential(
            nn.Flatten(start_dim=1, end_dim=-1),
            nn.LazyLinear(out_features=1024),
            nn.ReLU(),
            nn.LazyLinear(out_features=512),
            nn.ReLU(),
            nn.LazyLinear(out_features=n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.feature_extractor(x)
        y = self.classifier(y)
        return y


