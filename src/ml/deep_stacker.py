import logging

import torch
import torch.nn as nn
import torch.optim as optim

logger = logging.getLogger("ml.deep_stacker")


class TCNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation):
        super(TCNBlock, self).__init__()
        self.conv1 = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            padding=(kernel_size - 1) * dilation // 2,
            dilation=dilation,
        )
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        return self.dropout(self.relu(self.conv1(x)))


class DeepMetaStacker(nn.Module):
    """
    v24 Deep Learning Meta-Model.
    Stacks Signals using TCN (Temporal Convolutional Network) + Wide&Deep.
    """

    def __init__(self, num_models: int, num_context_features: int):
        super(DeepMetaStacker, self).__init__()

        # Branch 1: Time Series Processing of Signals (TCN)
        # Input: (Batch, Num_Models, Seq_Len=1)
        # Actually standard stacking treats models as features at T.
        # Let's assume we stack historical signals -> TCN

        self.tcn = nn.Sequential(
            TCNBlock(num_models, 32, kernel_size=3, dilation=1),
            TCNBlock(32, 16, kernel_size=3, dilation=2),
            nn.Flatten(),
        )

        # Branch 2: Context Features (Wide)
        self.dense_context = nn.Linear(num_context_features, 32)

        # Fusion
        self.fusion = nn.Linear(16 + 32, 16)
        self.output = nn.Linear(16, 1)  # Probability of Success
        self.sigmoid = nn.Sigmoid()

    def forward(self, signal_history, context):
        # signal_history: [Batch, Models, Seq_Len]
        t_out = self.tcn(signal_history)

        c_out = torch.relu(self.dense_context(context))

        combined = torch.cat([t_out, c_out], dim=1)
        fused = torch.relu(self.fusion(combined))
        return self.sigmoid(self.output(fused))


class StackerTrainer:
    def __init__(self):
        self.model = DeepMetaStacker(num_models=5, num_context_features=10)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion = nn.BCELoss()

    def train_step(self, signals, context, labels):
        # Stub loop
        self.optimizer.zero_grad()
        preds = self.model(signals, context)
        loss = self.criterion(preds, labels)
        loss.backward()
        self.optimizer.step()
        return loss.item()
