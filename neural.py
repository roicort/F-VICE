import time
import numpy as np
import torch
from torch import nn
from tqdm import tqdm

class DenseLSTM(nn.Module):
    def __init__(
        self, input_dim, hidden_dim, lstm_layers=1, bidirectional=False, dense=False
    ):
        super(DenseLSTM, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.layers = lstm_layers
        self.bidirectional = bidirectional
        self.dense = dense
        # define the LSTM layer
        self.lstm = nn.LSTM(
            input_size=self.input_dim,
            hidden_size=self.hidden_dim,
            num_layers=self.layers,
            bidirectional=self.bidirectional,
            batch_first=True,  # importante para que el batch esté en la primera dimensión
        )
        self.act1 = nn.ReLU()
        # change linear layer inputs depending on if lstm is bidrectional
        if not bidirectional:
            self.linear = nn.Linear(self.hidden_dim, self.hidden_dim)
        else:
            self.linear = nn.Linear(self.hidden_dim * 2, self.hidden_dim)
        self.act2 = nn.ReLU()
        # change linear layer inputs depending on if lstm is bidrectional and extra dense layer isn't added
        if bidirectional and not dense:
            self.final = nn.Linear(self.hidden_dim * 2, 1)
        else:
            self.final = nn.Linear(self.hidden_dim, 1)

    def forward(self, inputs):
        # inputs: (batch, seq_len, features)
        out, _ = self.lstm(inputs)    # (batch, seq_len, hidden)
        out = self.act1(out)
        if self.dense:
            out = self.linear(out)
            out = self.act2(out)
        out = self.final(out) # (batch, seq_len, 1) o (batch, 1)
        if out.dim() == 3:
            out = out[:, -1, :]   # (batch, 1, 1) -> (batch, 1)
        elif out.dim() == 2:
            out = out[:, -1]      # (batch, 1) -> (batch,)
        out = out.squeeze(-1)     # (batch, 1) -> (batch,)
        return out

    def fit(
        self,
        train_dataloader,
        test_dataloader,
        optimizer,
        criterion,
        device,
        epochs=10
    ):
        with tqdm(range(epochs)) as pbar:
            for epoch in pbar:
                self.train()
                start = time.time()
                epoch_loss = []
                for step, batch in enumerate(train_dataloader):
                    optimizer.zero_grad()
                    batch = tuple(t.to(device) for t in batch)
                    inputs, labels = batch
                    out = self(inputs)
                    loss = criterion(out, labels)
                    epoch_loss.append(loss.float().detach().cpu().numpy().mean())
                    loss.backward()
                    optimizer.step()
                test_epoch_loss = []
                end = time.time()
                self.eval()
                with torch.no_grad():
                    for step, batch in enumerate(test_dataloader):
                        batch = tuple(t.to(device) for t in batch)
                        inputs, labels = batch
                        out = self(inputs)
                        loss = criterion(out, labels)
                        test_epoch_loss.append(loss.float().detach().cpu().numpy().mean())
                pbar.set_postfix({
                    "train_loss": np.mean(epoch_loss),
                    "test_loss": np.mean(test_epoch_loss),
                    "time": end - start
                })
    
    def predict(self, dataloader, device):
        self.eval()
        predictions = []
        with torch.no_grad():
            for step, batch in enumerate(dataloader):
                batch = tuple(t.to(device) for t in batch)
                # Soporta batches con o sin labels
                if len(batch) == 2:
                    inputs, _ = batch
                else:
                    inputs = batch[0]
                out = self(inputs)
                predictions.append(out.float().detach().cpu().numpy())
        return np.concatenate(predictions, axis=0)
