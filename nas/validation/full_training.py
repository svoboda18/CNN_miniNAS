import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import os
import time
from model_builder import ModelBuilder

class FullTraining:
    def __init__(self, config: dict):
        # Access config via ValidationStrategy section as per YAML
        vs_cfg = config['ValidationStrategy']
        self.epochs = vs_cfg.get('epochs', 15)
        self.lr = vs_cfg.get('optimizer_params', {}).get('lr', 0.1)
        
        # Data paths and batch sizes from YAML
        self.train_url = vs_cfg['trn_data']['url']
        self.train_batch = vs_cfg['trn_data']['batch']
        self.test_url = vs_cfg['val_data']['url']
        self.test_batch = vs_cfg['val_data']['batch']
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _load_data(self, url, batch_size):
        # Resolve path relative to project root
        if not os.path.exists(url):
            raise FileNotFoundError(f"MNIST CSV not found at: {url}")
            
        df = pd.read_csv(url)
        labels = torch.tensor(df.iloc[:, 0].values, dtype=torch.long)
        pixels = torch.tensor(df.iloc[:, 1:].values, dtype=torch.float32) / 255.0
        pixels = pixels.view(-1, 1, 28, 28)
        
        return DataLoader(TensorDataset(pixels, labels), batch_size=batch_size, shuffle=True)

    def validate(self, architecture: dict) -> dict:
        train_loader = self._load_data(self.train_url, self.train_batch)
        test_loader = self._load_data(self.test_url, self.test_batch)
        
        model = ModelBuilder.build_from_dict(architecture).to(self.device)
        optimizer = optim.Adam(model.parameters(), lr=self.lr)
        criterion = nn.CrossEntropyLoss()
        
        loss_history = []
        start_time = time.time()

        print(f"  [FullTraining] Training for {self.epochs} epochs...")
        for epoch in range(self.epochs):
            model.train()
            epoch_loss = 0
            for x, y in train_loader:
                x, y = x.to(self.device), y.to(self.device)
                optimizer.zero_grad()
                loss = criterion(model(x), y)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            loss_history.append(epoch_loss / len(train_loader))

        # Evaluation
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(self.device), y.to(self.device)
                out = model(x)
                correct += (out.argmax(1) == y).sum().item()
                total += y.size(0)

        return {
            "acc_val": correct / total,
            "loss_history": loss_history,
            "model": model,
            "runtime": time.time() - start_time
        }