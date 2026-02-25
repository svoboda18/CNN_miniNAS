import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from typing import Optional
from tqdm import tqdm
import pandas as pd
import os
import time
from nas.model_builder import ModelBuilder

class FullTraining:

    # Metrics that can be requested via the YAML 'metrics' list
    SUPPORTED_METRICS = {"acc_val", "runtime"}

    def __init__(self, config: dict):
        vs_cfg = config['ValidationStrategy']
        self.epochs = vs_cfg.get('epochs', 15)
        self.lr = vs_cfg.get('optimizer_params', {}).get('lr', 0.1)
        self.use_full_dataset = vs_cfg.get('use_full_dataset', True)

        # Requested metrics (default: acc_val only)
        raw_metrics = vs_cfg.get('metrics', ['acc_val'])
        unknown = set(raw_metrics) - self.SUPPORTED_METRICS
        if unknown:
            raise ValueError(
                f"[FullTraining] Unknown metric(s): {unknown}. "
                f"Supported: {self.SUPPORTED_METRICS}"
            )
        self.metrics: list = list(raw_metrics)

        # Data paths and batch sizes from YAML
        self.train_url = vs_cfg['trn_data']['url']
        self.train_batch = vs_cfg['trn_data']['batch']
        self.train_max = vs_cfg['trn_data'].get('max_samples', None)

        self.test_url = vs_cfg['val_data']['url']
        self.test_batch = vs_cfg['val_data']['batch']
        self.test_max = vs_cfg['val_data'].get('max_samples', None)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _load_data(self, url: str, batch_size: int, max_samples: Optional[int]):
        if not os.path.exists(url):
            raise FileNotFoundError(f"MNIST CSV not found at: {url}")

        with tqdm(desc=f"    Loading {os.path.basename(url)}", unit=" rows",
                  bar_format="    {desc}: {n_fmt} rows read  [{elapsed}]",
                  total=None, leave=False) as pbar:
            df = pd.read_csv(url)
            pbar.update(len(df))

        total_rows = len(df)

        if not self.use_full_dataset and max_samples is not None:
            df = df.iloc[:max_samples]
            tqdm.write(f"    {os.path.basename(url)}: using {len(df)}/{total_rows} rows "
                       f"(max_samples={max_samples})")
        else:
            tqdm.write(f"    {os.path.basename(url)}: {total_rows} rows")

        labels = torch.tensor(df.iloc[:, 0].values, dtype=torch.long)
        pixels = torch.tensor(df.iloc[:, 1:].values, dtype=torch.float32) / 255.0
        pixels = pixels.view(-1, 1, 28, 28)

        return DataLoader(TensorDataset(pixels, labels), batch_size=batch_size, shuffle=True)

    def validate(self, architecture: dict) -> dict:
        train_loader = self._load_data(self.train_url, self.train_batch, self.train_max)
        test_loader  = self._load_data(self.test_url,  self.test_batch,  self.test_max)

        model = ModelBuilder.build_from_dict(architecture).to(self.device)
        optimizer = optim.Adam(model.parameters(), lr=self.lr)
        criterion = nn.CrossEntropyLoss()

        loss_history = []
        start_time = time.time()
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        tqdm.write(f"    device={self.device}  params={n_params:,}  "
                   f"epochs={self.epochs}  lr={self.lr}")

        epoch_bar = tqdm(
            range(self.epochs),
            desc="    train",
            unit="ep",
            bar_format="    {desc} {bar:25}| ep {n_fmt}/{total_fmt}  loss={postfix[0]:.4f}  [{elapsed}<{remaining}]",
            postfix=[float("inf")],
            leave=False,
        )
        for epoch in epoch_bar:
            model.train()
            epoch_loss = 0.0
            batch_bar = tqdm(
                train_loader,
                desc=f"      ep {epoch+1:02d}/{self.epochs}",
                unit="batch",
                bar_format="      {desc} {bar:20}| {n_fmt}/{total_fmt} batches  [{elapsed}]",
                leave=False,
            )
            for x, y in batch_bar:
                x, y = x.to(self.device), y.to(self.device)
                optimizer.zero_grad()
                loss = criterion(model(x), y)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                batch_bar.set_postfix(loss=f"{loss.item():.4f}")
            avg_loss = epoch_loss / len(train_loader)
            loss_history.append(avg_loss)
            epoch_bar.postfix[0] = avg_loss

        # Evaluation
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for x, y in tqdm(test_loader, desc="    eval ",
                             unit="batch", leave=False,
                             bar_format="    {desc} {bar:25}| {n_fmt}/{total_fmt} batches  [{elapsed}]"):
                x, y = x.to(self.device), y.to(self.device)
                out = model(x)
                correct += (out.argmax(1) == y).sum().item()
                total += y.size(0)

        # Build full result, then expose only requested metrics
        # ('loss_history' and 'model' are always included because
        #  io.save_results needs them regardless of metrics config)
        all_computed = {
            "acc_val": correct / total,
            "runtime": time.time() - start_time,
        }
        result = {k: v for k, v in all_computed.items() if k in self.metrics}
        result["loss_history"] = loss_history
        result["model"] = model

        if "acc_val" not in result:
            raise ValueError(
                "[FullTraining] 'acc_val' must be listed in metrics — "
                "it is required to rank architectures."
            )
        return result