import logging

import numpy as np
import torch
from torch import nn, optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from .model import ColorizationUNet
from .report import TrainingReport

log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
logging.basicConfig(level=logging.INFO, format=log_format)
torch.manual_seed(1337)


def train(
    experiment_name: str,
    train_dataloader: DataLoader,
    test_dataloader: DataLoader,
    batch_size: int = 16,
    nb_epochs: int = 10,
    init_learning_rate: float = 0.1,
    steplr_gamma: float = 0.5,
    steplr_step_size: int = 10,
) -> TrainingReport:
    log_dir = f"build/train/runs/experiment_{experiment_name}"
    writer = SummaryWriter(log_dir=log_dir)

    mean, std = 0, 0
    for inputs, _ in train_dataloader:
        mean += inputs.mean().item()
        std += inputs.std().item()
    mean /= len(train_dataloader)
    std /= len(train_dataloader)

    logging.info(f"Using mean = {mean:.4f} and std = {std:.4f}")
    # model = AutoEncoder(mean=mean, std=std)
    model = ColorizationUNet(mean=mean, std=std)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=init_learning_rate)
    scheduler = StepLR(optimizer, step_size=steplr_step_size, gamma=steplr_gamma)
    epochs = np.arange(1, nb_epochs + 1)

    # Train loop
    for epoch in epochs:
        train_loss = 0
        for bw_images, rgb_images in train_dataloader:
            outputs = model(bw_images)
            loss = criterion(outputs.flatten(), rgb_images.flatten())

            train_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        train_loss /= len(train_dataloader)

        # Parameters
        with torch.inference_mode():
            for name, param in model.named_parameters():
                writer.add_histogram(name, param, epoch)

        # Validation loop
        val_loss = 0.0
        with torch.inference_mode():
            for bw_images, rgb_images in test_dataloader:
                outputs = model(bw_images)
                loss = criterion(outputs, rgb_images)

                val_loss += loss.item()

            val_loss /= len(test_dataloader)

        writer.add_scalars(
            main_tag="Loss",
            tag_scalar_dict={"train/loss": train_loss, "val/loss": val_loss},
            global_step=epoch,
        )

        scheduler.step()
        logging.info(f"Epoch [{epoch}/{nb_epochs}] | Loss = {train_loss:.4f}")

    writer.flush()
    writer.close()

    return TrainingReport(model=model, batch_size=batch_size, log_dir=log_dir)
