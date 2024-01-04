import datetime
import os
import pickle

import fire
from torch.utils.data import DataLoader

from colornet.prepare import generate_dataset
from colornet.train import train

# from colornet.evaluate import evaluate


class Runner(object):
    def prepare(self):
        train_dataset, test_dataset = generate_dataset(path="data/images")

        path = R"build/prepare"
        os.makedirs(path, exist_ok=True)
        with open(os.path.join(path, "train_dataset.pkl"), "wb") as f:
            pickle.dump(train_dataset, f)
        with open(os.path.join(path, "test_dataset.pkl"), "wb") as f:
            pickle.dump(test_dataset, f)

    def train(
        self,
        batch_size: int = 16,
        nb_epochs: int = 10,
        learning_rate: float = 3.0e-4,
        steplr_step_size: int = 10,
    ):
        path = R"build/prepare"

        train_dataset = pickle.load(open(os.path.join(path, "train_dataset.pkl"), "rb"))
        test_dataset = pickle.load(open(os.path.join(path, "test_dataset.pkl"), "rb"))
        train_dataloader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, drop_last=True
        )
        test_dataloader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=True, drop_last=True
        )

        # Train
        today = datetime.datetime.today()
        experiment_name = today.strftime("%Y-%m-%d_%H-%M-%S")
        report = train(
            experiment_name=experiment_name,
            train_dataloader=train_dataloader,
            test_dataloader=test_dataloader,
            batch_size=batch_size,
            nb_epochs=nb_epochs,
            init_learning_rate=learning_rate,
            steplr_step_size=steplr_step_size,
        )
        report_folder = f"build/train/{experiment_name}"
        os.makedirs(report_folder, exist_ok=True)
        report.save_model(os.path.join(report_folder, "model_weights.pt"))  # save model
        report.to_csv(os.path.join(report_folder, "report.csv"))  # export full report
        report.save_fig(os.path.join(report_folder, "report.svg"))  # plot report as png

    # def evaluate(self, N: int, name: str):
    #    evaluate(N=N, experiment_name=name)


if __name__ == "__main__":
    fire.Fire(Runner)
