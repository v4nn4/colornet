# colornet

<a href="https://github.com/psf/black"><img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-000000.svg"></a>
</p>

A simple colorization model.

## Installation

```
conda env create -n colornet -f environment.yml
```

## Evaluation

```
conda activate colornet
python runner.py prepare
python runner.py train
```