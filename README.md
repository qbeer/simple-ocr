## Simple OCR for synthetically generated text images

This repository contains code for training and evaluating a simple OCR system on synthetically generated text images. The code is written in Python 3 and uses [PyTorch](https://pytorch.org/) as the deep learning framework. The data is hosted on [Kaggel](https://www.kaggle.com/competitions/physdl2023comp2/overview).

### Example images

| Image | Label |
| --- | --- |
| ![Image](./figures/60.jpg) | `60` |
| ![Image](./figures/789.jpg) | `789` |
| ![Image](./figures/72103.jpg) | `72103` |

We assume that the data is stored in the `data` directory. The data can be downloaded from [Kaggel](https://www.kaggle.com/competitions/physdl2023comp2/overview). The data should be extracted into the `data` directory. The directory structure should look like this:

```bash
data
├── 2
│   ├── train
│   │   ├── 0.jpg
│   │   ├── 1.jpg
...
│   │   └── 9999.jpg
│   └── test
│       ├── 0.jpg
│       ├── 1.jpg
...
│       └── 9999.jpg
├── 3
...
├── 5
...
├── train.csv
└── test.csv
```

### Task

The task is to train a model that can recognize the text in the images. The model should take an image as input and output the corresponding text. The text is always a sequence of digits and the length of the sequence is between 2 and 5. The model should be trained and validated then evaluated on the held-out test set. The evaluation metric is the [word error rate](https://en.wikipedia.org/wiki/Word_error_rate) (WER).

### Resources

- [PyTorch](https://pytorch.org/)
- [PyTorch Lightning](https://www.pytorchlightning.ai/)

### Environment setup

This project uses [Conda](https://docs.conda.io/en/latest/) to manage the environment. The following commands can be used to create and activate the environment.

```bash
conda create -n ocr python=3.9
conda activate ocr
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
conda install lightning -c conda-forge
conda install matplotlib numpy pandas
conda install jupyter notebook
conda install scikit-learn scipy
conda install tensorboard
```

You can also use the `environment.yml` file to create the environment.

```bash
conda env create -f environment.yml
```

## Training

Training is done using the `train_baseline.ipbynb` notebook. See the notebook for details.

## Logging

The training logs are saved in the `logs` directory. The logs can be visualized using TensorBoard.

```bash
tensorboard --logdir logs
```
