# Balancing Act: Constraining Disparate Impact in Sparse Models


## About
Official implementation for the paper [Balancing Act: Constraining Disparate Impact in Sparse Models](https://arxiv.org/abs/2310.20673).
This code enables fine-tuning sparse neural networks to achieve bounded disparate impact.

We use the [Cooper library](https://github.com/cooper-org/cooper) for implementing and solving non-convex, constrained optimization problems
(including problems with non-differentiable constraints).

## Dependencies
We use Python 3.9.15.

To install necessary dependencies, run the following command:
```bash
pip install -r requirements.txt
```


## WandB
We use [Weights and Biases](https://wandb.ai/) to track our experiments. To use WandB, you need to create an account and login.
Then setup a new project. 

To make authentication easier, we recommend setting up an environment variable for your API key.
You can find your API key in your account settings. Then add the following line to your .bashrc or .zshrc file:
```bash
export WANDB_API_KEY=<your_api_key>
```


## Environment Variables
We use environment variables to configure our experiments.
Set these variables in your terminal or add them to your .bashrc or .zshrc file.
```bash
export DATA_DIR=<path_to_data_dir>
export CHECKPOINT_DIR=<path_to_checkpoint_dir>

export WANDB_ENTITY=<wandb_entity>
export WANDB_PROJECT=<wandb_project>
```

## Pretrained models
- CIFAR100: we use the pretrained model from [here](https://github.com/chenyaofo/pytorch-CIFAR-models).
- FairFace: we use the pretrained model from [here](https://github.com/dchen236/FairFace).
- UTKFace: we trained the models ourselves, you can find the models [here](https://drive.google.com/drive/folders/1oEM9SpcqvRBsX0gFpdoi_NTa-KzCUr3N?usp=sharing).

Download the pretrained models and place them in the `path_to_checkpoint_dir` directory.


## Datasets
We use the following datasets:
- UTKFace
- FairFace
- CIFAR100

To run the experiments, you need to download the datasets and place them in the `path_to_data_dir` directory.

## Baseline Experiments
To run the Iterative Magnitude Pruning experiments, you first need to run the baseline/dense experiments.

### UTKFace
To run the UTKFace dense experiment, run the following command:

- Target attribute: race

```bash
python main.py \
 --config=src/configs/utkface/dense.py:utkface-no_mitigation \
 --config.train.seed=0 \
 --config.data.dataset_kwargs.target_attribute="race" \
 --config.model.num_classes=5
```
- Target attribute: gender

```bash
python main.py \
 --config=src/configs/utkface/dense.py:utkface-no_mitigation \
 --config.train.seed=0 \
 --config.data.dataset_kwargs.target_attribute="gender" \
 --config.model.num_classes=2
```

### FairFace
To run the FairFace dense experiment, run the following command:
- Target attribute: race

```bash
python main.py \
 --config=src/configs/fairface/dense.py:fairface-no_mitigation \
 --config.train.seed=0 \
 --config.data.dataset_kwargs.target_attribute="race" \
 --config.model.num_classes=7
```
- Target attribute: gender

```bash
python main.py \
 --config=src/configs/fairface/dense.py:fairface-no_mitigation \
 --config.train.seed=0 \
 --config.data.dataset_kwargs.target_attribute="gender" \
 --config.model.num_classes=2
```

### CIFAR100

To run the CIFAR100 dense experiment, run the following command:
```bash
python main.py \
 --config=src/configs/cifar100/dense.py:cifar100-no_mitigation \
 --config.train.seed=0 \
 --config.model.num_classes=100
```

## Main Experiments
After running the baseline experiments, you can run the main experiments.
```bash
python main.py \
 --config=src/configs/best/<dataset_name>/sparse_imp.py:<dataset_name>-<method_name> \
 --config.train.pretrained_model_runid=<pretrained_model_runid>
```
- dataset_name: utkface, fairface, cifar100
- method_name: naive_finetune, equalized_loss, ceag
- pretrained_model_runid: WandB runid of the pretrained model (from baseline experiments)

### Additional Configurations

- For UTKFace and FairFace datasets you can run change the target attribute by modiyfing the `target_attribute` parameter.

Example:
```bash
 --config.data.dataset_kwargs.target_attribute="race" \
 --config.model.num_classes=10  # make sure to change the number of classes accordingly
 ```

- For UTKFace and FairFace datasets you can run intersectional fairness by modiyfing the `target_attribute` parameter.
Example:
```bash
 --config.data.dataset_kwargs.protected_attributes='("race", "gender")' 
```

- For CEAG and Equalized Loss methods you can try different dual learning rates by modifying the dual.lr config parameter:
```bash
 --config.optim.dual.lr=<dual_lr>
```

- For CEAG (our) method you can try different tolerance ($\epsilon$) values by adding the following argument:
```bash
 --config.train.cmp_init_kwargs.tolerance=<tol>
```

### Full example:
```bash
python main.py \
 --config=src/configs/best/utkface/sparse_imp.py:utkface-ceag \
 --config.train.pretrained_model_runid=12
 --config.data.dataset_kwargs.target_attribute="race" \
 --config.model.num_classes=5
```
