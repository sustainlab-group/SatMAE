# SatMAE (NeurIPS 2022)
**[Project](https://sustainlab-group.github.io/SatMAE/)** | 
**[Paper](https://arxiv.org/abs/2207.08051)** | 
**[Video](https://recorder-v3.slideslive.com/?share=75759&s=4597a5f4-7f86-4e18-a11b-fbac51cb7616)**

This is the official repository for the NeurIPS 2022 paper 
"_SatMAE: Pre-training Transformers for Temporal and Multi-Spectral Satellite Imagery_".  

Authors: 
[Yezhen Cong](https://www.linkedin.com/in/yezhen-cong-60a449204/) <sup>1</sup>,
[Samar Khanna](https://www.linkedin.com/in/samar-khanna-133b8190/) <sup>1</sup>, 
[Chenlin Meng](https://chenlin9.github.io/), 
[Patrick Liu](https://web.stanford.edu/~pliu1/), 
[Erik Rozi](https://www.linkedin.com/in/erik-rozi/), 
[Yutong He](http://web.stanford.edu/~kellyyhe/),
[Marshall Burke](https://web.stanford.edu/~mburke/), 
[David B. Lobell](https://earth.stanford.edu/people/david-lobell#gs.5vndff), 
[Stefano Ermon](https://cs.stanford.edu/~ermon/).

<sub><sup>1</sup> Equal contribution, order determined via coin flip.</sub>

## Temporal SatMAE
Pre-training and finetuning on fMoW-Temporal are MEMORY-HEAVY. 
Please make sure you have enough memory.
For context, we ran our experiments on 8 NVIDIA V100 GPUs.

### fMoW-Temporal
You can download the fMoW dataset [here](https://github.com/fMoW/dataset). Then follow this piece of [code](https://github.com/fMoW/baseline/blob/master/code/data_ml_functions/dataFunctions.py#L107) to preprocess it. We will also upload the pre-processed dataset soon. The metadata files are [here](https://drive.google.com/drive/folders/1-xSXNpq0xJ4z3F7BPzEcZ04eZ7LqPbYD?usp=share_link).

After you download the dataset and metadata files, your directory should look like:
```
<PATH_TO_DATASET_ROOT_FOLDER>
--- train_62classes.csv
--- val_62classes.csv
--- fmow
------- train
---------- airport
---------- ...
------- val
---------- airport
---------- ...
```

### Pretraining
For pretraining, this is the default command:
```shell
python -m torch.distributed.launch --nproc_per_node=8 \
    --nnodes=1 --master_port=1234 main_pretrain.py \
    --batch_size 8 --accum_iter 16 \
    --norm_pix_loss --epochs 100 \
    --blr 1.5e-4 --mask_ratio 0.75 \
    --input_size 224 --patch_size 16 \
    --model mae_vit_large_patch16 \
    --model_type temporal \
    --dataset_type temporal \
    --train_path <PATH_TO_DATASET_ROOT_FOLDER>/train_62classes.csv
    --output_dir <PATH_TO_YOUR_OUTPUT_FOLDER> \
    --log_dir <PATH_TO_YOUR_OUTPUT_FOLDER> \
    --num_workers 8
```

Note that if you want to use wandb, login to wandb after activating conda 
and before running the code by doing `wandb login` in the shell, 
and add `--wandb <YOUR_WANDB_PROJECT_NAME>` to the command above.
This applies to all following commands.
You will also have to edit the entity name in `main_pretrain.py` and `main_finetune.py`.

### Finetuning
To finetune, the basic command is:
```shell
python -m torch.distributed.launch --nproc_per_node=8 \
    --nnodes=1 --master_port=1234 main_finetune.py \
    --output_dir <PATH_TO_YOUR_OUTPUT_FOLDER> \
    --log_dir <PATH_TO_YOUR_OUTPUT_FOLDER> \
    --batch_size 4 --accum_iter 4 \
    --model vit_large_patch16 --epochs 50 --blr 1e-3 --layer_decay 0.75 \
    --weight_decay 0.05 --drop_path 0.2 --reprob 0.25 \
    --mixup 0.8 --cutmix 1.0 --model_type temporal \
    --finetune <PATH_TO_YOUR_PRETRAIN_CHECKPOINT> \
    --dist_eval --num_workers 8 --dataset temporal \
    --train_path <PATH_TO_DATASET_ROOT_FOLDER>/train_62classes.csv \
    --test_path <PATH_TO_DATASET_ROOT_FOLDER>/val_62classes.csv
```

Note: If you are using our provided checkpoint, please add `--nb_classes 1000`. 
This is a legacy issue which won't affect the model performance since the  actual number of classes is less than 1000.
To resume a finetuning job, you can replace the 
`--finetune <PATH_TO_YOUR_PRETRAIN_CHECKPOINT>` to
`--resume <PATH_TO_YOUR_PRETRAIN_CHECKPOINT>` in the command above.
To finetune from scratch, simply omit the `--finetune` argument.

### Evaluation
To evaluate, the basic command is:
```shell
python -m torch.distributed.launch --nproc_per_node=8 \
    --nnodes=1 --master_port=1234 main_finetune.py \
    --output_dir <PATH_TO_YOUR_OUTPUT_FOLDER> \
    --log_dir <PATH_TO_YOUR_OUTPUT_FOLDER> \
    --batch_size 16 \
    --model vit_large_patch16 \
    --model_type temporal \
    --resume <PATH_TO_YOUR_FINEtune_CHECKPOINT>  \
    --dist_eval --eval --num_workers 8 --dataset fmow_temporal \
    --train_path <PATH_TO_DATASET_ROOT_FOLDER>/train_62classes.csv \
    --test_path <PATH_TO_DATASET_ROOT_FOLDER>/val_62classes.csv
```

Similarly, you are using our provided checkpoint, please add `--nb_classes 1000`.

### Model Weights
TODO


## Multi-Spectral SatMAE
Training multi-spectral SatMAE is similar to training 
temporal SatMAE.

### fMoW-Sentinel Dataset
You can access and download the fMoW-Sentinel dataset we collected [here](https://purl.stanford.edu/vg497cb6002). 
Try this [link](https://searchworks.stanford.edu/view/vg497cb6002) if the previous one doesn't display correctly.

Note that when loading the `train.csv` or `val.csv` files, you may have to preprocess a column
called `image_path`. The `image_path` for any row can be constructed like this:
```
fmow-sentinel/<split>/<category>/<category>_<location_id>/<category>_<location_id>_<image_id>.tif
```

### Pretraining
For pretraining, this is the default command:
```shell
python -m torch.distributed.launch --nproc_per_node=8 main_pretrain.py \
--wandb satmae_pretrain \
--batch_size 16 --accum_iter 32 --blr 0.0001 \
--epochs 200 --warmup_epochs 20 --num_workers 16 \
--input_size 96 --patch_size 8 \
--model_type group_c \
--dataset_type sentinel --dropped_bands 0 9 10 \
--train_path /home/fmow-sentinel-filtered-csv/train.csv \
--output_dir /home/experiments/pretrain \
--log_dir /home/experiments/pretrain
```

For an example of additional arguments, you can specify them like so:
```shell
python -m torch.distributed.launch --nproc_per_node=8 main_pretrain.py \
--wandb satmae_pretrain \
--batch_size 16 --accum_iter 32 --blr 0.0001 \
--epochs 200 --warmup_epochs 20 --num_workers 16 \
--input_size 96 --patch_size 8 \
--mask_ratio 0.9 --spatial_mask \
--norm_pix_loss \
--model_type group_c \
--dataset_type sentinel --dropped_bands 0 9 10 \
--grouped_bands 0 1 2 6 --grouped_bands 3 4 5 7 --grouped_bands 8 9 \
--train_path /home/fmow-sentinel-filtered-csv/train.csv \
--output_dir /home/experiments/pretrain \
--log_dir /home/experiments/pretrain \
--resume /home/experiments/pretrain/checkpoint-175.pth \
```



### Finetuning
To finetune, the basic command is:
```shell
python -m torch.distributed.launch --nproc_per_node=8 main_finetune.py \
--wandb satmae_finetune \
--batch_size 8 --accum_iter 16 --blr 0.0002 \
--epochs 30 --num_workers 16 \
--input_size 96 --patch_size 8  \
--weight_decay 0.05 --drop_path 0.2 --reprob 0.25 --mixup 0.8 --cutmix 1.0 \
--model_type group_c  \
--dataset_type sentinel --dropped_bands 0 9 10 \
--train_path /home/fmow-sentinel-filtered-csv/train.csv \
--test_path /home/fmow-sentinel-filtered-csv/val.csv \
--output_dir /home/experiments/finetune \
--log_dir /home/experiments/finetune \
--finetune /home/experiments/pretain/checkpoint-199.pth
```
To finetune from scratch, simply omit the `--finetune` argument.
To resume a finetuning job, you can replace `--finetune path/to/pretrained_checkpoint.pth` 
with `--resume path/to/finetune_checkpoint.pth` in the command above.

### Model Weights
TODO

## Acknowledgements
Code from this repository is inspired from the Masked Autoencoders (MAE) repository ([link](https://github.com/facebookresearch/mae)).

## Citation
If you found our project helpful, please cite our paper:
```
@inproceedings{
    satmae2022,
    title={Sat{MAE}: Pre-training Transformers for Temporal and Multi-Spectral Satellite Imagery},
    author={Yezhen Cong and Samar Khanna and Chenlin Meng and Patrick Liu and Erik Rozi and Yutong He and Marshall Burke and David B. Lobell and Stefano Ermon},
    booktitle={Thirty-Sixth Conference on Neural Information Processing Systems},
    year={2022},
    url={https://openreview.net/forum?id=WBhqzpF6KYH}
}
```
