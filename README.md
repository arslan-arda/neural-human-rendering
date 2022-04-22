# Virtual Humans Course Project: Vision Transformers for Neural Human Rendering

## Install Miniconda

```
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
chmod +x Miniconda3-latest-Linux-x86_64.sh
./Miniconda3-latest-Linux-x86_64.sh
```

Close the current terminal and open a new one.

## Setup Conda Environment

```
conda env create -f environment.yml
```

## Activate Conda Environment

```
conda activate virtual_humans
```

## Download and Process Data

```
cd scripts
chmod +x download_and_process_data.sh
./download_and_process_data.sh "--DATASETS_DIR=/path/to/data/directory"
```

## Run experiments

```
cd src
module load gcc/8.2.0 python_gpu/3.9.9
```

### Original Pix2Pix on Face dataset

```
bsub -n 4 -W 24:00 -R "rusage[mem=4096, ngpus_excl_p=1]" -R "select[gpu_mtotal0>=10240]" python main.py --datasets_dir /path/to/data/directory --dataset_type face --discriminator_type cnn --checkpoints_dir /path/to/checkpoints/directory
```

### Original Pix2Pix on Body dataset

```
bsub -n 4 -W 24:00 -R "rusage[mem=4096, ngpus_excl_p=1]" -R "select[gpu_mtotal0>=10240]" python main.py --datasets_dir /path/to/data/directory --dataset_type body_smplpix --discriminator_type cnn --checkpoints_dir /path/to/checkpoints/directory
```