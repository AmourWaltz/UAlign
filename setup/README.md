# Configuration

## Workspace Construction

### Directories

```shell
mkdir project model setup data project/githubs
```

### Docker

```shell
docker images
docker ps -a

docker run --gpus all --name reallm --network=host -v /home/swang/project/reallm:/workspace -it nvcr.io/nvidia/pytorch:23.10-py3 /bin/bash

docker start reallm
```


## General Setup Configurations

### Anaconda Download and Installation

Under `./setup`
```shell
curl -o Anaconda3.sh https://repo.anaconda.com/archive/Anaconda3-2024.06-1-Linux-x86_64.sh

bash Anaconda3.sh
source ~/.bashrc
rm Anaconda3.sh
```

### Git-LFS Download and Installation

Under `./setup`
```shell
wget -O git-lfs.tar.gz https://github.com/git-lfs/git-lfs/releases/download/v3.5.1/git-lfs-linux-amd64-v3.5.1.tar.gz

tar -zxvf git-lfs.tar.gz
rm git-lfs.tar.gz
bash git-lfs-3.5.1/install.sh

rm -rf git-lfs-3.5.1/
```

### LLM Downloads from HuggingFace

Under `./model`
```shell
# LLaMA-3.1-8B-Instruct
git lfs install
git clone https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct
rm -rf Meta-Llama-3.1-8B-Instruct/original

git clone https://huggingface.co/Qwen/Qwen2-7B-Instruct
```

## Project Configurations

### ReFact Environment Installation

```shell
cd ./project/
mkdir refact
git clone https://github.com/AmourWaltz/ReFact
cd ReFact

conda create -n refact python=3.11
conda activate refact

pip install -U -r requirements.txt
```