# Adversarial Continual Learning for Multi-Domain Hippocampal Segmentation

## Abstract
Deep learning for medical imaging suffers from temporal and privacy-related restrictions on data availability. To still obtain viable models, continual learning aims to train in sequential order, as and when data is available. The main challenge that continual learning methods face is to prevent catastrophic forgetting, i.e., a decrease in performance on the data encountered earlier. This issue makes continuous training of segmentation models for medical applications extremely difficult. Yet, often, data from at least two different domains is available which we can exploit to train the model in a way that it disregards domain-specific information. We propose an architecture that leverages the simultaneous availability of two or more datasets to learn a disentanglement between the content and domain in an adversarial fashion. The domain-invariant content representation then lays the base for continual semantic segmentation. Our approach takes inspiration from domain adaptation and combines it with continual learning for hippocampal segmentation in brain MRI. We showcase that our method reduces catastrophic forgetting and outperforms state-of-the-art continual learning methods.

For more information please refer to our [paper](https://arxiv.org/abs/2107.08751).

## Architecture
![alt text](https://github.com/memmelma/continual_adversarial_segmenter/raw/master/architecture.png)

## Qualitative Results
![alt text](https://github.com/memmelma/continual_adversarial_segmenter/raw/master/qualitative_results.png)
Legend: **VP MRI** (original), **GT** (groud truth segmentation), **ACS** (segmentation), **GAN O/P** (output of GAN generator)

## Setup
This repository builds on [medical_pytorch](https://github.com/camgbus/medical_pytorch) and [torchio](https://github.com/fepegar/torchio). Please install this repository as explained in [medical_pytorch](https://github.com/camgbus/medical_pytorch). We provide an implementation of the [continual adversarial segmenter (CAS)](https://arxiv.org/abs/2107.08751), and the baselines [memory aware synapses (MAS)](https://arxiv.org/abs/2005.00079), and [knowledge distillation (KD)](https://arxiv.org/abs/1907.13372).

Our core implementation consists of the following structure:
```
mp
├── agents
|   ├── CASAgent
|   ├── KDAgent
|   ├── MASAgent
|   ├── UNETAgent
├── data
|   ├── PytorchSeg2DDatasetDomain
├── models
|   ├── continual
|       ├── CAS
|       ├── MAS
|       ├── KD
├── *_train.py
```

## Usage
Use the `*_train.py` scripts to run the experiments and set arguments corresponding to `args.py`. We also provide the execution commands that we used to produce the results in `experiments.txt` and provide console logs of the runs in `logs/`.

## Datasets
Datasets should be placed in `storage/data/` and can be loaded via dataloaders provided in `mp.data.datasets` or by custom implementations. We use the following three datasets:
- _DecathlonHippocampus_ [A](http://medicaldecathlon.com/)
- _DryadHippocampus_ [B](https://www.nature.com/articles/sdata201559)
- _HarP_ [C](https://pubmed.ncbi.nlm.nih.gov/25616957/).

## Additional Features
- extensive logging via tensorboard
- load/save/resume training
- multi GPU support

## Acknowledgements
Supported by the _Bundesministerium für Gesundheit_ (BMG) with grant [ZMVI1- 2520DAT03A]
