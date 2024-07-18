# Spatio-temporal SSL

This is the official code repository for the paper "Context Matters: Leveraging Spatiotemporal Metadata for Semi-Supervised Learning on Remote Sensing Images", which was accepted ECAI 2024.

## Abstract
> Remote sensing projects typically generate large amounts of imagery that can be used to train powerful deep neural networks. However, the amount of labeled images is often small, as remote sensing applications generally require expert labelers. Thus, semi-supervised learning (SSL), i.e., learning with a small pool of labeled and a larger pool of unlabeled data, is particularly useful in this domain. Current SSL approaches generate pseudo-labels from model predictions for unlabeled samples. As the quality of these pseudo-labels is crucial for performance, utilizing additional information to improve pseudo-label quality yields a promising direction. For remote sensing images, geolocation and recording time are generally available and provide a valuable source of information as semantic concepts, such as land cover, are highly dependent on spatiotemporal context, e.g., due to seasonal effects and vegetation zones. In this paper, we propose to exploit spatiotemporal metainformation in SSL to improve the quality of pseudo-labels and, therefore, the final model performance. We show that directly adding the available metadata to the input of the predictor at test time degenerates the prediction quality for metadata outside the spatiotemporal distribution of the training set. Thus, we propose a teacher-student SSL framework where only the teacher network uses metainformation to improve the quality of pseudo-labels on the training set. Correspondingly, our student network benefits from the improved pseudo-labels but does not receive metadata as input, making it invariant to spatiotemporal shifts at test time. Furthermore, we propose methods for encoding and injecting spatiotemporal information into the model and introduce a novel distillation mechanism to enhance the knowledge transfer between teacher and student. Our framework dubbed Spatiotemporal SSL can be easily combined with several state-of-the-art SSL methods, resulting in significant and consistent improvements on the BigEarthNet and EuroSAT benchmarks. Code is publicly available at https://github.com/mxbh/spatiotemporal-ssl.

## Citation
```
@article{bernhard2024context,
  title={Context Matters: Leveraging Spatiotemporal Metadata for Semi-Supervised Learning on Remote Sensing Images},
  author={Bernhard, Maximilian and Hannan, Tanveer and Strau{\ss}, Niklas and Schubert, Matthias},
  journal={arXiv preprint arXiv:2404.18583},
  year={2024}
}
```
## Supplementary Material
[supplementary.pdf](https://github.com/mxbh/spatiotemporal-ssl/releases/download/supplementary/supplementary.pdf)

## Installation
This repository is based on [USB](https://github.com/microsoft/Semi-supervised-learning). USB is built on pytorch, with torchvision, torchaudio, and transformers.

To install the required packages, you can create a conda environment:
```
conda create --name usb python=3.8
```
then use pip to install required packages:
```
pip install -r requirements.txt
```
## Data Preparation
BigEarthNet can be downloaded [here](https://bigearth.net/).

EuroSAT can be downloaded [here](https://zenodo.org/records/7711810#.ZAm3k-zMKEA).

The `./data` folder should be structured as follows:
```
./data/
    bigearthnet/
      S2A_MSIL2A_20170717T113321_82_89/
        S2A_MSIL2A_20170717T113321_82_89_labels_metadata.json
        S2A_MSIL2A_20170717T113321_82_89_B04.tif
        S2A_MSIL2A_20170717T113321_82_89_B03.tif
        S2A_MSIL2A_20170717T113321_82_89_B02.tif
        ...
    eurosat/
        AnnualCrop/
        ...
    splits/
        bigearthnet/
            test.pkl
            val.pkl
            train.pkl
            train_0.01_lb_43.pkl
            train_0.01_ulb_43.pkl
            ...
        eurosat/
          lb_labels_20_1_seed0_idx.npy
          ulb_labels_20_1_seed0_idx.npy
          ...
    eurosat_meta.yaml
```
The split files for BigEarthNet can be accessed [here](https://github.com/mxbh/spatiotemporal-ssl/releases/tag/data).
We adopt the train-val-test for BigEarthNet from [Seasonal Contrast](https://github.com/ServiceNow/seasonal-contrast/blob/8285173ec205b64bc3e53b880344dd6c3f79fa7a/datasets/bigearthnet_dataset.py#L119) and exclude patches with cloud, shadow, and seasonal snow.

## Training
To train a model, e.g., FixMatch, run
```
python3 train.py --c config/bigearthnet/0.01_lb/vit_s/stssl_fixmatch/stssl_fixmatch_vits_b64u7_lr1e-4_th05_70_mse1.0/config.yaml
```

## Testing
After training, to test your model, run
```
python test.py \\
    --c config/bigearthnet/0.01_lb/vit_s/stssl_fixmatch/stssl_fixmatch_vits_b64u7_lr1e-4_th05_70_mse1.0/config.yaml \\
    --load_path runs/bigearthnet/0.01_lb/vit_s/stssl_fixmatch/stssl_fixmatch_vits_b64u7_lr1e-4_th05_70_mse1.0/model_best.pth
```

---
This code is based on [USB](https://github.com/microsoft/Semi-supervised-learning/tree/main), and contains parts of the code of [UDAL](https://openaccess.thecvf.com/content/WACV2023/papers/Lazarow_Unifying_Distribution_Alignment_as_a_Loss_for_Imbalanced_Semi-Supervised_Learning_WACV_2023_paper.pdf), [CAP](https://github.com/xiemk/SSMLL-CAP), and [CDMAD](https://github.com/LeeHyuck/CDMAD).