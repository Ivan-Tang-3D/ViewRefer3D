# ViewRefer: Grasp the Multi-view Knowledge for 3D Visual Grounding with GPT and Prototype Guidance

Official implementation of ['ViewRefer: Grasp the Multi-view Knowledge for 3D Visual Grounding with GPT and Prototype Guidance'](https://arxiv.org/pdf/2303.16894.pdf).

The paper has been accepted by **ICCV 2023**.

## News
* We release the GPT-expanded Sr3D dataset and the training code of ViewRefer 📌.
  
## Introduction
ViewRefer is a multi-view framework for 3D visual grounding, which grasps view knowledge to alleviate the challenging view discrepancy issue. For the text and 3D modalities, we respectively introduce LLM-expanded grounding texts and a fusion transformer for capturing multi-view information. We present multi-view prototypes to provide highlevel guidance to our framework, which contributes to superior 3D grounding performance.

<div align="center">
  <img src="pipeline.png"/>
</div>

## Requirements
Please refer to [referit3d](https://github.com/referit3d/referit3d) for the installation and data preparation.

We adopt pre-trained BERT from huggingface. Please install related packages:
```bash
pip install transformers
```
Download the [pre-trained BERT](https://huggingface.co/bert-base-uncased/tree/main), and put them into a folder, noted as PATH_OF_BERT.

Download [GPT-expanded Sr3D dataset](https://drive.google.com/file/d/1mb_XYZx_WB_einPNcbO-vqckh0DxJXws/view?usp=sharing), and put them into the folder './data'.

## Getting Started
### Training
* To train on Sr3D dataset, run:

```bash
    SR3D_GPT='./referit3d_3dvg/data/Sr3D_release.csv'
    PATH_OF_SCANNET_FILE='./referit3d_3dvg/data/keep_all_points_with_global_scan_alignment.pkl'
    PATH_OF_REFERIT3D_FILE=${SR3D_GPT}
    PATH_OF_BERT='./referit3d_3dvg/data/bert'

    VIEW_NUM=4
    EPOCH=100
    DATA_NAME=SR3D
    EXT=ViewRefer
    DECODER=4
    NAME=${DATA_NAME}_${VIEW_NUM}view_${EPOCH}ep_${EXT}
    TRAIN_FILE=train_referit3d

    python -u ./referit3d_3dvg/scripts/${TRAIN_FILE}.py \
    -scannet-file ${PATH_OF_SCANNET_FILE} \
    -referit3D-file ${PATH_OF_REFERIT3D_FILE} \
    --bert-pretrain-path ${PATH_OF_BERT} \
    --log-dir logs/results/${NAME} \
    --model 'referIt3DNet_transformer' \
    --unit-sphere-norm True \
    --batch-size 24 \
    --n-workers 8 \
    --max-train-epochs ${EPOCH} \
    --encoder-layer-num 3 \
    --decoder-layer-num ${DECODER} \
    --decoder-nhead-num 8 \
    --view_number ${VIEW_NUM} \
    --rotate_number 4 \
    --label-lang-sup True
```

* Refer to [this link](https://drive.google.com/drive/folders/1YqD7OklOl2rdXyG5aubLtEj6Jqth54Jc?usp=sharing) for the checkpoint and training log of ViewRefer on Sr3D dataset.

### Test
* To test on Sr3D dataset, run:

```bash
    SR3D_GPT='./referit3d_3dvg/data/Sr3D_release.csv'
    PATH_OF_SCANNET_FILE='./referit3d_3dvg/data/keep_all_points_with_global_scan_alignment.pkl'
    PATH_OF_REFERIT3D_FILE=${SR3D_GPT}
    PATH_OF_BERT='./referit3d_3dvg/data/bert'

    VIEW_NUM=4
    EPOCH=100
    DATA_NAME=SR3D
    EXT=ViewRefer_test
    DECODER=4
    NAME=${DATA_NAME}_${VIEW_NUM}view_${EPOCH}ep_${EXT}
    TRAIN_FILE=train_referit3d

    python -u ./referit3d_3dvg/scripts/${TRAIN_FILE}.py \
    --mode evaluate \
    -scannet-file ${PATH_OF_SCANNET_FILE} \
    -referit3D-file ${PATH_OF_REFERIT3D_FILE} \
    --bert-pretrain-path ${PATH_OF_BERT} \
    --log-dir logs/results/${NAME} \
    --resume-path "./checkpoints/best_model.pth"\
    --model 'referIt3DNet_transformer' \
    --unit-sphere-norm True \
    --batch-size 24 \
    --n-workers 8 \
    --max-train-epochs ${EPOCH} \
    --encoder-layer-num 3 \
    --decoder-layer-num ${DECODER} \
    --decoder-nhead-num 8 \
    --view_number ${VIEW_NUM} \
    --rotate_number 4 \
    --label-lang-sup True
```

## Acknowledgement
This repo benefits from [ReferIt3D](https://github.com/referit3d/referit3d) and [MVT-3DVG](https://github.com/sega-hsj/MVT-3DVG). Thanks for their wonderful works.

## Citation
```bash
@article{guo2023viewrefer,
  title={ViewRefer: Grasp the Multi-view Knowledge for 3D Visual Grounding with GPT and Prototype Guidance},
  author={Guo, Ziyu and Tang, Yiwen and Zhang, Renrui and Wang, Dong and Wang, Zhigang and Zhao, Bin and Li, Xuelong},
  journal={arXiv preprint arXiv:2303.16894},
  year={2023}
}
```

## Contact
If you have any questions about this project, please feel free to contact tangyiwen@pjlab.org.cn.
