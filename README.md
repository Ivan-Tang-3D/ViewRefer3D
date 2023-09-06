# ViewRefer: Grasp the Multi-view Knowledge for 3D Visual Grounding with GPT and Prototype Guidance

Official implementation of ['ViewRefer: Grasp the Multi-view Knowledge for 3D Visual Grounding with GPT and Prototype Guidance'](https://arxiv.org/pdf/2303.16894.pdf).

## ViewRefer
We propose ViewRefer, a multi-view framework for 3D visual grounding, which grasps view knowledge to alleviate the challenging view discrepancy issue. For the text and 3D modalities, we respectively introduce LLM-expanded grounding texts and a fusion transformer for capturing multi-view information. We present multi-view prototypes to provide highlevel guidance to our framework, which contributes to superior 3D grounding performance.

<div align="center">
  <img src="pipeline.png"/>
</div>

We release the code on the Sr3d Dataset soon.

## Installation and Data Preparation
Please refer the installation and data preparation from [referit3d](https://github.com/referit3d/referit3d).

We adopt bert-base-uncased from huggingface, which can be installed using pip as follows:
```Console
pip install transformers
```
you can download the pretrained weight in [this page](https://huggingface.co/bert-base-uncased/tree/main), and put them into a folder, noted as PATH_OF_BERT.

you can download the CSV of Sr3d from GPT in [this page](), and put them into the folder './data'.

you can download the result checkpoint and log of ViewRefer on Sr3d in [this page]() for test.

## Training
* To train on Sr3d dataset, use the following commands

```Console
    SR3D_GPT='./referit3d_3dvg/data/sr3d_gpt.csv'
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

    TYPE=reserved
    srun -p optimal --quotatype=${TYPE} --gres=gpu:1 -J bash  python -u ./referit3d_3dvg/scripts/${TRAIN_FILE}.py \
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
    --label-lang-sup True > ./logs/results/${NAME}.log 2>&1 &
```

## Validation
* After each epoch of the training, the program will automatically evaluate the performance of the current model. Our code will save the last model in the training as **last_model.pth**, and save the best model following the original Referit3D's repo as **best_model.pth**.

## Test
* At test time, the **analyze_predictions** will run following the original code of Referit3D. 
* The **analyze_predictions** will test the model multiple times, each time using a different random seed. With different random seeds, the sampled point clouds of each object are different. The average accuracy and std will be reported. 

* To test on Sr3d dataset, use the following commands

```Console
    SR3D_GPT='./referit3d_3dvg/data/sr3d_gpt.csv'
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

    TYPE=reserved
    srun -p optimal --quotatype=${TYPE} --gres=gpu:1 -J bash  python -u ./referit3d_3dvg/scripts/${TRAIN_FILE}.py \
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
    --label-lang-sup True > ./logs/results/${NAME}.log 2>&1 &
```

## Credits
The project is built based on the following repository:
* [ReferIt3D](https://github.com/referit3d/referit3d).
