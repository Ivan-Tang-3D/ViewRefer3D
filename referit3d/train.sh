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