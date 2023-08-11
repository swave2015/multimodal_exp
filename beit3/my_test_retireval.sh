python my_beitv3_model.py \
        --model beit3_base_patch16_384 \
        --input_size 384 \
        --task coco_retrieval \
        --batch_size 16 \
        --sentencepiece_model /home/caoxh/unilm/beit3/models/beit3.spm \
        --finetune /home/caoxh/unilm/beit3/models/beit3_base_patch16_384_coco_retrieval.pth \
        --data_path /home/caoxh/beitv3_dataset \
        --eval \
        --dist_eval
