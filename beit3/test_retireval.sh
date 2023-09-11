python -m torch.distributed.launch --nproc_per_node=1 run_beit3_finetuning.py \
        --model beit3_base_patch16_384 \
        --input_size 384 \
        --task coco_retrieval \
        --batch_size 16 \
        --sentencepiece_model /home/caoxh/unilm/beit3/models/beit3.spm \
        --finetune /home/caoxh/unilm/beit3/models/beit3_large_patch16_384_coco_retrieval.pth \
        --data_path /home/caoxh/beitv3_dataset \
        --eval \
        --dist_eval
