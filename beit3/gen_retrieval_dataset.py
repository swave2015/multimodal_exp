from datasets import RetrievalDataset
from transformers import XLMRobertaTokenizer

tokenizer = XLMRobertaTokenizer("/home/caoxh/unilm/beit3/models/beit3.spm")

RetrievalDataset.make_coco_dataset_index(
    data_path="/home/caoxh/beitv3_dataset",
    tokenizer=tokenizer,
)