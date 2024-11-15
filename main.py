from transformers import BertTokenizer, LineByLineTextDataset, DataCollatorForLanguageModeling, Trainer, TrainingArguments
from transformers import TrainerCallback, TrainerControl, PreTrainedTokenizerFast
import torch
from torch.utils.data import Dataset, DataLoader
import transformer_model
from torch import nn
import os
from transformers.trainer_utils import EvalPrediction
import mlflow
from tokenizers import Tokenizer



# Loading the pretrained tokenizer
tokenizer = PreTrainedTokenizerFast(tokenizer_file="resource/bpe_tokenizer_banking77.json")
tokenizer.mask_token = "[MASK]"
tokenizer.pad_token = "[PAD]"
tokenizer.cls_token = "[CLS]"
tokenizer.sep_token = "[SEP]"

#datasets for mlm
dataset = LineByLineTextDataset(
    tokenizer=tokenizer,
    file_path="data/banking77_corpus.txt",
    block_size=100,
)

test_dataset = LineByLineTextDataset(
    tokenizer=tokenizer,
    file_path="data/banking77_corpus.txt",
    block_size=100,
)

# the training function, training is done in DDP regimen
def train_ddp():
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=0.15)
    #trainLoader = DataLoader(dataset, collate_fn=data_collator, batch_size=8, shuffle=False)
    vocab_size = tokenizer.vocab_size
    # define transformer model for language modeling
    model = transformer_model.LMTransformer(vocab_size=vocab_size, embed_dim=600, num_classes=vocab_size, num_heads=6, num_layers=6)
    
    # number of trainable parameters in the model
    num_elements = sum(p.numel() for p in model.parameters())
    print(num_elements)

# configuration for training, for instance, where to save the model, batch size, evaluation etc.
    training_args = TrainingArguments(
            "basic-trainer-bpe",
            do_train=True,
            do_eval=False,
            evaluation_strategy="steps",
            per_device_train_batch_size=64,
            per_device_eval_batch_size=64,
            num_train_epochs=500,
            save_steps=500,
            remove_unused_columns=False,
            logging_dir="logs-bpe",
            logging_steps=100
        )

# training
    trainer = Trainer(
        model,
        training_args,
        train_dataset=dataset,
        eval_dataset=test_dataset,
        data_collator=data_collator
    )

    trainer.train()

train_ddp()
