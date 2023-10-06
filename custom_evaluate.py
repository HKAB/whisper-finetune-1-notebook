import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"

import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from transformers import TrainingArguments, Trainer

import datasets
from dataclasses import dataclass

from typing import Dict, List, Optional, Union
import numpy as np
import evaluate

metric = evaluate.load("wer")


@dataclass
class DataCollatorWhisperCTCEncoder:
    processor: WhisperProcessor
    padding: Union[bool, str] = True
    max_length: Optional[int] = None
    truncation: Optional[bool] = True
    max_length_labels: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    pad_to_multiple_of_labels: Optional[int] = None
    
    def __call__(self, features: List[Dict[str, np.ndarray]]) -> Dict[str, torch.Tensor]:
        batch_audio = []
        batch_label = []

        batch_size = len(features)

        for batch_idx in range(batch_size):
            batch_audio.append(features[batch_idx]['input_features'])
            batch_label.append(features[batch_idx]['label'])

        data = list(zip(batch_audio, batch_label))
        # random.shuffle(data)
        
        batch_audio = [item[0] for item in data]
        batch_label = [item[1] for item in data]

        batch = self.processor.feature_extractor(
            batch_audio,
            truncation=self.truncation,
            sampling_rate = 16000,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )
        
        batch_label_id = [self.processor.tokenizer(item, truncation=True, max_length=448)['input_ids'] for item in batch_label]
        # convert to numpy array as required by Whisper tokenizer
        label_features = [{"input_ids": np.asarray(item)} for item in batch_label_id]
        # must be longest, since we don't want to lose any transcript word
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        batch["labels"] = labels
        return batch

model = WhisperForConditionalGeneration.from_pretrained("/workspace/whisper/pretrain_base").to('cuda')
processor = WhisperProcessor.from_pretrained("openai/whisper-base", language="vi", task="transcribe")
processor.tokenizer.pad_token = processor.tokenizer.eos_token
processor.tokenizer.max_length = 448
processor.tokenizer.set_prefix_tokens(language="vi", task="transcribe")

all_dataset_test = datasets.load_from_disk('/workspace/whisper/dataset_hf_fleurs')

def prepare_dataset(examples):
    examples['label'] = examples["raw_transcription"]
    examples["input_features"] = examples["audio"]['array']
    # batch["label"] = batch["raw_transcription"]

    return examples

all_dataset_test_vectorized = all_dataset_test.map(
    prepare_dataset,
    num_proc=8,
    remove_columns=all_dataset_test.column_names,
) 

model.config.forced_decoder_ids = processor.tokenizer.get_decoder_prompt_ids(
    language="vi", task="transcribe"
)
model.config.suppress_tokens = []
model.generation_config.forced_decoder_ids = processor.tokenizer.get_decoder_prompt_ids(
    language="vi", task="transcribe"
)
model.generation_config.suppress_tokens = []

repo_name = '/workspace/whisper'
checkpoint_name = "pretrain_base"

batch_size = 1
eval_accumulation_steps=100

# total_steps = (total_samples / batch_size) * num_epochs
training_args = TrainingArguments(
        output_dir=f'{repo_name}/{checkpoint_name}',
        logging_dir=f'{repo_name}/{checkpoint_name}/log',
        per_device_eval_batch_size=batch_size,
        # gradient_accumulation_steps=accumulation_steps,
        eval_accumulation_steps=eval_accumulation_steps, # avoid OOM CUDA: https://discuss.huggingface.co/t/cuda-out-of-memory-during-evaluation-but-training-is-fine/1783
        metric_for_best_model='wer',
        greater_is_better=False,
        fp16=True, # CUDA only
        dataloader_num_workers=2,
        ignore_data_skip=True,
        label_names=["labels"],
    ) 

data_collator = DataCollatorWhisperCTCEncoder(
    processor=processor, 
)

def compute_wer(eval_prediction):
    pred_ids = eval_prediction.predictions[0] # shape (total eval sample, max_length, vocab size)
    label_ids = eval_prediction.label_ids  # shape (total eval sample, max_length)

    pred_ids = np.argmax(pred_ids, axis=-1) # -> to (total eval sample, max_length)

    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

    pred_str = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    wer = 100 * metric.compute(predictions=pred_str, references=label_str)

    return {'wer': wer}

trainer = Trainer(
        model=model,
        data_collator=data_collator,
        args=training_args,
        compute_metrics=compute_wer,
    )

trainer.evaluate(eval_dataset=all_dataset_test_vectorized)