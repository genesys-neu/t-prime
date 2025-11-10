from datasets import load_dataset
from transformers import ViTForImageClassification
from transformers import ViTFeatureExtractor
import numpy as np
from datasets import load_metric
from transformers import TrainingArguments
from transformers import Trainer
import torch


#ds = load_dataset("imagefolder", data_dir="/home/jgroen/NEU/DSTL_1_0_specgram_cropped")
ds = load_dataset("imagefolder", data_dir="/home/jgroen/NEU/DSTL_1_0_specgram_cropped", split="train")
print(ds)
ds_split_train_test = ds.train_test_split(test_size=0.20)
train_ds, val_ds = ds_split_train_test["train"], ds_split_train_test["test"]
#ex = ds['train'][4000]
#print(ex)
#print(len(train_ds))
#print(len(val_ds))

#label = ds['train'].features['label']
label = train_ds.features['label']
#print(label)


model_name_or_path = "/home/jgroen/NEU/dstl/transformer/outputs/"
feature_extractor = ViTFeatureExtractor.from_pretrained(model_name_or_path)
#print(feature_extractor)


def transform(example_batch):
    # Take a list of PIL images and turn them to pixel values
    inputs = feature_extractor([x for x in example_batch['image']], return_tensors='pt')

    # Don't forget to include the labels!
    inputs['label'] = example_batch['label']
    return inputs


#prepared_ds = ds.with_transform(transform)
prepared_train_ds = train_ds.with_transform(transform)
prepared_val_ds = val_ds.with_transform(transform)
#print(prepared_ds['train'][0:2])
#print(prepared_train_ds[0:2])


def collate_fn(batch):
  return {
    'pixel_values': torch.stack([x['pixel_values'] for x in batch]),
    'labels': torch.tensor([x['label'] for x in batch])
  }


metric = load_metric("accuracy")


def compute_metrics(p):
    return metric.compute(predictions=np.argmax(p.predictions, axis=1), references=p.label_ids)


labels = train_ds.features['label'].names
#labels = ds['train'].features['label'].names
#print(labels)

model = ViTForImageClassification.from_pretrained(
    model_name_or_path,
    num_labels=len(labels),
    id2label={str(i): c for i, c in enumerate(labels)},
    label2id={c: str(i) for i, c in enumerate(labels)}
)
print(train_ds[1])

training_args = TrainingArguments(
    output_dir="/home/jgroen/NEU/dstl/transformer/outputs/trained/",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    evaluation_strategy="epoch",
    num_train_epochs=100,
    fp16=True,
    save_strategy="epoch",
    eval_steps=100,
    logging_strategy='steps',
    logging_steps=10,
    learning_rate=1.5e-4,
    lr_scheduler_type='cosine',
    weight_decay=0.05,
    warmup_ratio=0.05,
    save_total_limit=3,
    remove_unused_columns=False,
    push_to_hub=False,
    report_to='tensorboard',
    load_best_model_at_end=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=collate_fn,
    compute_metrics=compute_metrics,
    train_dataset=prepared_train_ds,
    #train_dataset=prepared_ds["train"],
    eval_dataset=prepared_val_ds,
    tokenizer=feature_extractor,
)

train_results = trainer.train()
trainer.save_model()
trainer.log_metrics("train", train_results.metrics)
trainer.save_metrics("train", train_results.metrics)
trainer.save_state()

metrics = trainer.evaluate(prepared_val_ds)
#metrics = trainer.evaluate(prepared_ds['validation'])
trainer.log_metrics("eval", metrics)
trainer.save_metrics("eval", metrics)
