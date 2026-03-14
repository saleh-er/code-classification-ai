from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset

model_name = "microsoft/codebert-base"

tokenizer = AutoTokenizer.from_pretrained(model_name)

dataset = load_dataset("csv", data_files="data/processed/processed_data.csv")

def tokenize(example):
    return tokenizer(example["clean_code"], padding="max_length", truncation=True)

dataset = dataset.map(tokenize)

model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=6)

training_args = TrainingArguments(
    output_dir="./models",
    num_train_epochs=2,
    per_device_train_batch_size=8
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"]
)

trainer.train()
