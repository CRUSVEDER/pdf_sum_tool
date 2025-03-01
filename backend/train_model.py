from datasets import load_dataset, Dataset
from sklearn.model_selection import train_test_split
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, TrainingArguments, Trainer

# Load dataset from CSV
dataset = load_dataset("csv", data_files="data.csv")["train"]

# Convert dataset to Pandas DataFrame for easy manipulation
df = dataset.to_pandas()

# Split into training and evaluation sets (80% train, 20% eval)
train_texts, eval_texts = train_test_split(df, test_size=0.2)

# Convert back to Dataset format
train_dataset = Dataset.from_pandas(train_texts)
eval_dataset = Dataset.from_pandas(eval_texts)

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")
model = AutoModelForSeq2SeqLM.from_pretrained("facebook/bart-large-cnn")

# Tokenization function
def preprocess_data(batch):
    inputs = tokenizer(batch["text"], max_length=1024, truncation=True, padding="max_length")
    labels = tokenizer(batch["summary"], max_length=150, truncation=True, padding="max_length")
    return {"input_ids": inputs["input_ids"], "labels": labels["input_ids"]}

# Apply preprocessing
train_dataset = train_dataset.map(preprocess_data, batched=True)
eval_dataset = eval_dataset.map(preprocess_data, batched=True)

# Training Arguments
training_args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_dir="./logs",
    num_train_epochs=3
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset
)

# Train the model
trainer.train()

# Save the model
model.save_pretrained("my_summarizer")
tokenizer.save_pretrained("my_summarizer")

print("✅ Model trained and saved successfully!")
