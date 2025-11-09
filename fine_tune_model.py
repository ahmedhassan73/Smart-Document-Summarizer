# T5-Small Fine-tuning on CNN/DailyMail Dataset
# Google Colab GPU-Optimized Code - Error Fixed

# # Step 1: Install required libraries with specific versions
# !pip install -q transformers==4.35.0 datasets==2.14.0 accelerate sentencepiece torch

# Step 2: Import libraries
import torch
from transformers import (
    T5Tokenizer, 
    T5ForConditionalGeneration,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq
)
from datasets import load_dataset
import numpy as np

# Step 3: Check GPU availability
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
if device == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# Step 4: Load CNN/DailyMail dataset
print("\nLoading CNN/DailyMail dataset...")
dataset = load_dataset("cnn_dailymail", "3.0.0")

# Use smaller subset for 1-2 hour training
train_dataset = dataset["train"].select(range(5000))  # 5k samples
val_dataset = dataset["validation"].select(range(500))  # 500 samples

print(f"Train samples: {len(train_dataset)}")
print(f"Validation samples: {len(val_dataset)}")

# Step 5: Load T5-small model and tokenizer
print("\nLoading T5-small model and tokenizer...")
model_name = "t5-small"
tokenizer = T5Tokenizer.from_pretrained(model_name, legacy=False)
model = T5ForConditionalGeneration.from_pretrained(model_name)
model = model.to(device)

# Step 6: Preprocessing function
def preprocess_function(examples):
    """
    T5 ke liye input format: "summarize: <article>"
    Target: <summary>
    """
    inputs = ["summarize: " + doc for doc in examples["article"]]
    model_inputs = tokenizer(
        inputs, 
        max_length=256,
        truncation=True,
        padding="max_length"
    )
    
    # Setup the tokenizer for targets
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            examples["highlights"],
            max_length=64,
            truncation=True,
            padding="max_length"
        )
    
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

# Step 7: Preprocess datasets
print("\nPreprocessing datasets...")
tokenized_train = train_dataset.map(
    preprocess_function,
    batched=True,
    remove_columns=train_dataset.column_names,
    desc="Processing train dataset"
)

tokenized_val = val_dataset.map(
    preprocess_function,
    batched=True,
    remove_columns=val_dataset.column_names,
    desc="Processing validation dataset"
)

# Step 8: Data collator
data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    model=model,
    padding=True
)

# Step 9: Training arguments (Seq2SeqTrainingArguments - CORRECT CLASS)
print("\nSetting up training arguments...")
training_args = Seq2SeqTrainingArguments(
    output_dir="./t5-small-cnn-finetuned",
    eval_strategy="steps",  # Correct parameter name
    eval_steps=250,
    save_steps=500,
    learning_rate=5e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=2,
    weight_decay=0.01,
    save_total_limit=1,
    predict_with_generate=True,
    fp16=torch.cuda.is_available(),  # Auto-detect GPU
    logging_steps=50,
    load_best_model_at_end=True,
    metric_for_best_model="loss",
    greater_is_better=False,
    report_to="none",
    gradient_accumulation_steps=2,
    warmup_steps=100,
    save_strategy="steps"
)

# Step 10: Initialize Seq2SeqTrainer (CORRECT TRAINER CLASS)
print("\nInitializing Trainer...")
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_val,
    tokenizer=tokenizer,
    data_collator=data_collator
)

# Step 11: Start training
print("\nStarting training...")
print("=" * 50)
print("Training will take approximately 1-2 hours on T4 GPU")
print("=" * 50)

trainer.train()

# Step 12: Save the fine-tuned model
print("\nSaving fine-tuned model...")
model.save_pretrained("./t5-small-cnn-finetuned-final")
tokenizer.save_pretrained("./t5-small-cnn-finetuned-final")
print("✓ Model saved successfully!")

# Step 13: Test the model with a sample
print("\n" + "=" * 50)
print("Testing the fine-tuned model...")
print("=" * 50)

# Load the saved model for testing
model = T5ForConditionalGeneration.from_pretrained("./t5-small-cnn-finetuned-final")
model = model.to(device)
model.eval()

test_article = dataset["test"][0]["article"][:300]
print(f"\nOriginal Article (excerpt):\n{test_article}...")

input_text = "summarize: " + test_article
input_ids = tokenizer(
    input_text, 
    return_tensors="pt", 
    max_length=256, 
    truncation=True
).input_ids.to(device)

with torch.no_grad():
    outputs = model.generate(
        input_ids,
        max_length=64,
        num_beams=4,
        early_stopping=True,
        no_repeat_ngram_size=2
    )

summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(f"\nGenerated Summary:\n{summary}")

# Show original summary for comparison
print(f"\nOriginal Summary:\n{dataset['test'][0]['highlights']}")

print("\n" + "=" * 50)
print("✓ Fine-tuning complete!")
print(f"✓ Model location: ./t5-small-cnn-finetuned-final")
print("=" * 50)

# Optional: Download model to local machine
print("\nTo download model, uncomment these lines:")
print("# from google.colab import files")
print("# !zip -r model.zip ./t5-small-cnn-finetuned-final")
print("# files.download('model.zip')")

# Optional: Load model later
print("\n" + "=" * 50)
print("To use this model later, run:")
print("from transformers import T5Tokenizer, T5ForConditionalGeneration")
print("tokenizer = T5Tokenizer.from_pretrained('./t5-small-cnn-finetuned-final')")
print("model = T5ForConditionalGeneration.from_pretrained('./t5-small-cnn-finetuned-final')")
print("=" * 50)