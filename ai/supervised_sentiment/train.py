import torch
import pandas as pd
from datasets import Dataset
from transformers import (
    DistilBertTokenizerFast, 
    DistilBertForSequenceClassification, 
    Trainer, 
    TrainingArguments
)
from pathlib import Path

def train():
    # 1. Load your local data
    # Get the script's directory and construct absolute path
    script_dir = Path(__file__).parent
    data_path = script_dir / "../../data/processed/swt/reviews_balanced.json"
    df = pd.read_json(data_path)
    
    # Remove rows with missing text (412 reviews have null text)
    df = df.dropna(subset=['text']).reset_index(drop=True)
    
    # Rename 'label' to 'labels' (Trainer expects plural)
    df = df.rename(columns={'label': 'labels'})
    print(f"Loaded {len(df)} reviews (removed null texts)")
    print(f"Label distribution:\n{df['labels'].value_counts().sort_index()}")
    
    # Labels already in file: 0=negative, 1=neutral, 2=positive
    dataset = Dataset.from_pandas(df)

    # 2. Tokenizer (Fast version for your LOQ)
    tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')

    def tokenize_func(examples):
        return tokenizer(examples['text'], padding='max_length', truncation=True, max_length=128)

    tokenized_data = dataset.map(tokenize_func, batched=True)
    # Split: 80% train, 20% test
    split_data = tokenized_data.train_test_split(test_size=0.2)

    # 3. Model setup on GPU
    model = DistilBertForSequenceClassification.from_pretrained(
        'distilbert-base-uncased', 
        num_labels=3  # Ternary classification: 0=negative, 1=neutral, 2=positive
    ).to('cuda')

    # 4. RTX-Optimized Training Args
    checkpoint_dir = script_dir / "../../data/artifacts/checkpoints"
    training_args = TrainingArguments(
        output_dir=str(checkpoint_dir),
        eval_strategy="epoch",  # Changed from evaluation_strategy (deprecated)
        learning_rate=2e-5,
        per_device_train_batch_size=16, # Your 8GB VRAM can handle this easily
        num_train_epochs=3,
        weight_decay=0.01,
        fp16=True,             # ENABLE TENSOR CORES
        logging_steps=10,
        push_to_hub=False,
        report_to="none"       # Keeps it clean
    )

    # 5. The Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=split_data["train"],
        eval_dataset=split_data["test"],
    )

    # 6. Start the Burn
    print("🚀 LOQ GPU: Training Started...")
    trainer.train()

    # 7. Save to the artifact folder
    save_path = script_dir / "../../data/artifacts/sentiment_model"
    model.save_pretrained(str(save_path))
    tokenizer.save_pretrained(str(save_path))
    print(f"✅ Training Complete! Model saved to {save_path}")

if __name__ == "__main__":
    train()