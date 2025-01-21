
# Import necessary modules
import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
import numpy as np
import evaluate

# Load the dataset from a CSV file
csv_file = "ai_human_text_labels_chinese.csv"  # Replace with your CSV file path
data = pd.read_csv(csv_file)

# map the labels
data.label = data.label.map({'Human': 1, 'AI': 0}) # change as per the exact spelling of the labels even if they are in chinese

# Convert the data to a HuggingFace Dataset format
dataset = Dataset.from_pandas(data)

# Load tokenizer and define a tokenization function
tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

# Tokenize dataset
tokenized_datasets = dataset.map(tokenize_function, batched=True)

# Prepare training and evaluation datasets
train_dataset, eval_dataset = tokenized_datasets.train_test_split(test_size=0.2, seed=42).values()

# Load model with a classification head
model = AutoModelForSequenceClassification.from_pretrained("xlm-roberta-base", num_labels=2)

# Load accuracy metric
metric = evaluate.load("accuracy")
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

# Define training arguments
training_args = TrainingArguments(
    num_train_epochs=1,  # use a bigger number
    output_dir="test_trainer",
    evaluation_strategy="epoch",
    report_to=None,  # Prevents logging to external trackers like WandB
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    compute_metrics=compute_metrics,
)

# Train the model
trainer.train()

# Function to make predictions
def predict(text):
    # Ensure the model is on the same device as the inputs
    device = model.device  # Get the device where the model is loaded
    inputs = tokenizer(text, return_tensors="pt", padding="max_length", truncation=True, max_length=128).to(device)
    outputs = model(**inputs)
    predictions = outputs.logits.argmax(dim=-1).item()
    return predictions

# usage 
text = "The food was really BAD"
prediction = predict(text)
print(f"Predicted Label: {prediction}")

# Save the model and tokenizer
# this will save the model to the saved_model directory
model.save_pretrained("saved_model")
tokenizer.save_pretrained("saved_model")







