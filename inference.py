from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Load the saved model and tokenizer from the saved_model
loaded_model = AutoModelForSequenceClassification.from_pretrained("saved_model")
loaded_tokenizer = AutoTokenizer.from_pretrained("saved_model")

def predict(text):
    # Ensure the model is on the same device as the inputs
    device = loaded_model.device  # Get the device where the model is loaded
    inputs = loaded_tokenizer(text, return_tensors="pt", padding="max_length", truncation=True, max_length=128).to(device)
    outputs = loaded_model(**inputs)
    predictions = outputs.logits.argmax(dim=-1).item()
    return predictions

# Example usage
user_input = "The food was really BAD"
prediction = predict(user_input)
print(f"Predicted Label: {prediction}")