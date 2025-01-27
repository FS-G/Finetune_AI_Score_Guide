# from transformers import AutoTokenizer, AutoModelForSequenceClassification
# import torch.nn.functional as F  # Add this import for softmax
# import numpy as np

# # Load the saved model and tokenizer from the saved_model
# loaded_model = AutoModelForSequenceClassification.from_pretrained("saved_model")
# loaded_tokenizer = AutoTokenizer.from_pretrained("saved_model")








# # Function to make predictions and return probabilities
# def predict(text):
#     # Ensure the model is on the same device as the inputs
#     device = loaded_model.device  # Get the device where the model is loaded
#     inputs = loaded_tokenizer(text, return_tensors="pt", padding="max_length", truncation=True, max_length=128).to(device)
#     outputs = loaded_model(**inputs)
    
#     # Apply softmax to get probabilities
#     probabilities = F.softmax(outputs.logits, dim=-1).detach().cpu().numpy()
#     return {"AI": probabilities[0][0],"HUMAN": probabilities[0][1]}

# # Example usage
# user_input = "The food was really BAD"
# probabilities = predict(user_input)
# print(f"Probabilities: {probabilities}")

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch.nn.functional as F  # Add this import for softmax

# Load the saved model and tokenizer from the saved_model
loaded_model = AutoModelForSequenceClassification.from_pretrained("saved_model")
loaded_tokenizer = AutoTokenizer.from_pretrained("saved_model")

# Function to make predictions and return probabilities
def predict(text):
    # Ensure the model is on the same device as the inputs
    device = loaded_model.device  # Get the device where the model is loaded
    inputs = loaded_tokenizer(text, return_tensors="pt", padding="max_length", truncation=True, max_length=128).to(device)
    outputs = loaded_model(**inputs)
    
    # Apply softmax to get probabilities
    probabilities = F.softmax(outputs.logits, dim=-1).detach().cpu().numpy()[0]
    return {"AI": float(probabilities[0]), "HUMAN": float(probabilities[1])}

# Example usage
user_input = "The food was really BAD"
probabilities = predict(user_input)
print(f"Probabilities: {probabilities}")
