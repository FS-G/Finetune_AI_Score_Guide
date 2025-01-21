
# Fine-Tune and Inference - AI Score (MultiLingual)

This repository contains scripts to fine-tune a transformer model on a custom dataset and perform inference. The Model is multilingual and Supports both Chinese and English.

## Project Structure
- **finetune.py**: Script to fine-tune the model using a CSV dataset. The fine-tuned model is saved locally in the `saved_model` directory. The path to the CSV must be specified and it should have two columns i.e. text and label with labels marked as AI and HUMAN.
- **inference.py**: Script to perform inference using the fine-tuned model. Update the `user_input` variable in this file to input your desired text for prediction. You can later on create a RESTAPI for use in produciton.
- **requirements.txt**: Contains the list of dependencies required to run the scripts.

## Setup Instructions

### 1. Create a Virtual Environment
To keep the project dependencies isolated, create a virtual environment:

#### For Windows:
```bash
python -m venv venv
venv\Scripts\activate
```

#### For macOS/Linux:
```bash
python3 -m venv venv
source venv/bin/activate
```

### 2. Install Dependencies
Firstly clone the repository.
Once the virtual environment is activated, install the required dependencies:
```bash
pip install -r requirements.txt
```

### 3. Fine-Tune the Model
Prepare a CSV file with two columns: `text` (input text) and `label` (corresponding labels). Save the file in the same directory or provide its path.

Run the `finetune.py` script to train and save the model:
```bash
python finetune.py
```
The trained model will be saved in the `saved_model` directory.

### 4. Perform Inference
Update the `user_input` variable in the `inference.py` file with the text you want to predict.

Run the script to get predictions:
```bash
python inference.py
```



## Example CSV Format
The input CSV file should have the following format:
| text                   | label |
|------------------------|-------|
| This is a sample text. | AI    |
| Another example here.  | HUMAN |
```

The text within the "text" column can be chinese, but the column name and the label name should be in English, else we need to update t he finetune.py with the column and label names.