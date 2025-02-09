import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import AutoModel, AutoTokenizer
import pickle
import numpy as np


# # UNCOMMENT FOR LLAMA EMBEDDINGS

# model_name = "meta-llama/Llama-3.1-8B-Instruct"
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# tokenizer.pad_token = tokenizer.eos_token
# model = AutoModelForCausalLM.from_pretrained(model_name)

# device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
# model.to(device)
# model.eval()

# def get_embedding(text, tokenizer, model, device):
#     """Generates average-pooled embedding for a single text using Llama-3-8B."""
#     inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512).to(device)
#     with torch.no_grad():
#         outputs = model(**inputs, output_hidden_states=True)
#     last_hidden_states = outputs.hidden_states[-1]
#     attention_mask = inputs['attention_mask']
#     input_lengths = attention_mask.sum(dim=1)
#     sum_embeddings = torch.sum(last_hidden_states * attention_mask.unsqueeze(-1), dim=1)
#     average_pooled_embedding = sum_embeddings / input_lengths.unsqueeze(-1)
#     return average_pooled_embedding.cpu().numpy().flatten().tolist()



### UNCOMMENT FOR BIOBERT EMBEDDINGS

model_name = "dmis-lab/biobert-large-cased-v1.1"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

embedding_dim = 1024
model_embeddings = "BioBERT"

def get_embedding(text, tokenizer, model, device):
    """Generates CLS embedding for a single text using BioBERT."""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    cls_embedding_tensor = outputs.last_hidden_state[:, 0, :]
    return cls_embedding_tensor.cpu().numpy().flatten().tolist()


# The classifier and regressor hidden sizes need to be defined based on the saved checkpoint

classifier_hidden_size = 187
regressor_hidden_size = 88


class Stage1Classifier(nn.Module):
    def __init__(self, input_size, hidden_size=classifier_hidden_size, dropout_rate=0.5):
        super(Stage1Classifier, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.dropout1(out)
        out = self.fc2(out)
        out = self.sigmoid(out)
        return out

class Stage2Regressor(nn.Module):
    def __init__(self, input_size, hidden_size=regressor_hidden_size, dropout_rate=0.5):
        super(Stage2Regressor, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.dropout1(out)
        out = self.fc2(out)
        return out

def predict_rcr_two_stage(stage1_model, stage2_model, hypothesis_embedding, background_embedding):
    combined_embedding = torch.cat((torch.tensor(background_embedding).float(), torch.tensor(hypothesis_embedding).float()), dim=0).unsqueeze(0)
    combined_embedding = combined_embedding

    stage1_output = stage1_model(combined_embedding)
    stage1_prediction_binary = (stage1_output > 0.5).int()

    if stage1_prediction_binary.item() == 0:
        return 0.0
    else:
        stage2_output = stage2_model(combined_embedding)
        return stage2_output.item()


def get_rcr_for_hypothesis_background(hypothesis_text, background_text, stage1_model, stage2_model, tokenizer, model, device):
    """
    Generates embeddings for hypothesis and background, and predicts RCR using the two-stage model.
    """
    hypothesis_embedding_list = get_embedding(hypothesis_text, tokenizer, model, device)
    background_embedding_list = get_embedding(background_text, tokenizer, model, device)

    hypothesis_embedding = hypothesis_embedding_list
    background_embedding = background_embedding_list


    predicted_rcr = predict_rcr_two_stage(stage1_model, stage2_model, hypothesis_embedding, background_embedding)
    return predicted_rcr


if __name__ == '__main__':
    # --- Instantiate Stage 1 and Stage 2 Models ---
    input_size = embedding_dim * 2
    stage1_model = Stage1Classifier(input_size)
    stage2_model = Stage2Regressor(input_size)

    # --- Load Saved Stage 1 and Stage 2 Models ---
    stage1_model_path = f"gordonramsay_stage1_{model_embeddings}.pth"
    stage2_model_path = f"gordon_ramsay_stage2_{model_embeddings}.pth"

    stage1_model.load_state_dict(torch.load(stage1_model_path))
    stage2_model.load_state_dict(torch.load(stage2_model_path))
    stage1_model.eval()
    stage2_model.eval()
    print("Loaded Stage 1 and Stage 2 models with saved weights.")
    
    # --- Get Hypothesis and Background Text Input ---
    hypothesis1_text = input("Enter Hypothesis 1: ")
    background1_text = input("Enter background for Hypothesis 1: ")
    
    hypothesis2_text = input("Enter Hypothesis 2: ")
    background2_text = input("Enter background for Hypothesis 2: ")


    # --- Predict RCR for Input Hypothesis and Background ---
    predicted_rcr_score1 = get_rcr_for_hypothesis_background(hypothesis1_text, background1_text, stage1_model, stage2_model, tokenizer, model, device)
    predicted_rcr_score2 = get_rcr_for_hypothesis_background(hypothesis2_text, background2_text, stage1_model, stage2_model, tokenizer, model, device)

    if (predicted_rcr_score1 == 0) and (predicted_rcr_score2 == 0):
        print("Sorry, both of those hypotheses have a predicted RCR score of zero.")
    elif (predicted_rcr_score1 == predicted_rcr_score2):
        print("Both of those hypotheses have the same predicted RCR score!")
    elif (predicted_rcr_score1 > predicted_rcr_score2):
        print("Gordon predicts that Hypothesis 1 is stronger than Hypothesis 2!")
    elif (predicted_rcr_score2 > predicted_rcr_score1):
        print("Gordon predicts that Hypothesis 2 is stronger than Hypothesis 1!")