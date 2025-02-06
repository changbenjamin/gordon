import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
import pickle
import numpy as np

# --- 1. Load Llama-3-8B Model and Tokenizer ---
model_name = "meta-llama/Llama-3.1-8B-Instruct"  # Or "meta-llama/Llama-3-8B" if you prefer the base model
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(model_name)

# Move model to GPU if available
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)
model.eval()

embedding_dim = 4096 # Llama Embedding Dimension
model_embeddings = "llama" # For file paths


# --- 2. Model Definitions (Reusing from scoring model) ---
class Stage1Classifier(nn.Module):
    def __init__(self, input_size, hidden_size=121, dropout_rate=0.5): # Reusing best hidden_size from previous tuning
        super(Stage1Classifier, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(input_size, 1) # input_size here to fix potential error
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.dropout1(out)
        out = self.fc2(out)
        out = self.sigmoid(out)
        return out

class Stage2Regressor(nn.Module):
    def __init__(self, input_size, hidden_size=47, dropout_rate=0.5): # Reusing best hidden_size from previous tuning
        super(Stage2Regressor, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(input_size, hidden_size) # input_size here to fix potential error

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.dropout1(out)
        out = self.fc2(out)
        return out

# --- 3. Prediction Function using Two-Stage Scoring Model ---
def predict_rcr_two_stage(stage1_model, stage2_model, hypothesis_embedding, background_embedding):
    combined_embedding = torch.cat((torch.tensor(background_embedding).float(), torch.tensor(hypothesis_embedding).float()), dim=0).unsqueeze(0) # Prepare input
    combined_embedding = combined_embedding # No .unsqueeze(0) here

    stage1_output = stage1_model(combined_embedding)
    stage1_prediction_binary = (stage1_output > 0.5).int()

    if stage1_prediction_binary.item() == 0: # Stage 1 predicts zero RCR
        return 0.0
    else: # Stage 1 predicts non-zero RCR, use Stage 2
        stage2_output = stage2_model(combined_embedding)
        return stage2_output.item() # Return predicted RCR value

def get_llama_embedding(text, tokenizer, model, device):
    """Generates average-pooled embedding for a single text using Llama-3-8B."""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512).to(device)
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    last_hidden_states = outputs.hidden_states[-1]
    attention_mask = inputs['attention_mask']
    input_lengths = attention_mask.sum(dim=1)
    sum_embeddings = torch.sum(last_hidden_states * attention_mask.unsqueeze(-1), dim=1)
    average_pooled_embedding = sum_embeddings / input_lengths.unsqueeze(-1)
    return average_pooled_embedding.cpu().numpy().flatten().tolist() # Flatten to list

def get_rcr_for_hypothesis_background(hypothesis_text, background_text, stage1_model, stage2_model, tokenizer, model, device):
    """
    Generates Llama embeddings for hypothesis and background, and predicts RCR using the two-stage model.
    """
    hypothesis_embedding_list = get_llama_embedding(hypothesis_text, tokenizer, model, device)
    background_embedding_list = get_llama_embedding(background_text, tokenizer, model, device)

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
    stage2_model_path = f"gordonramsay_stage2_{model_embeddings}.pth"

    stage1_model.load_state_dict(torch.load(stage1_model_path))
    stage2_model.load_state_dict(torch.load(stage2_model_path))
    stage1_model.eval()
    stage2_model.eval()
    print("Loaded Stage 1 and Stage 2 models with saved weights.")


    # --- Get Hypothesis and Background Text Input ---
    hypothesis_text = input("Enter hypothesis text: ")
    background_text = input("Enter background text: ")

    # --- Predict RCR for Input Hypothesis and Background ---
    predicted_rcr_score = get_rcr_for_hypothesis_background(hypothesis_text, background_text, stage1_model, stage2_model, tokenizer, model, device)

    print(f"\n--- Predicted RCR Score ---")
    print(f"Hypothesis: {hypothesis_text}")
    print(f"Background: {background_text}")
    print(f"Predicted RCR: {predicted_rcr_score:.4f}")

    print("\n--- RCR Prediction Complete ---")