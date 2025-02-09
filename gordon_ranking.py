import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle
import time

# Choose your embedding below:

# ### LLAMA EMBEDDINGS
# embedding_dim = 1024
# background_pkl = 'llama_background_train.pkl'
# hypothesis_pkl = 'llama_hypothesis_train.pkl'
# model_embeddings = "llama"


### BIOBERT EMBEDDINGS
embedding_dim = 1024
background_pkl = 'biobert_background_train.pkl'
hypothesis_pkl = 'biobert_hypothesis_train.pkl'
model_embeddings = "BioBERT"


# ### RANDOM EMBEDDINGS
# embedding_dim = 1024
# background_pkl = 'random_background_train.pkl'
# hypothesis_pkl = 'random_hypothesis_train.pkl'
# model_embeddings = "random"



# --- 1. Data Loading and Preprocessing ---
def load_embeddings(embeddings_file):
    with open(embeddings_file, 'rb') as f:
        embeddings_dict = pickle.load(f)
    return embeddings_dict

def load_rcr_data(rcr_file):
    rcr_df = pd.read_csv(rcr_file)
    return rcr_df

def prepare_data(background_embeddings_file, hypothesis_embeddings_file, rcr_file):
    background_embeddings = load_embeddings(background_embeddings_file)
    hypothesis_embeddings = load_embeddings(hypothesis_embeddings_file)
    rcr_df = load_rcr_data(rcr_file)

    data = []
    for index, row in rcr_df.iterrows():
        pmid = int(row['PMID'])
        rcr = row['relative_citation_ratio']
        if pmid in background_embeddings and pmid in hypothesis_embeddings:
            background_embedding = background_embeddings[pmid]
            hypothesis_embedding = hypothesis_embeddings[pmid]
            data.append({'pmid': pmid,
                         'background_embedding': background_embedding,
                         'hypothesis_embedding': hypothesis_embedding,
                         'rcr': rcr})
    return pd.DataFrame(data)

def create_labels(df):
    df['is_zero_rcr'] = (df['rcr'] == 0).astype(int)
    return df

# --- 2. Model Definitions  ---
class Stage1Classifier(nn.Module):
    def __init__(self, input_size, hidden_size=239, dropout_rate=0.5):
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
    def __init__(self, input_size, hidden_size=64, dropout_rate=0.5):
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

# --- 3. Prediction Function using Two-Stage Scoring Model ---
def predict_rcr_two_stage(stage1_model, stage2_model, hypothesis_embedding, background_embedding):
    combined_embedding = torch.cat((torch.tensor(background_embedding).float(), torch.tensor(hypothesis_embedding).float()), dim=0).unsqueeze(0)

    stage1_output = stage1_model(combined_embedding)
    stage1_prediction_binary = (stage1_output > 0.5).int()

    if stage1_prediction_binary.item() == 0:
        return 0.0
    else:
        stage2_output = stage2_model(combined_embedding)
        return stage2_output.item()

if __name__ == '__main__':
    print(f"Evaluating Scoring Model as Ranking Model with {model_embeddings} Embeddings...")
    background_pkl = background_pkl
    hypothesis_pkl = hypothesis_pkl
    rcr_file = 'gordonramsay_data_processed.csv'

    processed_df = prepare_data(background_pkl, hypothesis_pkl, rcr_file)
    processed_df = create_labels(processed_df)

    train_val_stage1_df, overall_test_df = train_test_split(processed_df, test_size=0.1, random_state=42, stratify=processed_df['is_zero_rcr'])

    test_ranking_pairs = []
    rcr_threshold_eval = 1.0

    start_time_pairs = time.time()
    pair_count = 0
    papers_processed_count = 0
    total_papers_test = len(overall_test_df)
    print("Starting test pair creation...")

    for index_a, paper_a in overall_test_df.iterrows():
        papers_processed_count += 1

        for index_b, paper_b in overall_test_df.iterrows():
            if index_a == index_b:
                continue
            rcr_diff = abs(paper_a['rcr'] - paper_b['rcr'])
            if rcr_diff >= rcr_threshold_eval:
                if paper_a['rcr'] > paper_b['rcr']:
                    better_paper = paper_a
                    worse_paper = paper_b
                    label = 1 # Paper A is better
                else:
                    better_paper = paper_b
                    worse_paper = paper_a
                    label = 0 # Paper B is better
                test_ranking_pairs.append({
                    'paper_a': paper_a,
                    'paper_b': paper_b,
                    'ranking_label': label 
                })
                pair_count += 1
        if papers_processed_count % 10 == 0:
            elapsed_time = time.time() - start_time_pairs
            print(f"Processed {papers_processed_count+1}/{total_papers_test} papers, {pair_count} pairs created so far... ({elapsed_time:.2f} seconds)")

    test_ranking_pairs_df = pd.DataFrame(test_ranking_pairs)
    if test_ranking_pairs_df.empty:
        print("Warning: No test ranking pairs were created with RCR difference >", rcr_threshold_eval)

    print(f"Number of test ranking pairs with RCR difference > {rcr_threshold_eval}: {len(test_ranking_pairs_df)}")

    # --- Load Saved Stage 1 and Stage 2 Models ---
    stage1_model = Stage1Classifier(embedding_dim * 2)
    stage2_model = Stage2Regressor(embedding_dim * 2)

    stage1_model_path = f"gordon_stage1_{model_embeddings}.pth"
    stage2_model_path = f"gordon_stage2_{model_embeddings}.pth"

    stage1_model.load_state_dict(torch.load(stage1_model_path))
    stage2_model.load_state_dict(torch.load(stage2_model_path))
    stage1_model.eval()
    stage2_model.eval()


    # --- Predict Rankings and Evaluate Pairwise Accuracy ---
    correct_predictions = 0
    total_pairs = len(test_ranking_pairs_df)

    with torch.no_grad():
        for index, pair_row in test_ranking_pairs_df.iterrows():
            paper_a = pair_row['paper_a']
            paper_b = pair_row['paper_b']
            true_ranking_label = pair_row['ranking_label']

            # Get predicted RCR scores from scoring model
            predicted_rcr_a = predict_rcr_two_stage(stage1_model, stage2_model, paper_a['hypothesis_embedding'], paper_a['background_embedding'])
            predicted_rcr_b = predict_rcr_two_stage(stage1_model, stage2_model, paper_b['hypothesis_embedding'], paper_b['background_embedding'])

            if predicted_rcr_a > predicted_rcr_b:
                predicted_ranking_label = 0
            else:
                predicted_ranking_label = 1

            if predicted_ranking_label == true_ranking_label:
                correct_predictions += 1

    pairwise_accuracy = correct_predictions / total_pairs
    print(f"\nPairwise Accuracy of Scoring Model as Ranking Model (RCR diff > {rcr_threshold_eval}): {pairwise_accuracy:.6f}")

    print("\n--- Scoring Model as Ranking Model Evaluation Complete ---")