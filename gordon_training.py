import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, roc_auc_score, mean_squared_error, accuracy_score
import numpy as np
import optuna

# Choose your embedding below:

test_proportion = 0.1

### LLAMA EMBEDDINGS
embedding_dim = 4096
background_pkl = 'llama_background_train.pkl'
hypothesis_pkl = 'llama_hypothesis_train.pkl'
model_embeddings = "llama"


# ### BIOBERT EMBEDDINGS
# embedding_dim = 1024
# background_pkl = 'biobert_background_train.pkl'
# hypothesis_pkl = 'biobert_hypothesis_train.pkl'
# model_embeddings = "BioBERT"


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

class RCRDataset(Dataset):
    def __init__(self, dataframe, stage='stage1'):
        self.data = dataframe
        self.stage = stage

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        background_embedding = torch.tensor(self.data.iloc[idx]['background_embedding'], dtype=torch.float32)
        hypothesis_embedding = torch.tensor(self.data.iloc[idx]['hypothesis_embedding'], dtype=torch.float32)
        combined_embedding = torch.cat((background_embedding, hypothesis_embedding), dim=0)

        if self.stage == 'stage1':
            label = torch.tensor(self.data.iloc[idx]['is_zero_rcr'], dtype=torch.float32)
            return combined_embedding, label
        elif self.stage == 'stage2':
            rcr_value = torch.tensor(self.data.iloc[idx]['rcr'], dtype=torch.float32)
            return combined_embedding, rcr_value
        else:
            raise ValueError("Invalid stage specified for dataset.")

# --- 2. Define Models ---
class Stage1Classifier(nn.Module):
    def __init__(self, input_size, hidden_size=128, dropout_rate=0.5):
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
    def __init__(self, input_size, hidden_size=128, dropout_rate=0.5):
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

# --- 3. Training and Evaluation Functions ---

def train_stage1(model, train_loader, val_loader, epochs=10, learning_rate=0.001, class_weights=None, weight_decay=1e-5, dropout_rate=0.5, patience=3):
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    best_val_f1 = -np.inf
    epochs_no_improve = 0
    best_model_state = None

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for embeddings, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(embeddings)
            outputs = outputs.view(-1)
            labels = labels.view(-1)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        val_loss, val_f1, val_auc, val_accuracy = evaluate_stage1(model, val_loader, output_csv=None)
        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss/len(train_loader):.4f}, Val Loss: {val_loss:.4f}, Val F1: {val_f1:.4f}, Val AUC: {val_auc:.4f}, Val Accuracy: {val_accuracy:.4f}")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            epochs_no_improve = 0
            best_model_state = model.state_dict()
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping triggered after epoch {epoch+1} (No improvement in val F1 for {patience} epochs).")
                model.load_state_dict(best_model_state)
                break

    return model


def evaluate_stage1(model, data_loader, output_csv=f'{model_embeddings}_stage1_test_predictions.csv'):
    model.eval()
    val_loss = 0.0
    all_preds = []
    all_labels = []
    criterion = nn.BCELoss()

    with torch.no_grad():
        for embeddings, labels in data_loader:
            outputs = model(embeddings)
            loss = criterion(outputs.squeeze(), labels)
            val_loss += loss.item()

            predictions = outputs.squeeze().cpu().numpy()
            all_preds.extend(predictions)
            all_labels.extend(labels.cpu().numpy())

    val_auc = roc_auc_score(all_labels, all_preds)
    binary_preds = (np.array(all_preds) > 0.5).astype(int)
    val_f1 = f1_score(all_labels, binary_preds)
    val_accuracy = accuracy_score(all_labels, binary_preds)

    if output_csv:
        predictions_df = pd.DataFrame({'predicted': all_preds, 'truth': all_labels, 'binary_predicted': binary_preds})
        predictions_df.to_csv(output_csv, index=False)
        print(f"Stage 1 predictions saved to {output_csv}")

    return val_loss/len(data_loader), val_f1, val_auc, val_accuracy

def train_stage2(model, train_loader, val_loader, epochs=10, learning_rate=0.001, weight_decay=1e-5, dropout_rate=0.5, patience=5):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    best_val_rmse = np.inf
    epochs_no_improve = 0
    best_model_state = None


    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for embeddings, rcr_values in train_loader:
            optimizer.zero_grad()
            outputs = model(embeddings)
            loss = criterion(outputs.squeeze(), rcr_values)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        val_loss, val_rmse = evaluate_stage2(model, val_loader, output_csv=None)
        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss/len(train_loader):.4f}, Val Loss: {val_loss:.4f}, Val RMSE: {val_rmse:.4f}")

        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            epochs_no_improve = 0
            best_model_state = model.state_dict()
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping triggered after epoch {epoch+1} (No improvement in val RMSE for {patience} epochs).")
                model.load_state_dict(best_model_state)
                break
    return model


def evaluate_stage2(model, data_loader, output_csv=f'{model_embeddings}_stage2_test_predictions.csv'):
    model.eval()
    val_loss = 0.0
    all_preds = []
    all_labels = []
    criterion = nn.MSELoss()

    with torch.no_grad():
        for embeddings, rcr_values in data_loader:
            outputs = model(embeddings)
            loss = criterion(outputs.squeeze(), rcr_values)
            val_loss += loss.item()
            all_preds.extend(outputs.squeeze().cpu().numpy())
            all_labels.extend(rcr_values.cpu().numpy())

    val_rmse = np.sqrt(mean_squared_error(all_labels, all_preds))

    if output_csv:
        predictions_df = pd.DataFrame({'predicted': all_preds, 'truth': all_labels})
        predictions_df.to_csv(output_csv, index=False)
        print(f"Stage 2 predictions saved to {output_csv}")

    return val_loss/len(data_loader), val_rmse

def evaluate_overall_performance(stage1_model, stage2_model, overall_test_loader, output_csv=f'{model_embeddings}_overall_test_predictions.csv'):
    stage1_model.eval()
    stage2_model.eval()
    all_overall_preds = []
    all_overall_labels = []

    with torch.no_grad():
        for embeddings, labels in overall_test_loader:
            stage1_outputs = stage1_model(embeddings)
            stage1_predictions_binary = (stage1_outputs > 0.5).int()

            stage2_inputs_embeddings = embeddings[stage1_predictions_binary.squeeze() == 1]
            stage1_non_zero_indices = (stage1_predictions_binary.squeeze() == 1).nonzero(as_tuple=True)[0]

            stage2_rcr_preds = torch.zeros(len(embeddings))
            if len(stage2_inputs_embeddings) > 0:
                stage2_outputs = stage2_model(stage2_inputs_embeddings)
                stage2_rcr_preds[stage1_non_zero_indices] = stage2_outputs.squeeze().cpu()

            overall_predictions = stage2_rcr_preds.numpy()
            all_overall_preds.extend(overall_predictions)
            all_overall_labels.extend(labels.cpu().numpy())

    overall_rmse = np.sqrt(mean_squared_error(all_overall_labels, all_overall_preds))
    print(f"Overall Test RMSE: {overall_rmse:.4f}")

    if output_csv:
        predictions_df = pd.DataFrame({'predicted': all_overall_preds, 'truth': all_overall_labels})
        predictions_df.to_csv(output_csv, index=False)
        print(f"Overall predictions saved to {output_csv}")
    return overall_rmse


# --- 4. Main Training and Evaluation with Hyperparameter Tuning ---

def objective_stage1(trial, train_loader, val_loader, class_weights_tensor):
    lr = trial.suggest_float('lr_stage1', 1e-5, 1e-3, log=True)
    weight_decay = trial.suggest_float('weight_decay_stage1', 1e-6, 1e-4, log=True)
    dropout_rate = trial.suggest_float('dropout_rate_stage1', 0.2, 0.6)
    hidden_size = trial.suggest_int('hidden_size_stage1', 64, 256)
    patience_stage1 = trial.suggest_int('patience_stage1', 2, 5)

    # Instantiate Stage 1 model with suggested hyperparameters
    input_size = embedding_dim * 2
    model = Stage1Classifier(input_size, hidden_size=hidden_size, dropout_rate=dropout_rate)

    # Train Stage 1 model
    trained_model = train_stage1(model, train_loader, val_loader, epochs=10, learning_rate=lr, class_weights=class_weights_tensor, weight_decay=weight_decay, dropout_rate=dropout_rate, patience=patience_stage1)
    val_loss, val_f1, val_auc, val_accuracy = evaluate_stage1(trained_model, val_loader)

    return val_f1

def objective_stage2(trial, train_loader, val_loader):
    # Hyperparameter suggestions for Stage 2
    lr_stage2 = trial.suggest_float('lr_stage2', 1e-4, 1e-2, log=True)
    weight_decay_stage2 = trial.suggest_float('weight_decay_stage2', 1e-5, 1e-3, log=True)
    dropout_rate_stage2 = trial.suggest_float('dropout_rate_stage2', 0.1, 0.5)
    hidden_size_stage2 = trial.suggest_int('hidden_size_stage2', 32, 128)
    patience_stage2 = trial.suggest_int('patience_stage2', 3, 7)

    # Instantiate Stage 2 model with suggested hyperparameters
    input_size = embedding_dim * 2
    model = Stage2Regressor(input_size, hidden_size=hidden_size_stage2, dropout_rate=dropout_rate_stage2)

    # Train Stage 2 model
    trained_model = train_stage2(model, train_loader, val_loader, epochs=25, learning_rate=lr_stage2, weight_decay=weight_decay_stage2, dropout_rate=dropout_rate_stage2, patience=patience_stage2)
    val_loss, val_rmse = evaluate_stage2(trained_model, val_loader)

    return val_loss


if __name__ == '__main__':
    print(f"Using {model_embeddings} Embeddings...")
    # --- Data Preparation ---
    processed_df = prepare_data(background_pkl, hypothesis_pkl, 'gordonramsay_data_processed.csv')
    processed_df = create_labels(processed_df)

    # --- Initial Overall Split ---
    train_val_stage1_df, overall_test_df = train_test_split(processed_df, test_size=test_proportion, random_state=42, stratify=processed_df['is_zero_rcr'])

    # --- Stage 1 Data Split ---
    train_df_stage1, temp_df_stage1 = train_test_split(train_val_stage1_df, test_size=0.3, random_state=42, stratify=train_val_stage1_df['is_zero_rcr'])
    val_df_stage1, test_df_stage1 = train_test_split(temp_df_stage1, test_size=0.5, random_state=42, stratify=temp_df_stage1['is_zero_rcr'])

    train_dataset_stage1 = RCRDataset(train_df_stage1, stage='stage1')
    val_dataset_stage1 = RCRDataset(val_df_stage1, stage='stage1')
    test_dataset_stage1 = RCRDataset(test_df_stage1, stage='stage1')
    overall_test_dataset_stage1 = RCRDataset(overall_test_df, stage='stage1')

    train_loader_stage1 = DataLoader(train_dataset_stage1, batch_size=32, shuffle=True)
    val_loader_stage1 = DataLoader(val_dataset_stage1, batch_size=32, shuffle=False)
    test_loader_stage1 = DataLoader(test_dataset_stage1, batch_size=32, shuffle=False)
    overall_test_loader_stage1 = DataLoader(overall_test_dataset_stage1, batch_size=32, shuffle=False)

    # Calculate class weights for Stage 1
    zero_count = train_df_stage1['is_zero_rcr'].value_counts()[0]
    one_count = train_df_stage1['is_zero_rcr'].value_counts()[1]
    total_count = zero_count + one_count
    class_weights_tensor = torch.tensor([total_count / zero_count if zero_count > 0 else 1.0,
                                        total_count / one_count if one_count > 0 else 1.0], dtype=torch.float32)
    print(f"Class Weights Tensor for Stage 1: {class_weights_tensor}")

    # --- Hyperparameter Tuning for Stage 1 ---
    study_stage1 = optuna.create_study(direction='maximize') # I've found that maximize actually works better here
    study_stage1.optimize(lambda trial: objective_stage1(trial, train_loader_stage1, val_loader_stage1, class_weights_tensor), n_trials=10)

    print("--- Stage 1 Hyperparameter Tuning Results ---")
    print("  Best Trial:")
    trial_stage1 = study_stage1.best_trial

    print(f"    Value (Validation F1): {trial_stage1.value}")
    print("    Params:")
    for key, value in trial_stage1.params.items():
        print(f"      {key}: {value}")

    best_params_stage1 = trial_stage1.params

    # --- Stage 1 Training with Best Hyperparameters ---
    print("\n--- Stage 1 Training with Best Hyperparameters ---")
    input_size = embedding_dim * 2
    best_stage1_model = Stage1Classifier(input_size, hidden_size=best_params_stage1['hidden_size_stage1'], dropout_rate=best_params_stage1['dropout_rate_stage1'])
    best_stage1_model = train_stage1(best_stage1_model, train_loader_stage1, val_loader_stage1, epochs=10, class_weights=class_weights_tensor,
                 learning_rate=best_params_stage1['lr_stage1'], weight_decay=best_params_stage1['weight_decay_stage1'],
                 dropout_rate=best_params_stage1['dropout_rate_stage1'], patience=best_params_stage1.get('patience_stage1', 3))

    print("--- Stage 1 Test Evaluation with Best Hyperparameters ---")
    test_loss_stage1, test_f1_stage1, test_auc_stage1, test_accuracy_stage1 = evaluate_stage1(best_stage1_model, test_loader_stage1)
    print(f"Stage 1 Test Loss: {test_loss_stage1:.4f}, Test F1: {test_f1_stage1:.4f}, Test AUC: {test_auc_stage1:.4f}, Test Accuracy: {test_accuracy_stage1:.4f}")


        # --- Save Stage 1 Model ---
    stage1_model_path = f"gordon_stage1_{model_embeddings}.pth"
    torch.save(best_stage1_model.state_dict(), stage1_model_path)
    print(f"Stage 1 model saved to {stage1_model_path}")


    # --- Prepare Data for Stage 2 (Non-Zero RCR only) ---
    non_zero_rcr_df_stage2 = train_val_stage1_df[train_val_stage1_df['rcr'] > 0].copy()

    train_df_stage2, temp_df_stage2 = train_test_split(non_zero_rcr_df_stage2, test_size=0.3, random_state=42)
    val_df_stage2, test_df_stage2 = train_test_split(temp_df_stage2, test_size=0.5, random_state=42)

    train_dataset_stage2 = RCRDataset(train_df_stage2, stage='stage2')
    val_dataset_stage2 = RCRDataset(val_df_stage2, stage='stage2')
    test_dataset_stage2 = RCRDataset(test_df_stage2, stage='stage2')
    overall_test_dataset_stage2 = RCRDataset(overall_test_df[overall_test_df['rcr'] > 0].copy(), stage='stage2')

    train_loader_stage2 = DataLoader(train_dataset_stage2, batch_size=32, shuffle=True)
    val_loader_stage2 = DataLoader(val_dataset_stage2, batch_size=32, shuffle=False)
    test_loader_stage2 = DataLoader(test_dataset_stage2, batch_size=32, shuffle=False)
    overall_test_loader_stage2 = DataLoader(overall_test_dataset_stage2, batch_size=32, shuffle=False)


    # --- Hyperparameter Tuning for Stage 2 ---
    study_stage2 = optuna.create_study(direction='minimize')
    study_stage2.optimize(lambda trial: objective_stage2(trial, train_loader_stage2, val_loader_stage2), n_trials=10)

    print("\n--- Stage 2 Hyperparameter Tuning Results ---")
    print("  Best Trial:")
    trial_stage2 = study_stage2.best_trial

    print(f"    Value (Validation RMSE): {trial_stage2.value}")
    print("    Params:")
    for key, value in trial_stage2.params.items():
        print(f"      {key}: {value}")

    best_params_stage2 = trial_stage2.params

    # --- Stage 2 Training with Best Hyperparameters ---
    print("\n--- Stage 2 Training with Best Hyperparameters ---")
    input_size = embedding_dim * 2
    best_stage2_model = Stage2Regressor(input_size, hidden_size=best_params_stage2['hidden_size_stage2'], dropout_rate=best_params_stage2['dropout_rate_stage2'])
    best_stage2_model = train_stage2(best_stage2_model, train_loader_stage2, val_loader_stage2, epochs=25,
                 learning_rate=best_params_stage2['lr_stage2'], weight_decay=best_params_stage2['weight_decay_stage2'],
                 dropout_rate=best_params_stage2['dropout_rate_stage2'], patience=best_params_stage2.get('patience_stage2', 5))


    print("--- Stage 2 Test Evaluation with Best Hyperparameters ---")
    test_loss_stage2, test_rmse_stage2 = evaluate_stage2(best_stage2_model, test_loader_stage2)
    print(f"Stage 2 Test Loss: {test_loss_stage2:.4f}, Test RMSE: {test_rmse_stage2:.4f}")


    # --- Save Stage 2 Model ---
    stage2_model_path = f"gordon_stage2_{model_embeddings}.pth"
    torch.save(best_stage2_model.state_dict(), stage2_model_path)
    print(f"Stage 2 model saved to {stage2_model_path}")


    # --- Overall Two-Stage Model Evaluation on 10% Overall Test Set ---
    overall_test_dataset = RCRDataset(overall_test_df, stage='stage2')
    overall_test_loader = DataLoader(overall_test_dataset, batch_size=32, shuffle=False)

    print(f"\n--- Overall Two-Stage Model Evaluation on {test_proportion*100}% with {model_embeddings} Test Set ---")
    overall_rmse = evaluate_overall_performance(best_stage1_model, best_stage2_model, overall_test_loader)
    print(f"Overall Test RMSE: {overall_rmse:.4f}")

    print("\n--- Two-Stage Model Training and Evaluation Complete ---")