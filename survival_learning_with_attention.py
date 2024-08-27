import json
import os
import torch
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from pycox.models import CoxPH
from pycox.evaluation import EvalSurv
from lifelines.statistics import logrank_test
from lifelines import KaplanMeierFitter
import matplotlib.pyplot as plt
import torchtuples as tt
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm 



#module load StdEnv/2020 gcc/9.3.0 opencv python/3.8  scipy-stack hdf5 geos/3.10.2 arrow/7.0.0


# Define the AdditiveAttention model with masking
class AdditiveAttention(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(AdditiveAttention, self).__init__()
        self.W1 = nn.Linear(input_dim, 1)
        
    def forward(self, x, mask=None):
        # x is of shape (batch_size, seq_len, input_dim)
        score=self.W1(x)
        #score = self.V(torch.tanh(self.W1(x) + self.W2(x)))  # (batch_size, seq_len, 1)

        if mask is not None:
            # Apply the mask before the softmax
            score = score.masked_fill(mask == 0, float('-inf'))

        attention_weights = torch.softmax(score, dim=1)  # (batch_size, seq_len, 1)
        context_vector = torch.sum(attention_weights * x, dim=1)  # (batch_size, input_dim)
        
        return context_vector  # (batch_size, input_dim)

# Define the complete model with attention and MLP
class AttentionMLPModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_nodes, out_features, batch_norm, dropout, output_bias):
        super(AttentionMLPModel, self).__init__()
        self.attention = AdditiveAttention(input_dim, hidden_dim)
        self.mlp = tt.practical.MLPVanilla(input_dim, num_nodes, out_features, batch_norm, dropout, output_bias=output_bias)

    def forward(self, x):
        print(x.shape)
        # Generate mask (batch_size, seq_len, 1) - assuming padded elements are 0
        mask = (x.sum(dim=2) != 0).unsqueeze(-1)  # shape (batch_size, seq_len, 1)

        # Apply the attention mechanism with mask
        #print(x.shape)
        x = self.attention(x, mask)  # Now x is (batch_size, input_dim)

        # Apply the MLP
        x = self.mlp(x)  # Final shape depends on out_features

        return x
    
# Set seeds for reproducibility
def set_seeds(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

set_seeds(42)

# Load the JSON file

json_file_path = 'svs_patient_map_DFS.json'
#json_file_path = 'svs_patient_map_DFS_clinical_labels.json'
#embeddings_folder = 'refined_embeddings_folder'
#embeddings_folder = 'refined_embeddings_folder'
#embeddings_folder = 'last_cluster_label_head_7'
#embeddings_folder = 'emb_1_refined'
#embeddings_folder = "last_cluster_label_head_1"
#embeddings_folder = "TCGA-HNSC-embeddings-flatten-per-patient"
embeddings_folder = "TCGA-HNSC-embeddings-small"

with open(json_file_path, 'r') as f:
    metadata = json.load(f)

# Prepare data
count = 0
data_list = []
for entry in metadata:
    if pd.isnull(entry['censoring']) or pd.isnull(entry['time_to_event']):
        continue
    file_name = entry['file_name'].replace('.svs', '.pt')
    file_path = os.path.join(embeddings_folder, file_name)
    if os.path.exists(file_path):  # Check if file exists
        embedding = torch.load(file_path, map_location='cpu').numpy().astype(np.float32)  # Load embedding as (n x 192)
        censoring = int(entry['censoring'])
        time_to_death = float(entry['time_to_event'])
        # Store the embedding and labels together in a dictionary
        data_list.append({'embedding': embedding, 'time': time_to_death, 'event': censoring})
        count += 1
        if count % 10 == 0:
            print(count)
    else:
        print(f"File {file_path} does not exist.")

print("done loading the data")
print("total slides is:", count)

# Convert data_list to DataFrame for easier handling of labels
labels_df = pd.DataFrame([{'time': item['time'], 'event': item['event']} for item in data_list])

# Separate events and non-events
events = labels_df[labels_df['event'] == 1].reset_index(drop=True)
non_events = labels_df[labels_df['event'] == 0].reset_index(drop=True)

# Shuffle the data
events = events.sample(frac=1, random_state=42).reset_index(drop=True)
non_events = non_events.sample(frac=1, random_state=42).reset_index(drop=True)

# Create stratified folds manually
folds = 5
non_event_folds = np.array_split(non_events, folds)

c_indices = []
logrank_p_values = []

# Define the neural network architecture
num_nodes = [192, 128, 64]  # Number of neurons in each hidden layer
hidden_dim = 256  # Hidden dim of attention mechanism 
out_features = 1  # Output dimension (single risk score)
batch_norm = True  # Use batch normalization
dropout = 0.40  # Dropout rate to prevent overfitting
output_bias = False  # Bias term in the output layer

for fold in range(folds):
    # Create training and test sets
    test_non_events = non_event_folds[fold]
    train_non_events = pd.concat([non_event_folds[i] for i in range(folds) if i != fold])

    # Use a distinct subset of events for the test set
    test_events = events.iloc[fold * (len(events) // folds):(fold + 1) * (len(events) // folds)]
    
    # Use the remaining events for the training set and resample
    remaining_events = pd.concat([events.iloc[:fold * (len(events) // folds)], events.iloc[(fold + 1) * (len(events) // folds):]])
    train_events = remaining_events.sample(len(train_non_events), replace=True, random_state=42)

    df_train = pd.concat([train_events, train_non_events])
    df_test = pd.concat([test_events, test_non_events])

    # Extract embeddings and labels from the data list
    X_train = [torch.tensor(data_list[i]['embedding']) for i in df_train.index]
    X_test = [torch.tensor(data_list[i]['embedding']) for i in df_test.index]
    y_train = (torch.tensor(df_train['time'].values, dtype=torch.float32), torch.tensor(df_train['event'].values, dtype=torch.float32))
    y_test = (torch.tensor(df_test['time'].values, dtype=torch.float32), torch.tensor(df_test['event'].values, dtype=torch.float32))
    
    # Pad sequences so that they all have the same length
    X_train_padded = pad_sequence(X_train, batch_first=True)
    X_test_padded = pad_sequence(X_test, batch_first=True)
    # Standardize the input data
    # (Standardization is done within the model, so we skip it here)

    #print(f"X_train_padded: {X_train_padded.shape}, X_test_padded: {X_test_padded.shape}, y_train: {len(y_train[0])}, y_test: {len(y_test[0])}")

    # Initialize the attention-based MLP model
    n_input=192
    attention_mlp_model = AttentionMLPModel(n_input, hidden_dim=hidden_dim, num_nodes=num_nodes, out_features=out_features, batch_norm=batch_norm, dropout=dropout, output_bias=output_bias)

    # Convert model to pycox compatible model
    model = CoxPH(attention_mlp_model, tt.optim.Adam(lr=1e-3))
    # Training
    batch_size = 64
    #epochs = 100
    epochs = 30
    #callbacks = [tt.callbacks.EarlyStopping()]
    callbacks = []
    verbose = False
    
    print(X_train_padded.shape)

    model.fit(X_train_padded, y_train, batch_size, epochs, callbacks, verbose, val_data=(X_test_padded, y_test))

    # Compute baseline hazards after training
    model.compute_baseline_hazards()

    # Calculate risk scores for the test set



    risk_scores = model.predict(X_test_padded).flatten()
    #assert 1==0

    # Move risk scores to CPU before converting to numpy
    risk_scores = risk_scores.cpu().numpy()

    # Determine threshold using the median risk score
    threshold = np.median(risk_scores)
    high_risk = risk_scores >= threshold
    low_risk = risk_scores < threshold

    # Evaluate the separation of the two groups using the log-rank test
    T_high, E_high = y_test[0][high_risk], y_test[1][high_risk]
    T_low, E_low = y_test[0][low_risk], y_test[1][low_risk]
    results = logrank_test(T_high, T_low, event_observed_A=E_high, event_observed_B=E_low)

    # Store the log-rank test p-value
    logrank_p_values.append(results.p_value)

    # Concordance Index for the fold

    #assert 1==0


    durations = y_test[0].cpu().numpy()
    events_np = y_test[1].cpu().numpy()


    surv = model.predict_surv_df(X_test_padded)
    ev = EvalSurv(surv, durations, events_np, censor_surv='km')
    c_index = ev.concordance_td('antolini')
    c_indices.append(c_index)

    # Plot Kaplan-Meier ccurves
    kmf_high = KaplanMeierFitter()
    kmf_low = KaplanMeierFitter()

    plt.figure(figsize=(10, 6))
    kmf_high.fit(T_high, E_high, label='High Risk')
    kmf_low.fit(T_low, E_low, label='Low Risk')
    kmf_high.plot_survival_function(ci_show=True)
    kmf_low.plot_survival_function(ci_show=True)
    plt.title(f'Kaplan-Meier Curves for Fold {fold+1} (p-value: {results.p_value:.4f})')
    plt.xlabel('Time')
    plt.ylabel('Survival Probability')
    plt.legend()

    # Save the plot
    plot_filename = f'Kaplan_Meier_Curve_Fold_{fold+1}.png'
    plt.savefig(plot_filename)
    plt.close()

    # Print number of data points with events and without events in this fold
    num_events_train = df_train['event'].sum()
    num_non_events_train = len(df_train) - num_events_train
    num_events_test = df_test['event'].sum()
    num_non_events_test = len(df_test) - num_events_test

    print(f'Fold {fold+1} - Training set: {num_events_train} events, {num_non_events_train} non-events')
    print(f'Fold {fold+1} - Test set: {num_events_test} events, {num_non_events_test} non-events')

# Average Concordance Index and average log-rank p-value
mean_c_index = np.mean(c_indices)
mean_logrank_p_value = np.mean(logrank_p_values)

print(f'5-Fold Cross-Validated Concordance Index: {mean_c_index:.4f}')
print(f'Average Log-Rank Test p-value: {mean_logrank_p_value:.7f}')