from transformers import AutoModel, AutoTokenizer
import torch
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm

# Load model and tokenizer directly
model_name = "dmis-lab/biobert-large-cased-v1.1"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# Move model to GPU
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu") # Use GPU if available, otherwise CPU
model.to(device)
model.eval()

rcr_background = pd.read_csv("gordonramsay_data_processed.csv")
background = rcr_background["clean_hypothesis"].tolist()
pmids = rcr_background["PMID"].tolist()



def get_cls_embeddings_batched(texts, tokenizer, model, device):
    """
    Generates CLS embeddings for a batch of texts using BioBERT on the specified device.

    Args:
        texts (list of str): A list of input text summaries (batch).
        tokenizer (AutoTokenizer): The BioBERT tokenizer.
        model (AutoModel): The BioBERT model.
        device (torch.device): The device to run the model on (e.g., 'cuda' or 'cpu').

    Returns:
        torch.Tensor: A tensor of CLS embeddings for the batch, moved to CPU.
                       Shape: [batch_size, hidden_size]
    """
    # Tokenize the batch of texts, padding and truncation are handled automatically
    # Explicitly set max_length and truncation strategy
    inputs = tokenizer(texts,
                       return_tensors="pt",
                       truncation=True,
                       padding=True,
                       max_length=512) # Explicitly set max_length to 512
    inputs = inputs.to(device)

    # Check for any input_ids that are still longer than 512 (for debugging)
    for input_ids_example in inputs['input_ids']:
        if len(input_ids_example) > 512:
            print("WARNING: Input sequence is still longer than 512 after tokenization and truncation!")
            print(f"Length: {len(input_ids_example)}")


    # Get model outputs
    with torch.no_grad():
        outputs = model(**inputs)

    # CLS embeddings for the entire batch
    cls_embeddings = outputs.last_hidden_state[:, 0, :]  # Shape: [batch_size, hidden_size]

    return cls_embeddings.cpu() # Move embeddings back to CPU


# Batch size - adjust this based on your GPU memory. Start with 32 or 64 and increase if possible.
batch_size = 32

embedding_dict = {}
num_batches = len(background) // batch_size + (1 if len(background) % batch_size != 0 else 0) # Calculate number of batches for tqdm

with tqdm(total=num_batches, desc="Generating Embeddings (Batched)") as pbar:
    for i in range(0, len(background), batch_size):
        batch_texts = background[i:i + batch_size]
        batch_pmids = pmids[i:i + batch_size] # Get corresponding PMIDs for the batch

        # --- Automated Verification: Check Alignment for the First Batch ---
        if i == 0: # Perform verification only for the first batch
            print("\n--- Automated Verification: PMID and Background Alignment for First Batch ---")
            for batch_index in range(len(batch_texts)):
                list_pmid = batch_pmids[batch_index]
                list_background = batch_texts[batch_index]
                df_pmid = rcr_background.iloc[batch_index]['PMID'] # Access DataFrame using .iloc
                df_background = rcr_background.iloc[batch_index]['clean_hypothesis'] # Access DataFrame using .iloc

                if df_pmid != list_pmid:
                    raise AssertionError(f"PMID Mismatch at batch index {batch_index}: DataFrame PMID '{df_pmid}' != List PMID '{list_pmid}'")
                if df_background != list_background:
                    df_background_preview = df_background[:50] + "..." if len(df_background) > 50 else df_background
                    list_background_preview = list_background[:50] + "..." if len(list_background) > 50 else list_background
                    raise AssertionError(f"Background Mismatch at batch index {batch_index}: DataFrame Background (starts with '{df_background_preview}') != List Background (starts with '{list_background_preview}')")
            print("--- Automated Verification Passed for First Batch ---")
        # --- End of Automated Verification ---


        try: # Added try-except block to catch the RuntimeError and print more info
            batch_embeddings_tensor = get_cls_embeddings_batched(batch_texts, tokenizer, model, device)
            batch_embeddings_numpy = batch_embeddings_tensor.numpy() # Convert batch tensor to numpy

            # Store embeddings in the dictionary, mapping PMID to embedding
            for j in range(len(batch_pmids)):
                pmid = batch_pmids[j]
                embedding = batch_embeddings_numpy[j]
                embedding_dict[pmid] = embedding
        except RuntimeError as e:
            print(f"RuntimeError encountered in batch starting at index {i}: {e}")
            for text_index_in_batch, text_summary in enumerate(batch_texts):
                tokenized_input = tokenizer(text_summary, return_tensors="pt", truncation=False, padding=False) # Tokenize without truncation to see full length
                print(f"  Text index in batch: {text_index_in_batch}")
                print(f"  PMID: {batch_pmids[text_index_in_batch]}")
                print(f"  Original text (truncated for display): {text_summary[:200]}...")
                print(f"  Length of original text: {len(text_summary)}")
                print(f"  Length of tokenized input (without truncation): {len(tokenized_input['input_ids'][0])}") # Length BEFORE truncation
            raise e # Re-raise the exception to stop the script

        pbar.update(1) # Update progress bar after each batch

# Save the dictionary to a pickle file
output_file = "biobert_embeddings_hypothesis.pkl"
with open(output_file, 'wb') as f:
    pickle.dump(embedding_dict, f)

print(f"Batched embeddings saved to {output_file}")


# --- Code to load the embeddings later ---
# Load the embeddings from the pickle file
loaded_embedding_dict = {}
with open(output_file, 'rb') as f:
    loaded_embedding_dict = pickle.load(f)

print(f"Number of embeddings loaded: {len(loaded_embedding_dict)}")
example_pmid = pmids[0] # Example PMID to retrieve
loaded_embedding = loaded_embedding_dict[example_pmid]
print(f"Shape of loaded embedding for {example_pmid}: {loaded_embedding.shape}")
print(f"Type of loaded embedding: {type(loaded_embedding)}")
