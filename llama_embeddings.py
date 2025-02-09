from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm

model_name = "meta-llama/Llama-3.1-8B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)

tokenizer.pad_token = tokenizer.eos_token
# print(f"Padding token set to: {tokenizer.pad_token}")

model = AutoModelForCausalLM.from_pretrained(model_name)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)
model.eval()

rcr_background = pd.read_csv("gordonramsay_data_processed.csv")
background = rcr_background["clean_hypothesis"].tolist()
pmids = rcr_background["PMID"].tolist()

def get_llama_embeddings_batched(texts, tokenizer, model, device):
    """
    Generates average-pooled embeddings for a batch of texts using Llama-3-8B.

    Args:
        texts (list of str): A list of input text summaries (batch).
        tokenizer (AutoTokenizer): The Llama-3-8B tokenizer.
        model (AutoModelForCausalLM): The Llama-3-8B model.
        device (torch.device): The device to run the model on (e.g., 'cuda' or 'cpu').

    Returns:
        torch.Tensor: A tensor of average-pooled embeddings for the batch, moved to CPU.
                       Shape: [batch_size, hidden_size]
    """
    inputs = tokenizer(texts,
                       return_tensors="pt",
                       truncation=True,
                       padding=True,
                       max_length=512)
    inputs = inputs.to(device)

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True) 

    last_hidden_states = outputs.hidden_states[-1] 

    attention_mask = inputs['attention_mask']
    input_lengths = attention_mask.sum(dim=1) 

    sum_embeddings = torch.sum(last_hidden_states * attention_mask.unsqueeze(-1), dim=1)
    average_pooled_embeddings = sum_embeddings / input_lengths.unsqueeze(-1)
    return average_pooled_embeddings.cpu() 

batch_size = 4

embedding_dict = {}
num_batches = len(background) // batch_size + (1 if len(background) % batch_size != 0 else 0)

with tqdm(total=num_batches, desc="Generating Llama Embeddings (Batched)") as pbar:
    for i in range(0, len(background), batch_size):
        batch_texts = background[i:i + batch_size]
        batch_pmids = pmids[i:i + batch_size]

        # --- Automated Verification (same as before) ---
        if i == 0:
            print("\n--- Automated Verification: PMID and Background Alignment for First Batch ---")
            for batch_index in range(len(batch_texts)):
                list_pmid = batch_pmids[batch_index]
                list_background = batch_texts[batch_index]
                df_pmid = rcr_background.iloc[batch_index]['PMID']
                df_background = rcr_background.iloc[batch_index]['clean_hypothesis']

                if df_pmid != list_pmid:
                    raise AssertionError(f"PMID Mismatch at batch index {batch_index}: DataFrame PMID '{df_pmid}' != List PMID '{list_pmid}'")
                if df_background != list_background:
                    df_background_preview = df_background[:50] + "..." if len(df_background) > 50 else df_background
                    list_background_preview = list_background[:50] + "..." if len(list_background) > 50 else list_background
                    raise AssertionError(f"Background Mismatch at batch index {batch_index}: DataFrame Background (starts with '{df_background_preview}') != List Background (starts with '{list_background_preview}')")
            print("--- Automated Verification Passed for First Batch ---")
        # --- End of Automated Verification ---


        try:
            batch_embeddings_tensor = get_llama_embeddings_batched(batch_texts, tokenizer, model, device)
            batch_embeddings_numpy = batch_embeddings_tensor.numpy()

            for j in range(len(batch_pmids)):
                pmid = batch_pmids[j]
                embedding = batch_embeddings_numpy[j]
                embedding_dict[pmid] = embedding
        except RuntimeError as e:
            print(f"RuntimeError encountered in batch starting at index {i}: {e}")
            for text_index_in_batch, text_summary in enumerate(batch_texts):
                tokenized_input = tokenizer(text_summary, return_tensors="pt", truncation=False, padding=False)
                print(f"  Text index in batch: {text_index_in_batch}")
                print(f"  PMID: {batch_pmids[text_index_in_batch]}")
                print(f"  Original text (truncated for display): {text_summary[:200]}...")
                print(f"  Length of original text: {len(text_summary)}")
                print(f"  Length of tokenized input (without truncation): {len(tokenized_input['input_ids'][0])}")
            raise e

        pbar.update(1)

output_file = "llama3.1_embeddings_hypothesis.pkl"
with open(output_file, 'wb') as f:
    pickle.dump(embedding_dict, f)

print(f"Batched Llama embeddings saved to {output_file}")

loaded_embedding_dict = {}
with open(output_file, 'rb') as f:
    loaded_embedding_dict = pickle.load(f)

print(f"Number of embeddings loaded: {len(loaded_embedding_dict)}")
example_pmid = pmids[0]
loaded_embedding = loaded_embedding_dict[example_pmid]
print(f"Shape of loaded embedding for {example_pmid}: {loaded_embedding.shape}")
print(f"Type of loaded embedding: {type(loaded_embedding)}")