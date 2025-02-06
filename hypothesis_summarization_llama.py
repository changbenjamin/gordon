import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from tqdm import tqdm
import torch
import time
from datasets import Dataset

# 1. Load the CSV file and extract the abstracts.
input_csv = "test_sample.csv"
df = pd.read_csv(input_csv)
print("Starting processing abstracts")
abstracts = df["abstract"].tolist()

# 2. Load the Llama-3 Chat model
model_name = "meta-llama/Llama-3.1-8B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
pipe = pipeline(
    "text-generation",
    model=model_name,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    tokenizer=tokenizer
)

prompt_template = ("Summarize the core hypothesis/question from the abstract as ONE very detailed and very scientific question. Just give ONE very scientific and detailed sentence which should start with ##QUESTION: [What, Why, Can, How, Where] and the response should end in ? ")

# 3. Create a list of prompts for all abstracts
prompts = [prompt_template + str(abstract) for abstract in abstracts]

# 4. Create a Hugging Face Dataset
dataset = Dataset.from_dict({"prompt": prompts})

# 5. Process abstracts using pipeline with the Dataset
print("Processing abstracts using pipeline with Dataset...")
start_time = time.time()  # Record start time here

outputs = pipe(dataset['prompt'], max_new_tokens=100, return_full_text=False)

end_time = time.time()    # Record end time here
processing_time = end_time - start_time # Calculate elapsed time
print(f"Dataset processed in: {processing_time:.2f} seconds") # Print the time

print("Dataset processed.\n")

hypothesis_summaries = []
for output in tqdm(outputs, desc="Extracting Hypothesis Summaries"): # Iterate through the OUTPUTS now, not abstracts
    full_response = output[0]["generated_text"].strip()
    summary = full_response

    start_marker = "#QUESTION:"
    end_marker = "?"

    start_index = full_response.find(start_marker)
    end_index = full_response.find(end_marker)

    if start_index != -1 and end_index != -1:
        summary = full_response[start_index + len(start_marker):end_index+1].strip()
    else:
        summary = full_response

    hypothesis_summaries.append(summary)
    print(f"Extracted hypothesis: {summary}\n")


# 6. Add the summaries to the DataFrame and save
df['hypothesis'] = hypothesis_summaries
output_csv = "generated_hypothesis_dataset.csv"
df.to_csv(output_csv, index=False)
print(f"\nNew CSV with hypothesis summaries saved as '{output_csv}'.")