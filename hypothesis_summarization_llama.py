import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from tqdm import tqdm
import torch
import time
from datasets import Dataset

# 1. Load the CSV file and extract the abstracts.
input_csv = "clean_randomized_rcr_data.csv"
df = pd.read_csv(input_csv)
# df = df[:500] # for testing, remove to process all
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
output_csv = "rcr_data_w_hypothesis_dataset9.csv"
df.to_csv(output_csv, index=False)
print(f"\nNew CSV with hypothesis summaries saved as '{output_csv}'.")


# import pandas as pd
# from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
# from tqdm import tqdm
# import torch

# # 1. Load the CSV file and extract the abstracts.
# input_csv = "clean_randomized_rcr_data.csv"
# df = pd.read_csv(input_csv)
# # df = df[:50]
# abstracts = df["abstract"].tolist()

# # 2. Load the Llama-3 Chat model in mixed precision for your H100.
# model_name = "meta-llama/Llama-3.1-8B-Instruct"
# tokenizer = AutoTokenizer.from_pretrained(model_name)

# # Ensure a pad token is set (if not, set it to the eos_token)
# if tokenizer.pad_token is None:
#     tokenizer.pad_token = tokenizer.eos_token

# pipe = pipeline(
#     "text-generation",
#     model=model_name,
#     torch_dtype=torch.bfloat16, # Recommended for H100 for faster and memory-efficient computation
#     device_map="auto", # Automatically utilize available GPUs (including your H100)
#     tokenizer=tokenizer
# )

# # prompt_template = ("Summarize the core hypothesis/question from this abstract into ONE scientific sentence that captures the main hypothesis/question. Do not have any intro fluff in your sentence, just give the pure hypothesis/question: ")

# prompt_template = ("Summarize the core hypothesis/question from the abstract as ONE very detailed and very scientific question. Just give ONE very scientific and detailed sentence which should start with ##QUESTION: [What, Why, Can, How, Where] and the response should end in ?")

# # 5. Process abstracts one by one.
# hypothesis_summaries = []

# for abstract in tqdm(abstracts, desc="Processing abstracts"):
#     # Create a prompt string for the current abstract.
#     prompt = prompt_template + str(abstract)

#     # Generate response for the single abstract.
#     outputs = pipe(prompt, max_new_tokens=100, return_full_text=False) # Increased max_new_tokens for detailed question

#     # Extract the generated response.
#     full_response = outputs[0]["generated_text"].strip()
#     summary = full_response  # Default to full response

#     start_marker = "#QUESTION:"
#     end_marker = "?"

#     start_index = full_response.find(start_marker)
#     end_index = full_response.find(end_marker)

#     if start_index != -1 and end_index != -1:
#         # Extract the substring between "##QUESTION:" and "?" (inclusive of "?")
#         summary = full_response[start_index + len(start_marker):end_index+1].strip()
#     else:
#         # print(f"Markers '##QUESTION:' or '?' not found in response: {full_response}. Using full response.")
#         summary = full_response # Fallback to full response if markers are not found

#     print(f"Extracted hypothesis: {summary}\n")
#     hypothesis_summaries.append(summary)

# # 6. Add the summaries as a new column to the DataFrame.
# df['hypothesis'] = hypothesis_summaries

# # 7. Write the updated DataFrame to a new CSV file.
# output_csv = "rcr_data_w_hypothesis5.csv"
# df.to_csv(output_csv, index=False)
# print(f"\nNew CSV with hypothesis summaries has been saved as '{output_csv}'.")