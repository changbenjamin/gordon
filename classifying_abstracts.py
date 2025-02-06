import spacy
import pandas as pd
from spacy.language import Language
from spacy.tokens import Doc
from tqdm import tqdm  # Import tqdm

# Load the trained spaCy model from the saved directory
output_dir = "abstract_classifier"  # Replace with the actual path if different
try:
    nlp: Language = spacy.load(output_dir)
    print(f"Loaded model from directory: {output_dir}")
except OSError as e:
    print(f"Error loading model from directory: {output_dir}")
    print(e)
    exit()

# Load the CSV file into a pandas DataFrame
csv_file = "randomized_rcr_data.csv"  # Replace with the actual path to your CSV file
try:
    df = pd.read_csv(csv_file)
    print(f"Loaded CSV file: {csv_file}")
except FileNotFoundError:
    print(f"Error: CSV file not found: {csv_file}")
    exit()

# Function to split abstract into sentences and classify each sentence
def classify_abstract_sentences(abstract_text: str, nlp: Language) -> list:
    if not isinstance(abstract_text, str) or not abstract_text.strip():
        return []  # Return empty list for non-string or empty abstract

    # Split by ". "
    sentences = abstract_text.split('. ')

    classified_sentences = []
    for sentence in sentences:
        # Remove leading/trailing whitespace
        sentence = sentence.strip()
        if not sentence: # Skip empty sentences
            continue

        # **Still process each segment with spaCy for classification**
        doc_sentence: Doc = nlp(sentence)
        categories = doc_sentence.cats
        classified_sentences.append({'sentence': sentence, 'categories': categories})
    return classified_sentences

# Process each row in the DataFrame
background_sentences_list = [] # List to store background sentences for each abstract

for index, row in tqdm(df.iterrows(), total=len(df), desc="Processing Abstracts"): # Wrap df.iterrows() with tqdm
    abstract_text = row['abstract']
    if not isinstance(abstract_text, str):
        background_sentences_list.append("") # Handle non-string abstracts
        continue

    classified_sentences_data = classify_abstract_sentences(abstract_text, nlp)
    background_sentences_for_abstract = []
    last_background_sentence_index = -1 # Initialize to -1, indicating no background sentence found yet

    for i, item in enumerate(classified_sentences_data): # Enumerate to get sentence index
        sentence = item['sentence']
        categories = item['categories']

        # Check if 'background' category score is above a threshold (e.g., 0.5)
        background_score = categories.get('background', 0.0) # Default to 0 if 'background' not in cats
        if background_score > 0.5: # You can adjust this threshold
            last_background_sentence_index = i # Update the index of the last background sentence

    if last_background_sentence_index != -1: # If at least one background sentence was found
        # Take all sentences up to and including the last background sentence
        for i in range(last_background_sentence_index + 1):
            background_sentences_for_abstract.append(classified_sentences_data[i]['sentence'])

    # Join the background sentences for this abstract into a single string, **using ". " as separator**
    joined_background_sentences = ". ".join(background_sentences_for_abstract) # Changed to ". "
    background_sentences_list.append(joined_background_sentences)

# Add the 'background' sentences as a new column to the DataFrame
df['background'] = background_sentences_list

# Save the updated DataFrame to a new CSV file (or overwrite the original)
output_csv_file = "rcr_data_w_background.csv" # Choose a new name or use the original to overwrite
df.to_csv(output_csv_file, index=False)
print(f"Saved classified data to: {output_csv_file}")




# import spacy
# import pandas as pd
# from spacy.language import Language
# from spacy.tokens import Doc
# from tqdm import tqdm  # Import tqdm

# # Load the trained spaCy model from the saved directory
# output_dir = "abstract_classifier"  # Replace with the actual path if different
# try:
#     nlp: Language = spacy.load(output_dir)
#     print(f"Loaded model from directory: {output_dir}")
# except OSError as e:
#     print(f"Error loading model from directory: {output_dir}")
#     print(e)
#     exit()

# # Load the CSV file into a pandas DataFrame
# csv_file = "randomized_rcr_data.csv"  # Replace with the actual path to your CSV file
# try:
#     df = pd.read_csv(csv_file)
#     df = df[:50]
#     print(f"Loaded CSV file: {csv_file}")
# except FileNotFoundError:
#     print(f"Error: CSV file not found: {csv_file}")
#     exit()

# # Function to split abstract into sentences and classify each sentence
# def classify_abstract_sentences(abstract_text: str, nlp: Language) -> list:
#     if not isinstance(abstract_text, str) or not abstract_text.strip():
#         return []  # Return empty list for non-string or empty abstract

#     # Split by ". "
#     sentences = abstract_text.split('. ')

#     classified_sentences = []
#     for sentence in sentences:
#         # Remove leading/trailing whitespace
#         sentence = sentence.strip()
#         if not sentence: # Skip empty sentences
#             continue

#         # **Still process each segment with spaCy for classification**
#         doc_sentence: Doc = nlp(sentence)
#         categories = doc_sentence.cats
#         classified_sentences.append({'sentence': sentence, 'categories': categories})
#     return classified_sentences

# # Process each row in the DataFrame
# background_sentences_list = [] # List to store background sentences for each abstract

# for index, row in tqdm(df.iterrows(), total=len(df), desc="Processing Abstracts"): # Wrap df.iterrows() with tqdm
#     abstract_text = row['abstract']
#     if not isinstance(abstract_text, str):
#         background_sentences_list.append("") # Handle non-string abstracts
#         continue

#     classified_sentences_data = classify_abstract_sentences(abstract_text, nlp)
#     background_sentences_for_abstract = []

#     for item in classified_sentences_data:
#         sentence = item['sentence']
#         categories = item['categories']

#         # Check if 'background' category score is above a threshold (e.g., 0.5)
#         background_score = categories.get('background', 0.0) # Default to 0 if 'background' not in cats
#         if background_score > 0.5: # You can adjust this threshold
#             background_sentences_for_abstract.append(sentence)

#     # Join the background sentences for this abstract into a single string
#     joined_background_sentences = " ".join(background_sentences_for_abstract)
#     background_sentences_list.append(joined_background_sentences)

# # Add the 'background' sentences as a new column to the DataFrame
# df['background'] = background_sentences_list

# # Save the updated DataFrame to a new CSV file (or overwrite the original)
# output_csv_file = "rcr_data_w_background.csv" # Choose a new name or use the original to overwrite
# df.to_csv(output_csv_file, index=False)
# print(f"Saved classified data to: {output_csv_file}")