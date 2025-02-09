import spacy
import pandas as pd
from spacy.language import Language
from spacy.tokens import Doc
from tqdm import tqdm

# Putting the trained spaCy model to actually classify sentences now

# Load the trained spaCy model
output_dir = "abstract_classifier"
try:
    nlp: Language = spacy.load(output_dir)
    print(f"Loaded model from directory: {output_dir}")
except OSError as e:
    print(f"Error loading model from directory: {output_dir}")
    print(e)
    exit()

csv_file = "randomized_rcr_data.csv"
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

    # Split by ". " for each sentence... better than splitting via spaCy
    sentences = abstract_text.split('. ')

    classified_sentences = []
    for sentence in sentences:
        # Remove leading/trailing whitespace
        sentence = sentence.strip()
        if not sentence: # Skip empty sentences
            continue

        doc_sentence: Doc = nlp(sentence)
        categories = doc_sentence.cats
        classified_sentences.append({'sentence': sentence, 'categories': categories})
    return classified_sentences

# Process each row in the DataFrame
background_sentences_list = []

for index, row in tqdm(df.iterrows(), total=len(df), desc="Processing Abstracts"):
    abstract_text = row['abstract']
    if not isinstance(abstract_text, str):
        background_sentences_list.append("")
        continue

    classified_sentences_data = classify_abstract_sentences(abstract_text, nlp)
    background_sentences_for_abstract = []
    last_background_sentence_index = -1 # Initialize to -1, indicating no background sentence found yet

    for i, item in enumerate(classified_sentences_data): # Enumerate to get sentence index
        sentence = item['sentence']
        categories = item['categories']

        # Check if 'background' category score is 0.5 threshold.. works well enough
        background_score = categories.get('background', 0.0)
        if background_score > 0.5:
            last_background_sentence_index = i

    # Basically I assume that the first sentence(s) will be background, so will take all of the sentences from
    # the beginning to the last identified "background" sentence as background
    
    if last_background_sentence_index != -1:
        for i in range(last_background_sentence_index + 1):
            background_sentences_for_abstract.append(classified_sentences_data[i]['sentence'])

    # Join the background sentences for this abstract into a single string, **using ". " as separator**
    joined_background_sentences = ". ".join(background_sentences_for_abstract) # Changed to ". "
    background_sentences_list.append(joined_background_sentences)

# Add the 'background' sentences as a new column to the DataFrame
df['background'] = background_sentences_list

output_csv_file = "rcr_data_w_background.csv"
df.to_csv(output_csv_file, index=False)
print(f"Saved classified data to: {output_csv_file}")