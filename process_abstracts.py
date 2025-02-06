import nltk
nltk.download('punkt')
from nltk.tokenize import sent_tokenize
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch


abstract = """
The search for vaccines that protect from severe morbidity and mortality as a result of infection with severe acute respiratory syndrome coronavirus 2 (SARS-CoV-2), the virus that causes coronavirus disease 2019 (COVID-19) is a race against the clock and the virus. Several vaccine candidates are currently being tested in the clinic. Inactivated virus and recombinant protein vaccines can be safe options but may require adjuvants to induce robust immune responses efficiently. In this work we describe the use of a novel amphiphilic imidazoquinoline (IMDQ-PEG-CHOL) TLR7/8 adjuvant, consisting of an imidazoquinoline conjugated to the chain end of a cholesterol-poly(ethylene glycol) macromolecular amphiphile). This amphiphile is water soluble and exhibits massive translocation to lymph nodes upon local administration, likely through binding to albumin. IMDQ-PEG-CHOL is used to induce a protective immune response against SARS-CoV-2 after single vaccination with trimeric recombinant SARS-CoV-2 spike protein in the BALB/c mouse model. Inclusion of amphiphilic IMDQ-PEG-CHOL in the SARS-CoV-2 spike vaccine formulation resulted in enhanced immune cell recruitment and activation in the draining lymph node. IMDQ-PEG-CHOL has a better safety profile compared to native soluble IMDQ as the former induces a more localized immune response upon local injection, preventing systemic inflammation. Moreover, IMDQ-PEG-CHOL adjuvanted vaccine induced enhanced ELISA and in vitro microneutralization titers, and a more balanced IgG2a/IgG1 response. To correlate vaccine responses with control of virus replication in vivo, vaccinated mice were challenged with SARS-CoV-2 virus after being sensitized by intranasal adenovirus-mediated expression of the human angiotensin converting enzyme 2 (ACE2) gene. Animals vaccinated with trimeric recombinant spike protein vaccine without adjuvant had lung virus titers comparable to non-vaccinated control mice, whereas animals vaccinated with IMDQ-PEG-CHOL-adjuvanted vaccine controlled viral replication and infectious viruses could not be recovered from their lungs at day 4 post infection. In order to test whether IMDQ-PEG-CHOL could also be used to adjuvant vaccines currently licensed for use in humans, proof of concept was also provided by using the same IMDQ-PEG-CHOL to adjuvant human quadrivalent inactivated influenza virus split vaccine, which resulted in enhanced hemagglutination inhibition titers and a more balanced IgG2a/IgG1 antibody response. Enhanced influenza vaccine responses correlated with better virus control when mice were given a lethal influenza virus challenge. Our results underscore the potential use of IMDQ-PEG-CHOL as an adjuvant to achieve protection after single immunization with recombinant protein and inactivated vaccines against respiratory viruses, such as SARS-CoV-2 and influenza viruses.
"""

abstract_sentences = sent_tokenize(abstract)


# Load your fine-tuned MiniLM model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("your-fine-tuned-minilm-model")
model = AutoModelForSequenceClassification.from_pretrained("your-fine-tuned-minilm-model")

def classify_sentence(sentence):
    inputs = tokenizer(sentence, return_tensors="pt", truncation=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    predicted_class = torch.argmax(logits, dim=-1).item()
    # Map predicted_class to a label like "Background", "Hypothesis", or "Methods/Tools Based"
    label_map = {0: "Background", 1: "Hypothesis", 2: "Methods/Tools Based"}
    return label_map.get(predicted_class, "Unknown")

classifications = [classify_sentence(sentence) for sentence in abstract_sentences]




# from transformers import AutoTokenizer, AutoModelForCausalLM

# # Load model and tokenizer
# # tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
# # model = AutoModelForCausalLM.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")

# # Load model directly
# from transformers import AutoModel
# model = AutoModel.from_pretrained("allenai/scibert_scivocab_uncased")

# # Abstract to process
# abstract = """
# The search for vaccines that protect from severe morbidity and mortality as a result of infection with severe acute respiratory syndrome coronavirus 2 (SARS-CoV-2), the virus that causes coronavirus disease 2019 (COVID-19) is a race against the clock and the virus. Several vaccine candidates are currently being tested in the clinic. Inactivated virus and recombinant protein vaccines can be safe options but may require adjuvants to induce robust immune responses efficiently. In this work we describe the use of a novel amphiphilic imidazoquinoline (IMDQ-PEG-CHOL) TLR7/8 adjuvant, consisting of an imidazoquinoline conjugated to the chain end of a cholesterol-poly(ethylene glycol) macromolecular amphiphile). This amphiphile is water soluble and exhibits massive translocation to lymph nodes upon local administration, likely through binding to albumin. IMDQ-PEG-CHOL is used to induce a protective immune response against SARS-CoV-2 after single vaccination with trimeric recombinant SARS-CoV-2 spike protein in the BALB/c mouse model. Inclusion of amphiphilic IMDQ-PEG-CHOL in the SARS-CoV-2 spike vaccine formulation resulted in enhanced immune cell recruitment and activation in the draining lymph node. IMDQ-PEG-CHOL has a better safety profile compared to native soluble IMDQ as the former induces a more localized immune response upon local injection, preventing systemic inflammation. Moreover, IMDQ-PEG-CHOL adjuvanted vaccine induced enhanced ELISA and in vitro microneutralization titers, and a more balanced IgG2a/IgG1 response. To correlate vaccine responses with control of virus replication in vivo, vaccinated mice were challenged with SARS-CoV-2 virus after being sensitized by intranasal adenovirus-mediated expression of the human angiotensin converting enzyme 2 (ACE2) gene. Animals vaccinated with trimeric recombinant spike protein vaccine without adjuvant had lung virus titers comparable to non-vaccinated control mice, whereas animals vaccinated with IMDQ-PEG-CHOL-adjuvanted vaccine controlled viral replication and infectious viruses could not be recovered from their lungs at day 4 post infection. In order to test whether IMDQ-PEG-CHOL could also be used to adjuvant vaccines currently licensed for use in humans, proof of concept was also provided by using the same IMDQ-PEG-CHOL to adjuvant human quadrivalent inactivated influenza virus split vaccine, which resulted in enhanced hemagglutination inhibition titers and a more balanced IgG2a/IgG1 antibody response. Enhanced influenza vaccine responses correlated with better virus control when mice were given a lethal influenza virus challenge. Our results underscore the potential use of IMDQ-PEG-CHOL as an adjuvant to achieve protection after single immunization with recombinant protein and inactivated vaccines against respiratory viruses, such as SARS-CoV-2 and influenza viruses.
# """

# # Function to generate the output dictionary
# def extract_background_and_hypothesis(abstract):
#     # Prompt for background extraction
#     background_prompt = f"""
#     Read the following scientific abstract and extract the background section. Simply extract word-for-word the background literature review and problem statement section, not including what the results or hypothesis is.
    
#     Abstract: 
#     {abstract}

#     Background:
#     """
#     inputs_background = tokenizer(background_prompt, return_tensors="pt", truncation=True, max_length=512)
#     outputs_background = model.generate(**inputs_background, max_length=512)
#     background = tokenizer.decode(outputs_background[0], skip_special_tokens=True)
    
#     # Prompt for hypothesis extraction
#     hypothesis_prompt = f"""
#     Read the following scientific abstract and summarize the main hypothesis or research question of the study in one sentence. The hypothesis should describe what the study is attempting to prove or investigate. Avoid including the results or methods—focus purely on the central claim or proposition the study tests or addresses.

#     Abstract: 
#     {abstract}

#     Hypothesis:
#     """
#     inputs_hypothesis = tokenizer(hypothesis_prompt, return_tensors="pt", truncation=True, max_length=512)
#     outputs_hypothesis = model.generate(**inputs_hypothesis, max_length=512)
#     hypothesis = tokenizer.decode(outputs_hypothesis[0], skip_special_tokens=True)
    
#     # Prompt for determining if it's methods/tool based
#     methods_prompt = f"""
#     Based on the following abstract, determine if the paper primarily focuses on new methods, techniques, or tools. Answer with "Yes" if it's a methods or tool-based paper, "Unclear" if it's unclear, and "No" if the paper does not focus on methods/tools. Only return one word: "Yes" "Unclear" or "No".
    
#     Abstract: {abstract}
    
#     Methods/Tools Based: 
#     """
#     inputs_methods = tokenizer(methods_prompt, return_tensors="pt", truncation=True, max_length=512)
#     outputs_methods = model.generate(**inputs_methods, max_length=512)
#     methods_based = tokenizer.decode(outputs_methods[0], skip_special_tokens=True)
    
#     # Return as dictionary
#     return {"Background": background, "Hypothesis": hypothesis, "Methods/Tools Based": methods_based}

# # Test the function with the provided abstract
# result = extract_background_and_hypothesis(abstract)

# # Print the result
# print(result)