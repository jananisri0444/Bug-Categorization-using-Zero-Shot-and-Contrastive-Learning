import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration

# Load the T5 model and tokenizer
model_name = "google/flan-t5-large"  
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

# Load extracted keywords
with open("top_keywords.txt", "r") as f:
    keywords = [line.strip() for line in f.readlines()]

# Function to classify keywords
def classify_keyword(keyword):
    prompt = f"Classify the software bug-related keyword '{keyword}' into a relevant category."
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids

    # Generate category name
    with torch.no_grad():
        output_ids = model.generate(input_ids)
    category = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return category

# Classify all extracted keywords
keyword_category_mapping = {keyword: classify_keyword(keyword) for keyword in keywords}

# Save categorized keywords
import json
with open("keyword_categories.json", "w") as f:
    json.dump(keyword_category_mapping, f, indent=4)

print("✔️ Categorized keywords using Google T5 model!")

import json

# Load keyword-category mapping
with open("keyword_categories.json", "r") as f:
    keyword_category_mapping = json.load(f)

# Function to categorize bug reports based on extracted keywords
def categorize_bug(description):
    words = description.lower().split()  # Tokenize the description
    assigned_categories = []

    for word in words:
        if word in keyword_category_mapping:
            assigned_categories.append(keyword_category_mapping[word])

    if assigned_categories:
        return max(set(assigned_categories), key=assigned_categories.count)  # Most frequent category
    return "Uncategorized"

# Apply categorization to each bug report
df["Predicted Category"] = df["Short Description"].apply(categorize_bug)

# Save categorized bug reports
df.to_csv("categorized_bug_reports.csv", index=False)

print("✔️ Bug reports categorized based on extracted keywords!")
