import numpy as np

# Calculate Keyword Overlap Accuracy (How many reports contain strong category words)
def keyword_overlap_accuracy(row):
    description = str(row["Short Description"]).lower()
    category = row["Category"]

    if category in dynamic_category_mapping_5000:
        matched_words = [word for word in dynamic_category_mapping_5000[category] if word in description]
        return len(matched_words) / max(len(dynamic_category_mapping_5000[category]), 1)  # Normalize score

    return 0  # If no match, return 0

# Apply the keyword overlap metric
df_bugs["Keyword Overlap Score"] = df_bugs.apply(keyword_overlap_accuracy, axis=1)

# Calculate overall accuracy as the average keyword match score
keyword_overlap_accuracy_score = np.mean(df_bugs["Keyword Overlap Score"])

# Return the computed metric
keyword_overlap_accuracy_score

# Calculate Category Confidence Score (How strongly a report matches its assigned category)
def category_confidence_score(row):
    description = str(row["Short Description"]).lower()
    category = row["Category"]

    if category in dynamic_category_mapping_5000:
        matched_words = [word for word in dynamic_category_mapping_5000[category] if word in description]
        return len(matched_words)  # More matches = higher confidence

    return 0

# Apply the category confidence metric
df_bugs["Category Confidence Score"] = df_bugs.apply(category_confidence_score, axis=1)

# Calculate overall confidence as the mean score
category_confidence_score_avg = np.mean(df_bugs["Category Confidence Score"])

# Return the computed metric
category_confidence_score_avg

from scipy.stats import entropy

# Compute category distribution for entropy-based consistency
category_counts = df_bugs["Category"].value_counts(normalize=True)  # Normalize for probability distribution

# Compute entropy (Lower entropy = better distinct categorization)
entropy_score = entropy(category_counts)

# Return entropy score
entropy_score