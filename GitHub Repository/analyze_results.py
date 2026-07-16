import pandas as pd

# -----------------------------
# Load Results
# -----------------------------

df = pd.read_csv("Results/biomedclip_results.csv")

print("\nUnique disease labels:")
for disease in sorted(df["Expected_Diagnosis"].unique()):
    print(repr(disease))

print("="*70)
print("BIOMEDCLIP ANALYSIS")
print("="*70)

# -----------------------------
# Overall Accuracy
# -----------------------------

overall = df["Correct"].mean()*100

print(f"\nOverall Accuracy: {overall:.2f}%")

# -----------------------------
# Accuracy by Generator
# -----------------------------

generator_accuracy = (
    df.groupby("Generator")["Correct"]
      .mean()
      .mul(100)
      .round(2)
)

print("\nAccuracy by Generator")
print(generator_accuracy)

# -----------------------------
# Accuracy by Requested MST
# -----------------------------

mst_accuracy = (
    df.groupby("Requested_MST")["Correct"]
      .mean()
      .mul(100)
      .round(2)
)

print("\nAccuracy by Requested MST")
print(mst_accuracy)

# -----------------------------
# Accuracy by Disease
# -----------------------------

disease_accuracy = (
    df.groupby("Expected_Diagnosis")["Correct"]
      .mean()
      .mul(100)
      .round(2)
)

print("\nAccuracy by Disease")
print(disease_accuracy)

# -----------------------------
# Accuracy by Prompt
# -----------------------------

prompt_accuracy = (
    df.groupby("Prompt")["Correct"]
      .mean()
      .mul(100)
      .round(2)
)

print("\nAccuracy by Prompt")
print(prompt_accuracy)

# -----------------------------
# Save Tables
# -----------------------------

generator_accuracy.to_csv(
    "Results/accuracy_by_generator.csv"
)

mst_accuracy.to_csv(
    "Results/accuracy_by_mst.csv"
)

disease_accuracy.to_csv(
    "Results/accuracy_by_disease.csv"
)

prompt_accuracy.to_csv(
    "Results/accuracy_by_prompt.csv"
)

print("\nSaved summary tables.")

# -----------------------------
# Standardize disease names
# -----------------------------

disease_map = {
    "Atopic Dermaitits": "Atopic Dermatitis",
    "Atopic Dermatitis Gemini": "Atopic Dermatitis",
    "Atopic Dermatitis": "Atopic Dermatitis",
    "Basal Cell Carcinoma": "Basal Cell Carcinoma",
    "Focal Acaral Hyperkeratosis": "Focal Acral Hyperkeratosis",
    "Focal Acral Hyperkeratosis": "Focal Acral Hyperkeratosis",
    "Molluscum Contagiosum": "Molluscum Contagiosum",
    "Psoriasis Vulgaris": "Psoriasis Vulgaris"
}

df["Expected_Diagnosis"] = (
    df["Expected_Diagnosis"]
    .replace(disease_map)
    .str.strip()
)

print(sorted(df["Expected_Diagnosis"].unique()))

bad = df[
    ~df["Expected_Diagnosis"].isin([
        "Atopic Dermatitis",
        "Basal Cell Carcinoma",
        "Focal Acral Hyperkeratosis",
        "Molluscum Contagiosum",
        "Psoriasis Vulgaris"
    ])
]

print(bad[["Filename", "Expected_Diagnosis"]])