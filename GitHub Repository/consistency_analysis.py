import pandas as pd

# -----------------------------------
# Load BioMedCLIP Results
# -----------------------------------

df = pd.read_csv("Results/biomedclip_results.csv")

# -----------------------------------
# Clean disease names
# -----------------------------------

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

df["Expected_Diagnosis"] = df["Expected_Diagnosis"].replace(disease_map)

# -----------------------------------
# Analyze each experimental group
# -----------------------------------

results = []

groups = df.groupby(
    [
        "Generator",
        "Expected_Diagnosis",
        "Prompt"
    ]
)

for (generator, disease, prompt), group in groups:

    total = len(group)

    correct = group["Correct"].sum()

    accuracy = correct / total * 100

    # Most common prediction
    mode_prediction = group["Predicted_Diagnosis"].mode()[0]

    agreement = (
        (group["Predicted_Diagnosis"] == mode_prediction)
        .mean()
        * 100
    )

    unique_predictions = group["Predicted_Diagnosis"].nunique()

    mean_confidence = group["Confidence"].mean()

    results.append({

        "Generator": generator,

        "Disease": disease,

        "Prompt": prompt,

        "Correct": correct,

        "Total": total,

        "Accuracy (%)": round(accuracy,2),

        "Prediction Agreement (%)": round(agreement,2),

        "Unique Predictions": unique_predictions,

        "Dominant Prediction": mode_prediction,

        "Mean Confidence": round(mean_confidence,3)

    })

results = pd.DataFrame(results)

results.to_csv(
    "Results/diagnostic_consistency.csv",
    index=False
)

print(results)

print("\nAverage Agreement")

print(
    results.groupby("Generator")["Prediction Agreement (%)"].mean()
)

print("\nAverage Diagnostic Accuracy")

print(
    results.groupby("Generator")["Accuracy (%)"].mean()
)