import os
import re
import cv2
import numpy as np
import pandas as pd

# ----------------------------------
# Data folders
# ----------------------------------

DATASETS = [
    "Gemini",
    "ChatGPT"
]

# ----------------------------------
# Official Monk Skin Tone RGB values
# ----------------------------------

MONK_RGB = {
    1:(246,237,228),
    2:(243,231,219),
    3:(247,234,208),
    4:(234,218,186),
    5:(215,189,150),
    6:(160,126,86),
    7:(130,92,67),
    8:(96,65,52),
    9:(58,49,42),
    10:(41,36,32)
}

def rgb_to_lab(rgb):

    rgb = np.uint8([[rgb]])

    return cv2.cvtColor(
        rgb,
        cv2.COLOR_RGB2LAB
    )[0][0].astype(float)


MONK_LAB = {
    mst: rgb_to_lab(rgb)
    for mst, rgb in MONK_RGB.items()
}

def detect_skin(image):

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)

    lower_hsv = np.array([0,30,60],dtype=np.uint8)
    upper_hsv = np.array([25,255,255],dtype=np.uint8)

    mask1 = cv2.inRange(
        hsv,
        lower_hsv,
        upper_hsv
    )

    lower_y = np.array([0,133,77],dtype=np.uint8)
    upper_y = np.array([255,173,127],dtype=np.uint8)

    mask2 = cv2.inRange(
        ycrcb,
        lower_y,
        upper_y
    )

    mask = cv2.bitwise_and(mask1,mask2)

    kernel = np.ones((5,5),np.uint8)

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        kernel
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        kernel
    )

    return mask

def representative_skin_lab(image, mask):

    rgb = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    lab = cv2.cvtColor(
        rgb,
        cv2.COLOR_RGB2LAB
    )

    pixels = lab[mask > 0]

    if len(pixels) == 0:
        return None

    return np.median(
        pixels,
        axis=0
    )

def nearest_mst(lab):

    best = None

    best_distance = 1e9

    for mst, ref in MONK_LAB.items():

        d = np.linalg.norm(
            lab-ref
        )

        if d < best_distance:

            best_distance = d

            best = mst

    return best, best_distance

# ----------------------------------
# Evaluate every image
# ----------------------------------

results = []

for generator in DATASETS:

    folder = os.path.join("Data", generator)

    for filename in sorted(os.listdir(folder)):

        if not filename.lower().endswith((".png", ".jpg", ".jpeg")):
            continue

        path = os.path.join(folder, filename)

        image = cv2.imread(path)

        if image is None:
            continue

        # -----------------------
        # Requested MST
        # -----------------------

        match = re.search(r"mst0*(\d+)", filename.lower())

        if not match:
            continue

        requested = int(match.group(1))

        # -----------------------
        # Detect skin
        # -----------------------

        mask = detect_skin(image)

        lab = representative_skin_lab(image, mask)

        if lab is None:
            continue

        measured, delta_e = nearest_mst(lab)

        results.append({

            "Filename": filename,

            "Generator": generator,

            "Requested_MST": requested,

            "Measured_MST": measured,

            "Exact_Match": requested == measured,

            "Absolute_Error": abs(requested-measured),

            "Bias": measured-requested,

            "DeltaE": round(delta_e,2)

        })

df = pd.DataFrame(results)

os.makedirs("Results", exist_ok=True)

df.to_csv(
    "Results/mst_results.csv",
    index=False
)

print(df.head())

print(f"\nProcessed {len(df)} images.")

print("\n===================================")
print("OVERALL")
print("===================================")

exact = df["Exact_Match"].mean()*100

within1 = (df["Absolute_Error"]<=1).mean()*100

within2 = (df["Absolute_Error"]<=2).mean()*100

mae = df["Absolute_Error"].mean()

bias = df["Bias"].mean()

print(f"Exact Match: {exact:.2f}%")
print(f"Within ±1: {within1:.2f}%")
print(f"Within ±2: {within2:.2f}%")
print(f"Mean Absolute Error: {mae:.2f}")
print(f"Mean Bias: {bias:.2f}")

print("\n===================================")
print("BY GENERATOR")
print("===================================")

for generator in DATASETS:

    subset = df[df["Generator"] == generator]

    print(f"\n{generator}")

    print(f"Exact Match: {subset['Exact_Match'].mean()*100:.2f}%")

    print(f"Within ±1: {(subset['Absolute_Error']<=1).mean()*100:.2f}%")

    print(f"Within ±2: {(subset['Absolute_Error']<=2).mean()*100:.2f}%")

    print(f"MAE: {subset['Absolute_Error'].mean():.2f}")

    print(f"Bias: {subset['Bias'].mean():.2f}")

print("\n===================================")
print("CALIBRATION")
print("===================================")

calibration = (
    df.groupby(["Generator", "Requested_MST"])
      .agg(
          Mean_Measured=("Measured_MST","mean"),
          Std=("Measured_MST","std"),
          Count=("Measured_MST","count")
      )
      .round(2)
)

print(calibration)

calibration.to_csv(
    "Results/mst_calibration.csv"
)

from sklearn.metrics import confusion_matrix

print("\n===================================")
print("CONFUSION MATRIX")
print("===================================")

cm = confusion_matrix(
    df["Requested_MST"],
    df["Measured_MST"],
    labels=range(1,11)
)

cm_df = pd.DataFrame(
    cm,
    index=[f"Req_{i}" for i in range(1,11)],
    columns=[f"Pred_{i}" for i in range(1,11)]
)

print(cm_df)

cm_df.to_csv(
    "Results/mst_confusion_matrix.csv"
)

