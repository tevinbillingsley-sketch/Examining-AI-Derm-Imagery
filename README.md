# Examining Dermatology Image Generation Capabilities of Generative AI Across Skin Tones

## Abstract
Accurate representation of dermatological conditions across skin tones is critical to equitable diagnosis and medical education because darker skin tones remain underrepresented in training imagery. Although generative artificial intelligence (AI) has shown increasing capability in synthesizing dermatological images, systematic evaluations of whether these tools preserve both requested skin tone and condition-specific features when only skin tone changes remain scarce. To address this gap, we conduct a counterfactual evaluation of OpenAI's ChatGPT and Google's Gemini, holding all prompt attributes constant except the requested Monk Skin Tone (MST) classification, and generate 500 synthetic dermatology images across five conditions. Requested skin-tone accuracy was 52% for lighter tones (MST 01--05) and 44.8% for darker tones (MST 06--10). Among 242 correct-MST images, BioMedCLIP condition-matching accuracy was 68.5\% for lighter tones and 57.1% for darker tones. These notable discrepancies show that current text-to-image models can reproduce representational inequities, requiring evaluation of both skin-tone fidelity and BioMedCLIP condition-matching accuracy before considering generated images in medical education.

Paper DOI: https://doi.org/10.1145/3805696.3847294

## Framework
The following diagram illustrates the framework behind "Examining Dermatology Image Generation Capabilities of Generative AI Across Skin Tones":

<img width="895" height="427" alt="image" src="https://github.com/user-attachments/assets/d1a5500b-7493-423e-acbe-9b14ddde3207" />

## Key Features

| Feature | Description |
|---------|-------------|
| **Prompt Templates** | Utilizes 5 Conditions with 5 different templates and 7 fixed attributes. |
| **Synthetic Image Generation** | 2 Image generators to create 250 images each |
| **Skin-Tone Fidelity** | Generated image is tested for accuracy by matching |
| **Disease Fidelity** | Determines the similarity between the generated image and the requested condition. |
| **Joint Stratified Analysis** | Image set is put through an ablation study and separate accuracy analysis |

## Key Results
The figures below depict the overall trends in accuracy for both the requested skin tone and the condition accuracy. The first chart shows the requested skin-tone accuracy, which was 2% for lighter tones (MST 01–05) and 44.8% for darker tones (MST 06–10). The second chart shows condition accuracy, in which among 242 correct-MST images, BioMedCLIP condition-matching accuracy was 68.5% for lighter tones and 57.1% for darker tones.

<img width="219" height="126" alt="image" src="https://github.com/user-attachments/assets/6ff1e611-b6bd-4e08-bd3f-a6083b661566" />

<img width="217" height="126" alt="image" src="https://github.com/user-attachments/assets/b4bcaacb-5c0a-48fd-bebb-628a087a32ea" />



## Installation

```bash
pip install -r requirements.txt
```

---

## Citation

If you use this repository, please cite:
 ```bibtex
@inproceedings{billingsley2026jcdl,
  title={Examining Dermatology Image Generation Capabilities of Generative AI Across Skin Tones},
  author={Billingsley, Tevin and Li, Mingchen and Feng, Yunhe},
  booktitle={The 2026 ACM/IEEE Joint Conference on Digital Libraries (JCDL ’26)},
  pages={6 pages},
  year={2026},
  organization={ACM}
}
```

## Acknowledgments
This work was supported in part by the U.S. National Science Foundation (NSF) under Grant No. CCF-2447834. Research conducted at the University of North Texas.
