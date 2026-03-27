# Machine Learning & Fuzzy Logic — Healthcare Projects

Two notebooks implementing classification and decision-support systems in the healthcare domain.

---

## Repository Structure

```
├── BCANN-Notebook.ipynb       # Breast Cancer Classification (Random Forest & MLP)
├── IGDFuzz-Notebook.ipynb     # ER Patient Priority System (Fuzzy Logic)
└── README.md
```

---

## 1. BCANN-Notebook.ipynb — Breast Cancer Classification

### Overview
This notebook builds two binary classifiers to predict whether a breast tumor is **Malignant** or **Benign** using the Breast Cancer Wisconsin Diagnostic dataset.

### Dataset
| Attribute | Details |
|---|---|
| Source | Breast Cancer Wisconsin Diagnostic (`wdbc.data`) |
| Samples | 569 patients |
| Features | 30 numerical measurements of cell nuclei |
| Label | `M` (Malignant) -> `1`, `B` (Benign) -> `0` |
| Class Distribution | ~37% Malignant, ~63% Benign |

### Preprocessing
- The `id` column is dropped — it is a patient identifier with no predictive value
- Labels are encoded from text (`M`/`B`) to integers (`1`/`0`)
- Data is split **80% training / 20% testing** with `stratify=y` to preserve class proportions
- Features are normalized using `StandardScaler` — fitted only on training data, then applied to both sets

### Models

#### Random Forest
| Parameter | Value |
|---|---|
| `n_estimators` | 100 trees |
| `random_state` | 42 |

#### MLP (Multilayer Perceptron)
```
Input(30) -> Dense(64, ReLU) -> Dropout(0.3)
          -> Dense(32, ReLU) -> Dropout(0.2)
          -> Dense(16, ReLU)
          -> Dense(1, Sigmoid)
```
| Parameter | Value |
|---|---|
| Optimizer | Adam |
| Loss Function | Binary Crossentropy |
| Max Epochs | 200 |
| Batch Size | 16 |
| Validation Split | 20% |
| Early Stopping patience | 15 epochs |

### Results

| Model | Accuracy | AUC-ROC | Runtime |
|---|---|---|---|
| Random Forest | 97.37% | 0.9929 | ~0.43s |
| MLP | 98.25% | 0.9960 | ~6.23s |

> MLP scores slightly higher on both metrics, but takes roughly 14x longer to run than Random Forest.

### Evaluation Methods
- **Confusion Matrix** — breaks down predictions into TP, TN, FP, FN
- **ROC Curve** — plots True Positive Rate vs False Positive Rate across all thresholds
- **AUC Score** — single number summarizing model discrimination ability (closer to 1.0 is better)

### Dependencies
```
numpy, pandas, matplotlib, seaborn, tensorflow, keras, scikit-learn
```

---

## 2. IGDFuzz-Notebook.ipynb — ER Patient Priority System

### Overview
This notebook builds a **Fuzzy Logic** system to determine the examination priority of mild emergency room (ER) patients based on their clinical condition. Three defuzzification methods are implemented and compared.

### System Inputs
| Variable | Range | Description |
|---|---|---|
| Pain Level | 0 – 10 | Patient-reported pain score |
| Body Temperature | 32 – 42 C | Measured body temperature |
| Waiting Time | 0 – 180 minutes | How long the patient has been waiting |

### System Output
| Variable | Range | Description |
|---|---|---|
| Examination Priority | 0 – 100 | Priority score for patient examination |

### Membership Functions

**Pain Level:**
| Label | Range |
|---|---|
| Mild | 0 – 4 |
| Moderate | 3 – 7 |
| Severe | 6 – 10 |

**Body Temperature:**
| Label | Range |
|---|---|
| Hypothermia | < 36 C |
| Normal | 36 – 37.5 C |
| Fever | 37.5 – 39 C |
| High | > 39 C |

**Waiting Time:**
| Label | Range |
|---|---|
| Short | < 30 minutes |
| Moderate | 30 – 90 minutes |
| Long | > 90 minutes |

**Priority Output:**
| Label | Score Range |
|---|---|
| Low | 0 – 25 |
| Moderate | 25 – 50 |
| High | 50 – 75 |
| Critical | 75 – 100 |

### Fuzzy Rules
The system uses **36 IF-THEN rules** (3 pain levels x 4 temperature levels x 3 wait levels). Examples:
- `IF pain=Severe AND temp=Hypothermia AND wait=Long -> Priority=Critical`
- `IF pain=Mild AND temp=Normal AND wait=Short -> Priority=Low`
- `IF pain=Moderate AND temp=Fever AND wait=Moderate -> Priority=Moderate`

### Defuzzification Methods
| Method | Description |
|---|---|
| Mamdani (Centroid) | Center of mass of the aggregated fuzzy area |
| Max Value (LOM) | Rightmost point of the maximum membership value |
| Sugeno (Weighted Average) | Weighted average of singleton outputs per rule |

### Dependencies
```
numpy, matplotlib, scikit-fuzzy
```

---

## Getting Started

### Requirements
```bash
pip install numpy pandas matplotlib seaborn tensorflow scikit-learn scikit-fuzzy
```

### Running on Google Colab
1. Upload the `.ipynb` file to [Google Colab](https://colab.research.google.com)
2. For BCANN, also upload `Breast Cancer Wisconsin Diagnostic.zip`
3. Run each cell in order from top to bottom

### Running Locally
```bash
jupyter notebook BCANN-Notebook.ipynb
# or
jupyter notebook IGDFuzz-Notebook.ipynb
```

---

## Tech Stack

![Python](https://img.shields.io/badge/Python-3.x-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![scikit-learn](https://img.shields.io/badge/scikit--learn-latest-green)
![scikit-fuzzy](https://img.shields.io/badge/scikit--fuzzy-0.5.0-yellow)
