# WhatsApp B2C Order Recommender

**B2C Order Generation Platform via Instant Messaging and Intelligent Automation**
Master's Thesis (TFM) — Master's Degree in Artificial Intelligence, Universidad Internacional de La Rioja (UNIR)

- **Author:** Silvina Nora Gattino
- **Thesis director:** Carlos Manuel Moreno Negrin
- **Type:** Experimental pilot · September 2026
- **Full thesis (PDF, in Spanish):** [`Silvina_Nora_Gattino_Deposito_Ordinaria.pdf`](Silvina_Nora_Gattino_Deposito_Ordinaria.pdf)

---

## 1. Overview

Many wholesale (B2B/B2C) businesses receive customer orders as free-text messages over WhatsApp, which staff must then read, interpret and re-type into the management system. This project proposes an AI-based pipeline that:

1. **Interprets** a natural-language product request (e.g. *"pasta dental infantil"* — kids' toothpaste),
2. **Recommends** the best-matching products (and alternatives) from a structured catalog, and
3. **Suggests complementary products** learned from real purchase history,

so that the order can be generated automatically in the company's management system.

The repository contains the two experimental notebooks of the thesis: a **supervised hybrid recommender** and an **unsupervised complementary-product model**.

> **Language note.** The product catalog and the test queries belong to a Spanish-speaking market, so queries, product names and some column names (e.g. `Denominacion`, `Subcategoria`, `Marca`) remain in Spanish inside the code and in the original output figures. All documentation, comments and markdown are in English.

## 2. Architecture

```
Natural-language query (WhatsApp)
        │
        ▼
1. Preprocessing & normalization ── accents/symbols cleanup · synonym dictionary
        │
        ▼
2. Category detection ── RapidFuzz (partial_ratio ≥ 75%)  ──fallback──▶  TF-IDF cosine (≥ 0.40)
        │
        ▼
3. Semantic scoring ── Query Score (SC) + Attribute Score (SA)  →  Semantic Score SS = 0.70·SC + 0.30·SA
        │
        ▼
4. Structural K-NN ── One-Hot Encoding + cosine similarity  →  ScoreKNN
        │
        ▼
5. Hybrid weighted fusion ── S_H = α·KNN_n + β·SC_n + γ·SA_n  (scenarios A / B / C)
        │
        ▼
Top-5 recommended products   +   complementary products (clustering on sales history)
```

## 3. Methodology

### 3.1 Supervised hybrid recommender (`Main.ipynb`)

| Stage | Technique |
|---|---|
| Text normalization | Accent/symbol removal, unit normalization (g, ml), synonym dictionary (e.g. *kids → INFANTIL*, *charcoal → CARBON*) |
| Category retrieval | **RapidFuzz** `partial_ratio` over categories; **TF-IDF + cosine similarity** fallback when no match is found |
| Query Score (SC) | RapidFuzz `token_set_ratio` against an enriched product text × query-token coverage |
| Attribute Score (SA) | Weighted sum of attribute/value matches: `SA = Σ pa_i × pv_i` (business-rule weights) |
| Semantic Score (SS) | `SS = SC × 0.70 + SA × 0.30` |
| Structural K-NN | One-Hot Encoding + `StandardScaler` for numeric variables; cosine similarity between query vector and product vectors (`ScoreKNN`) |
| Hybrid fusion | `S_H = α·S_KNN_n + β·S_C_n + γ·S_A_n` with MinMax-normalized components (α+β+γ = 1) |
| Scenarios | **A** Balanced (α=0.25, β=0.50, γ=0.25) · **B** Strong semantic (0.10, 0.70, 0.20) · **C** Strong structural (0.50, 0.30, 0.20) |
| Evaluation | Precision, Recall, Accuracy and F1-Score on top-5 results against a manually defined **ground truth** |

### 3.2 Unsupervised complementary recommendation (`Main_NS.ipynb`)

- Starting point: the **last six months** of sales history; the best-selling products are used as reference (target) products.
- Co-purchased products are split into two groups:
  - **Similar products** (same category, different subcategory) → clustering by subcategory and brand with **Agglomerative Hierarchical Clustering** (Ward linkage, dendrogram used to choose *K*) and **K-Modes** (Huang initialization); top products per cluster by purchase frequency.
  - **Diverse products** (other categories) → direct purchase-frequency analysis (top-3 per subcategory and brand), since clustering loses discriminative power on highly heterogeneous groups.
- Cluster quality is checked with Silhouette, Davies-Bouldin and Calinski-Harabasz indices (K-Prototypes is used to obtain a Silhouette on mixed variables).

## 4. Results

### 4.1 Evaluation of the hybrid model

Macro-average over 6 test queries (top-5 results vs. manual ground truth):

| Model | Precision | Recall | Accuracy | F1-Score |
|---|---|---|---|---|
| Original scoring | 0.83 | 0.81 | 0.96 | 0.79 |
| Scenario A — Balanced | 0.80 | 0.79 | 0.95 | 0.77 |
| Scenario B — Strong semantic | 0.80 | 0.79 | 0.95 | 0.77 |
| Scenario C — Strong structural | 0.70 | 0.71 | 0.93 | 0.68 |

![Average metrics per model](Figures/02_average_metrics_per_model.png)
*Average Precision / Recall / Accuracy / F1 per model (labels in Spanish: "Esc." = Scenario, "Balanceado" = Balanced, "Semántico" = Semantic, "Estructural" = Structural).*

![F1-Score per query](Figures/03_f1_per_query.png)
*F1-Score per test query and model (C1 Infantil = kids, C2 Charcoal, C3 Genérica = generic, C4 Kolynos, C5 Anticaries = anti-cavity, C6 Sensitive).*

![F1 heatmap](Figures/04_f1_heatmap.png)
*F1-Score heatmap by query and model.*

### 4.2 Exploratory analysis of the structured attributes

![Attribute distributions](Figures/01_eda_attribute_distributions.png)
*Distribution of the generated product attributes (Demografia, Beneficio, Sabor, Componentes, Extras, Marca, UnidadPeso, PrecioCaja). "NULO" = null.*

### 4.3 Unsupervised complementary recommendation

![Sales EDA](Figures/05_sales_eda.png)
*Exploratory analysis of the sales table (category, subcategory, brand, price).*

![Silhouette K-Prototypes](Figures/06_silhouette_kprototypes.png)
*Silhouette score as a function of the number of clusters (K-Prototypes).*

![Dendrogram](Figures/07_dendrogram_hierarchical.png)
*Dendrogram of the hierarchical clustering on similar products ("Distancia" = distance, "Productos" = products); used to select K.*

![Clusters of similar products](Figures/08_clusters_similar_products.png)
*Similar products by subcategory·brand, and by K-Modes and Hierarchical cluster (x: purchase frequency, y: normalized price).*

![PCA diverse products](Figures/09_pca_diverse_products.png)
*2D PCA of the diverse products group.*

![Diverse products](Figures/10_diverse_products_scatter.png)
*Diverse products by subcategory·brand (frequency vs. normalized price).*

![PCA all products](Figures/11_pca_all_products.png)
*2D PCA of all products with the target product highlighted.*

## 5. Repository structure

```
whatsapp-b2c-order-recommender/
├── README.md
├── requirements.txt
├── .gitignore
├── Main.ipynb                       # Supervised hybrid recommender (NLP + RapidFuzz + TF-IDF + K-NN)
├── Main_NS.ipynb                    # Unsupervised complementary recommender (Hierarchical + K-Modes)
├── Figures/                         # Figures extracted from the notebook outputs
└── Silvina_Nora_Gattino_Deposito_Ordinaria.pdf   # Full thesis (Spanish)
```

## 6. How to run

The notebooks were developed in **Google Colab** (Python 3). The underlying datasets come from a real wholesale company and are **not included** in this repository for confidentiality reasons.

1. Install the dependencies: `pip install -r requirements.txt`
2. Provide your own CSV files with the same structure as the original datasets (products, product categories, attributes, category attributes, attribute values and sales), and set `base_path` in the data-loading cell of each notebook (the default `drive/MyDrive/TFM/` is a Google Drive path used in Colab).
3. Set `consulta_query` (the user query) and run the cells in order.

Expected input tables: `Productos_Proveedor_828_718.csv`, `Productos_Categorias.csv`, `Atributos.csv`, `Categorias_Atributos.csv`, `Categorias_Atributos_Valor.csv` (`Main.ipynb`) and `Ventas_117.txt`, `Productos_Categorias.csv` (`Main_NS.ipynb`).

The outputs stored in the notebooks are those of the original experiment run.

## 7. Main technologies

Python · pandas · NumPy · scikit-learn · RapidFuzz · kmodes · SciPy · Matplotlib · Seaborn · Plotly

## 8. Citation

> Gattino, S. N. (2026). *B2C Order Generation Platform via Instant Messaging and Intelligent Automation* [Master's thesis, Universidad Internacional de La Rioja]. Director: C. M. Moreno Negrin.
