# FoodRec 
Restaurant Recommendation Engine using Neural Collaborative Filtering

A deep learning–based recommendation system that predicts which restaurants a user is most likely to engage with, trained on **implicit feedback** (real user to restaurant interactions) rather than explicit ratings; mirroring how production recommender systems at food-delivery platforms like Swiggy, Zomato, and DoorDash actually operate.

# Interface Screenshots

<img width="1276" height="708" alt="Screenshot 2026-07-16 at 2 39 18 PM" src="https://github.com/user-attachments/assets/c00ff815-d599-4f10-8db9-780cf479f2ca" />
<img width="1278" height="710" alt="Screenshot 2026-07-16 at 2 39 41 PM" src="https://github.com/user-attachments/assets/6259b3ad-5b73-426d-8ea5-a42ba1f2a9b3" />
<img width="1279" height="712" alt="Screenshot 2026-07-16 at 2 39 55 PM" src="https://github.com/user-attachments/assets/36ee93c7-1a83-498b-a8d2-8c8de58787a0" />


## Why this project

Most recommender system tutorials use explicit rating data (e.g., 1–5 stars) and assume it's dense and reliable. In practice, most production systems; especially in food delivery, e-commerce, and content platforms; face **sparse, implicit signals**: a user either engaged with an item (ordered, clicked, viewed) or didn't. This project is built around that realistic constraint.

Instead of relying on a pretrained model, the recommendation model here — **Neural Collaborative Filtering (NCF)** — is trained from scratch, learning user and item embeddings purely from interaction data.


## Problem Statement

Given a user and a catalog of restaurants, predict a ranked list of restaurants the user is most likely to want to order from, using only implicit interaction history.


## Dataset

**Source:** [Yelp Open Dataset](https://www.yelp.com/dataset) (free, official release)

| File | Used for |
|---|---|
| `yelp_academic_dataset_business.json` | Restaurant metadata: name, cuisine/category, location, price range |
| `yelp_academic_dataset_review.json` | Real user–restaurant interactions (used to construct implicit feedback) |
| `yelp_academic_dataset_user.json` | User metadata |

**Implicit feedback construction:**
- A review and tip by `user_id` for `business_id` is treated as a **positive interaction** (engagement), regardless of star rating — reflecting the reality that food-delivery platforms track *orders*, not ratings, as the primary signal.
- **Negative sampling**: for each positive interaction, a small number of restaurants the user never reviewed are randomly sampled as negatives, following standard practice for implicit-feedback recommender training.
- Dataset is filtered to a subset of cities/categories (restaurants only) to keep training tractable on a single machine.


## Model Architecture — Neural Collaborative Filtering (NCF)

```text
User ID ──► User Embedding ─┐
                            ├─► Concatenate ──► MLP (64 → 32 → 1) ──► Sigmoid ──► Probability
Business ID ──► Item Embedding ─┘
```

- **User and item embeddings**: learned dense vectors for each user and restaurant, initialized randomly and optimized during training.
- **MLP head**: combines the two embeddings through a small multi-layer perceptron with ReLU activations to learn non-linear interactions.
- **Output**: a score between 0 and 1 representing the likelihood that a user will engage with a restaurant.
- **Training setup**: the model is trained with binary cross-entropy loss on implicit feedback pairs built from reviews and tips.
- **Sampling strategy**: each positive interaction is paired with random negative samples so the model learns to distinguish observed from unobserved interactions.


## Evaluation

Since this is a ranking problem, standard classification accuracy is not meaningful. The following ranking-focused metrics are used:

- **Hit Rate@K** — was the true (held-out) interacted item present in the top-K recommendations?
- **NDCG@K** — rewards correctly ranking the true item higher within the top-K list.
- **Baseline comparison**: model performance is compared against a **popularity-based baseline** (recommend the most-reviewed restaurants to every user) to demonstrate that personalization adds real value beyond simple popularity ranking.

| Model | Hit Rate@10 | NDCG@10 |
|---|---|---|
| Popularity Baseline | TBD | TBD |
| Neural Collaborative Filtering | TBD | TBD |

*(Results to be filled in after training.)*


## Handling the Cold-Start Problem

New users/restaurants have no learned embedding at inference time. This system falls back to a **popularity-based recommendation** for new users, and can incorporate restaurant metadata (cuisine, price range, location) as a content-based fallback for new restaurants — a common hybrid strategy used in production recommender systems.


## Tech Stack

| Layer | Tool |
|---|---|
| Model training | PyTorch |
| Data processing | Pandas, NumPy |
| API serving | FastAPI |
| Frontend/demo | Streamlit |
| Deployment | AWS Lambda / EC2, API Gateway, S3, CloudFront |
| Storage | Amazon S3 (model artifacts, processed data) |


## Project Structure

```text
FoodRec/
├── app.py                    # Main Flask app entry point
├── api/
│   └── main.py               # FastAPI service for recommendation endpoints
├── app/
│   └── streamlit_app.py      # Streamlit demo interface
├── assets/
├── data/
│   ├── features/             # Feature-mapped IDs and metadata tables
│   ├── interim/              # Intermediate preprocessing outputs
│   ├── processed/            # Cleaned business, review, user, and tip data
│   └── raw/
│       └── yelp_json/        # Yelp raw JSON datasets
├── models/
│   └── ncf_model.pt          # Trained neural collaborative filtering model
├── notebooks/
│   ├── 01_data_loading.ipynb
│   ├── 02_eda.ipynb
│   ├── 03_preprocessing.ipynb
│   ├── 04_feature_engineering.ipynb
│   └── 05_ncf_model.ipynb
├── src/
│   └── recommend.py          # Recommendation inference logic
├── static/
│   └── css/                  # Frontend stylesheets
├── templates/
│   ├── index.html
│   └── recommend.html
├── requirements.txt
└── README.md
```


## How to Run Locally

```bash
# 1. Clone the repository and install dependencies
git clone https://github.com/Manjushwarofficial/FoodRec.git
cd FoodRec
pip install -r requirements.txt

# 2. Run the main Flask web app
python app.py

```

> The main user-facing app is the Flask interface in app.py. 

## AWS Deployment

| Component | Service |
|---|---|
| Model inference | AWS EC2 |
| API | Amazon API Gateway → EC2 |
| Frontend | Amazon S3 (static hosting) + CloudFront |
| Model & data storage | Amazon S3 |


## Key Concepts Demonstrated

- Learned embeddings vs. pretrained embeddings
- Matrix factorization intuition and how NCF generalizes it with an MLP
- Implicit feedback modeling and negative sampling
- Ranking metrics (Hit Rate@K, NDCG@K) vs. classification accuracy
- Cold-start problem and hybrid fallback strategies
- Production considerations: baseline comparison, latency vs. accuracy tradeoffs, scaling recommendation serving


## Future Improvements

- Incorporate restaurant metadata (cuisine, price, location) as side features for a hybrid content + collaborative model
- Add approximate nearest neighbor search (e.g., FAISS) for scaling to large item catalogs
- Experiment with sequence-aware recommendation (recent order history) using a lightweight sequential model


## License

This project uses the Yelp Open Dataset, which is provided by Yelp for academic and personal use under their [dataset license terms](https://www.yelp.com/dataset).
