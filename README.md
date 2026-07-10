# FoodRec 
Restaurant Recommendation Engine using Neural Collaborative Filtering

A deep learning–based recommendation system that predicts which restaurants a user is most likely to engage with, trained on **implicit feedback** (real user to restaurant interactions) rather than explicit ratings; mirroring how production recommender systems at food-delivery platforms like Swiggy, Zomato, and DoorDash actually operate.


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
- A review by `user_id` for `business_id` is treated as a **positive interaction** (engagement), regardless of star rating — reflecting the reality that food-delivery platforms track *orders*, not ratings, as the primary signal.
- **Negative sampling**: for each positive interaction, a small number of restaurants the user never reviewed are randomly sampled as negatives, following standard practice for implicit-feedback recommender training.
- Dataset is filtered to a subset of cities/categories (restaurants only) to keep training tractable on a single machine.


## Model Architecture — Neural Collaborative Filtering (NCF)

```
User ID ──► Embedding Layer ─┐
                              ├─► Concatenate ──► MLP (Dense + ReLU layers) ──► Sigmoid ──► Interaction Score
Item ID ──► Embedding Layer ─┘
```

- **User & Item Embeddings**: learned dense vector representations (latent factors), initialized randomly and trained end-to-end; no pretrained embeddings used.
- **MLP layers**: capture non-linear interactions between user and item embeddings, generalizing beyond simple matrix factorization (dot product).
- **Output**: probability that a user would engage with a given restaurant.
- **Loss function**: Binary Cross-Entropy (implicit feedback framed as binary classification: interacted vs. not).


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

```
foodrec/
├── data/
│   ├── raw/                  # Yelp raw JSON files (not committed)
│   └── processed/            # Cleaned interaction tables
├── notebooks/
│   └── eda.ipynb             # Exploratory data analysis
├── src/
│   ├── preprocess.py         # Builds (user, item, label) triples + negative sampling
│   ├── dataset.py            # PyTorch Dataset/DataLoader
│   ├── model.py              # NCF model definition
│   ├── train.py              # Training loop
│   ├── evaluate.py           # Hit Rate@K, NDCG@K, baseline comparison
│   └── recommend.py          # Inference: top-N recommendations for a user
├── api/
│   └── main.py                # FastAPI app exposing /recommend/{user_id}
├── app/
│   └── streamlit_app.py       # Demo interface
├── requirements.txt
└── README.md
```


## How to Run Locally

```bash
# 1. Clone repo and install dependencies
git clone <repo-url>
cd foodrec
pip install -r requirements.txt

# 2. Preprocess data (constructs implicit interaction dataset)
python src/preprocess.py

# 3. Train the model
python src/train.py

# 4. Evaluate against baseline
python src/evaluate.py

# 5. Run the API
uvicorn api.main:app --reload

# 6. Run the demo interface
streamlit run app/streamlit_app.py
```


## AWS Deployment

| Component | Service |
|---|---|
| Model inference | AWS Lambda (or EC2 for larger models) |
| API | Amazon API Gateway → Lambda |
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