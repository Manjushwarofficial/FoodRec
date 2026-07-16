import pandas as pd
import numpy as np
import torch
import torch.nn as nn

FEATURE_DIR = "data/features"
MODEL_DIR = "models"

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

user_id_map = pd.read_csv(f"{FEATURE_DIR}/user_id_map.csv", index_col=0)["user_idx"].to_dict()
item_id_map = pd.read_csv(f"{FEATURE_DIR}/item_id_map.csv", index_col=0)["item_idx"].to_dict()
item_idx_to_id = {v: k for k, v in item_id_map.items()}

N_USERS = len(user_id_map)
N_ITEMS = len(item_id_map)

df_restaurants = pd.read_csv(f"{FEATURE_DIR}/restaurant_metadata.csv")
df_train = pd.read_csv(f"{FEATURE_DIR}/train_interactions.csv")


class NCF(nn.Module):
    def __init__(self, n_users, n_items, embedding_dim=16):
        super().__init__()
        self.user_embedding = nn.Embedding(n_users, embedding_dim)
        self.item_embedding = nn.Embedding(n_items, embedding_dim)

        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim * 2, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, user_idx, item_idx):
        user_vec = self.user_embedding(user_idx)
        item_vec = self.item_embedding(item_idx)
        x = torch.cat([user_vec, item_vec], dim=1)
        logit = self.mlp(x)
        return torch.sigmoid(logit).squeeze()


model = NCF(N_USERS, N_ITEMS, embedding_dim=16).to(device)
model.load_state_dict(torch.load(f"{MODEL_DIR}/ncf_model.pt", map_location=device))
model.eval()

popularity_ranking = (
    df_train[df_train["label"] == 1]["item_idx"]
    .value_counts()
    .index.tolist()
)


def recommend(user_id, top_n=10):
    if user_id not in user_id_map:
        print(f"User '{user_id}' not seen during training — using popularity fallback.")
        top_item_indices = popularity_ranking[:top_n]
    else:
        user_idx = user_id_map[user_id]
        already_seen = set(df_train[df_train["user_idx"] == user_idx]["item_idx"])
        all_item_indices = np.array([i for i in range(N_ITEMS) if i not in already_seen])

        with torch.no_grad():
            user_tensor = torch.tensor([user_idx] * len(all_item_indices), dtype=torch.long).to(device)
            item_tensor = torch.tensor(all_item_indices, dtype=torch.long).to(device)
            scores = model(user_tensor, item_tensor).cpu().numpy()

        ranked = all_item_indices[np.argsort(-scores)]
        top_item_indices = ranked[:top_n]

    business_ids = [item_idx_to_id[i] for i in top_item_indices]
    results = df_restaurants[df_restaurants["business_id"].isin(business_ids)][
        ["business_id", "name", "city", "categories"]
    ].copy()

    results["rank"] = results["business_id"].apply(lambda b: business_ids.index(b))
    results = results.sort_values("rank").drop(columns="rank").reset_index(drop=True)

    return results