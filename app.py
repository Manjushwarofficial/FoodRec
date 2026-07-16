from flask import Flask, render_template, request
from src.recommend import recommend
import pandas as pd

app = Flask(__name__)

df_users = pd.read_csv("data/processed/user_clean.csv")
users_list = df_users[['user_id', 'name']].sample(n=100, random_state=42).to_dict('records')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/recommend', methods=['GET', 'POST'])
def get_recommendations():
    recommendations = None
    searched = False

    if request.method == 'POST':
        user_id = request.form.get('user_id', '').strip() or 'guest'
        cuisine = request.form.get('cuisine', '').strip().lower()

        results_df = recommend(user_id, top_n=20)

        if cuisine:
            results_df = results_df[
                results_df['categories'].str.lower().str.contains(cuisine, na=False)
            ]

        results_df = results_df.head(10)
        recommendations = results_df.to_dict('records')
        searched = True

    return render_template('recommend.html', users=users_list, recommendations=recommendations, searched=searched)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=80)