import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from flask import Flask, request, render_template_string

app = Flask(__name__)

# --------------------------------
# Restaurant Dataset
# --------------------------------
restaurants = [
    {"name": "Paradise", "location": "Hyderabad", "cuisine": "South Indian", "food_item": "Biryani", "rating": 4.5, "budget": 500},
    {"name": "Bawarchi", "location": "Hyderabad", "cuisine": "South Indian", "food_item": "Biryani", "rating": 4.4, "budget": 450},
    {"name": "Shah Ghouse", "location": "Hyderabad", "cuisine": "North Indian", "food_item": "Mandi", "rating": 4.3, "budget": 600},
    {"name": "Haldirams", "location": "Rajasthan", "cuisine": "North Indian", "food_item": "Sweets", "rating": 4.3, "budget": 300},
    {"name": "Pizza Hut", "location": "Mumbai", "cuisine": "American Food", "food_item": "Pizza", "rating": 4.1, "budget": 800},
    {"name": "Burger King", "location": "Mumbai", "cuisine": "American Food", "food_item": "Burger", "rating": 4.2, "budget": 500},
    {"name": "Empire Restaurant", "location": "Bangalore", "cuisine": "American Food", "food_item": "Burger", "rating": 4.2, "budget": 400},
    {"name": "Karachi Bakery", "location": "Charminar", "cuisine": "South Indian", "food_item": "Cake", "rating": 4.5, "budget": 500},
    {"name": "Cafe Coffee Day", "location": "Secunderabad", "cuisine": "South Indian", "food_item": "Beverages", "rating": 4.0, "budget": 350}
]

df = pd.DataFrame(restaurants)

# Combine features for ML
df["features"] = df["cuisine"] + " " + df["location"] + " " + df["food_item"]

# TF-IDF
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(df["features"])

# Similarity
similarity_matrix = cosine_similarity(tfidf_matrix)

# --------------------------------
# HTML Template with Background
# --------------------------------
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Restaurant Recommendation</title>
    <style>
        body {
            margin: 0;
            font-family: Arial;
            background: url('https://images.unsplash.com/photo-1504674900247-0877df9cc836') no-repeat center center fixed;
            background-size: cover;
        }
        .overlay {
            background: rgba(0,0,0,0.7);
            min-height: 100vh;
            padding: 40px;
            color: white;
            text-align: center;
        }
        select, button {
            padding: 10px;
            margin: 10px;
            width: 220px;
            border-radius: 5px;
            border: none;
        }
        button {
            background: #e74c3c;
            color: white;
            cursor: pointer;
        }
        table {
            margin: auto;
            margin-top: 20px;
            border-collapse: collapse;
            background: white;
            color: black;
        }
        th, td {
            padding: 10px;
            border: 1px solid #ddd;
        }
        th {
            background: #e74c3c;
            color: white;
        }
    </style>
</head>
<body>
<div class="overlay">

<h1>🍽 Restaurant Recommendation System</h1>

<form method="POST">
    <select name="restaurant">
        {% for r in restaurants %}
        <option value="{{r}}">{{r}}</option>
        {% endfor %}
    </select>
    <button type="submit">Get Recommendation</button>
</form>

{% if selected %}
<h2>Selected Restaurant</h2>
<table>
<tr>
<th>Name</th><th>Location</th><th>Cuisine</th><th>Food</th><th>Rating ⭐</th><th>Budget ₹</th>
</tr>
<tr>
<td>{{selected.name}}</td>
<td>{{selected.location}}</td>
<td>{{selected.cuisine}}</td>
<td>{{selected.food_item}}</td>
<td>{{selected.rating}}</td>
<td>{{selected.budget}}</td>
</tr>
</table>

<h2>Recommended Restaurants</h2>
<table>
<tr>
<th>Name</th><th>Location</th><th>Cuisine</th><th>Food</th><th>Rating ⭐</th><th>Budget ₹</th>
</tr>
{% for rec in recommendations %}
<tr>
<td>{{rec.name}}</td>
<td>{{rec.location}}</td>
<td>{{rec.cuisine}}</td>
<td>{{rec.food_item}}</td>
<td>{{rec.rating}}</td>
<td>{{rec.budget}}</td>
</tr>
{% endfor %}
</table>
{% endif %}

</div>
</body>
</html>
"""

# --------------------------------
# Flask Route
# --------------------------------
@app.route("/", methods=["GET", "POST"])
def index():
    selected = None
    recommendations = []

    if request.method == "POST":
        name = request.form.get("restaurant")

        if name in df["name"].values:
            idx = df[df["name"] == name].index[0]
            selected = df.iloc[idx].to_dict()

            similarity_scores = list(enumerate(similarity_matrix[idx]))
            similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)

            for i in similarity_scores[1:4]:  # skip itself
                recommendations.append(df.iloc[i[0]].to_dict())

    return render_template_string(
        HTML_TEMPLATE,
        restaurants=df["name"].tolist(),
        selected=selected,
        recommendations=recommendations
    )

if __name__ == "__main__":
    app.run(debug=True)