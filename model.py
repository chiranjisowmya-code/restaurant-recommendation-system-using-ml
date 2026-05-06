import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load data
df = pd.read_csv("restaurants.csv")

# Combine important features
df["features"] = df["location"] + " " + df["cuisine"] + " " + df["food_item"]

# Convert text to vectors
cv = CountVectorizer()
matrix = cv.fit_transform(df["features"])

# Calculate similarity
similarity = cosine_similarity(matrix)

def recommend(location, cuisine, food_item):
    user_input = location + " " + cuisine + " " + food_item
    user_vector = cv.transform([user_input])
    scores = cosine_similarity(user_vector, matrix)
    scores = scores.flatten()

    top_indices = scores.argsort()[-5:][::-1]

    results = df.iloc[top_indices][
        ["name", "location", "cuisine", "food_item", "rating", "budget"]
    ]

    return results