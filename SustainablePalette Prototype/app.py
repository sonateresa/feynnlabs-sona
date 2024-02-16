from flask import Flask, render_template, request
import pandas as pd
from sklearn.cluster import KMeans

app = Flask(__name__)

# Load the necessary datasets and pre-trained models
food_production_df = pd.read_csv("Food_Production.csv")
nutritional_info_df = pd.read_csv("NUTRITIONAL INFO.csv")

# Preprocess the data and train the KMeans clustering model
# ... (code for preprocessing and training)

# Define functions for recommendation
def assign_cluster_labels(data, kmeans_model):
    """Assign cluster labels to each data point"""
    cluster_labels = kmeans_model.predict(data)
    return cluster_labels

def recommend_products(product_name, cluster_labels, products_df):
    """Recommend other products from the same cluster as the given product"""
    product_cluster = cluster_labels[products_df.index[products_df['food product'] == product_name].tolist()[0]]
    recommended_products = products_df[cluster_labels == product_cluster]['food product']
    recommended_products = recommended_products[recommended_products != product_name]  # Exclude the given product
    return recommended_products

# Route for the home page
@app.route('/')
def index():
    return render_template('index.html')

# Route to handle form submission
@app.route('/recommend', methods=['POST'])
def recommend():
    product_name = request.form['product_name']
    cluster_labels = assign_cluster_labels(nutritional_info_df.drop(columns=['food product']), kmeans_model)
    recommended_products = recommend_products(product_name, cluster_labels, nutritional_info_df)
    return render_template('result.html', product_name=product_name, recommended_products=recommended_products)

if __name__ == '__main__':
    app.run(debug=True)
