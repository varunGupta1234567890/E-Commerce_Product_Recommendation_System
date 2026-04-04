# Product Recommendation System (FastAPI)

A **content-based product recommendation system** built using **FastAPI, Machine Learning, and Jinja2 templates**.
It recommends similar products based on user input using **TF-IDF + Cosine Similarity**.

---

## Features

* Search products by name
* Content-based recommendations using ML
* Fast similarity computation (precomputed cosine similarity)
* Trending products section on homepage
* Product images display
* Ratings & review count
* Background video support (UI enhancement)
* Clean UI using HTML, CSS & Jinja2
* Backend powered by FastAPI

---

## How It Works

1. Dataset is loaded (`product_recommendation.csv`)
2. Text data (Tags) is converted into vectors using TF-IDF
3. Cosine similarity is computed between all products
4. User enters a product name
5. System finds closest match
6. Returns top similar products

---

## Tech Stack

* FastAPI 
* Python 
* Pandas 
* Scikit-learn 
* Jinja2 Templates 
* HTML/CSS 

---

## Project Structure

├── app.py                          # FastAPI backend
├── product_recommendation.csv      # Dataset
├── templates/
│   ├── index.html                 # Homepage (trending products)
│   └── main.html                  # Recommendation page
├── static/
│   ├── v.mp4                      # Background video
│   └── assets...                  # CSS / images
├── README.md                      # Documentation


---

## How to Run Locally

### 1. Clone the repository

git clone https://github.com/your-username/product-recommender-fastapi.git
cd product-recommender-fastapi


### 2. Install dependencies

pip install -r requirements.txt


### 3. Run the server

uvicorn app:app --reload


### 4. Open in browser

http://127.0.0.1:8000

---

## Sample Input / Output

* Input: "laptop"
* Output: List of similar products with:

  * Image
  * Rating
  * Brand
  * Review count

---

## Author

**Varun Gupta**

---

