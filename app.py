from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


app = FastAPI()

# ================= STATIC + TEMPLATES =================
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")


# ================= LOAD DATA =================
train_data = pd.read_csv("product_recommendation.csv")

# clean dataset
train_data = train_data.drop(columns=["Unnamed: 0"], errors='ignore')
train_data = train_data.dropna(subset=['Name','Tags'])

# trending products (top 20)
trending_products = train_data.head(20)


# ================= PRECOMPUTE =================
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(train_data['Tags'])
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)


# ================= UTIL =================
def truncate(text, length=25):
    return text[:length] + "..." if len(text) > length else text


# ================= RECOMMEND =================
def content_based_recommendations(item_name, top_n=10):

    item_name = item_name.strip().lower()

    matches = train_data[train_data['Name'].str.lower().str.contains(item_name, na=False)]

    if matches.empty:
        print("❌ No match found:", item_name)
        return pd.DataFrame()

    idx = matches.index[0]

    similar_items = list(enumerate(cosine_sim[idx]))
    similar_items = sorted(similar_items, key=lambda x: x[1], reverse=True)

    top_items = similar_items[1:top_n+1]
    indices = [i[0] for i in top_items]

    return train_data.iloc[indices][['Name','ReviewCount','Brand','ImageURL','Rating']]
# ================= ROUTES =================

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {
        "request": request,
        "trending_products": trending_products.to_dict(orient="records"),
        "video_url": "/static/v.mp4",
        "truncate": truncate
    })


@app.get("/main", response_class=HTMLResponse)
async def main_page(request: Request):
    return templates.TemplateResponse("main.html", {"request": request})


# ================= RECOMMEND =================
@app.post("/recommendations", response_class=HTMLResponse)
async def recommendations(
    request: Request,
    prod: str = Form(...),
    nbr: int = Form(...)
):
    recs = content_based_recommendations(prod, nbr)

    if recs.empty:
        return templates.TemplateResponse("main.html", {
            "request": request,
            "message": "No recommendations found"
        })

    return templates.TemplateResponse("main.html", {
        "request": request,
        "content_based_rec": recs.to_dict(orient="records"),
        "truncate": truncate
    })


# ================= RUN =================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=8000)