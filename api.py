import uvicorn
from fastapi import FastAPI, Query, Path, Body
from pydantic import BaseModel, Field
from main import main

app = FastAPI()
article_kb_class, emb_dumper = main()

class Article(BaseModel):
    title: str

@app.get("/hello")
def hello():
   return {"Hello": "World!"}

@app.post("/request/")
async def predict_nearest_title(article: Article):
    top_titles = article_kb_class.search_with_emb(
        emb=emb_dumper.predictor.predict(article.title)['encoded_embeddings'])
    return top_titles