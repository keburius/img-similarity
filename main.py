from typing import Optional

from fastapi import FastAPI, Depends
import uvicorn
import tensorflow as tf
from pydantic import BaseModel
from starlette.middleware.cors import CORSMiddleware
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing import image as tf_image
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from fastapi.encoders import jsonable_encoder
import numpy as np
import requests
import uuid
from io import BytesIO
from qdrant_client.models import Distance, VectorParams
from fastapi import HTTPException

from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct

# client = QdrantClient(url="http://localhost:6333")
# http://34.199.16.216:6333/dashboard#/collections/test_collection
client = QdrantClient(url="http://34.199.16.216:6333")
COLLECTION_NAME = 'test_collection'


class CollectionData(BaseModel):
    collection_name: str


# Load ResNet50 model
resnet50 = tf.keras.applications.ResNet50(weights='imagenet', include_top=False, pooling='avg')
resnet50.trainable = False

app = FastAPI()

# CORS middleware setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_image_embedding(img_data):
    img = tf_image.load_img(img_data, target_size=(720, 720))
    img_array = tf_image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    embedding = resnet50.predict(img_array)
    # embedding = vgg16.predict(img_array)
    return embedding.squeeze()


def get_embedding_from_url(url: str):
    response = requests.get(url)
    response.raise_for_status()
    image_data = BytesIO(response.content)
    return get_image_embedding(image_data)


def generate_unique_id():
    return hash(uuid.uuid4()) & ((1<<64) - 1)


class ImageData(BaseModel):
    ss_id: int
    path: str


class SearchRequestType(BaseModel):
    img_id: int
    img_path: str
    statement_id: Optional[int] = None
    limit: Optional[int] = 1
    minimal_score: Optional[float] = None


@app.post("/search/")
async def search_image(data: ImageData):
    # Get the embedding for the image
    query_vector = get_embedding_from_url(data.path)

    # Search for similar images
    hits = client.search(
        collection_name="test",
        query_vector=query_vector,
        limit=10
    )

    print(hits)

    # Check if we got hits and extract the top match's score
    top_score = hits[0].score if hits else 0.0

    # Determine the similar_id
    if hits and hits[0].score > 0.87:
        # If a similar image is found, set similar_id to ss_id of the similar image
        similar_id = hits[0].payload["ss_id"]
        img_url = hits[0].payload["img_path"]
        response_msg = {"message": "Image added with a similar image found.", "similar_id": similar_id,
                        "img_url": img_url, "score": top_score}
    else:
        # No similar image found
        similar_id = 0
        response_msg = {"message": "Image added without a similar match.", "score": top_score}

    # Generate a unique ID for the new image
    unique_id = generate_unique_id()

    # Always upsert the image to the database
    client.upsert(
        collection_name="test",
        points=[
            PointStruct(
                id=unique_id,
                vector=query_vector.tolist(),
                payload={"ss_id": data.ss_id, "similar_id": similar_id, "img_path": data.path}
            )
        ]
    )

    return response_msg


@app.post("/search-v1/")
async def search_v1(request_data: SearchRequestType):
    try:
        # Get the embedding for the image
        img_vector = get_embedding_from_url(request_data.img_path)

        # Search for similar images
        hits = client.search(
            collection_name=COLLECTION_NAME,
            query_vector=img_vector,
            limit=request_data.limit,
            score_threshold=request_data.minimal_score
        )

        similar_img_id = hits[0].id if hits and hits[0] else None
        # similar_img_statement_id = hits[0].payload['statement_id'] if hits and hits[0] else None
        similar_img_statement_id = 0

        # Always upsert the image to the database
        client.upsert(
            collection_name=COLLECTION_NAME,
            points=[
                PointStruct(
                    id=request_data.img_id,
                    vector=img_vector.tolist(),
                    payload={
                        "statement_id": request_data.statement_id,
                        "similar_img_id": similar_img_id,
                        "img_path": request_data.img_path
                    }
                )
            ]
        )

        # If everything works well
        return {
            "success": True,
            "data": {
                "similar_images": hits,
                "entered_img_id": request_data.img_id,
                "similar_img_id": similar_img_id,
                "similar_img_statement_id": similar_img_statement_id
            }
        }

    except Exception as e:
        print(str(e))
        # Generic error handling
        raise HTTPException(status_code=500, detail={"status": "error", "message": str(e)})


@app.post("/create_collection/")
async def create_collection(data: CollectionData):
    try:
        client.recreate_collection(
            collection_name=data.collection_name,
            vectors_config=VectorParams(
                size=2048,             # default size
                distance=Distance.COSINE,   # default distance
                on_disk=True          # default on_disk value
            ),
        )
        return {"message": f"Collection '{data.collection_name}' created successfully!"}
    except Exception as e:
        return {"error": str(e)}


@app.get("/")
async def root():
    return {"message": "Hello World"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
