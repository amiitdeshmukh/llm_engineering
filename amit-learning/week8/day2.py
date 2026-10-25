import os, logging, re, chromadb
from dotenv import load_dotenv
from huggingface_hub import login
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.manifold import TSNE
import plotly.graph_objects as go
from litellm import completion
from tqdm import tqdm
from agents.evaluator import evaluate
from agents.items import Item

load_dotenv()
DB_NAME="products_vectorstore"

hf_token = os.getenv("HF_TOKEN")
login(hf_token, add_to_git_credential=True)

LITE_MODE = True

username= "ed-donner"
dataset = f"{username}/items_lite" if LITE_MODE else f"{username}/items_full"

train, val, test = Item.from_hub(dataset)
print(f"Loaded {len(train):,} training items, {len(val):,} validation items, {len(test):,} test items")

client = chromadb.PersistentClient(path=DB_NAME)

# SentenceTransformer Encoding LLM

encoder = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# vector = encoder.encode(["A proficient AI engineer who has almost reached the finale of AI Engineering Core Track!"])[0]

# print(vector.shape)
# print(vector)


collection_name = "products"
existing_collection_name = [collection.name for collection in client.list_collections()]

if collection_name not in existing_collection_name:
    collection = client.create_collection(collection_name)
    for i in tqdm(range(0,len(train), 100)):
        documents = [item.summary for item in train[i: i+100]]
        vectors = encoder.encode(documents).astype(float).tolist()
        metadatas = [{"category": item.category, "price": item.price} for item in train[i: i+100]]
        ids = [f"doc_{j}" for j in range(i, i+100)]
        ids = ids[:len(documents)]
        collection.add(ids=ids, documents=documents, embeddings=vectors, metadatas=metadatas)

collection = client.get_or_create_collection(collection_name)

# MAXIMUM_DATAPOINTS = 10_000

# CATEGORIES = ['Appliances', 'Automotive', 'Cell_Phones_and_Accessories', 'Electronics','Musical_Instruments', 'Office_Products', 'Tools_and_Home_Improvement', 'Toys_and_Games']
# COLORS = ['cyan', 'blue', 'brown', 'orange', 'yellow', 'green' , 'purple', 'red']

# result = collection.get(include=['embeddings', 'documents', 'metadatas'], limit=MAXIMUM_DATAPOINTS)
# vectors = np.array(result['embeddings'])
# documents = result['documents']
# categories = [metadata['category'] for metadata in result['metadatas']]
# colors = [COLORS[CATEGORIES.index(c)] for c in categories]

# tsne = TSNE(n_components=3, random_state=42)
# reduced_vectors = tsne.fit_transform(vectors)

# # Create the 3D scatter plot
# fig = go.Figure(data=[go.Scatter3d(
#     x=reduced_vectors[:, 0],
#     y=reduced_vectors[:, 1],
#     z=reduced_vectors[:, 2],
#     mode='markers',
#     marker=dict(size=2, color=colors, opacity=0.7),
#     text=[f"Category: {c}<br>Text: {d[:50]}..." for c, d in zip(categories, documents)],
#     hoverinfo='text'
# )])

# fig.update_layout(
#     title='3D Chroma Vector Store Visualization',
#     scene=dict(xaxis_title='x', yaxis_title='y', zaxis_title='z'),
#     width=1200,
#     height=800,
#     margin=dict(r=20, b=10, l=10, t=40)
# )

# fig.show()


def vector(item):
    return encoder.encode(item.summary)

def find_similars(item):
    vec = vector(item)
    results = collection.query(query_embeddings=vec.astype(float).tolist(), n_results=5)
    documents = results['documents'][0][:]
    prices =[m['price'] for m in results['metadatas'][0][:]]
    return documents, prices

# print(find_similars(test[0]))

def make_context(similars, prices):
    message = "For context, here are some other items that might be similar to the item you need to estimate.\n\n"
    for similar, price in zip(similars, prices):
        message += f"Potentially related product:\n{similar}\nPrice is ${price:.2f}\n\n"
    return message


def message_for(item, similar, prices):
    message = f"Estimate the price of this product. Respond with the price, no explanation\n\n{item.summary}\n\n"
    message += make_context(similar, prices)
    return [{"role": "user", "content": message}]

def gpt_6_rag(item):
    documents, prices = find_similars(item)
    response = completion(model="gpt-6-luna", messages=message_for(item, documents, prices), reasoning_effort="low", seed=42)
    return response.choices[0].message.content

print(test[0].price)

print(gpt_6_rag(test[0]))

evaluate(gpt_6_rag, test)