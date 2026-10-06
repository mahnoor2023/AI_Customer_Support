from pathlib import Path
import json

import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from crewai.tools import tool


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

FAISS_PATH = DATA_DIR / "faiss.index"
CHUNKS_PATH = DATA_DIR / "chunks.json"
ORDERS_PATH = DATA_DIR / "orders.xlsx"


# Load the existing company knowledge index
faiss_index = faiss.read_index(str(FAISS_PATH))

with open(CHUNKS_PATH, "r", encoding="utf-8") as file:
    chunks_data = json.load(file)


# This MUST be the same embedding model used when
# the original FAISS index was created.
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")


# Load the simulated order database
orders_df = pd.read_excel(ORDERS_PATH)

# Normalize column names
orders_df.columns = [
    str(column).strip()
    for column in orders_df.columns
]


@tool("company_knowledge_search")
def company_knowledge_search(question: str) -> str:
    """
    Search the company's internal knowledge base.

    Use this tool when the customer asks about company policies,
    products, services, troubleshooting, returns, refunds,
    shipping policies, warranties, FAQs, or other company information.
    """

    question_embedding = embedding_model.encode(
        [question],
        convert_to_numpy=True
    )

    question_embedding = question_embedding.astype("float32")

    # Search the five most relevant chunks
    distances, indices = faiss_index.search(
        question_embedding,
        5
    )

    results = []

    for distance, index in zip(distances[0], indices[0]):

        if index < 0:
            continue

        chunk = chunks_data[index]

        results.append(
            {
                "similarity_distance": float(distance),
                "text": chunk
            }
        )

    if not results:
        return "No relevant company information was found."

    formatted_results = []

    for number, result in enumerate(results, start=1):

        formatted_results.append(
            f"""
Result {number}

Similarity distance:
{result["similarity_distance"]}

Content:
{result["text"]}
"""
        )

    return "\n".join(formatted_results)


@tool("order_lookup")
def order_lookup(order_id: str) -> str:
    """
    Look up a customer's order using the order ID.

    Use this tool when the customer asks about a specific order,
    including its status, product, order date, shipping address,
    or other order information.
    """

    requested_order_id = order_id.strip().upper()

    # Try exact match first
    matches = orders_df[
        orders_df["Order ID"]
        .astype(str)
        .str.strip()
        .str.upper()
        == requested_order_id
    ]

    if matches.empty:
        return (
            f"No order was found with Order ID "
            f"{requested_order_id}."
        )

    order = matches.iloc[0]

    order_information = []

    for column in orders_df.columns:
        value = order[column]

        if pd.isna(value):
            value = "Not available"

        order_information.append(
            f"{column}: {value}"
        )

    return "\n".join(order_information)
