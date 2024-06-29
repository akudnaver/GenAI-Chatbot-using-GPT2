import argparse
import torch
# import numpy as np

from pdfminer.high_level import extract_text
from sentence_transformers import SentenceTransformer, CrossEncoder, util

from text_generation import Client

# Verify CUDA availability
print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("Current CUDA device:", torch.cuda.current_device())
    print("CUDA device name:", torch.cuda.get_device_name(torch.cuda.current_device()))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fname", type=str, required=True)
    parser.add_argument("--top-k", type=int, default=32)
    parser.add_argument("--window-size", type=int, default=128)
    parser.add_argument("--step-size", type=int, default=100)
    return parser.parse_args()


def embed(fname, window_size, step_size):
    text = extract_text(fname)
    text = " ".join(text.split())
    text_tokens = text.split()

    sentences = []
    for i in range(0, len(text_tokens), step_size):
        window = text_tokens[i : i + window_size]
        if len(window) < window_size:
            break;
        sentences.append(window)

    paragraphs = [" ".join(s) for s in sentences]
    print(paragraphs)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2", device=device)

    model.max_seq_length = 512
    cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", device=device)
    
    embeddings = model.encode(paragraphs, show_progress_bar=True, convert_to_tensor=True)

    return model, cross_encoder, embeddings, paragraphs


def search(query, model, cross_encoder, embeddings, paragraphs, top_k):
    query_embeddings = model.encode(query, convert_to_tensor=True)
    query_embeddings = query_embeddings.cuda() if torch.cuda.is_available() else query_embeddings
    hits = util.semantic_search(
        query_embeddings,
        embeddings,
        top_k=top_k,
    )[0]

    cross_inputs = [[query, paragraphs[hit["corpus_id"]]] for hit in hits]
    cross_scores = cross_encoder.predict(cross_inputs)
    for idx in range(len(cross_scores)):
        hits[idx]["cross_score"] = cross_scores[idx]
    
    results = []
    hits = sorted(hits, key=lambda x: x["cross_score"], reverse=True)
    for hit in hits[:5]:
        results.append(paragraphs[hit["corpus_id"]].replace("\n", " "))
    return results


if __name__ == "__main__":
    args = parse_args()
    model, cross_encoder, embeddings, paragraphs = embed(
        args.fname, 
        args.window_size, 
        args.step_size,
    )
    print(embeddings.shape)
    while True:
        print("\n")
        query = input("Please enter your query:")
        results = search(
            query,
            model,
            cross_encoder,
            embeddings,
            paragraphs,
            top_k=args.top_k,
        )
        print(results)
