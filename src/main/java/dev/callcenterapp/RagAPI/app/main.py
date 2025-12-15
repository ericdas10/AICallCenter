from fastapi import FastAPI
from pydantic import BaseModel
from app.embeddings import Embedder
from app.generator import generate_with_ollama
from app.utils.similarity import cosine_similarity

app = FastAPI(title="Local RAG API")

embedder = Embedder()
conversation_cache = []

SIMILARITY_THRESHOLD_STRICT = 0.90     # identical / nearly identical
SIMILARITY_THRESHOLD_APPROX = 0.75     # paraphrased / similar meaning

class AskRequest(BaseModel):
    query: str


@app.post("/ask")
def ask(body: AskRequest):
    query = body.query
    
    # 1. Compute embedding
    query_embedding = embedder.model.encode([query])[0]

    # 2. Search conversation cache
    best_match = None
    best_score = 0

    for item in conversation_cache:
        score = cosine_similarity(query_embedding, item["embedding"])
        
        if score > best_score:
            best_score = score
            best_match = item

    # 3. Strict similarity (≥ 0.90) → return cached answer instantly
    if best_score >= SIMILARITY_THRESHOLD_STRICT:
        return {
            "answer": best_match["answer"],
            "cached": True,
            "match_type": "strict",
            "similarity_score": float(best_score)
        }


    # 4. Approx similarity (≥ 0.75) → refine using cached answer
    if best_score >= SIMILARITY_THRESHOLD_APPROX:
        previous_answer = best_match["answer"]

        refine_prompt = f"""
        A similar question was asked before. Here is the previous answer:

        {previous_answer}

        Now answer the new question:

        {query}

        Use the previous answer as context, but improve and adapt it to the new question.
        """

        refined_answer = generate_with_ollama(refine_prompt)

        # Add refined result to cache
        conversation_cache.append({
            "question": query,
            "embedding": query_embedding.tolist(),
            "answer": refined_answer
        })

        return {
            "answer": refined_answer,
            "cached": True,
            "match_type": "refined_from_cache",
            "similarity_score": float(best_score)
        }


    # 5. No similarity → use RAG
    results = embedder.search(query, top_k=2)

    if not results["documents"]:
        return {"answer": "No relevant documents found."}

    context = " ".join(results["documents"][0])

    prompt = f"""
    Using ONLY the context below, answer the question directly.

    Context:
    {context}

    Question: {query}
    Answer:
    """

    answer = generate_with_ollama(prompt)

    # store new answer in cache
    conversation_cache.append({
        "question": query,
        "embedding": query_embedding.tolist(),
        "answer": answer
    })

    return {
        "answer": answer,
        "cached": False,
        "match_type": "none"
    }
