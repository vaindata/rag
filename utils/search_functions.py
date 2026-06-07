from qdrant_client.http import models
#Helper function for semantic search
def semantic_search(client, collection_name: str, query: str, embedding_model, top_k: int = 5):

    # ✅ Embed query
    query_dense = embedding_model.embed_query(query)

    # ✅ Query Qdrant (dense vector)
    results = client.query_points(
        collection_name=collection_name,
        query=query_dense,
        using="dense",         # <-- force semantic vector search
        with_payload=True,
        limit=top_k
    )

    # ✅ Format results
    output = []
    for point in results.points:
        output.append({
            "id": str(point.id),
            "score": point.score,
            "text": point.payload.get("text", "")
        })
    
    return output

#Helper function for hybrid_search
def hybrid_search(client, collection_name: str, query: str, embedding_model, bm25_encoder, top_k: int = 5, prefetch_limit: int = 20):
    
    # Generate dense embedding
    query_dense = embedding_model.embed_query(query)
    
    # Generate sparse BM25 embedding
    query_sparse = list(bm25_encoder.passage_embed([query]))[0].as_object()
    
    # Setup prefetch queries for both dense and sparse search
    prefetch = [
        models.Prefetch(
            query=query_dense,
            using="dense",  # Dense vector name in your collection
            limit=prefetch_limit,
        ),
        models.Prefetch(
            query=models.SparseVector(**query_sparse),
            using="bm25",   # Sparse vector name in your collection
            limit=prefetch_limit,
        ),
    ]
    
    # Execute hybrid search with RRF fusion
    results = client.query_points(
        collection_name,
        prefetch=prefetch,
        query=models.FusionQuery(
            fusion=models.Fusion.RRF,  # Reciprocal Rank Fusion
        ),
        with_payload=True,
        limit=top_k,
    )
    
    # Format results
    output = []
    for point in results.points:
        output.append({
            "id": str(point.id),
            "score": point.score,
            "text": point.payload.get("text", "")
        })
    
    return output