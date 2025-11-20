from src.data_loader import load_all_documents
from src.vectorstore import FaissVectorStore
from src.search import RAGSearch

# Example usage
if __name__ == "__main__":
    
    docs = load_all_documents("data")
    store = FaissVectorStore("faiss_store")
    store.build_from_documents(docs)  # ← Uncomment this to build the index first
    store.save()  # ← Add this to save the index
    # store.load()  # ← Comment out or use after building
    
    rag_search = RAGSearch()
    query = "Who is Messi?"
    summary = rag_search.search_and_summarize(query, top_k=3)
    print("Summary:", summary)