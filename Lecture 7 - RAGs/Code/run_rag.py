from sdu_news_rag.rag import sduNewsRag

# Suppose we have some financial news snippets (load larger library for better results)
news_corpus = [
    "The company's share price soared after positive quarterly earnings.",
    "European markets opened lower today, signaling investor pessimism.",
    "The newly appointed CEO announced major cost-cutting measures.",
    "US tech stocks rallied following strong consumer demand.",
    "NVIDIA's acquisition of ARM is facing regulatory scrutiny.",
    "Apple's new product launch was met with mixed reviews.",
]

rag = sduNewsRag()
rag.vector_lib(news_corpus)

user_question = "What is the sentiment around Apple today?"
conclusion, sentiment, response_text, retrieved_docs = rag.query_rag(user_question)
print("User Question:", user_question)
print("Retrieved Documents:", retrieved_docs)
print("LLM Response:", response_text)
print("LLM Response sentiment:", sentiment)
print("LLM Response conclusion:", conclusion)