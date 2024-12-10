from .llm import LLM
from transformers import AutoTokenizer, AutoModel, pipeline
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import torch

class sduNewsRag():
    def __init__(self, model_context="You are a finance expert. You pick up on trends in the market", model_name_embedding="google-bert/bert-base-uncased"):
        # Initialize the LLM with the given context
        self.finance_expert = LLM(context=model_context)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_embedding)
        self.model = AutoModel.from_pretrained(model_name_embedding).to("cuda")
        self.news = []
        self.news_vectors = None
        self.zero_shot_classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")


    def embed_text(self, text):
        """
        Embed a single piece of text using the transformers model.
        """
        inputs = self.tokenizer(text, return_tensors='pt', truncation=True, padding=True).to("cuda")
        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()

    def vector_lib(self, news_list):
        """
        Store the news articles and pre-compute their embeddings.
        """
        self.news = news_list
        # Embed all news documents
        self.news_vectors = np.array([self.embed_text(x) for x in news_list])

    def retrieve_relevant_docs(self, query, n_docs=3):
        """
        Retrieve the most relevant documents from the news corpus based on semantic similarity.
        """
        query_vector = self.embed_text(query).reshape(1, -1)

        # Compute cosine similarities with all news vectors
        sims = cosine_similarity(query_vector, self.news_vectors)[0]
        best_indices = np.argsort(sims)[-n_docs:][::-1]
        return [(self.news[idx], sims[idx]) for idx in best_indices]
    
    def censoring(self, prompt, retrieved_docs, response, timeout=2):
        """
        Censor the text using the LLM.
        """
        iter = 0
        while iter < timeout:
            iter += 1
            retrieved_docs_text = "\n".join([f"Document {i+1}: {doc}" for i, (doc, _) in enumerate(retrieved_docs)])
            censored_text = self.finance_expert(
                f"Prompt: {prompt};\nContext: {retrieved_docs_text};\nResponse: {response}.\n\nIs this analysis sensible? Only say 'yes' or 'no'. I will fire you if you say anything else than 'yes' or 'no'. I will promote you if you follow my instruction."
            )
            censor_response = censored_text.split("<|assistant|>")[-1].strip()
            if "yes" in censor_response.lower()[0:100]: # "Yes" in the first 100 characters
                return "yes"
            elif "no" in censor_response.lower()[0:100]: # "No" in the first 100 characters
                return "no"
        
        return f"failed_to_censor_after_{timeout}_attempts"
    
    def bull_bear_sentiment(self, response):
        """
        Determine the sentiment of the retrieved document based on the user query.
        """
        sentiment = self.finance_expert(f"Prompt: {response}. Is this analysis bullish, neutral or bearish? Please answer 'bull', 'neut' or 'bear'. If you don't follow my instruction, I will fire you. I will promote you if you follow my instruction. Only say 'bull', 'neut' or 'bear'.") 
        sentiment = sentiment.split("<|assistant|>")[-1].strip()

        return sentiment

    def query_rag(self, user_query, max_iter=1, n_docs=1):
        """
        Perform a RAG-style query:
        1. Retrieve the most relevant documents from the news corpus.
        2. Combine the user query and the retrieved documents.
        3. Use the LLM to generate a response.
        """
        if len(self.news) == 0 or self.news_vectors is None:
            raise ValueError("No news corpus found. Please call `vector_lib(news_list)` first.")
        
        retrieved_docs = self.retrieve_relevant_docs(user_query, n_docs)

        combined_prompt = (
            f"The following is a financial news analysis task.\n\n"
            f"User Query:\n{user_query}\n\n"
            f"Relevant News Documents:\n" +
            "\n".join([f"Document {i+1}: {doc}" for i, (doc, _) in enumerate(retrieved_docs)]) +
            f"\n\nBased on the relevant context and the user query, provide an insightful answer:"
        )

        censor_response = "no"
        iters = 0
        while censor_response == "no": # Run until censor is satisfied, "yes"

            # Generate response from the LLM
            response = self.finance_expert(combined_prompt)

            # Extract answer only
            response_text = response.split("<|assistant|>")[-1].strip()

            # Switch context to censoring context
            context0 = self.finance_expert.context

            self.finance_expert.context = """
            You are a financial expert. Your role is to make sure that the new fancy LLM, 
            that the company has just acquired, is working as expected. You are asked, 
            in our expert opinion whether the analysis is sensible. You answer with 'yes' or 'no'.
            You don't add any comments. 
            """

            # Censor the response
            censor_response = self.censoring(user_query, retrieved_docs, response_text)

            # Rest original context
            self.finance_expert.context = context0
            
            # Stop if max iter
            iters += 1
            if iters > max_iter:
                break

        # Determine sentiment
        sentiment = self.bull_bear_sentiment(response_text)

        # One shot classification of sentiment
        self.zero_shot_classifier(
            "I have a problem with my iphone that needs to be resolved asap!",
            candidate_labels=["bull", "neutral", "bear"],
        )

        return sentiment, response_text, retrieved_docs

if __name__ == "__main__":
    # Suppose we have some financial news snippets
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

    user_question = "What is the sentiment around the American tech industry right now?"
    sentiment, response, retrieved_docs = rag.query_rag(user_question)
    print("User Question:", user_question)
    print("Retrieved Documents:", retrieved_docs)
    print("LLM Response:", response)