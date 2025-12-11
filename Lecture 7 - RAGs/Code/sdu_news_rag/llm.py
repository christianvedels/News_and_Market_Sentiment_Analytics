import torch
from transformers import pipeline

class LLM():

    def __init__(self, context = "You are a an expert of financial news and I am a student who wants to learn more about the stock market."):
        self.context = context
        self.pipe = self.load_model()

    def load_model(self):
        """
        Load a pretrained language model and tokenizer using the Hugging Face pipeline.
        """
        # Initialize the text generation pipeline with the specified model
        pipe = pipeline("text-generation", model="HuggingFaceH4/zephyr-7b-beta", torch_dtype=torch.bfloat16, device_map="auto")
        return pipe

    def construct_prompt(self, context, prompt):
        """
        Construct a prompt from the given context and prompt.
        """
        # We use the tokenizer's chat template to format each message - see https://huggingface.co/docs/transformers/main/en/chat_templating
        messages = [
            {
                "role": "system",
                "content": context,
            },
            {
                "role": "user", 
                "content": prompt
            },
        ]
        prompt = self.pipe.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        return prompt

    def __call__(self, text, temperature=0.7):
        """
        Generate text based on the given prompt using the loaded model.
        """
        # Generate text using the model
        prompt = self.construct_prompt(self.context, text)
        outputs = self.pipe(prompt, max_new_tokens=256, do_sample=True, temperature=temperature, top_k=50, top_p=0.95)
        return outputs[0]["generated_text"]
    
if __name__ == "__main__":
    llm_instance = LLM()
    results = [
        llm_instance("How are the capital markets in France doing recently?", temperature=0.7),
        llm_instance("How are the capital markets in France doing recently?", temperature=0.001),
        llm_instance("How are the capital markets in France doing recently?", temperature=1.0)
    ]
    
    for x in results:
        print(x)
