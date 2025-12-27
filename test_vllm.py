from openai import OpenAI

"""
First, run the vllm server with the command:

vllm serve google/gemma-3-1b-it \                            
  --enable-lora \
  --lora-modules ner=gemma-ner-lora \
--tokenizer gemma-ner-lora
"""

client = OpenAI(base_url="http://localhost:8000/v1", api_key="EMPTY")

test_input = "This is a demo of how vLLM can be used to serve LLMs easily."

raw_prompt = f"Extract entities and relationships from the text below as JSON.\n\nInput: {test_input}\n\nJSON Output:\n"

response = client.completions.create(
    model="ner",
    prompt=raw_prompt,
    max_tokens=200,
    temperature=0.1
)

print(response.choices[0].text)