from openai import OpenAI
import os
import json
import re
import time

def extract_json(text: str):
    # Remove triple-backtick fences like ```json or ``` anything
    cleaned = re.sub(r"```(?:json)?", "", text, flags=re.IGNORECASE).strip()
    
    # If extra trailing backticks remain
    cleaned = cleaned.replace("```", "").strip()
    
    match = re.search(r"(\{.*\}|\[.*\])", cleaned, flags=re.DOTALL)
    if match:
        cleaned = match.group(1)

    # Try parsing normally
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        # Last-chance cleanup: remove trailing commas
        cleaned = re.sub(r",\s*([}\]])", r"\1", cleaned)
        return json.loads(cleaned)  # will raise again if still invalid

PROMPT = """
You are a synthetic data generation model that extracts entities and relationship triplets from sentences. Generate a JSON list of inputs and outputs in the following format:
{
"input": "Mukul is working on deploying the Mattermost server on AWS."
"output": {
"entities": ["Mukul", "Mattermost", "AWS"],
"relationships": [
["Mukul", "DEPLOYS", "Mattermost"],
["Mattermost", "DEPLOYED_ON", "AWS"]
]
}
}

Generate only the json and nothing else. You are part of a larger pipeline and it will break if you generate anything other than the json.

Do not generate the example I provided

generate as many synthetic data pairs as you can
"""

client = OpenAI(
    api_key=os.environ.get("GEMINI_API_KEY"),
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)


def generate_content():
    response = client.chat.completions.create(
    model="models/gemini-flash-lite-latest",
    messages=[
        {
            "role": "user",
            "content": PROMPT
        }
    ]
)

    return response.choices[0].message.content

def append_json(FILE_NAME: str, json_content):
    if json_content is None:
        return

    # Ensure directory exists
    os.makedirs(os.path.dirname(FILE_NAME) or ".", exist_ok=True)

    # Load old items
    existing = []
    if os.path.exists(FILE_NAME):
        try:
            raw = open(FILE_NAME, "r", encoding="utf-8").read().strip()
            if raw:
                parsed = extract_json(raw)
                existing = parsed if isinstance(parsed, list) else [parsed]
        except BaseException as e:
            print(f"ERROR: {e}")
            existing = []

    new_items = json_content if isinstance(json_content, list) else [json_content]
    combined = existing + new_items
    with open(FILE_NAME, "w", encoding="utf-8") as wf:
        json.dump(combined, wf, ensure_ascii=False, indent=2)

FILE_NAME = "entities.json"

def main():
    try:
        json_content = extract_json(generate_content())
    except Exception as e:
        print(f"ERROR: {e}")
    append_json(FILE_NAME, json_content)

if __name__ == "__main__":
    count = 0
    while True:
        try:
            print(f"Request No. {count}")
            main()
            count += 1
            print("\tRequest Completed")
        except Exception as e:
            print(f"ERROR: {e}")

        time.sleep(3)
