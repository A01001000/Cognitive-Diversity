import os
import requests
from dotenv import load_dotenv

load_dotenv(override=True)
naga_key = os.getenv("NAGAAI_API_KEY")

headers = {"Authorization": f"Bearer {naga_key}"}
response = requests.get("https://api.naga.ac/v1/models", headers=headers)

if response.status_code == 200:
    models = response.json().get("data", [])
    # Filter ONLY for models that have 'free' in their ID
    free_models = [m["id"] for m in models if "free" in m["id"].lower()]
    
    print("Available FREE models on NagaAI:")
    for m in free_models:
        print(f"- {m}")
else:
    print("Failed to fetch models.")