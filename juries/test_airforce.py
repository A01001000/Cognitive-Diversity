import os
import requests
from dotenv import load_dotenv

load_dotenv(override=True)
airforce_key = os.getenv("AIRFORCE_API_KEY")

headers = {"Authorization": f"Bearer {airforce_key}"}
response = requests.get("https://api.airforce/v1/models", headers=headers)

if response.status_code == 200:
    models = response.json().get("data", [])
    
    print("Available 'Mini' weight class models on Api.Airforce:")
    for m in models:
        model_id = m["id"].lower()
        # Look for the typical small-weight identifiers
        if "8b" in model_id or "9b" in model_id or "mistral" in model_id or "haiku" in model_id:
            print(f"- {m['id']}")
else:
    print("Failed to fetch models.")