import requests

prompt = "AI를 간단하게 설명해줘"

response = requests.post(
    "http://localhost:11434/api/generate",
    json={
        "model": "EEVE-Korean-10.8B",
        "prompt": prompt,
        "options": {
            "temperature": 0.7,
            "top_p": 0.9
        },
        "stream": True
    }
)

print(response.json()["response"])
