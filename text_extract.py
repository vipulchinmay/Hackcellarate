import requests
import json

def extract_text(messages, text):
    api_key = "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"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://192.168.108.133:8080"  # Replace with your site URL
    }

    messages.append({"role": "user", "content": text})
    # Create your request payload
    payload = {
        "model": "meta-llama/llama-3.1-8b-instruct:free",
        "messages": messages, 
        "temperature": 0.7,
        "max_tokens": 150
    }

    # Make the API request
    response = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers=headers,
        json = payload
    )

    # Parse the response
    result = response.json()
    print(result)
    
    # Extract just the model's response text (nothing else)
    if "choices" in result and len(result["choices"]) > 0:
        assistant_response = result["choices"][0]["message"]["content"]
        print(assistant_response)
        return assistant_response
    else:
        print("Error or unexpected response format:")
        print(json.dumps(result, indent=2))
        return "Sorry, I encountered an error."