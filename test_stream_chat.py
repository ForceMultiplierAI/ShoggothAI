import httpx

# Configuration
API_URL = "http://localhost:7000/chat/completions"
UPSTREAM_URL = "https://openrouter.ai/api/v1"
MODEL_NAME = "qwen/qwen3-coder:floor"

# Request body
REQUEST_BODY = {
    "model": MODEL_NAME,
    "messages": [
        {"role": "user", "content": "Write a Python script that outputs 'Hello, World!'\n\n"}
    ],
    "stream": True,
    "temperature": 0.7,
    "max_tokens": 1000
}

# Query parameter (UPSTREAM_URL to proxy)
REQUEST_URL = f"{API_URL}?p={UPSTREAM_URL}"

def test_stream():
    with httpx.Client() as client:
        print(f"Connecting to proxy: {REQUEST_URL}")
        with client.stream("POST", REQUEST_URL, json=REQUEST_BODY, timeout=60) as response:
            if response.status_code != 200:
                print(f"Error: {response.status_code} - {response.text}")
                return

            print("Streaming started...\n")
            for chunk in response.iter_text():
                print(chunk, end="")  # Print token-by-token as it streams

    print("\n\nStreaming complete.")

if __name__ == "__main__":
    test_stream()