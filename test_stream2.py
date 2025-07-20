import httpx

# ✅ LOCAL VLLM CONFIGURATION
VLLM_URL_BASE = "http://10.0.0.2:8000/v1"
PROXY_URL = "http://localhost:7000/chat/completions"

# ✅ MODEL (From your `curl` response to `/models`)
VLLM_MODEL_NAME = "hello"  # Model ID returned from `curl /models`

REQUEST_BODY = {
    "model": VLLM_MODEL_NAME,
    "messages": [
        {"role": "user", "content": "Write a Python script that outputs 'Hello, World!'\n\n"}
    ],
    "stream": True,
    "temperature": 0.7,
    "max_tokens": 1000
}

# Use proxy with upstream pointing to your VLLM instance
REQUEST_URL = f"{PROXY_URL}?p={VLLM_URL_BASE}"

def test_stream():
    try:
        with httpx.Client() as client:
            print(f"Connecting to proxy with upstream VLLM: {REQUEST_URL}")
            with client.stream("POST", REQUEST_URL, json=REQUEST_BODY, timeout=60) as response:
                if response.status_code != 200:
                    print(f"🚨 Proxy error: {response.status_code} - {response.text}")
                    return

                print("🚀 Streaming response:\n")
                for chunk in response.iter_text():
                    print(chunk, end="")

            print("\n\n✅ Streaming complete.")
    except Exception as e:
        print(f"❌ Test failed: {e}")

if __name__ == "__main__":
    test_stream()