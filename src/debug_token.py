# debug_token.py - Diagnose HuggingFace token issues

import os
from dotenv import load_dotenv
from huggingface_hub import InferenceClient
import requests

# Load environment
load_dotenv()
token = os.getenv("HF_API_KEY")

print("="*80)
print("HUGGINGFACE TOKEN DIAGNOSTIC")
print("="*80)

# Step 1: Check if token exists
print("\n1️⃣ Token Detection:")
if token:
    print(f"   ✅ Token found: {token[:10]}...{token[-5:]}")
    print(f"   Length: {len(token)} characters")
    print(f"   Starts with 'hf_': {token.startswith('hf_')}")
else:
    print("   ❌ No token found in .env file!")
    print("   Make sure your .env file has: HF_API_KEY=hf_your_token")
    exit(1)

# Step 2: Test token validity with API
print("\n2️⃣ Token Validity Test:")
headers = {"Authorization": f"Bearer {token}"}
try:
    response = requests.get(
        "https://huggingface.co/api/whoami-v2",
        headers=headers
    )
    if response.status_code == 200:
        user_info = response.json()
        print(f"   ✅ Token is valid!")
        print(f"   User: {user_info.get('name', 'Unknown')}")
        print(f"   Type: {user_info.get('type', 'Unknown')}")
        
        # Check token permissions
        auth = user_info.get('auth', {})
        print(f"\n   Permissions:")
        print(f"   - Read: {auth.get('accessToken', {}).get('role') == 'read' or 'read' in str(auth)}")
        print(f"   - Write: {'write' in str(auth).lower()}")
    else:
        print(f"   ❌ Token validation failed: {response.status_code}")
        print(f"   Response: {response.text}")
        exit(1)
except Exception as e:
    print(f"   ❌ Error validating token: {e}")
    exit(1)

# Step 3: Test Inference API with simple model
print("\n3️⃣ Inference API Test:")

# Try different models - some might work better than others
test_models = [
    "meta-llama/Llama-3.2-1B-Instruct",
    "google/gemma-2-2b-it",
    "microsoft/Phi-3-mini-4k-instruct",
]

client = InferenceClient(token=token)

for model_name in test_models:
    print(f"\n   Testing: {model_name}")
    try:
        response = client.chat_completion(
            messages=[{"role": "user", "content": "Say hello"}],
            model=model_name,
            max_tokens=10,
            timeout=10
        )
        answer = response.choices[0].message.content
        print(f"   ✅ SUCCESS! Response: {answer[:50]}")
        break
    except Exception as e:
        error_msg = str(e)
        if "401" in error_msg:
            print(f"   ❌ 401 Unauthorized - Token issue")
        elif "403" in error_msg:
            print(f"   ❌ 403 Forbidden - Model access denied")
        elif "404" in error_msg:
            print(f"   ❌ 404 Not Found - Model doesn't exist")
        elif "429" in error_msg:
            print(f"   ❌ 429 Rate Limited")
        elif "503" in error_msg:
            print(f"   ❌ 503 Model Loading/Unavailable")
        else:
            print(f"   ❌ Error: {error_msg[:100]}")

# Step 4: Alternative - Try text_generation instead of chat_completion
print("\n4️⃣ Alternative API Test (text_generation):")
try:
    response = client.text_generation(
        "Hello, how are you?",
        model="google/gemma-2-2b-it",
        max_new_tokens=10
    )
    print(f"   ✅ text_generation works! Response: {response[:50]}")
except Exception as e:
    print(f"   ❌ text_generation failed: {str(e)[:100]}")

print("\n" + "="*80)
print("DIAGNOSIS COMPLETE")
print("="*80)

# Recommendations
print("\n💡 RECOMMENDATIONS:")
print("\n1. If token is invalid:")
print("   - Go to https://huggingface.co/settings/tokens")
print("   - Create a new token with 'read' access")
print("   - Update your .env file")
print("   - Restart your terminal/Python")

print("\n2. If specific models fail:")
print("   - Some models require accepting their license first")
print("   - Visit the model page on HuggingFace")
print("   - Example: https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct")
print("   - Click 'Agree and access repository'")

print("\n3. If 401 persists for all models:")
print("   - Token might not be properly formatted")
print("   - Make sure .env has: HF_API_KEY=hf_your_token (no quotes)")
print("   - Try copying token directly into code temporarily for testing")

print("\n4. If rate limited:")
print("   - Wait a few minutes")
print("   - Consider HuggingFace Pro ($9/month)")

print("\n5. Alternative solution:")
print("   - Use text_generation() instead of chat_completion()")
print("   - Some models work better with one API than the other")