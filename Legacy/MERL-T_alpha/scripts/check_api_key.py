#!/usr/bin/env python3
"""Check if OpenRouter API key is loaded"""

import os
from dotenv import load_dotenv

# Load .env file
load_dotenv()

api_key = os.environ.get("OPENROUTER_API_KEY")

if api_key:
    print(f"✅ API key loaded: {api_key[:15]}...{api_key[-4:]}")
    print(f"   Length: {len(api_key)} chars")
else:
    print("❌ API key NOT loaded")
    print("   Checking .env file...")

    if os.path.exists(".env"):
        print("   .env file exists")
        with open(".env") as f:
            for line in f:
                if "OPENROUTER_API_KEY" in line:
                    print(f"   Found in .env: {line.strip()[:50]}...")
    else:
        print("   .env file NOT FOUND")
