from google import genai
from google.genai import types
import os
from dotenv import load_dotenv

load_dotenv()

def test_models_and_caching():
    """Test available models and caching support"""
    print("Testing Google AI Models and Caching Support")
    print("=" * 60)

    try:
        client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

        print("🔍 Listing available models...")
        models = client.models.list()  # returns a Pager, not an object with .models

        caching_supported_models = []

        # Iterate directly over the pager
        for model in models:
            print(f"Model: {model.name}")
            print(f"  Display Name: {getattr(model, 'display_name', 'N/A')}")
            print(f"  Description: {getattr(model, 'description', 'N/A')}")

            methods = getattr(model, "supported_generation_methods", [])
            print(f"  Supported Methods: {methods}")

            if any("cache" in str(m).lower() for m in methods):
                caching_supported_models.append(model.name)
                print("  ✅ Caching: SUPPORTED")
            else:
                print("  ❌ Caching: NOT SUPPORTED")

            print("-" * 60)

        print("\n🎯 Models that support caching:")
        if caching_supported_models:
            for model_name in caching_supported_models:
                print(f"  ✅ {model_name}")
        else:
            print("  ❌ None found with explicit caching support")

        # Test caching
        print("\n🧪 Testing caching with gemini-1.5-flash...")
        try:
            test_cache = client.caches.create(
                model="models/gemini-1.5-flash",
                config=types.CreateCachedContentConfig(
                    system_instruction="You are a helpful assistant.",
                    ttl=300,
                ),
            )
            print(f"✅ Cache created: {test_cache.name}")
            client.caches.delete(name=test_cache.name)
            print("🗑️  Cache deleted successfully")
        except Exception as e:
            print(f"❌ Caching test failed: {e}")

    except Exception as e:
        print(f"❌ Error testing models: {e}")


if __name__ == "__main__":
    test_models_and_caching()
