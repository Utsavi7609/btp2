from google import genai

# Paste your key here
GOOGLE_API_KEY = "AIzaSyBp-tlfJcr6nX5mL_yWLwCw3s_X2ywvjFc"
client = genai.Client(api_key=GOOGLE_API_KEY)

print("Checking available models for your key...")
try:
    # This asks the server: "What models can I use?"
    for model in client.models.list():
        print(f" - {model.name}")
except Exception as e:
    print(f"Error checking models: {e}")