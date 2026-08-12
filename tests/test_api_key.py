import os

import google.generativeai as genai

api_key = os.getenv("GEMINI_API_KEY", "")
model_name = os.getenv("GEMINI_MODEL", "gemini-3.6-flash")

genai.configure(api_key=api_key)

try:
    model = genai.GenerativeModel(model_name)

    response = model.generate_content('Greet me and tell me whether the connection is successful or not.')

    print("\n Connection successful!")
    print(f"Model response: {response.text}")

except Exception as e:
    print("\n Error connecting to Gemini API:")
    print(e)
