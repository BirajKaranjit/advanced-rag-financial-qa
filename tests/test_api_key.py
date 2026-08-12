
## To check whether the api key is working or not
# import os
# from dotenv import load_dotenv
# import google.generativeai as genai
# load_dotenv()
#
#
# api_key = os.getenv("GEMINI_API_KEY", "")
# model_name = os.getenv("GEMINI_MODEL", "gemini-3.6-flash")
#
# genai.configure(api_key=api_key)
#
# try:
#     model = genai.GenerativeModel(model_name)
#
#     response = model.generate_content('Greet me and tell me whether the connection is successful or not.')
#
#     print("\n Connection successful!")
#     print(f"Model response: {response.text}")
#
# except Exception as e:
#     print("\n Error connecting to Gemini API:")
#     print(e)




##to check supported models
from google import genai
import os
from dotenv import load_dotenv

load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

for model in client.models.list():
    if "generateContent" in (model.supported_actions or []):
        print(model.name)
