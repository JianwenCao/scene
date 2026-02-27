import os
import google.generativeai as genai

api_key = os.environ.get("GEMINI_API_KEY", "AIzaSyDzcaxpz7Qm5m6vWZxOZCs4lPHw6OF_S5U")
genai.configure(api_key=api_key)

model = genai.GenerativeModel('gemini-2.0-flash')
try:
    response = model.generate_content("Extract the main object name from this query: 'berber carpet in the room near the rack'. Output only the object name.")
    print(f"Response: {response.text.strip()}")
except Exception as e:
    print(f"Error: {e}")
