import requests
import logging

OLLAMA_API_URL = "http://localhost:11434/api/generate" 

def get_response(user_input):
    payload = {
        "model": "llama3", 
        "prompt": user_input,
        "stream": False  
    }
    headers = {
        "Content-Type": "application/json"
    }
    try:
        response = requests.post(OLLAMA_API_URL, json=payload, headers=headers)
        print(response.status_code, response.text)  
        logging.info(f"Request to {OLLAMA_API_URL} with payload {payload}")
        if response.status_code == 200:
            return response.json().get("response", "No response found.")
        else:
            logging.error(f"Error: Received status code {response.status_code}")
            return f"Error: Unable to generate a response. Status code: {response.status_code}, Response: {response.text}"
    except Exception as e:
        logging.error(f"Error during API request: {str(e)}")
        return f"Error: {str(e)}"
