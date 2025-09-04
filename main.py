import requests

resp = requests.post("http://localhost:8000/predict", json={
    "ticket_text": "Hi, can you help me switch from monthly to annual billing?"
})
print(resp.json())
print("\n"+"="*100+"\n")
resp = requests.post("http://localhost:8000/predict", json={
    "ticket_text": "Your app crashed after the update. I lost hours of work—this is unacceptable and I need help NOW."
})
print(resp.json())



# API endpoint
url = "http://localhost:8000/batch"

# Example batch of tickets
payload = {
    "tickets": [
        "Your app crashed after the update. I lost hours of work—please help ASAP!",
        "Hi, can you help me switch from monthly to annual billing?",
        "Everything works great, just wanted to say thanks!"
    ]
}

# Make POST request
resp = requests.post(url, json=payload)

# Print JSON response
print(resp.json())
