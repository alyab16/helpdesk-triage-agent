import time
import requests

start_total = time.perf_counter()  # start total timer

# First single ticket
t0 = time.perf_counter()
resp = requests.post("http://localhost:8000/predict", json={
    "ticket_text": "Hi, can you help me switch from monthly to annual billing?"
})
t1 = time.perf_counter()
print(resp.json())
print(f"[timing] Request 1 took {t1 - t0:.4f}s (requests.elapsed: {resp.elapsed.total_seconds():.4f}s)")
print("\n" + "=" * 100 + "\n")

# Second single ticket
t0 = time.perf_counter()
resp = requests.post("http://localhost:8000/predict", json={
    "ticket_text": "Your app crashed after the update. I lost hours of work—this is unacceptable and I need help NOW."
})
t1 = time.perf_counter()
print(resp.json())
print(f"[timing] Request 2 took {t1 - t0:.4f}s (requests.elapsed: {resp.elapsed.total_seconds():.4f}s)")
print("\n" + "=" * 100 + "\n")

# Batch request
url = "http://localhost:8000/batch"
payload = {
    "tickets": [
        "Your app crashed after the update. I lost hours of work—please help ASAP!",
        "Hi, can you help me switch from monthly to annual billing?",
        "Everything works great, just wanted to say thanks!"
    ]
}

t0 = time.perf_counter()
resp = requests.post(url, json=payload)
t1 = time.perf_counter()
print(resp.json())
print(f"[timing] Batch request took {t1 - t0:.4f}s (requests.elapsed: {resp.elapsed.total_seconds():.4f}s)")

# Total script time
end_total = time.perf_counter()
print(f"\n[total timing] Entire script took {end_total - start_total:.4f}s")
