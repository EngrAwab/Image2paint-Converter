import requests

API_URL = "http://localhost:5000/api/sum"

def sum_via_get(a, b):
    """
    Sends a GET request to /api/sum?a=<a>&b=<b>
    """
    params = {'a': a, 'b': b}
    resp = requests.get(API_URL, params=params)
    resp.raise_for_status()
    data = resp.json()
    print(f"GET sum of {a} + {b} = {data.get('sum')}")

def sum_via_post(a, b):
    """
    Sends a POST request with JSON body { "a": <a>, "b": <b> }
    """
    payload = {'a': a, 'b': b}
    resp = requests.post(API_URL, json=payload)
    resp.raise_for_status()
    data = resp.json()
    print(f"POST sum of {a} + {b} = {data.get('sum')}")

if __name__ == "__main__":
    x, y = 12, 30

    # Using GET
    sum_via_get(x, y)

    # Using POST
    sum_via_post(x, y)
