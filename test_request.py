# This module is used to test the backend, it need the requests module

import requests
url = "https://www.floydlabs.com/serve/Vse3e7aYeR5aemScRDo7um"
# url = "http://localhost:5000"
res = requests.post(url, json={'text': 'ciao'})
print(res.json()['embedded_text'])
