import requests as re

parameters = {
    "lat": 40.71,
    "lon": -74
}

response = re.get("https://api.open-notify.org/iss-pass.json", params=parameters)
print (response.status_code)
print (response.json())
