import requests

url = 'http://127.0.0.1:5000/upload-zip'
files = {'zipfile': open('Archive.zip', 'rb')}
data = {'username': 'dt'}

response = requests.post(url, files=files, data=data)
print(response.json())
