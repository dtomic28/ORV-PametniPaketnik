# How to build image and run with docker

```
docker build -t orv-api .

docker run -p 8080:8080 \
  -e MONGO_USER=dt \
  -e MONGO_PASS=yourpassword \
  -e MONGO_DB=PametniPaketnik \
  -e MONGO_URI_TEMPLATE="mongodb+srv://{user}:{password}@cluster0.nu4cj3h.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0" \
  orv-api
```
