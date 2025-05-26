import os
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure
from dotenv import load_dotenv

# Load from .env only if not running in Docker
if not os.getenv("MONGO_USER"):
    print("Loading .env")
    load_dotenv()

# Required environment variables
required_vars = ["MONGO_USER", "MONGO_PASS", "MONGO_DB", "MONGO_URI_TEMPLATE"]
missing = [var for var in required_vars if not os.environ.get(var)]

if missing:
    raise EnvironmentError(
        f"Missing required environment variables: {', '.join(missing)}"
    )

# Read credentials
MONGO_USER = os.environ["MONGO_USER"]
MONGO_PASS = os.environ["MONGO_PASS"]
MONGO_DB = os.environ.get("MONGO_DB", "test")
MONGO_URI_TEMPLATE = os.environ["MONGO_URI_TEMPLATE"]

# Construct final URI
MONGO_URI = MONGO_URI_TEMPLATE.format(user=MONGO_USER, password=MONGO_PASS)

# Connect and verify
try:
    client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
    client.admin.command("ping")  # Quick check to verify connection
    print("✅ MongoDB Atlas connection successful.")
    db = client[MONGO_DB]
except ConnectionFailure as e:
    print("❌ MongoDB Atlas connection failed:", str(e))
    db = None
