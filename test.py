import os

token = os.getenv("GITHUB_TOKEN")
print("Token cargado:", token is not None)
