from huggingface_hub import upload_folder
import huggingface_hub
from dotenv import load_dotenv
import os

load_dotenv()

huggingface_hub.login(os.environ['HF_API_KEY'])

# Your Hugging Face username/org and Space name
repo_id = "Markoskokos/CLIP-Fashion-Multi-Modal-Image-Text-Search"  # 👈 must match the Space name
repo_type = "space"  # 👈 required for Spaces

# Push everything inside my_app/ to the Space
upload_folder(
    folder_path="app",    # local folder containing app.py + requirements.txt
    repo_id=repo_id,
    repo_type=repo_type,
)
