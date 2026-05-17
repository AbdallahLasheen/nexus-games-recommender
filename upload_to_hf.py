import os
from huggingface_hub import HfApi

def upload_to_huggingface():
    print("="*60)
    print("🚀 NEXUS Games - Hugging Face Deployment")
    print("="*60)
    print("\nTo deploy to Hugging Face Spaces, you need an Access Token with 'Write' permissions.")
    print("You can get one here: https://huggingface.co/settings/tokens\n")
    
    token = input("Enter your Hugging Face Access Token: ").strip()
    if not token:
        print("Error: Token cannot be empty.")
        return
        
    space_id = input("Enter your Hugging Face Space ID (e.g., username/nexus-recommender): ").strip()
    if not space_id:
        print("Error: Space ID cannot be empty.")
        return
        
    print(f"\nUploading project to {space_id}...")
    print("Please wait, this will take some time as it uploads large model files (~8GB).")
    
    api = HfApi(token=token)
    
    # Path to the current directory
    folder_path = os.path.dirname(os.path.abspath(__file__))
    
    try:
        # Create the repo if it doesn't exist
        api.create_repo(repo_id=space_id, repo_type="space", space_sdk="docker", exist_ok=True)
        
        # Upload the entire folder
        api.upload_folder(
            folder_path=folder_path,
            repo_id=space_id,
            repo_type="space",
            ignore_patterns=[
                "venv/*", ".venv/*", "env/*", "__pycache__/*", "*.pyc", 
                ".git/*", ".github/*", "upload_to_hf.py"
            ]
        )
        print("\n✅ Upload successful!")
        print(f"Your project is now being built at: https://huggingface.co/spaces/{space_id}")
    except Exception as e:
        print(f"\n❌ Error during upload: {e}")

if __name__ == "__main__":
    upload_to_huggingface()
