
import wandb
import os
import shutil
from pathlib import Path

# Config
ENTITY = "daniele5"
REGISTRY = "model-registry"
COLLECTION = "thumbnail-classifier"
ALIAS = "production"
FILENAME = "model.onnx"

def download_prod_model():
    print(f"🚀 Fetching Production Model from Registry...")
    
    api = wandb.Api()
    artifact_path = f"{ENTITY}/{REGISTRY}/{COLLECTION}:{ALIAS}"
    
    try:
        artifact = api.artifact(artifact_path)
        print(f"✅ Found Artifact: {artifact.name} (ID: {artifact.id})")
        
        # Define output path (assets/ folder relative to this script)
        script_dir = Path(__file__).parent
        assets_dir = script_dir.parent / "assets"
        assets_dir.mkdir(exist_ok=True)
        
        # Download strictly the ONNX file
        # Note: artifact.download() downloads everything, get_path().download() downloads specific file
        print(f"⬇️  Downloading {FILENAME}...")
        file_path = artifact.get_path(FILENAME).download(root=str(assets_dir))
        
        print(f"✅ Model saved to: {file_path}")
        
        # Save Metadata
        import json
        metadata = {
            "version": artifact.version,
            "updated_at": artifact.updated_at,
            "name": artifact.name
        }
        with open(assets_dir / "model_metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
        print(f"📝 Metadata saved to: {assets_dir / 'model_metadata.json'}")
        
    except Exception as e:
        print(f"❌ Failed to download model: {e}")
        exit(1)

if __name__ == "__main__":
    download_prod_model()
