"""
Model Downloader - Downloads model from cloud storage if not present locally
"""
import os
import requests
import hashlib
import re
from pathlib import Path

# Configuration - CHANGE THIS TO YOUR MODEL URL
MODEL_URL = os.environ.get('MODEL_URL', 'https://drive.google.com/uc?export=download&confirm=t&id=1rOjcWyaLmIuDkO-AfYDTD-ivB8I5hFPp')
MODEL_FILENAME = 'best_model.pth'
MODEL_DIR = Path('models')
MODEL_PATH = MODEL_DIR / MODEL_FILENAME

def download_from_google_drive(url, output_path):
    """Download file from Google Drive, handling virus scan warning"""
    session = requests.Session()
    
    # First request to get confirmation token
    response = session.get(url, stream=True)
    
    # Check if we got a virus scan warning page
    if 'confirm=' in response.url or 'downloadWarning' in response.text:
        # Extract confirm token from the warning page
        for key, value in response.cookies.items():
            if key.startswith('download_warning'):
                # Construct URL with confirmation
                confirm_token = value
                url = url.replace('&confirm=t', f'&confirm={confirm_token}')
                break
    
    # Download the actual file
    response = session.get(url, stream=True)
    
    total_size = int(response.headers.get('content-length', 0))
    downloaded = 0
    
    with open(output_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
                downloaded += len(chunk)
                if total_size > 0:
                    percent = (downloaded / total_size) * 100
                    print(f"   Progress: {percent:.1f}% ({downloaded}/{total_size} bytes)", end='\r')
    
    return downloaded

def ensure_model_exists():
    """Ensure model file exists, download if not"""
    MODEL_DIR.mkdir(exist_ok=True)
    
    if MODEL_PATH.exists():
        print(f"✅ Model already exists at: {MODEL_PATH}")
        return str(MODEL_PATH)
    
    print(f"📥 Model not found locally. Downloading from cloud...")
    print(f"   URL: {MODEL_URL[:50]}...")
    
    try:
        # Check if it's a Google Drive URL
        if 'drive.google.com' in MODEL_URL:
            download_from_google_drive(MODEL_URL, MODEL_PATH)
        else:
            # Standard download for other URLs
            response = requests.get(MODEL_URL, stream=True, timeout=300)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            downloaded = 0
            
            with open(MODEL_PATH, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        if total_size > 0:
                            percent = (downloaded / total_size) * 100
                            print(f"   Progress: {percent:.1f}% ({downloaded}/{total_size} bytes)", end='\r')
        
        print(f"\n✅ Model downloaded successfully: {MODEL_PATH}")
        print(f"   Size: {MODEL_PATH.stat().st_size / (1024*1024):.2f} MB")
        
        return str(MODEL_PATH)
        
    except Exception as e:
        print(f"❌ Failed to download model: {e}")
        # Clean up partial download
        if MODEL_PATH.exists():
            MODEL_PATH.unlink()
        raise

def get_model_path():
    """Get model path, downloading if necessary"""
    return ensure_model_exists()

# For Vercel/serverless - use /tmp directory
def get_model_path_serverless():
    """Get model path for serverless environment (Vercel)"""
    # In serverless, use /tmp which is writable
    tmp_dir = Path('/tmp/models')
    tmp_dir.mkdir(exist_ok=True)
    tmp_model_path = tmp_dir / MODEL_FILENAME
    
    if tmp_model_path.exists():
        return str(tmp_model_path)
    
    print(f"📥 Downloading model to /tmp...")
    try:
        response = requests.get(MODEL_URL, stream=True, timeout=300)
        response.raise_for_status()
        
        with open(tmp_model_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        
        print(f"✅ Model downloaded to: {tmp_model_path}")
        return str(tmp_model_path)
        
    except Exception as e:
        print(f"❌ Download failed: {e}")
        if tmp_model_path.exists():
            tmp_model_path.unlink()
        raise

if __name__ == '__main__':
    # Test download
    path = get_model_path()
    print(f"\nModel ready at: {path}")
