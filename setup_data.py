import os
import sys
import zipfile
import requests
from tqdm import tqdm

DRIVE_LINK = "YOUR_GOOGLE_DRIVE_LINK_HERE"
DATA_ZIP = "data.zip"
DATA_DIR = "data"

def download_from_drive(url, output):
    """Download file from Google Drive with progress bar"""
    
    # Extract file ID from various Drive URL formats
    if '/file/d/' in url:
        file_id = url.split('/file/d/')[1].split('/')[0]
    elif 'id=' in url:
        file_id = url.split('id=')[1].split('&')[0]
    else:
        print("Invalid Google Drive URL format")
        return False
    
    # Direct download URL
    download_url = f"https://drive.google.com/uc?export=download&id={file_id}"
    
    print(f"Downloading {output}...")
    
    session = requests.Session()
    response = session.get(download_url, stream=True)
    
    # Handle large file confirmation
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            download_url = f"https://drive.google.com/uc?export=download&id={file_id}&confirm={value}"
            response = session.get(download_url, stream=True)
    
    total_size = int(response.headers.get('content-length', 0))
    
    with open(output, 'wb') as f, tqdm(
        desc=output,
        total=total_size,
        unit='B',
        unit_scale=True,
        unit_divisor=1024,
    ) as pbar:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
                pbar.update(len(chunk))
    
    print(f"Downloaded {output}")
    return True

def extract_zip(zip_path, extract_to='.'):
    """Extract zip file with progress bar"""
    print(f"Extracting {zip_path}...")
    
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        members = zip_ref.namelist()
        for member in tqdm(members, desc="Extracting"):
            zip_ref.extract(member, extract_to)
    
    print(f"Extracted to {extract_to}")

def main():
    # Check if data already exists
    if os.path.exists(DATA_DIR) and os.path.isdir(DATA_DIR):
        print(f"{DATA_DIR} already exists. Skipping download.")
        return
    
    # Download data.zip if not present
    if not os.path.exists(DATA_ZIP):
        if DRIVE_LINK == "YOUR_GOOGLE_DRIVE_LINK_HERE":
            print("ERROR: Please update DRIVE_LINK in setup_data.py")
            print("Get the shareable link from Google Drive and paste it in this script")
            return
        
        success = download_from_drive(DRIVE_LINK, DATA_ZIP)
        if not success:
            print("Download failed")
            return
    else:
        print(f"{DATA_ZIP} already exists. Skipping download.")
    
    # Extract data
    extract_zip(DATA_ZIP)
    print("Setup complete!")

if __name__ == "__main__":
    main()
