import os
import zipfile
import gdown

DRIVE_LINK = "https://drive.google.com/file/d/1zju4xcWNwbeSmW99gaQDF623Gyjf3SAC/view?usp=drive_link"
DATA_ZIP = "data.zip"
DATA_DIR = "data"

def download_from_drive(url, output):
    """Download file from Google Drive using gdown (handles large files)"""
    print(f"Downloading {output} from Google Drive...")
    print("This may take a while for large files...")
    
    try:
        gdown.download(url, output, quiet=False, fuzzy=True)
        print(f"Downloaded {output}")
        return True
    except Exception as e:
        print(f"Download failed: {e}")
        return False

def extract_zip(zip_path, extract_to='.'):
    """Extract zip file"""
    print(f"Extracting {zip_path}...")
    
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    
    print(f"Extracted to {extract_to}")

def main():
    if os.path.exists(DATA_DIR) and os.path.isdir(DATA_DIR):
        print(f"{DATA_DIR} already exists. Skipping download.")
        return
    
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
    
    extract_zip(DATA_ZIP)
    print("Setup complete!")
    print(f"Data extracted to: {os.path.abspath(DATA_DIR)}")

if __name__ == "__main__":
    main()
