import shutil
import os

if os.path.exists('src.zip'):
    os.remove('src.zip')

shutil.make_archive('src', 'zip', 'src')
print("src.zip created successfully")
