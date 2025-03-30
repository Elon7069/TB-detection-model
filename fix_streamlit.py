"""
Fix for the missing imghdr module in Python 3.13 for Streamlit.
This script patches the Streamlit image.py file to work without imghdr.
"""

import os
import sys
import importlib.util
from pathlib import Path

def get_streamlit_path():
    """Get the path to the Streamlit installation."""
    try:
        # Try using pip to get the location
        import subprocess
        result = subprocess.run([sys.executable, "-m", "pip", "show", "streamlit"], 
                                capture_output=True, text=True, check=True)
        for line in result.stdout.split('\n'):
            if line.startswith('Location:'):
                return Path(line.split('Location:')[1].strip()) / 'streamlit'
    except:
        pass
    
    # If that fails, try to get it from sys.path
    for path in sys.path:
        streamlit_path = Path(path) / 'streamlit'
        if streamlit_path.exists():
            return streamlit_path
    
    return None

def patch_streamlit_image():
    """
    Patch the Streamlit image.py file to work without imghdr.
    
    Replaces the imghdr dependency with a simple function that detects image types
    based on file magic numbers.
    """
    streamlit_path = get_streamlit_path()
    
    if not streamlit_path:
        print("❌ Error: Could not find Streamlit installation.")
        return False
    
    image_file = streamlit_path / 'elements' / 'image.py'
    
    if not image_file.exists():
        print(f"❌ Error: Could not find {image_file}")
        return False
    
    # Backup the original file
    backup_file = image_file.with_suffix('.py.bak')
    if not backup_file.exists():
        with open(image_file, 'r') as f:
            original_content = f.read()
        
        with open(backup_file, 'w') as f:
            f.write(original_content)
        
        print(f"✅ Created backup at {backup_file}")
    
    # Read the file content
    with open(image_file, 'r') as f:
        content = f.read()
    
    # Replace the imghdr import
    if 'import imghdr' in content:
        # Add our custom imghdr replacement
        imghdr_replacement = """
# imghdr replacement for Python 3.13
def what(file, h=None):
    \"\"\"Test the image data in a file to see what format it is.\"\"\"
    if h is None:
        if isinstance(file, str):
            with open(file, 'rb') as f:
                h = f.read(32)
        else:
            location = file.tell()
            h = file.read(32)
            file.seek(location)
    
    if len(h) >= 4:
        # PNG
        if h.startswith(b'\\x89PNG\\r\\n\\x1a\\n'):
            return 'png'
        # GIF
        if h[:6] in (b'GIF87a', b'GIF89a'):
            return 'gif'
        # JPEG
        if h.startswith(b'\\xff\\xd8'):
            return 'jpeg'
        # BMP
        if h.startswith(b'BM'):
            return 'bmp'
        # TIFF
        if h.startswith(b'II*\\0') or h.startswith(b'MM\\0*'):
            return 'tiff'
        # WebP
        if h.startswith(b'RIFF') and h[8:12] == b'WEBP':
            return 'webp'
    
    return None
"""
        
        # Replace the import with our implementation
        content = content.replace('import imghdr', imghdr_replacement)
        
        # Write the patched content back
        with open(image_file, 'w') as f:
            f.write(content)
        
        print(f"✅ Successfully patched {image_file}")
        return True
    else:
        print("⚠️ Could not find 'import imghdr' in the Streamlit image.py file.")
        print("It's possible that the file structure has changed or the issue has been fixed in a newer version.")
        return False

if __name__ == "__main__":
    print("🔧 Patching Streamlit to work without imghdr...")
    if patch_streamlit_image():
        print("✅ Patch complete! Streamlit should now work with Python 3.13.")
    else:
        print("❌ Failed to apply the patch.") 