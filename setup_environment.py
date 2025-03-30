#!/usr/bin/env python
"""
Environment Setup Script for TB Detection with Federated Learning

This script helps set up the environment for the TB Detection project by:
1. Checking Python version
2. Installing required dependencies
3. Creating necessary directories
4. Verifying installations
"""

import sys
import os
import subprocess
import platform
import shutil
from importlib import util

def print_section(title):
    """Print a section title."""
    print("\n" + "=" * 60)
    print(f" {title}")
    print("=" * 60)

def check_python_version():
    """Check if Python version is compatible."""
    print_section("CHECKING PYTHON VERSION")
    
    python_version = sys.version_info
    print(f"Python version: {python_version.major}.{python_version.minor}.{python_version.micro}")
    
    if python_version.major == 3 and python_version.minor >= 6:
        print("✓ Python version is compatible")
        return True
    else:
        print("✗ Python version is not compatible")
        print("This project requires Python 3.6 or higher")
        return False

def install_dependencies():
    """Install required dependencies."""
    print_section("INSTALLING DEPENDENCIES")
    
    # List of required packages
    packages = [
        "numpy",
        "pandas",
        "scikit-learn",
        "matplotlib",
        "streamlit",
        "opencv-python",
        "pillow"
    ]
    
    # Special handling for TensorFlow on Windows
    if platform.system() == "Windows":
        packages.append("tensorflow-directml-plugin")
    else:
        packages.append("tensorflow")
    
    # Determine pip command
    pip_cmd = "pip3" if shutil.which("pip3") else "pip"
    
    # Install packages
    print(f"Installing packages using {pip_cmd}...")
    try:
        for package in packages:
            print(f"\nInstalling {package}...")
            result = subprocess.run(
                [pip_cmd, "install", package],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            if result.returncode == 0:
                print(f"✓ Successfully installed {package}")
            else:
                print(f"✗ Failed to install {package}")
                print(result.stderr)
    except subprocess.CalledProcessError as e:
        print(f"Error installing packages: {e}")
        print(e.stderr)
        return False
    
    return True

def create_directories():
    """Create necessary directories for the project."""
    print_section("CREATING DIRECTORIES")
    
    dirs = [
        'data/train/normal',
        'data/train/tb',
        'data/test',
        'fl_models',
        'visualizations'
    ]
    
    for dir_path in dirs:
        os.makedirs(dir_path, exist_ok=True)
        print(f"Created directory: {dir_path}")
    
    return True

def verify_installation():
    """Verify that required packages are installed."""
    print_section("VERIFYING INSTALLATION")
    
    packages_to_check = {
        "numpy": "numpy",
        "pandas": "pandas",
        "sklearn": "scikit-learn",
        "matplotlib": "matplotlib",
        "streamlit": "streamlit",
        "cv2": "opencv-python",
        "PIL": "pillow"
    }
    
    # Special check for TensorFlow
    if platform.system() == "Windows":
        packages_to_check["tensorflow"] = "tensorflow-directml-plugin"
    else:
        packages_to_check["tensorflow"] = "tensorflow"
    
    # Check each package
    all_installed = True
    for module_name, package_name in packages_to_check.items():
        if util.find_spec(module_name):
            print(f"✓ {package_name} is installed")
        else:
            print(f"✗ {package_name} is not installed")
            all_installed = False
    
    return all_installed

def main():
    """Main function."""
    print_section("TB DETECTION ENVIRONMENT SETUP")
    print("This script will set up the environment for the TB Detection project.")
    
    # Check Python version
    if not check_python_version():
        print("\nPlease install Python 3.6 or higher and try again.")
        return False
    
    # Install dependencies
    if not install_dependencies():
        print("\nFailed to install dependencies. Please try installing them manually.")
        print("Run: pip install numpy pandas scikit-learn matplotlib streamlit opencv-python pillow tensorflow")
        return False
    
    # Create directories
    if not create_directories():
        print("\nFailed to create directories. Please check file permissions.")
        return False
    
    # Verify installation
    if not verify_installation():
        print("\nSome packages may not be properly installed.")
        print("Please try installing them manually using pip.")
        return False
    
    print_section("SETUP COMPLETED SUCCESSFULLY")
    print("You can now use the TB Detection system:")
    print("1. Place TB X-ray images in data/train/tb/")
    print("2. Place normal X-ray images in data/train/normal/")
    print("3. Place test images in data/test/")
    
    print("\nRun one of the following commands:")
    print("- python main.py train    (to train the model)")
    print("- python main.py predict  (to generate predictions)")
    print("- python main.py streamlit (to launch the web interface)")
    
    print("\nOr use the interactive guide:")
    if platform.system() == "Windows":
        print("- Run 'tb_hackathon_guide.bat' for a step-by-step guide")
    else:
        print("- Run 'bash tb_hackathon_guide.sh' for a step-by-step guide")
    
    return True

if __name__ == "__main__":
    main() 