# Setup script for creating virtual environment and installing dependencies
# Run this script with: python setup_venv.py

import subprocess
import sys
import os

def run_command(command, description):
    """Run a shell command and print status"""
    print(f"\n{'='*60}")
    print(f"{description}")
    print(f"{'='*60}")
    print(f"Command: {command}")
    
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
        print(f"✓ {description} - SUCCESS")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ {description} - FAILED")
        print(f"Error: {e}")
        print(f"Output: {e.output}")
        return False

def main():
    print("\n" + "="*60)
    print("FRAUD DETECTION PROJECT - VIRTUAL ENVIRONMENT SETUP")
    print("="*60)
    
    # Check if venv already exists
    venv_path = "venv"
    if os.path.exists(venv_path):
        print(f"\n⚠ Virtual environment '{venv_path}' already exists!")
        response = input("Do you want to delete and recreate it? (y/n): ")
        if response.lower() == 'y':
            import shutil
            shutil.rmtree(venv_path)
            print(f"✓ Deleted existing '{venv_path}'")
        else:
            print("Skipping venv creation...")
            return
    
    # Step 1: Create virtual environment
    if not run_command(
        f"{sys.executable} -m venv venv",
        "Step 1: Creating virtual environment"
    ):
        return
    
    # Step 2: Upgrade pip
    pip_path = os.path.join("venv", "Scripts", "pip.exe")
    if not run_command(
        f"{pip_path} install --upgrade pip",
        "Step 2: Upgrading pip"
    ):
        return
    
    # Step 3: Install requirements
    if not run_command(
        f"{pip_path} install -r requirements.txt",
        "Step 3: Installing dependencies from requirements.txt"
    ):
        return
    
    print("\n" + "="*60)
    print("✓ SETUP COMPLETE!")
    print("="*60)
    print("\nTo activate the virtual environment:")
    print("  Windows (PowerShell): .\\venv\\Scripts\\Activate.ps1")
    print("  Windows (CMD):        .\\venv\\Scripts\\activate.bat")
    print("\nTo deactivate:")
    print("  deactivate")
    print("\nTo run your app:")
    print("  .\\venv\\Scripts\\python.exe app.py")
    print("="*60)

if __name__ == "__main__":
    main()
