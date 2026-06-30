#!/usr/bin/env python
"""Push Crops project to GitHub"""

from git import Repo
from pathlib import Path

PROJECT_PATH = Path.cwd()
GITHUB_URL = "https://github.com/kguruvishnuteja/Crop_Yield-Prediction-using-ML-.git"

try:
    # Initialize repository
    print("Initializing git repository...")
    repo = Repo.init(PROJECT_PATH)
    
    # Configure git user
    with repo.config_writer() as git_config:
        git_config.set_value("user", "name", "Vishnu").release()
        git_config.set_value("user", "email", "kguruvishnuteja@example.com").release()
    
    print("Adding all files...")
    repo.index.add([item.relative_to(PROJECT_PATH) for item in PROJECT_PATH.rglob("*") 
                    if item.is_file() and ".git" not in item.parts and "__pycache__" not in item.parts])
    
    print("Creating initial commit...")
    repo.index.commit("Initial commit: Crop Yield Prediction ML project")
    
    print("Adding remote origin...")
    origin = repo.create_remote("origin", GITHUB_URL)
    
    print("Pushing to GitHub...")
    origin.push(refspec="HEAD:main")
    
    print("✅ Successfully pushed to GitHub!")
    
except Exception as e:
    print(f"❌ Error: {e}")
    print("\nTroubleshooting:")
    print("1. Make sure Git is installed: https://git-scm.com/download/win")
    print("2. Make sure you have GitHub credentials configured")
    print("3. Check your internet connection")
