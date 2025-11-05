#!/usr/bin/env python3
"""
 Solar System Recommendation Platform - Streamlit Launcher
"""
import subprocess
import sys
import os

def check_requirements():
    """Check if required packages are installed"""
    required_packages = [
        'streamlit', 'plotly', 'pandas', 'numpy', 
        'scikit-learn', 'xgboost'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f" Missing packages: {', '.join(missing_packages)}")
        print(" Installing missing packages...")
        
        for package in missing_packages:
            try:
                subprocess.check_call([sys.executable, '-m', 'pip', 'install', package])
                print(f" Installed {package}")
            except subprocess.CalledProcessError:
                print(f" Failed to install {package}")
                return False
    
    return True

def launch_streamlit():
    """Launch the Streamlit application"""
    try:
        print(" Starting Solar System Recommendation Platform...")
        print(" Opening browser at http://localhost:8501")
        print(" Press Ctrl+C to stop the application")
        print("=" * 60)
        
        # Launch Streamlit
        subprocess.run([
            sys.executable, '-m', 'streamlit', 'run', 'streamlit_app.py',
            '--server.port', '8501',
            '--server.address', '0.0.0.0',
            '--browser.gatherUsageStats', 'false'
        ])
        
    except KeyboardInterrupt:
        print("\n Application stopped by user")
    except Exception as e:
        print(f" Error launching application: {e}")

if __name__ == "__main__":
    print(" AI-Powered Solar System Recommendation Platform")
    print("=" * 60)
    
    # Check requirements
    if check_requirements():
        print(" All requirements satisfied")
        launch_streamlit()
    else:
        print(" Failed to install requirements")
        print(" Try running: pip install -r streamlit_requirements.txt")
