import subprocess
import sys

def download_spacy_model():
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "spacy"])
        print("Spacy installed successfully")
        subprocess.check_call([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])
        print("English language model downloaded successfully")
    except subprocess.CalledProcessError as e:
        print(f"Error occurred: {e}")
        sys.exit(1)

if __name__ == "__main__":
    download_spacy_model()