import gdown
import zipfile
import os
import shutil

# === Настройки ===
FILE_ID = "1Indcjd8TfgPFMpV03wWUo13vhcqhwjIW"
ZIP_PATH = "model.zip"
MODEL_DIR = "model_files"

def clean_previous(zip_path=ZIP_PATH, model_dir=MODEL_DIR):
    if os.path.exists(zip_path):
        os.remove(zip_path)
        print(f"🪚 Удалён старый архив: {zip_path}")
    if os.path.exists(model_dir):
        shutil.rmtree(model_dir)
        print(f"🪚 Удалена старая папка модели: {model_dir}")

def download_model(file_id=FILE_ID, output_path=ZIP_PATH):
    url = f"https://drive.google.com/uc?id={file_id}"
    print("⬇️ Скачивание модели...")
    gdown.download(url, output_path, quiet=False)

def extract_model_zip(zip_path=ZIP_PATH, extract_to=MODEL_DIR):
    if zipfile.is_zipfile(zip_path):
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
        print(f"✅ Архив распакован в: {extract_to}")
    else:
        raise ValueError("❌ Файл не является ZIP-архивом!")

if __name__ == "__main__":
    clean_previous()
    download_model()
    extract_model_zip()
