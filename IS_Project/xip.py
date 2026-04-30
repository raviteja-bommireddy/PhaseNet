import os
import zipfile

# List of files and folders to include
items_to_zip = [
    "IS_Project/baselines",
    "IS_Project/model",
    "IS_Project/preprocessing",
    "IS_Project/config.py",
    "IS_Project/dataset.py",
    "IS_Project/.gitignore",
    "IS_Project/evaluate.py",
    "IS_Project/phaseaware1.ipynb",
    "IS_Project/phaseaware2.ipynb",
    "IS_Project/inference.py",
    "IS_Project/requirements.txt",
    "IS_Project/run_kaggle.py",
    "IS_Project/train.py"
]

zip_file_name = "IS_Project.zip"

with zipfile.ZipFile(zip_file_name, 'w', zipfile.ZIP_DEFLATED) as zipf:
    for item in items_to_zip:
        if os.path.isdir(item):
            # Walk through folder, skipping __pycache__ directories
            for root, dirs, files in os.walk(item):
                dirs[:] = [d for d in dirs if d != "__pycache__"]  # Exclude __pycache__
                for file in files:
                    file_path = os.path.join(root, file)
                    zipf.write(file_path, arcname=file_path)
        else:
            # Add individual file
            zipf.write(item, arcname=item)

print(f"{zip_file_name} created successfully!")