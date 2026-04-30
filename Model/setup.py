
import os
import sys

class ProjectSetup:
    """"""    
    def __init__(
        self,
        drive_name = "LLM-Groupe-S",
        repo_url   = "https://github.com/rayanesahnoune/ProjetLLM.git"
    ):
        self.drive_name   = drive_name
        self.repo_url     = repo_url
        self.project_root = f"/content/drive/MyDrive/{drive_name}"
        self.model_dir    = os.path.join(self.project_root, "models")
        self.data_dir     = os.path.join(self.project_root, "data")
        self.model_path   = os.path.join(self.model_dir, "best_smallgpt.keras")

    def mount_drive(self):
        from google.colab import drive
        drive.mount('/content/drive')
        print(" Drive monté")

    def clone_or_update(self):
        if not os.path.exists(self.project_root):
            print(" Clonage du repo...")
            os.system(f"git clone {self.repo_url} {self.project_root}")
            print(" Repo cloné")
        else:
            print("Mise à jour du repo...")
            os.system(f"cd {self.project_root} && git pull")
            print(" Repo à jour")

    def create_dirs(self):
        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.data_dir,  exist_ok=True)
        print(" Dossiers créés")

    def add_to_path(self):
        model_dir = os.path.join(self.project_root, "Model")
        if model_dir not in sys.path:
            sys.path.append(model_dir)
        print(" Path mis à jour")

    def setup(self):
        self.mount_drive()
        self.clone_or_update()
        self.create_dirs()
        self.add_to_path()

        print(f"\n Projet : {self.project_root}")
        print(f" Models : {self.model_dir}")
        print(f" Data   : {self.data_dir}")
        print(f" Path   : {sys.path[-1]}")