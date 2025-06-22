import shutil
import os

def deploy():
    repo_url = "https://huggingface.co/IssaBachir/sentiment"
    repo_local_dir = "./models/hf_repo"

    # Clone ou crée le repo localement
    repo = Repository(local_dir=repo_local_dir, clone_from=repo_url, use_auth_token=os.environ["HF_API_KEY"])

    # Copier les fichiers du modèle dans repo_local_dir
    shutil.copy("outputs/pytorch_model.bin", repo_local_dir)
    shutil.copy("outputs/config.json", repo_local_dir)
    shutil.copy("outputs/tokenizer_config.json", repo_local_dir)

    # Ajouter, commit et push
    repo.git_add(auto_lfs_track=True)
    repo.git_commit("Déploiement automatique depuis pipeline CI/CD")
    repo.git_push()
