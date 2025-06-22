# scripts/deploy.py
from huggingface_hub import HfApi, Repository
import os

def main():
    model_dir = './models/distilbert-sentiment'
    repo_id = "IssaBachir/sentiment-model"  # Modifie ici avec ton repo HF si besoin
    token = os.getenv("HF_API_KEY")

    if token is None:
        raise ValueError("Le token HF_API_KEY n'est pas défini dans les variables d'environnement.")

    api = HfApi()
    
    # Créer le repo sur HF si pas déjà créé (ignore l'erreur sinon)
    try:
        api.create_repo(token=token, name="sentiment-model", private=False)
        print("Repo créé sur Hugging Face.")
    except Exception as e:
        print(f"Repo déjà existant ou erreur: {e}")

    # Pousser les fichiers du modèle
    repo = Repository(local_dir=model_dir, clone_from=repo_id, use_auth_token=token)
    repo.push_to_hub(commit_message="Déploiement automatique via CI/CD")

if __name__ == "__main__":
    main()
