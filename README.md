# ProjetLLM

## Prérequis

- Python 3.9+
- pip

## Installation

### 1. Cloner le projet

```bash
git clone https://github.com/rayanesahnoune/ProjetLLM.git
cd ProjetLLM
```

### 2. Installer les dépendances

```bash
pip install tensorflow flask werkzeug numpy requests gdown
```
### 3. Structure attendue du projet
ProjetLLM/
├── Backend/
│   ├── app.py
│   ├── inference.py
│   ├── saved_model/(tokenizer_oz.pkl et max_sequence_len.npy
│   │   
│   └── users.db                ← généré automatiquement
├── Model/
│   ├── smallGPT.py
│   ├── prepData.py
│   ├── trainer.py
│   ├── attention.py
│   ├── decoder.py
│   └── best_smallgpt_oz.keras  ← téléchargé automatiquement
└── Frontend/
    ├── login.html
    ├── register.html
    └── chat.html





## Lancer l'application

```bash
cd Backend
python app.py
```

Ouvrir le navigateur sur :
http://127.0.0.1:5000



## Comptes disponibles par défaut

| Utilisateur | Mot de passe |
|-------------|--------------|
| yanis       | 0000         |
| sylia       | 1234         |
| rayane      | 123          |
| itria       | itria2024    |

## Notes

- Au premier lancement, le modèle est téléchargé automatiquement depuis Google Drive
- La base de données `users.db` est créée automatiquement au premier lancement



