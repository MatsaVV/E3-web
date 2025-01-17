# Utilisation d'une image Python légère
FROM python:3.10-slim

# Définir le répertoire de travail
WORKDIR /app

# Copier les fichiers nécessaires
COPY requirements.txt .

# Installer les dépendances
RUN pip install --no-cache-dir -r requirements.txt

# Copier le code de l'application
COPY web/ .

# Copier les données de test
COPY data/ ./data

# Exposer le port utilisé par Streamlit
EXPOSE 8501

# Lancer Streamlit en précisant le bon chemin
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
