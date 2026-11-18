"""
app.py : test de base de ChromaDB (base de données vectorielle).

Ce script :
    1. crée une collection en mémoire ;
    2. y stocke 3 phrases (ChromaDB les transforme en vecteurs) ;
    3. cherche les phrases les plus proches d'une requête.

Usage :
    python3 app.py
"""

import chromadb

# Client en mémoire : toutes les données disparaissent à la fin du script.
# Pour garder les données, on utiliserait chromadb.PersistentClient(path="./db").
chroma_client = chromadb.Client()

# get_or_create : récupère la collection si elle existe, sinon la crée.
# Cela évite une erreur quand on relance le script.
collection_name = "TestCollection"
collection = chroma_client.get_or_create_collection(name=collection_name)

# Données de test : chaque document a un identifiant unique et un texte.
documents = [
    {"id": "doc1", "text": "Hello, world!"},
    {"id": "doc2", "text": "How are you today?"},
    {"id": "doc3", "text": "Goodbye, see you later!"},
]

# Chroma attend des LISTES (ids et documents), même pour un seul élément.
# upsert = ajoute le document, ou le met à jour si l'id existe déjà.
# L'embedding (vecteur de 384 dimensions par défaut) est calculé automatiquement.
for doc in documents:
    collection.upsert(ids=[doc["id"]], documents=[doc["text"]])

# Recherche sémantique : la requête est aussi transformée en vecteur,
# puis comparée aux vecteurs stockés.
query_text = "Hello, world!"
results = collection.query(query_texts=[query_text], n_results=3)

# 'distances' : plus la valeur est petite, plus le document est proche.
print("Query Results:")
print(results)