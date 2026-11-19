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
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
import tempfile
import shutil
from dotenv import load_dotenv
from chromadb.utils.embedding_functions import DefaultEmbeddingFunction


load_dotenv()

#embeddings_model = OpenAIEmbeddings(model="text-embedding-3-small")
# Pas de crédit sur l'API OpenAI, donc on ne peut pas utiliser OpenAIEmbeddings.
# À la place, on utilise le modèle d'embedding local et gratuit de Chroma
# (DefaultEmbeddingFunction, vecteurs de 384 dimensions).
#
# LangChain attend un objet avec deux méthodes :
#   - embed_documents(texts) : transforme une liste de textes en vecteurs
#   - embed_query(text)      : transforme la requête en un seul vecteur
# Chroma n'expose pas ces méthodes directement, donc cette classe sert
# d'adaptateur (wrapper) entre la fonction d'embedding de Chroma et LangChain.
class LocalEmbeddings:
    def __init__(self):
        self.fn = DefaultEmbeddingFunction()
    def embed_documents(self, texts):
        return [list(map(float, v)) for v in self.fn(texts)]
    def embed_query(self, text):
        return list(map(float, self.fn([text])[0]))

embeddings_model = LocalEmbeddings()

# Sample documents
SAMPLE_DOCS = [
    Document(
        page_content="LangChain is a framework for developing applications powered by language models.",
        metadata={"source": "langchain_docs", "topic": "overview"},
    ),
    Document(
        page_content="LangGraph is a library for building stateful, multi-actor applications with LLMs.",
        metadata={"source": "langgraph_docs", "topic": "overview"},
    ),
    Document(
        page_content="Vector stores are databases optimized for storing and searching embeddings.",
        metadata={"source": "vector_guide", "topic": "database"},
    ),
    Document(
        page_content="RAG combines retrieval with generation for more accurate LLM responses.",
        metadata={"source": "rag_guide", "topic": "architecture"},
    ),
    Document(
        page_content="Embeddings convert text into numerical vectors for semantic similarity.",
        metadata={"source": "embeddings_guide", "topic": "fundamentals"},
    ),
    Document(
        page_content="Chroma is an open-source embedding database for AI applications.",
        metadata={"source": "chroma_docs", "topic": "database"},
    ),
    Document(
        page_content="FAISS is a library for efficient similarity search developed by Facebook.",
        metadata={"source": "faiss_docs", "topic": "database"},
    ),
    Document(
        page_content="Pinecone is a managed vector database service for production workloads.",
        metadata={"source": "pinecone_docs", "topic": "database"},
    ),
]


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

def similarity_search_with_scores():
    """
    Démontre une recherche sémantique avec scores sur un vector store Chroma.

    Étapes :
        1. Crée un vector store Chroma temporaire à partir de SAMPLE_DOCS.
           Chaque document est transformé en vecteur par `embeddings_model`.
        2. Transforme la requête en vecteur avec le même modèle.
        3. Récupère les k documents les plus proches (ici k=3), avec leur distance.
        4. Convertit chaque distance en un score de similarité et l'affiche.

    Le dossier temporaire (`tempfile.TemporaryDirectory`) est supprimé
    automatiquement à la sortie du bloc `with` : rien n'est conservé sur le disque.

    Affiche, pour chaque résultat : le texte, le score et la source (métadonnée).
    Ne retourne rien.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        # Crée le vector store : chaque document est converti en vecteur (embedding)
        # puis stocké dans Chroma. persist_directory pointe vers le dossier temporaire.
        vectorstore = Chroma.from_documents(
            documents=SAMPLE_DOCS, embedding=embeddings_model, persist_directory=tmpdir
        )

        # Recherche sémantique : la requête est vectorisée, puis comparée aux
        # vecteurs stockés. k=3 limite le résultat aux 3 documents les plus proches.
        # La méthode renvoie des tuples (Document, distance).
        query = "Explain vector stores."
        results_with_scores = vectorstore.similarity_search_with_score(query, k=3)

        print(f"Top 3 results with scores for query '{query}':")
        for i, (doc, score) in enumerate(results_with_scores):
            # `score` est une DISTANCE : plus elle est petite, plus le document
            # est proche. On la convertit en similarité entre 0 et 1 :
            # distance 0 -> similarité 1 (identique), grande distance -> proche de 0.
            final_score = 1 / (1 + score)
            print(
                f"Result {i+1}: {doc.page_content} (Score: {final_score:.4f}, Source: {doc.metadata['source']})"
            )


# Formules possibles pour convertir une distance en similarité :
#
#   similarity = 1 / (1 + distance)
#       -> toujours entre 0 et 1, simple, sans connaître la distance maximale.
#          C'est celle utilisée ici.
#
#   similarity = 1 - (distance / max_distance)
#       -> normalisation linéaire : demande de connaître (ou d'estimer)
#          la distance maximale, sinon le résultat peut devenir négatif.
#
# Ce ne sont que des conversions de commodité, pas une vraie similarité cosinus.
# Elles servent surtout à classer les résultats et à rendre le score lisible.




def metadata_filtering():
    """Démontre le filtrage par métadonnées dans une recherche vectorielle (Chroma).

    La fonction compare deux recherches de similarité sur le même jeu de documents :
      1. une recherche sémantique classique, sans filtre ;
      2. une recherche sémantique restreinte aux documents dont les métadonnées
         correspondent à un critère (ici ``topic == "database"``).

    Le filtrage permet de réduire l'espace de recherche *avant* le classement par
    similarité, ce qui améliore la pertinence des résultats et évite de remonter
    des documents hors sujet.

    Dépendances (définies ailleurs dans le module) :
        SAMPLE_DOCS: liste de ``Document`` LangChain, chacun avec un ``page_content``
            et des ``metadata`` contenant au minimum les clés ``source`` et ``topic``.
        embeddings_model: modèle d'embeddings utilisé pour vectoriser les documents
            et la requête.

    Returns:
        None. Les résultats sont affichés dans la console.
    """
    # Dossier temporaire pour la persistance de Chroma : il est supprimé
    # automatiquement à la sortie du bloc `with`, ce qui évite de laisser des
    # fichiers de test sur le disque.
    with tempfile.TemporaryDirectory() as tmpdir:
        # Création de la base vectorielle : chaque document est transformé en
        # vecteur par `embeddings_model`, puis stocké avec ses métadonnées.
        vectorstore = Chroma.from_documents(
            documents=SAMPLE_DOCS,
            embedding=embeddings_model,
            persist_directory=tmpdir,
        )

        # Requête en langage naturel utilisée pour les deux recherches.
        query = "What databases are available?"

        # --- 1) Recherche SANS filtre -------------------------------------
        # Retourne les k=5 documents sémantiquement les plus proches de la
        # requête, quel que soit leur sujet : des documents hors thème
        # "database" peuvent apparaître.
        results = vectorstore.similarity_search(query, k=5)
        print(f"Results without metadata filtering for query '{query}':")
        for i, doc in enumerate(results):
            print(
                f"Result {i+1}: {doc.page_content} (Source: {doc.metadata['source']})"
            )

        # --- 2) Recherche AVEC filtre sur les métadonnées -----------------
        # Seuls les documents dont `metadata["topic"] == "database"` sont
        # considérés ; le classement par similarité s'applique ensuite
        # uniquement à ce sous-ensemble.
        filter_criteria = {"topic": "database"}
        filtered_results = vectorstore.similarity_search(
            query, k=5, filter=filter_criteria
        )
        print(f"\nResults with metadata filtering for query '{query}':")
        for i, doc in enumerate(filtered_results):
            print(
                f"Result {i+1}: {doc.page_content} (Source: {doc.metadata['source']})"
            )
    
if __name__ == "__main__":
    similarity_search_with_scores()
    metadata_filtering()