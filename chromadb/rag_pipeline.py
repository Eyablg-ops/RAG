"""
Construction d'un pipeline RAG (Retrieval-Augmented Generation).

Principe du RAG : au lieu de laisser le LLM répondre de mémoire, on
  1. découpe une base de connaissances en morceaux (chunks) ;
  2. transforme chaque chunk en vecteur (embedding) et le stocke dans Chroma ;
  3. à chaque question, on récupère les chunks les plus proches en sens ;
  4. on les donne au LLM comme contexte pour qu'il réponde uniquement avec eux.
"""

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_anthropic import ChatAnthropic
from langchain_community.embeddings import FastEmbedEmbeddings

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import BaseModel, Field
from typing import List
from dotenv import load_dotenv
import tempfile

# Charge les variables d'environnement depuis le fichier .env
# (notamment ANTHROPIC_API_KEY, nécessaire pour le LLM).
load_dotenv()

# Modèle d'embeddings local et gratuit : convertit un texte en vecteur numérique.
# Utilisé à la fois pour indexer les documents et pour vectoriser les questions.
embeddings_model = FastEmbedEmbeddings()

# Base de connaissances d'exemple (texte Markdown sur LangChain).
# Dans un vrai projet, elle viendrait de fichiers, d'une base de données, etc.
KNOWLEDGE_BASE = """# LangChain Framework
...
"""  # (contenu inchangé)

# Modèle de langage qui génère la réponse finale.
# temperature=0.2 : réponses peu aléatoires, donc plus fidèles au contexte.
llm = ChatAnthropic(model="claude-haiku-5-5", temperature=0.2)


def create_kb():
    """Construit la base vectorielle à partir de KNOWLEDGE_BASE.

    Étapes :
        1. Découpage du texte en chunks de ~500 caractères, avec 50 caractères
           de chevauchement pour ne pas couper une idée en deux.
        2. Vectorisation de chaque chunk et stockage dans Chroma, avec la
           métadonnée ``source`` (nom du fichier d'origine).

    Returns:
        Chroma: la base vectorielle, stockée dans un dossier temporaire
        (créé par ``mkdtemp``, non supprimé automatiquement).
    """
    # Le splitter essaie de couper aux séparateurs naturels
    # (paragraphes, lignes, phrases) avant de couper brutalement.
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)

    # Le texte entier est encapsulé dans un Document avec sa métadonnée.
    doc = Document(
        page_content=KNOWLEDGE_BASE, metadata={"source": "langchain_knowledge_base.md"}
    )

    # Chaque chunk hérite de la métadonnée du document d'origine.
    chunks = splitter.split_documents([doc])

    # Calcule les embeddings des chunks et les indexe dans Chroma.
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings_model,
        persist_directory=tempfile.mkdtemp(),
    )
    return vector_store


def demo_basic_rag():
    """Démontre une chaîne RAG de base et l'exécute sur 3 questions.

    Flux d'une question :
        question → retriever (2 chunks les plus proches) → format_docs
                 → prompt (contexte + question) → LLM → texte final
    """
    vector_store = create_kb()

    # Le retriever fait une recherche sémantique (par similarité de vecteurs)
    # et renvoie les k=2 chunks les plus proches de la question.
    retriever = vector_store.as_retriever(
        search_type="similarity", search_kwargs={"k": 2}
    )

    # Template du prompt : {context} et {question} seront remplis à l'exécution.
    # Le LLM est obligé de répondre uniquement avec le contexte fourni.
    prompt = ChatPromptTemplate.from_template(
        """
Answer the question based only on the following context:

{context}

Question: {question}

Answer:


Make sure to answer in a concise manner, 
and if you don't know the answer, just say "I don't know."""
    )

    def format_docs(docs):
        """Fusionne le texte des chunks récupérés en un seul bloc de contexte."""
        return "\n\n".join([doc.page_content for doc in docs])

    # Chaîne RAG (syntaxe LCEL, les étapes sont reliées par « | ») :
    #  1. Le dict prépare les deux variables du prompt :
    #       - "context"  : la question passe dans le retriever puis format_docs
    #       - "question" : la question est transmise telle quelle
    #         (RunnablePassthrough)
    #  2. prompt          : remplit le template
    #  3. llm             : génère la réponse
    #  4. StrOutputParser : extrait le texte brut de la réponse du LLM
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    questions = [
        "What is LangChain?",
        "Who created LangChain?",
        "What is LangGraph used for?",
    ]

    print("Basic RAG Demo:\n")
    for q in questions:
        # invoke(q) exécute toute la chaîne pour une question
        answer = rag_chain.invoke(q)
        print(f"Q: {q}")
        print(f"A: {answer}\n")


if __name__ == "__main__":
    demo_basic_rag()