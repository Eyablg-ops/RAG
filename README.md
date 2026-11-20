# RAG — projet d'apprentissage

> Public learning project: building the foundations of a **RAG** (Retrieval-Augmented Generation) system with **LangChain**, step by step.

**Statut : en cours.** Le projet couvre maintenant le chargement de documents, la vectorisation locale et une première chaîne RAG complète. Ce README sert à la fois de présentation publique et de pense-bête pour reprendre le projet plus tard.

---

## 1. C'est quoi ce projet ?

Un système RAG sert à répondre à des questions en s'appuyant sur **ses propres documents** : on charge des fichiers, on les découpe, on les transforme en vecteurs, puis on récupère les passages pertinents avant de demander la réponse à un LLM.

Le projet avance en trois étapes :

1. **Chargement** (`document_loaders.py`) : lire un `.txt` ou un `.pdf` et produire des objets `Document`.
2. **Indexation** (`chromadb/app.py`) : transformer du texte en vecteurs et les stocker dans Chroma.
3. **Chaîne RAG** (`chromadb/rag_pipeline.py`) : question → retrieval → prompt → LLM → réponse.

Les embeddings sont **locaux et gratuits**, donc aucune clé n'est nécessaire pour vectoriser. En revanche, la génération de la réponse utilise un LLM distant et nécessite une clé API avec des crédits.

## 2. Ce qui a été fait

- [x] Initialisation du projet avec **uv** et Python 3.11+ (`pyproject.toml`, `uv.lock`).
- [x] Installation des paquets LangChain, LangGraph, OpenAI, Anthropic et Chroma.
- [x] `load_text_file()` : crée un petit fichier `.txt` **temporaire**, le lit avec `TextLoader`, affiche le résultat, puis le supprime.
- [x] `pdf_loader()` : lit `docs/langchain_demo.pdf` avec `PyPDFLoader` (dépendance **`pypdf`** ajoutée au projet).
- [x] `chromadb/app.py` : test de Chroma avec l'embedding local intégré (`DefaultEmbeddingFunction`).
- [x] `chromadb/rag_pipeline.py` : chunking, embeddings `FastEmbedEmbeddings`, stockage Chroma et chaîne LCEL complète.
- [x] `src/rag/main.py` : test de connexion aux fournisseurs de modèles (nécessite des clés API valides et créditées).
- [x] Corrections apprises en chemin : dépendance `pypdf` manquante, erreur d'indentation, résumé affiché plusieurs fois à cause d'une boucle imbriquée, imports OpenAI dupliqués.

## 3. Structure du dépôt

```
.
├── document_loaders.py         # démo : TextLoader + PyPDFLoader
├── chromadb/
│   ├── app.py                  # test Chroma avec embeddings locaux intégrés
│   └── rag_pipeline.py         # chaîne RAG complète (FastEmbed + Chroma + Claude)
├── docs/
│   └── langchain_demo.pdf      # PDF de test (3 pages)
├── src/rag/main.py             # test des connexions OpenAI / Anthropic + versions des paquets
├── pyproject.toml              # dépendances et métadonnées du projet
├── uv.lock                     # versions exactes figées par uv
└── README.md                   # ce fichier
```

## 4. Installation

Prérequis : Python >= 3.11 et [uv](https://docs.astral.sh/uv/).

```bash
uv sync
```

Crée ensuite un fichier `.env` **à la racine** (il ne doit jamais être publié) :

```text
OPENAI_API_KEY=ta_cle_openai
ANTHROPIC_API_KEY=ta_cle_anthropic
```

> Le fichier `.env` est listé dans `.gitignore` : ne le commite jamais. Si tu publies le dépôt, vérifie avec `git check-ignore .env`.

## 5. Utilisation

Toutes les commandes se lancent **depuis la racine du dépôt**.

### Chargement des documents (aucune clé API nécessaire)

```bash
uv run python document_loaders.py
```

Sortie attendue, en résumé :

```text
Loaded 1 document(s)                     # le fichier .txt temporaire = 1 seul document
Hello, this is a sample text file.
This file is used to

Loaded 3 document(s)                     # le PDF = 3 pages = 3 documents
Document 1:
Content: LangChain Document Loaders - Demo Document ...
Metadata: {... 'total_pages': 3, 'page': 0, 'page_label': '1'}
Document 2: ...
Document 3: ...
```

**Les 3 documents ne sont pas une erreur** : `PyPDFLoader` crée un `Document` par page. `page: 0` est l'index interne (commence à 0), `page_label: '1'` est le numéro de page affiché.

### Test de Chroma (aucune clé API nécessaire)

```bash
uv run python chromadb/app.py
```

### Chaîne RAG complète (clé Anthropic créditée nécessaire)

```bash
uv run chromadb/rag_pipeline.py
```

Le script découpe la base de connaissances en chunks de 500 caractères, les indexe dans Chroma avec `FastEmbedEmbeddings` (modèle local téléchargé au premier lancement dans `~/.cache/fastembed`), puis répond à 3 questions d'exemple.

> Si tu vois `Error code: 400 - Your credit balance is too low`, c'est que la clé Anthropic n'a pas de crédits. Les embeddings restent gratuits ; seule la génération nécessite un compte crédité (ou un fournisseur alternatif comme Groq).

### Test des modèles (clés API créditées nécessaires)

```bash
uv run python -m rag.main
```

## 6. Glossaire — ce qu'il faut retenir

| Terme | Définition simple |
|---|---|
| **RAG** | Retrieval-Augmented Generation : chercher dans ses documents avant de laisser le LLM répondre. |
| **LLM** | Modèle de langage (GPT, Claude) : une connexion configurée vers le modèle, appelée `llm` dans le code. |
| **LangChain** | Bibliothèque qui fournit des briques toutes prêtes (loaders, splitters, vector stores, chains). |
| **Document** | Objet LangChain = `page_content` (le texte) + `metadata` (informations : source, page...). |
| **Document loader** | Classe qui lit un format de fichier et renvoie une **liste de Documents**. |
| **TextLoader** | Loader pour les fichiers `.txt` → généralement 1 seul Document. |
| **PyPDFLoader** | Loader pour les `.pdf` → 1 Document **par page**. Nécessite le paquet `pypdf`. |
| **page_content** | Le texte brut contenu dans un Document. |
| **metadata** | Les infos sur l'origine : `source`, `page`, `total_pages`, `creator`... Utiles pour les citations. |
| **Chunking** | Découper les Documents en morceaux plus petits pour une meilleure recherche (`RecursiveCharacterTextSplitter`). |
| **Embeddings** | Transformer le texte en vecteurs de nombres. Ici : `FastEmbedEmbeddings`, gratuit et local. |
| **Vector store** | Base de données de vecteurs (Chroma ici) où l'on stocke les embeddings. |
| **Retrieval** | Récupérer les morceaux les plus proches de la question posée. |
| **LCEL** | LangChain Expression Language : assembler les étapes avec `|` dans une chaîne. |
| **uv** | Gestionnaire de paquets Python : `uv sync` installe, `uv run` exécute dans l'environnement du projet. |
| **Indentation** | Les espaces (4 en Python) au début d'une ligne : ils définissent les blocs (`if`, `for`, `def`). Une ligne trop indentée sans bloc ouvert provoque une `IndentationError`. |

## 7. Leçons des erreurs rencontrées

1. **`ImportError: pypdf package not found`** — `PyPDFLoader` délègue la lecture du PDF à `pypdf`, une dépendance optionnelle de `langchain-community` : `uv add pypdf` résout le problème.
2. **`IndentationError: unexpected indent`** — après avoir supprimé une boucle, le `for` suivant était resté à 8 espaces au lieu de 4. En Python, on **désindente** (on retire des espaces) pour sortir d'un bloc, on ne supprime pas la boucle.
3. **Résumé affiché 3 fois** — les lignes d'affichage du résumé étaient à l'intérieur de la première boucle (`for doc in documents`). Les déplacer **après** la boucle suffit ; rien n'était à supprimer.
4. **`No such file or directory` avec `uv run chromadb/rag_pipeline.py`** — la commande était lancée depuis le dossier parent. Il faut toujours se placer à la racine de `RAG/`.
5. **`openai.RateLimitError: 429 - You have no credits remaining`** — les embeddings OpenAI coûtent des crédits. Remplacés par des embeddings locaux (`FastEmbedEmbeddings`).
6. **`anthropic.BadRequestError: 400 - Your credit balance is too low`** — la clé Anthropic n'est pas créditée. Le code est valide ; il faut ajouter des crédits ou changer de fournisseur.
7. **Appels en double** — `demo_basic_rag()` construisait la base vectorielle, et le bloc `if __name__ == "__main__"` la reconstruisait juste avant. Un seul appel suffit.

Deux avertissements sans gravité apparaissent au lancement : `langchain-community` est déprécié (le projet migrera vers des paquets dédiés) et `USER_AGENT` n'est pas défini (proviens de l'import de `WebBaseLoader`).

## 8. Prochaines étapes (roadmap)

- [x] Découper les documents avec `RecursiveCharacterTextSplitter` (chunking).
- [x] Générer des embeddings et les stocker dans un vector store (Chroma).
- [x] Créer une chaîne de retrieval : question → passages pertinents → LLM → réponse.
- [ ] Passer le LLM sur une option réellement gratuite (Groq ou Ollama) pour supprimer la dépendance aux crédits.
- [ ] Brancher la chaîne sur les vrais documents (`document_loaders.py`) au lieu du texte d'exemple.
- [ ] Afficher les sources utilisées dans la réponse.
- [ ] Remplacer les `print` par une petite interface (CLI ou Streamlit).
- [ ] Ajouter des tests et nettoyer les imports inutilisés (`Path`, `WebBaseLoader`, `DirectoryLoader`, `RunnableParallel`).

---

Auteur : Eyaaa.blg — projet en cours, n'hésitez pas à revenir plus tard pour la suite.
