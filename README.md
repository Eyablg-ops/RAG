# RAG — projet d'apprentissage

> Public learning project: building the foundations of a **RAG** (Retrieval-Augmented Generation) system with **LangChain**, step by step.

**Statut : en cours.** Étape actuelle : chargement et inspection de documents (`.txt` et `.pdf`). Ce README sert à la fois de présentation publique et de pense-bête pour reprendre le projet plus tard.

---

## 1. C'est quoi ce projet ?

Un système RAG sert à répondre à des questions en s'appuyant sur **ses propres documents** : on charge des fichiers, on les découpe, on les transforme en vecteurs, puis on récupère les passages pertinents avant de demander la réponse à un LLM.

Pour l'instant, le projet s'arrête à la **première étape** : transformer des fichiers en objets `Document` que LangChain peut manipuler. Rien n'est encore envoyé à un modèle dans `document_loaders.py`.

## 2. Ce qui a été fait

- [x] Initialisation du projet avec **uv** et Python 3.11+ (`pyproject.toml`).
- [x] Installation des paquets LangChain, LangGraph, OpenAI et Anthropic.
- [x] `load_text_file()` : crée un petit fichier `.txt` **temporaire**, le lit avec `TextLoader`, affiche le resultat, puis le supprime.
- [x] `pdf_loader()` : lit `docs/langchain_demo.pdf` avec `PyPDFLoader` (dépendance **`pypdf`** ajoutée au projet).
- [x] Test de connexion aux deux fournisseurs de modèles dans `src/rag/main.py` (nécessite des clés API valides).
- [x] Corrections apprises en chemin : dépendance `pypdf` manquante, erreur d'indentation, résumé affiché plusieurs fois à cause d'une boucle imbriquée.

## 3. Structure du dépôt

```
.
├── document_loaders.py     # démo : TextLoader + PyPDFLoader (point d'entrée principal)
├── docs/
│   └── langchain_demo.pdf  # PDF de test (3 pages)
├── src/rag/main.py         # test des connexions OpenAI / Anthropic + versions des paquets
├── pyproject.toml          # dépendances et métadonnées du projet
├── uv.lock                 # versions exactes figées par uv
└── README.md               # ce fichier
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

### Chargement des documents (aucune clé API nécessaire)

```bash
uv run python document_loaders.py
```

Sortie attendue, en résumé :

```
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

### Test des modèles (clés API nécessaires)

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
| **Chunking** | Étape future : couper les Documents en morceaux plus petits pour une meilleure recherche. |
| **Embeddings** | Étape future : transformer le texte en vecteurs de nombres. |
| **Vector store** | Étape future : base de données de vecteurs (FAISS, Chroma...) où l'on stocke les embeddings. |
| **Retrieval** | Étape future : récupérer les morceaux les plus proches de la question posée. |
| **uv** | Gestionnaire de paquets Python : `uv sync` installe, `uv run` exécute dans l'environnement du projet. |
| **Indentation** | Les espaces (4 en Python) au début d'une ligne : ils définissent les blocs (`if`, `for`, `def`). Une ligne trop indentée sans bloc ouvert provoque une `IndentationError`. |

## 7. Leçons des erreurs rencontrées

1. **`ImportError: pypdf package not found`** — `PyPDFLoader` délègue la lecture du PDF à `pypdf`, une dépendance optionnelle de `langchain-community` : `uv add pypdf` résout le problème.
2. **`IndentationError: unexpected indent`** — après avoir supprimé une boucle, le `for` suivant était resté à 8 espaces au lieu de 4. En Python, on **désindente** (on retire des espaces) pour sortir d'un bloc, on ne supprime pas la boucle.
3. **Résumé affiché 3 fois** — les lignes d'affichage du résumé étaient à l'intérieur de la première boucle (`for doc in documents`). Les déplacer **après** la boucle suffit ; rien n'était à supprimer.

Deux avertissements sans gravité apparaissent au lancement : `langchain-community` est déprécié, et `USER_AGENT` n'est pas défini (proviens de l'import de `WebBaseLoader`).

## 8. Prochaines étapes (roadmap)

- [ ] Découper les documents avec `RecursiveCharacterTextSplitter` (chunking).
- [ ] Générer des embeddings et les stocker dans un vector store (FAISS ou Chroma).
- [ ] Créer une chaîne de retrieval : question → passages pertinents → LLM → réponse avec sources.
- [ ] Remplacer les `print` par une petite interface (CLI ou Streamlit).
- [ ] Ajouter des tests et nettoyer les imports inutilisés (`Path`, `WebBaseLoader`, `DirectoryLoader`).

---

Auteur : Eyaaa.blg — projet en cours, n'hésitez pas à revenir plus tard pour la suite.
