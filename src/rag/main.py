# --- 1. Charger les clés secrètes depuis le fichier .env ---
from dotenv import load_dotenv

load_dotenv()  # lit .env et place OPENAI_API_KEY / ANTHROPIC_API_KEY
               # dans les variables d'environnement du programme

# --- 2. Lire les versions des paquets installés ---
from importlib.metadata import version

core_version = version("langchain-core")  # nom du paquet (avec un tiret)
lg_version = version("langgraph")

# --- 3. Importer les deux "clients" de modèles de langage ---
from langchain_openai import ChatOpenAI        # pour parler à OpenAI (GPT)
from langchain_anthropic import ChatAnthropic  # pour parler à Anthropic (Claude)

# --- 4. Afficher les versions ---
print(f"langchain-core version: {core_version}")
print(f"langgraph version: {lg_version}")

# --- 5. Test OpenAI ---
# On crée un objet "llm" : une connexion configurée vers le modèle
llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)
# On envoie une question ; le modèle renvoie un objet "réponse"
response = llm.invoke("Say 'setup complete!' in one word")
print(f"Response from ChatOpenAI: {response}")

# --- 6. Test Anthropic (même principe) ---
llm = ChatAnthropic(model_name="claude-sonnet-5-5", temperature=0)
response = llm.invoke("Say 'setup complete!' in one word")
print(f"Response from ChatAnthropic: {response}")

# --- 7. Point d'entrée du programme ---
def main():
    print("Hello from rag!")

if __name__ == "__main__":
    main()