from flask import Flask, request, jsonify
from flask_cors import CORS  # Permet les requêtes depuis le navigateur
from gpt4all import GPT4All

app = Flask(__name__)
CORS(app, resources={r"/describe_cluster/*": {"origins": "*"}}, supports_credentials=True)

# Charger le modèle GPT4All localement
model_path = "Nous-Hermes-2-Mistral-7B-DPO.Q4_0.gguf"  
model = GPT4All(model_path)

@app.route('/describe_cluster/<int:cluster_id>', methods=['OPTIONS'])
def options_cluster(cluster_id):
    """Gérer les requêtes OPTIONS préflight pour éviter les erreurs CORS."""
    response = jsonify({"message": "CORS preflight accepted"})
    response.headers.add("Access-Control-Allow-Origin", "*")
    response.headers.add("Access-Control-Allow-Methods", "POST, OPTIONS")
    response.headers.add("Access-Control-Allow-Headers", "Content-Type")
    return response

def generate_description(tfidf_words, associated_words):
    """Génère une description avec GPT4All."""
    prompt = f"""
    Voici des mots-clés décrivant un endroit à Lyon.
    - Mots-clés avec score TF-IDF élevé : {', '.join(tfidf_words)}
    - Mots issus de règles d'associations : {', '.join(associated_words)}
    Il faut parvenir à décrire cet endroit. Donne donc un seul titre et une seule mini-description de cet endroit à partir des mots à ta disposition.
    Rédige ta réponse sous la forme suivante : 
    - Titre : ...
    - Mini-description : Cette endroit pourrait correspondre à ...
    """

    response = model.generate(prompt, max_tokens=150)
    return response

@app.route('/describe_cluster/<int:cluster_id>', methods=['POST'])
def describe_cluster(cluster_id):
    try:
        data = request.json
        tfidf_words = data.get("tfidf_words", [])
        associated_words = data.get("associated_words", [])

        if not tfidf_words:
            return jsonify({"error": "Aucun mot-clé fourni"}), 400

        description = generate_description(tfidf_words, associated_words)

        response = jsonify({"description": description})
        response.headers.add("Access-Control-Allow-Origin", "*")
        return response

    except Exception as e:
        print(f"Erreur API : {e}")  # Debug sur Flask
        return jsonify({"error": "Erreur serveur"}), 500

if __name__ == '__main__':
    app.run(debug=True)