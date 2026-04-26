
import os
from flask import Flask, render_template, request, redirect, url_for, session, jsonify
import sqlite3
from werkzeug.security import generate_password_hash, check_password_hash
import traceback
from inference import predict_next_words

BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
FRONTEND  = os.path.abspath(os.path.join(BASE_DIR, "..", "Frontend"))
DATABASE  = os.path.join(BASE_DIR, "users.db")


app = Flask(__name__, template_folder=FRONTEND, static_folder=FRONTEND)
app.secret_key = os.environ.get("SECRET_KEY", "itria-secret-key-change-me")

DEFAULT_USERS = {
    "yanis":  "0000",
    "sylia":  "1234",
    "rayane": "123",
    "itria":  "itria2024",
}


def get_db():
    conn = sqlite3.connect(DATABASE)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    with get_db() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                username      TEXT    NOT NULL UNIQUE,
                password_hash TEXT    NOT NULL
            )
        """)
        conn.commit()


def create_default_users():
    with get_db() as conn:
        for username, password in DEFAULT_USERS.items():
            exists = conn.execute(
                "SELECT 1 FROM users WHERE username = ?", (username,)
            ).fetchone()
            if not exists:
                conn.execute(
                    "INSERT INTO users (username, password_hash) VALUES (?, ?)",
                    (username, generate_password_hash(password))
                )
        conn.commit()


init_db()
create_default_users()



@app.route("/")
def index():
    if "user" in session:
        return redirect(url_for("chat"))
    return redirect(url_for("login"))


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "GET":
        if "user" in session:
            return redirect(url_for("chat"))
        return render_template("login.html")

    data     = request.get_json(silent=True) or {}
    username = data.get("username", "").strip().lower()
    password = data.get("password", "")

    if not username or not password:
        return jsonify(success=False, error="Veuillez remplir tous les champs."), 400

    with get_db() as conn:
        user = conn.execute(
            "SELECT username, password_hash FROM users WHERE username = ?", (username,)
        ).fetchone()

    if user and check_password_hash(user["password_hash"], password):
        session["user"] = user["username"]
        return jsonify(success=True), 200

    return jsonify(success=False, error="Identifiant ou mot de passe incorrect."), 401


@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "GET":
        if "user" in session:
            return redirect(url_for("chat"))
        return render_template("register.html")

    data     = request.get_json(silent=True) or {}
    username = data.get("username", "").strip().lower()
    password = data.get("password", "")

    if not username or not password:
        return jsonify(success=False, error="Veuillez remplir tous les champs."), 400

    with get_db() as conn:
        exists = conn.execute(
            "SELECT 1 FROM users WHERE username = ?", (username,)
        ).fetchone()
        if exists:
            return jsonify(success=False, error="Ce nom d'utilisateur existe déjà."), 409

        conn.execute(
            "INSERT INTO users (username, password_hash) VALUES (?, ?)",
            (username, generate_password_hash(password))
        )
        conn.commit()

    return jsonify(success=True), 201


@app.route("/chat")
def chat():
    if "user" not in session:
        return redirect(url_for("login"))
    return render_template("chat.html", username=session["user"])


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))


import traceback  


@app.route("/predict", methods=["POST"])
def predict():
    if "user" not in session:
        return jsonify(error=True, message="Non authentifié."), 401

    data = request.get_json(silent=True) or {}
    prompt = data.get("prompt", "").strip()

    if not prompt:
        return jsonify(error=True, message="Prompt vide."), 400

    try:
        generated_words = []
        current_text = prompt
        prompt_words = set(prompt.lower().split())  # uniquement les mots du prompt bloqués

        while len(generated_words) < 5:
            next_word = predict_next_words(current_text)
            if not next_word:
                break
            # Saute uniquement si c'est un mot du prompt original
            if next_word in prompt_words:
                current_text = current_text + " " + next_word  # contexte avance quand même
                continue
            generated_words.append(next_word)
            current_text = current_text + " " + next_word

        return jsonify({
            "prompt": prompt,
            "generated": generated_words
        })
    except Exception as e:
        return jsonify(error=True, message=f"Erreur serveur : {str(e)}")


if __name__ == "__main__":
    app.run(debug=True)