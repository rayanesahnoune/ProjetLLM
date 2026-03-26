<<<<<<< HEAD
from flask import Flask, render_template, request, jsonify
import psycopg2
import psycopg2.extras

app = Flask(__name__)
app.secret_key = 'une_cle_secrete'

def get_db():
    return psycopg2.connect(
        dbname   = 'utilisateur',
        user     = 'ray',
        password = '1234',
        host     = 'localhost',
        port     = '5432'
    )

@app.route('/', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '')

        if not username or not password:
            return jsonify(success=False, error='Veuillez remplir tous les champs.')

        try:
            conn   = get_db()
            cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
            cursor.execute(
                "SELECT * FROM users WHERE username = %s AND password = %s",
                (username, password)
            )
            user = cursor.fetchone()
            cursor.close()
            conn.close()

            if user:
                return jsonify(success=True)
            else:
                return jsonify(success=False, error='Identifiants incorrects !')

        except Exception as e:
            return jsonify(success=False, error=f'Erreur BDD : {str(e)}')

    return render_template('login.html')

if __name__ == '__main__':
    app.run(debug=True)
=======
from flask import Flask, render_template, request, redirect, url_for, session, jsonify
import os

# Pointe vers ../Frontend/ depuis le dossier Backend/
FRONTEND = os.path.join(os.path.dirname(__file__), '..', 'Frontend')

app = Flask(__name__, template_folder=FRONTEND)
app.secret_key = "itria-secret-key-change-me"  # ← change en prod

# ── Utilisateurs (remplace par ta DB) ──────────────────────────────────
USERS = {
    "yanis": "0000",
    "sylia": "1234",
    "rayane": "123",
    "itria": "itria2024"
}

# ── Routes ─────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return redirect(url_for("login"))


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "GET":
        if "user" in session:
            return redirect(url_for("chat"))
        return render_template("login.html")

    # POST — appelé en JSON par le fetch() du frontend
    data     = request.get_json(force=True)
    username = data.get("username", "").strip()
    password = data.get("password", "")

    if not username or not password:
        return jsonify(success=False, error="Veuillez remplir tous les champs.")

    if USERS.get(username) == password:
        session["user"] = username
        return jsonify(success=True)

    return jsonify(success=False, error="Identifiant ou mot de passe incorrect.")


@app.route("/chat")
def chat():
    if "user" not in session:
        return redirect(url_for("login"))
    return render_template("chat.html", username=session["user"])


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))


# ── Point d'entrée ─────────────────────────────────────────────────────
if __name__ == "__main__":
    app.run(debug=True)
>>>>>>> 16f6a14ea0f78f241c03ca405a3e0fe45bbae0b8
