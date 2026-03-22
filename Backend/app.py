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
