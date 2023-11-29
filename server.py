from flask import Flask, request
from main import translate_process

app = Flask(__name__)

@app.route('/', methods=['POST'])
def index():
    if request.method == 'POST':
        data = request.json  # Get JSON data from the request
        text = data.get('text', '')
        language = data.get('language', 'english')
        dialect = data.get('dialect', 'dialect1')
        print(dialect)
        output = translate_process(text, language, dialect)

        return {'translated_text': output}

if __name__ == '__main__':
    app.run(host = '0.0.0.0', port=4321, debug=True)
