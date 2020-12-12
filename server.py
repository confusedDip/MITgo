from flask import Flask, render_template, request
from queryParsing import parsed_result

app = Flask(__name__)


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/<keyword>')
def get_results_from_url(keyword):
    result = parsed_result(query=keyword)
    return render_template('results.html', results=result, query=keyword)


@app.route('/results', methods=["POST"])
def get_results():
    query = request.form["query"]
    result = parsed_result(query)
    return render_template('results.html', results=result, query=query)


if __name__ == "__main__":
    app.run(debug=True)
