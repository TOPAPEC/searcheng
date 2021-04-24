from flask import Flask, render_template, request
from search import build_index, index_search, sort_1k_top
from time import time

app = Flask(__name__, template_folder='.')
build_index()


@app.route('/', methods=['GET'])
def index():
    start_time = time()
    query = request.args.get('query')
    if query is None:
        query = ''
    documents = sort_1k_top(query)
    print(documents)
    print(len(index_search))
    results = [(((index_search[doc[1]]).format(query)) + [doc[0]]) for doc in documents]
    return render_template(
        'index.html',
        time="%.2f" % (time()-start_time),
        query=query,
        search_engine_name='Tsundere',
        results=results
    )


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=80)
