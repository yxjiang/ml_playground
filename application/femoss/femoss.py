"""THe demo of femoss. Conversation with non-human being.
"""
from flask import Flask, render_template, request
from flask_socketio import SocketIO
import webbrowser

app = Flask(__name__)
socketio = SocketIO(app)

@app.route('/')
def index():
    return render_template('femoss.html')


@app.route('/update', methods=['POST'])
def update():
    data = request.get_json()
    print(data)
    socketio.emit('update_page', data)
    return 'Success'


if __name__ == '__main__':
    port = 2222
    webbrowser.open_new(f'http://127.0.0.1:{port}')
    app.run(port=port)