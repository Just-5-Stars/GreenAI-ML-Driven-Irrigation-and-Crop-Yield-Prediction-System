from flask import Flask
from application.config import Config

app = None

def create_app():
    app = Flask(__name__, template_folder = "templates")
    app.app_context().push()
    app.config.from_object(Config)

    return app

app = create_app()

if __name__ == "__main__":
    from application.controllers import *
    app.run()
