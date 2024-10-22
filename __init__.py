from flask_migrate import Migrate
from .database import db

migrate = Migrate()

def create_app():
    app = Flask(__name__)
    init_db(app)
    migrate.init_app(app, db)
    return app
