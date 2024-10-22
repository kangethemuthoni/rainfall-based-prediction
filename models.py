from database import db  

class Review(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(100), nullable=False)
    review = db.Column(db.Text, nullable=False)
    sentiment = db.Column(db.String(10), nullable=False) 
    confidence = db.Column(db.Float, nullable=False)

    def __repr__(self):
        return f'<Review {self.title}>'
