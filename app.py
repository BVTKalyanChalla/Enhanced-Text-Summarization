import os
import nltk
import PyPDF2
import re
from flask import Flask, render_template, request, redirect, url_for, session, send_file
import io
from sentence_transformers import SentenceTransformer
import numpy as np
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast
from rouge import Rouge
import pyttsx3
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash

# Initialize Flask app
app = Flask(__name__)

# Secret key for sessions
app.secret_key = os.urandom(24)

# Configure SQLite database
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
db = SQLAlchemy(app)

# User model for login/signup
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(150), unique=True, nullable=False)
    password = db.Column(db.String(150), nullable=False)
    reviews = db.relationship('Review', backref='user', lazy=True)

# Review model to store user reviews
class Review(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    review_text = db.Column(db.Text, nullable=False)

# Create database tables
with app.app_context():
    db.create_all()

# Text-to-speech engine
engine = pyttsx3.init()


def clean_text(text):
    # Clean text but preserve paragraph breaks
    cleaned_text = re.sub(r'[^A-Za-z0-9.,!? \n]+', ' ', text)  # Keep letters, numbers, punctuation, and newlines
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text)  # Remove multiple spaces
    cleaned_text = re.sub(r'([.!?])\s+', r'\1\n\n', cleaned_text)  # Treat sentence-end punctuation as paragraph breaks
    return cleaned_text.strip()

def extract_text_from_pdf(pdf_path):
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        text = ""
        for page in range(len(reader.pages)):
            page_text = reader.pages[page].extract_text()
            text += page_text if page_text else ""

    # Clean and preserve paragraph structure
    cleaned_text = clean_text(text)
    return cleaned_text

def preprocess_text(text):
    sentences = nltk.sent_tokenize(text)
    return [sent.strip() for sent in sentences if sent]

# Load SentenceTransformer model for sentence embeddings
sentence_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

def build_sentence_graph(sentences):
    embeddings = sentence_model.encode(sentences)
    similarity_matrix = cosine_similarity(embeddings)

    graph = nx.Graph()
    for i in range(len(sentences)):
        graph.add_node(i, sentence=sentences[i])

    for i in range(len(sentences)):
        for j in range(i + 1, len(sentences)):
            if similarity_matrix[i, j] > 0:
                graph.add_edge(i, j, weight=similarity_matrix[i, j])

    return graph

def textrank(graph, num_sentences):
    scores = nx.pagerank(graph, weight='weight')
    ranked_sentences = sorted(((scores[i], i) for i in graph.nodes()), reverse=True)
    return [graph.nodes[idx]['sentence'] for _, idx in ranked_sentences[:num_sentences]]

# Load the saved mBART model and tokenizer
tokenizer = MBart50TokenizerFast.from_pretrained('./saved_mbart_model')
model = MBartForConditionalGeneration.from_pretrained('./saved_mbart_model')

def mbart_refine(sentences, max_length=256):
    input_text = ' '.join(sentences)
    
    # Tokenize and encode the input text
    inputs = tokenizer(input_text, max_length=1024, padding='max_length', truncation=True, return_tensors='pt')

    # Generate the summary
    summary_ids = model.generate(
        input_ids=inputs['input_ids'],
        attention_mask=inputs['attention_mask'],
        max_length=max_length,
        num_beams=5,  # Adjusted to improve quality
        early_stopping=True
    )

    # Decode and return the summary
    refined_summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)
    return refined_summary

def hybrid_summarization(text, num_sentences=20, max_length=512):  # Increased num_sentences to 20
    sentences = preprocess_text(text)
    graph = build_sentence_graph(sentences)
    key_sentences = textrank(graph, num_sentences)
    refined_summary = mbart_refine(key_sentences, max_length)
    return refined_summary

def evaluate_summarization(original_text, summarized_text):
    rouge = Rouge()
    scores = rouge.get_scores(summarized_text, original_text)
    formatted_scores = {
        "rouge-1": scores[0]['rouge-1'],
        "rouge-2": scores[0]['rouge-2'],
        "rouge-l": scores[0]['rouge-l'],
    }
    return formatted_scores

@app.route('/')
def home():
    if 'user_id' in session:
        return redirect(url_for('index'))
    return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        user = User.query.filter_by(username=username).first()
        if user and check_password_hash(user.password, password):
            session['user_id'] = user.id
            return redirect(url_for('index'))
        else:
            return render_template('login.html', error='Invalid username or password.')
    return render_template('login.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form['username']
        password = generate_password_hash(request.form['password'])
        new_user = User(username=username, password=password)
        try:
            db.session.add(new_user)
            db.session.commit()
            session['user_id'] = new_user.id
            return redirect(url_for('index'))
        except:
            return render_template('signup.html', error='Username already exists.')
    return render_template('signup.html')

@app.route('/logout')
def logout():
    session.pop('user_id', None)
    return redirect(url_for('login'))

@app.route('/index')
def index():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    user = User.query.filter_by(id=session['user_id']).first()
    return render_template('index.html', user=user)

@app.route('/summarize', methods=['POST'])
def summarize():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    input_text = request.form['input_text']
    voice_type = request.form['voice_type']
    
    # Check if PDF file uploaded
    pdf_file = request.files.get('pdf_file')
    if pdf_file:
        pdf_path = os.path.join('uploads', pdf_file.filename)
        pdf_file.save(pdf_path)
        input_text = extract_text_from_pdf(pdf_path)  # Reuse your extract_text_from_pdf() function

    summary = hybrid_summarization(input_text, num_sentences=20, max_length=512)
    
    audio_file = create_audio(summary, voice_type)
    rouge_scores = evaluate_summarization(input_text, summary)
    
    return render_template('summary.html', summarized_text=summary, audio_file=audio_file, rouge_scores=rouge_scores, user_id=session['user_id'])

@app.route('/review', methods=['POST'])
def review():
    if 'user_id' not in session:
        return redirect(url_for('login'))

    review_text = request.form['review_text']
    new_review = Review(user_id=session['user_id'], review_text=review_text)
    db.session.add(new_review)
    db.session.commit()
    
    return redirect(url_for('index'))

def create_audio(text, voice_type):
    audio_file = "static/summary.mp3"
    
    # Set voice based on selection
    voices = engine.getProperty('voices')
    if voice_type == 'female':
        engine.setProperty('voice', voices[1].id)  # Usually, female voice is at index 1
    else:
        engine.setProperty('voice', voices[0].id)  # Usually, male voice is at index 0

    engine.save_to_file(text, audio_file)
    engine.runAndWait()
    return audio_file

@app.route('/play/<filename>', methods=['GET'])
def play_audio(filename):
    return send_file(filename)

@app.route('/download')
def download():
    text = request.args.get('text')
    file_name = 'summary.txt'
    memory_file = io.StringIO(text)
    
    return send_file(memory_file, as_attachment=True, attachment_filename=file_name, mimetype='text/plain')

if __name__ == '__main__':
    if not os.path.exists('uploads'):
        os.makedirs('uploads')
    app.run(debug=True)
