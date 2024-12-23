This repository contains a Flask web application that performs text summarization and translation. The application integrates several features such as user authentication, text and PDF summarization, audio playback of the summary, and user reviews.

Features

User Authentication:

Signup and login functionality with secure password hashing using werkzeug.

Session management to ensure user-specific operations.

Text and PDF Summarization:

Users can input text or upload a PDF file for summarization.

Text is cleaned and processed to generate key sentences using the TextRank algorithm.

Summaries are refined using the mBART model for improved coherence.

Audio Playback:

Summaries can be converted into audio using the pyttsx3 library.

Male and female voice options are available.

User Reviews:

Users can provide feedback on the summarization.

Reviews are stored in the database for future reference.

Evaluation Metrics:

ROUGE scores are displayed to evaluate the quality of the summary.

Download and Playback:

Summaries can be downloaded as text files.

Audio files of the summary can be played directly from the app.

Star Rating System:

Users can rate the generated summary using a star rating system.

Ratings are displayed along with a "Thank you" message and emoji.

Technology Stack

Backend: Flask

Database: SQLite

Machine Learning Models: SentenceTransformer, mBART

Libraries: PyPDF2, NLTK, NetworkX, Rouge, pyttsx3

Frontend: HTML templates rendered via Flask

Installation

Clone the repository:

git clone https://github.com/yourusername/flask-summarization-app.git
cd flask-summarization-app

Set up a virtual environment:

python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

Install the required dependencies:

pip install -r requirements.txt

Download the pretrained mBART model and SentenceTransformer:

Place the mBART model in the ./saved_mbart_model directory.

Ensure paraphrase-MiniLM-L6-v2 is downloaded for SentenceTransformer.

Set up the database:

flask shell
>>> from app import db
>>> db.create_all()
>>> exit()

Run the application:

python app.py

Usage

Open the app in your browser at http://127.0.0.1:5000.

Sign up or log in.

Navigate to the summarization page to input text or upload a PDF file.

View the summary and its ROUGE scores.

Play or download the audio of the summary.

Provide a review for the generated summary.

Rate the summary using the star rating system.

File Structure

app.py: Main application file.

templates/: HTML templates for the web interface.

static/: Static files (e.g., CSS, JavaScript, audio files).

uploads/: Directory for storing uploaded PDF files.

saved_mbart_model/: Directory containing the pretrained mBART model.

requirements.txt: List of Python dependencies.

Future Enhancements

Add support for more languages in summarization and translation.

Implement additional evaluation metrics.

Integrate advanced text-to-speech capabilities.

Deploy the application to a cloud service like AWS or Heroku.

Acknowledgments

Hugging Face for the mBART model.

Sentence Transformers for sentence embeddings.

Flask for the web framework.
