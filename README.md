````markdown
# 🌐 Social Media Sentiment Analyzer

This project is a comprehensive, multi-modal sentiment analysis tool built with Streamlit. It can ingest a URL from various social media platforms (like Reddit, YouTube, TikTok, etc.), extract text, media (images/videos/audio), and comments, and perform a deep analysis on each component.

The application uses an ensemble of transformer models to analyze sentiment from text, captions, audio transcriptions, and even emotions detected in media. All results are then intelligently fused into an overall sentiment score, saved in a persistent database, and displayed in a rich, interactive dashboard.

## ✨ Key Features

* **Multi-Modal Analysis:** Performs sentiment analysis on:
    * **Text:** Post titles, captions, and body text.
    * **Images:** Analyzes text via OCR, scene context via VLM (image captioning), and human emotion via facial recognition.
    * **Videos:** Samples frames and performs the same multi-modal image analysis on each.
    * **Audio:** Transcribes speech to text (ASR) for sentiment analysis and performs separate audio-based emotion recognition.
    * **Comments:** Fetches and individually-analyzes top comments (currently supports Reddit).
* **Ensemble AI Models:** Avoids reliance on a single model by using an ensemble approach for:
    * **Text Sentiment:** `twitter-roberta-base-sentiment`, `distilbert-sst-2-english`, and `siebert-roberta-large-english`.
    * **Facial Emotion:** `fer` (MTCNN & Haar) and `deepface` (RetinaFace, OpenCV, etc.).
    * **Audio Analysis:** `openai/whisper` for ASR and `wav2vec2-lg-xlsr` for emotion.
* **Adaptive Sentiment Fusion:** Implements a `fuse_sentiments_adaptive` function that combines all component scores (text, media, comments) using a weighted average based on confidence. It can detect and penalize "discordance" (e.g., positive text with negative-emotion media) to improve accuracy.
* **Persistent History:** Uses a local **SQLite** database (`sentiment_history.db`) to save every analysis run, allowing for trend tracking.
* **Evaluation & Metrics:**
    * Includes a `sample.csv` to pre-load the database with sample data.
    * Allows uploading a "Ground Truth" CSV to compare analysis results against human-labeled data.
    * Calculates and displays "Internal Consistency" metrics (Precision, Recall, F1, Confusion Matrix) by comparing component sentiments against the final fused sentiment from the history.
* **Interactive UI:**
    * Built with Streamlit, featuring a clean, multi-tab interface (Input, Results, History).
    * Includes a custom, theme-aware (Light/Dark) CSS implementation for a polished look.
    * Provides rich data visualizations (bar charts, pie charts, histograms, and trend lines) using Matplotlib & Seaborn.

## 📸 Demo

<img width="1919" height="914" alt="Screenshot 2025-10-27 222047" src="https://github.com/user-attachments/assets/d87e65e4-a6a7-43ca-9229-8e4f00ea45c9" />

<img width="1919" height="914" alt="Screenshot 2025-10-27 222047" src="https://github.com/user-attachments/assets/4ac24d00-dc07-4d9d-ada5-49445b2e5e0e" />

``

## 🛠️ Technology Stack

### Core Framework & Data

* **Framework:** Streamlit
* **Data Handling:** Pandas, NumPy
* **Database:** SQLite3 (built-in)

### AI/ML Models & Libraries

* **Text Sentiment (Transformers):**
    * `cardiffnlp/twitter-roberta-base-sentiment-latest`
    * `distilbert-base-uncased-finetuned-sst-2-english`
    * `siebert/sentiment-roberta-large-english`
* **VLM (Image Captioning):**
    * `Salesforce/blip-image-captioning-large`
* **ASR (Speech-to-Text):**
    * `openai/whisper-tiny`
* **Audio Emotion Recognition:**
    * `ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition`
* **OCR (Optical Character Recognition):**
    * `easyocr`
* **Facial Emotion Recognition:**
    * `fer`
    * `deepface`
* **Metrics:**
    * `scikit-learn`

### Media & Web

* **Media Processing:** MoviePy, Pillow (PIL), Librosa, SoundFile
* **Web/API:** Requests, PRAW (Reddit API), yt-dlp

### Visualization

* **Plotting:** Matplotlib, Seaborn

## ⚙️ How It Works: The Analysis Pipeline

1.  **Input:** The user provides a social media URL in the "Analysis Input" tab.
2.  **Data Fetching:** The `fetch_social_media_data` function identifies the platform.
    * **Reddit:** Uses `praw` to fetch the submission title, self-text, comments, and direct media URL (image, video, or gallery).
    * **Other Platforms (YouTube, TikTok, etc.):** Uses the `social-media-master` RapidAPI to fetch the post caption/title and a downloadable media URL.
3.  **Component Analysis:** The app analyzes each component it successfully fetched.
    * **Text (Title/Caption):** The text is cleaned and passed to `analyze_text_ensemble`, which averages the sentiment scores from the three different text-based transformer models.
    * **Comments:** Each comment's text is individually analyzed by the same text ensemble.
    * **Media:**
        * The media URL is passed to `download_media`, which uses `requests` (for images) or `yt-dlp` (for video/audio) to save a temporary local file.
        * **If Image:** `analyze_image_advanced` is called.
            1.  `easyocr` extracts any text from the image.
            2.  `BLIP` (VLM) generates a caption describing the scene.
            3.  This text (OCR + Caption) is analyzed by the text ensemble.
            4.  `detect_emotions_ensemble` (`fer` + `deepface`) finds all faces and averages their emotion scores.
            5.  The final image score is a weighted fusion of the *text-derived sentiment* and the *emotion-derived sentiment*.
        * **If Video:** `analyze_media` samples 5-15 frames from the video clip (using `moviepy`). It runs the full `analyze_image_advanced` pipeline on each frame and averages all the resulting scores.
        * **If Audio:** `analyze_media` is called.
            1.  `Whisper` transcribes the audio to text, which is then analyzed by the text ensemble.
            2.  `Wav2Vec2` analyzes the audio for emotional tone.
            3.  The scores from the transcript sentiment and the audio emotion are averaged.
4.  **Sentiment Fusion:** The `fuse_sentiments_adaptive` function receives the individual sentiment scores (from 0.0 to 1.0) for Text, Media, and Comments.
    * It calculates a "confidence" for each score (how far it is from the neutral 0.5).
    * It computes a confidence-weighted average to get the `overall_score`.
    * It checks for "discordance" (e.g., highly positive text + highly negative media) and reduces the confidence of conflicting components if found.
5.  **Storage & Display:**
    * The complete analysis (all scores, timings, weights, etc.) is saved to the `sentiment_history.db` SQLite database.
    * The results are displayed in the "Analysis Results" tab, powering all the metrics and visualizations.

## 🚀 Setup and Installation

### 1. Clone the Repository

```bash
git clone [https://github.com/SaiKarthik547/Multimodal-sentimental-analysis.git](https://github.com/SaiKarthik547/Multimodal-sentimental-analysis.git)
cd Multimodal-sentimental-analysis
````

### 2\. Create a Virtual Environment (Recommended)

```bash
# Windows
python -m venv venv
.\venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### 3\. Install Dependencies

The provided `requirements.txt` file contains most dependencies. The `app.py` file also explicitly uses `pandas` and `scikit-learn`, which are included in the list below.

```bash
# Install from the provided requirements file
pip install -r requirements.txt

# Install packages imported in app.py but missing from the file
pip install pandas scikit-learn
```

**Full `requirements.txt` content (for reference):**

```
streamlit
numpy
requests
praw
yt-dlp
pillow
moviepy
soundfile
librosa
transformers
torch-directml # Note: This is for DirectML on AMD. If using NVIDIA, you'd want 'torch' and 'torchvision' with CUDA.
bitsandbytes
easyocr
fer
deepface
matplotlib
seaborn
python-dotenv
validators
emoji
google-api-python-client
retry
tf-keras
sentencepiece
```

*(Note: `torch-directml` implies this is set up for an AMD GPU. If you have an NVIDIA GPU, you should install PyTorch with CUDA support. If you have no GPU, you can install the CPU version of PyTorch (`pip install torch torchvision torchaudio`))*

### 4\. Set Up API Keys

This project requires API keys for data fetching. Create a file named `.env` in the root of the project directory.

```bash
touch .env
```

Now, add the following keys to the `.env` file.

```ini
# Get from: [https://rapidapi.com/V-K-Apis/api/social-media-master](https://rapidapi.com/V-K-Apis/api/social-media-master)
RAPIDAPI_KEY="your_rapidapi_key_here"

# Get from: [https://www.reddit.com/prefs/apps](https://www.reddit.com/prefs/apps)
REDDIT_CLIENT_ID="your_reddit_client_id"
REDDIT_CLIENT_SECRET="your_reddit_client_secret"
REDDIT_USER_AGENT="streamlit_app/1.0 (by /u/your_username)"

# This key is loaded in the app but not actively used in the fetching logic
YOUTUBE_API_KEY="your_google_api_key_here"
```

### 5\. (Optional) Add Sample Data

To pre-load the history database, you can create a `sample.csv` file in the root directory. The app will load this data *only* if the database is newly created.

The CSV **must** contain these columns:
`url, true_sentiment, text_label, text_score, media_label, media_score, comment_label, comment_avg_score, overall_sentiment, overall_score, media_dominant_emotion, media_emotions_json, platform, title, accuracy_est`

## ▶️ How to Run

Once all dependencies are installed and your `.env` file is configured, run the Streamlit app:

```bash
streamlit run app.py
```

Your browser should automatically open to the application.

1.  Paste a supported social media URL into the text box on the **"Analysis Input"** tab.
2.  Click **"Analyze Now ✨"**.
3.  Wait for the analysis to complete (this can take 1-2 minutes for videos).
4.  View the detailed results, metrics, and plots on the **"Analysis Results"** tab.
5.  View past runs on the **"History"** tab.

## 📁 Project Structure

```
.
├── 📄 .env                 # (Must be created) API keys
├── 📜 app.py                # The main Streamlit application
├── 📦 logs/                  # Directory for log files (created automatically)
├── 📄 requirements.txt        # Python dependencies
├── 📄 sample.csv             # (Optional) Sample data for pre-loading
├── 🗃️ sentiment_history.db   # (Created automatically) SQLite database
└── 📁 temp_media/            # (Created automatically) Temporary media files
```

## ⚖️ License

This project is open-sourced under the **MIT License**. See the `LICENSE` file for more details.

```
```
