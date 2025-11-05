import streamlit as st
import os
import gc
import re
import numpy as np
import pandas as pd # Added for dataframes and CSV reading
from collections import defaultdict
import validators
import requests
import praw
from prawcore.exceptions import NotFound, Forbidden, PrawcoreException
import yt_dlp
from PIL import Image, UnidentifiedImageError
from moviepy.editor import VideoFileClip
import soundfile as sf
import librosa
from transformers import pipeline, BlipProcessor, BlipForConditionalGeneration
import easyocr
from fer import FER
from deepface import DeepFace
import psutil
import time
import logging
import traceback
import matplotlib.pyplot as plt
import seaborn as sns
from dotenv import load_dotenv
from datetime import datetime, timezone # Use timezone for consistency 
import emoji
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
import shutil 

# --- Added for Metrics ---
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, ConfusionMatrixDisplay

# --- End Added ---

# --- Added for Database ---
import sqlite3
import json
# --- End Added ---





# --- 1. CONFIGURATION & API KEYS ---
load_dotenv()
RAPIDAPI_KEY = os.getenv('RAPIDAPI_KEY', 'default_rapidapi_key') # Use default if not set
REDDIT_CLIENT_ID = os.getenv('REDDIT_CLIENT_ID', 'default_reddit_id') # Use default if not set
REDDIT_CLIENT_SECRET = os.getenv('REDDIT_CLIENT_SECRET', 'default_reddit_secret') # Use default if not set
REDDIT_USER_AGENT = os.getenv('REDDIT_USER_AGENT', 'streamlit_app/1.0') # Use default if not set
YOUTUBE_API_KEY = os.getenv('YOUTUBE_API_KEY', 'default_youtube_key') # Use default if not set
DB_NAME = "sentiment_history.db" # Database file name
SAMPLE_CSV_PATH = "sample.csv" # --- FIX: Corrected filename ---


# Configure logging
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
# --- FIX: Expanded logging config ---
# Use try-except for file handler creation in case of permission issues
log_handlers: list = [logging.StreamHandler()]
try:
    file_handler = logging.FileHandler(os.path.join(log_dir, f"analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"))
    log_handlers.append(file_handler)
except Exception as log_err:
    logging.error(f"Could not create log file handler: {log_err}")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=log_handlers
)


# Streamlit config
st.set_page_config(page_title="Social Media Sentiment Analyzer", layout="wide", initial_sidebar_state="expanded")

# --- 2. UTILITY FUNCTIONS ---
def clean_text(text):
    """Cleans text by removing URLs, emojis, and special characters."""
    if not text or not isinstance(text, str):
        return ""
    # Remove URLs first
    text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
    # Remove emojis
    try:
        text = emoji.replace_emoji(text, replace='')
    except ImportError:
        logging.warning("Emoji library not installed or failed to import. Skipping emoji removal.")
    # Remove most non-alphanumeric characters, keep basic punctuation
    text = re.sub(r'[^\w\s.,!?-]', '', text)
    # Normalize whitespace
    text = ' '.join(text.split())
    return text



def resize_image(image, max_size=(512, 512)):
    """Resizes image to reduce memory usage, handling potential errors."""
    try:
        img_copy = image.copy()
        # Ensure image has channels before thumbnailing (handle grayscale)
        if img_copy.mode == 'L':
             img_copy = img_copy.convert('RGB')
        elif img_copy.mode == 'LA':
             img_copy = img_copy.convert('RGBA') # Preserve alpha if present
        # Use appropriate resampling filter - fallback approach
        if hasattr(Image, 'Resampling'):
            resample_filter = Image.Resampling.LANCZOS
        else:
            # For older versions, use a simple approach
            resample_filter = 1  # NEAREST resampling
        img_copy.thumbnail(max_size, resample_filter)
        return img_copy
    except Exception as e:
        logging.error(f"Resize fail: {e}")
        return image



def normalize_audio(audio, sr):
    """Normalizes audio amplitude."""
    try:
        # Ensure input is float for librosa normalize
        if not np.issubdtype(audio.dtype, np.floating):
            audio = audio.astype(np.float32) / np.iinfo(audio.dtype).max # Convert int to float -1 to 1
        return librosa.util.normalize(audio)
    except Exception as e:
        logging.error(f"Normalize fail: {e}")
        return audio



def clean_temp_dir():
    """Cleans up the temporary media directory on startup."""
    temp_dir = "temp_media"
    if os.path.exists(temp_dir):
        logging.info(f"Cleaning up temp dir: {temp_dir}")
        for f in os.listdir(temp_dir):
            file_path = os.path.join(temp_dir, f)
            try:
                if os.path.isfile(file_path):
                    os.remove(file_path)
            except Exception as e:
                logging.error(f"Failed clean {file_path}: {e}")
        try:
            # Check again if dir exists before trying to remove
            # Ensure directory is empty before removing
            # Add a small delay in case files are still locked
            time.sleep(0.1)
            if not os.listdir(temp_dir):
                os.rmdir(temp_dir)
                logging.info(f"Removed empty temp dir: {temp_dir}")
            else:
                logging.warning(f"Temp dir {temp_dir} not empty, cannot remove.")
        except PermissionError:
            logging.warning(f"Perm fail remove {temp_dir}.")
        except OSError as e:
            # Ignore "Directory not empty" error if files couldn't be deleted
            if e.errno != 39: # errno 39 is Directory not empty on Windows
                logging.warning(f"OS fail remove {temp_dir}: {e}")
            else:
                 logging.warning(f"Temp dir {temp_dir} not empty (files likely in use), cannot remove dir.")



# --- FIX: ADDED MISSING FUNCTION ---
@st.cache_resource
def create_secure_session():
    """Creates a requests session with a user agent."""
    session = requests.Session()
    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'Accept': '*/*',
        'Connection': 'keep-alive'
        })
    return session
# --- END ADDED FUNCTION ---



# --- FIX: MOVED score_to_label HERE TO FIX NameError ---
# Define labels for metrics consistently
METRIC_LABELS = ['NEGATIVE', 'NEUTRAL', 'POSITIVE']



def score_to_label(score):
    """Converts a numerical score (0-1) to a sentiment label."""
    if score is None:
        return 'NEUTRAL' # Handle None case safely
    if not isinstance(score, (int, float)):
        logging.warning(f"Invalid score type for score_to_label: {type(score)}. Returning NEUTRAL.")
        return 'NEUTRAL'
    if score > 0.6:
        return 'POSITIVE'
    if score < 0.4:
        return 'NEGATIVE'
    return 'NEUTRAL'
# --- END MOVED FUNCTION ---



# --- 3. DATABASE FUNCTIONS ---
def init_db():
    """Initializes the SQLite database and creates/updates the history table."""
    conn = None
    table_created = False
    try:
        conn = sqlite3.connect(DB_NAME, timeout=10) # Added timeout
        cursor = conn.cursor()

        # Check if table exists first
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='analysis_history';")
        table_exists = cursor.fetchone()



        if not table_exists:
            # Create table with all columns, including new ones
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS analysis_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    url TEXT NOT NULL UNIQUE, -- Ensure URLs are unique
                    platform TEXT,
                    timestamp TEXT NOT NULL,
                    overall_sentiment TEXT,
                    overall_score REAL,
                    accuracy_est REAL,
                    title TEXT,
                    analysis_components_json TEXT,
                    weights_json TEXT,
                    timings_json TEXT,
                    data_source TEXT DEFAULT 'live_analysis',
                    true_sentiment TEXT, -- For evaluation data
                    -- Add expanded component columns
                    text_label TEXT,
                    text_score REAL,
                    media_label TEXT,
                    media_score REAL,
                    media_dominant_emotion TEXT,
                    media_emotions_json TEXT,
                    comment_label TEXT,
                    comment_avg_score REAL
                )
            ''')
            conn.commit()
            table_created = True
            logging.info(f"Database table 'analysis_history' created in '{DB_NAME}'.")
        else:
            logging.info(f"Database table 'analysis_history' already exists.")
            # --- Check and Add new columns if they don't exist (Migration) ---
            cursor.execute("PRAGMA table_info(analysis_history)")
            existing_columns = [col[1] for col in cursor.fetchall()]

            new_columns_to_add = {
                "data_source": "TEXT DEFAULT 'live_analysis'",
                "true_sentiment": "TEXT",
                "text_label": "TEXT",
                "text_score": "REAL",
                "media_label": "TEXT",
                "media_score": "REAL",
                "media_dominant_emotion": "TEXT",
                "media_emotions_json": "TEXT",
                "comment_label": "TEXT",
                "comment_avg_score": "REAL"
            }



            for col_name, col_type in new_columns_to_add.items():
                if col_name not in existing_columns:
                    try:
                        cursor.execute(f"ALTER TABLE analysis_history ADD COLUMN {col_name} {col_type}")
                        logging.info(f"Added column '{col_name}' to database.")
                    except sqlite3.OperationalError as alter_err:
                        # This might happen in rare race conditions, safe to ignore
                        if "duplicate column name" in str(alter_err).lower():
                            logging.info(f"Column '{col_name}' already exists (race condition).")
                        else:
                            raise alter_err # Re-raise other errors
            conn.commit()
            logging.info(f"Database schema in '{DB_NAME}' verified/updated.")



    except sqlite3.Error as e:
        logging.error(f"Database initialization/update error: {e}")
        st.error(f"Fatal Error: Could not initialize/update database: {e}")
        # Optionally, provide more context or stop the app
        # st.stop() # Uncomment to stop if DB is critical
    finally:
        if conn:
            conn.close() # Essential for releasing the DB lock
    return table_created # Return whether the table was newly created



def preload_data_from_csv(csv_filepath):
    """Reads the sample CSV and inserts its data into the database if the table is empty."""
    if not os.path.exists(csv_filepath):
        logging.warning(f"Sample data file not found: {csv_filepath}. Skipping preload.")
        st.warning(f"Sample data file '{csv_filepath}' not found. History will start empty.")
        return



    conn = None
    try:
        conn = sqlite3.connect(DB_NAME, timeout=10)
        cursor = conn.cursor()



        # Check if table is empty *before* preloading
        cursor.execute("SELECT COUNT(*) FROM analysis_history WHERE data_source = 'preloaded_sample'")
        count = cursor.fetchone()[0]
        if count > 0:
            logging.info("Database already contains preloaded sample data. Skipping preload.")
            return



        logging.info(f"Preloading data from {csv_filepath} into database...")
        df = pd.read_csv(csv_filepath)
        # Ensure required columns exist
        required_cols = [
            'url', 'true_sentiment', 'text_label', 'text_score', 'media_label',
            'media_score', 'comment_label', 'comment_avg_score', 'overall_sentiment',
            'overall_score', 'media_dominant_emotion', 'media_emotions_json',
            'platform', 'title', 'accuracy_est'
            ]

        # Check for missing columns
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            logging.error(f"CSV file is missing required columns: {missing_cols}")
            st.error(f"Sample CSV is missing columns: {', '.join(missing_cols)}. Cannot preload.")
            return



        inserted_count = 0
        skipped_count = 0

        # --- Expanded loop for readability ---
        for index, row in df.iterrows():
            # Construct the sentiment_data dict from CSV columns
            text_result_preload = None
            # Use a simpler approach instead of pd.notna
            if row['text_label'] is not None and str(row['text_label']).lower() != 'nan':
                text_result_preload = {'pred_label': str(row['text_label']), 'score': float(row['text_score'])}

            media_result_preload = None
            # Use a simpler approach instead of pd.notna
            if row['media_label'] is not None and str(row['media_label']).lower() != 'nan':
                dominant_emotion = str(row['media_dominant_emotion']) if (row['media_dominant_emotion'] is not None and str(row['media_dominant_emotion']).lower() != 'nan') else None
                media_result_preload = {
                    'pred_label': str(row['media_label']),
                    'score': float(row['media_score']),
                    'dominant_emotion': dominant_emotion
                }

            # Parse media emotions JSON safely
            media_emotions_preload = None
            media_emotions_json_str = row['media_emotions_json']
            # Use a simpler approach instead of pd.notna
            if media_emotions_json_str is not None and str(media_emotions_json_str).lower() != 'nan' and str(media_emotions_json_str).upper() != 'NULL':
                if isinstance(media_emotions_json_str, str):
                    # Handle potential string 'NaN', 'inf', '-inf' values
                    json_string = media_emotions_json_str.replace("NaN", "null").replace("inf", "null").replace("-inf", "null").replace("NULL", "null")
                    try:
                        media_emotions_preload = json.loads(json_string)
                        # Ensure values are valid floats
                        if isinstance(media_emotions_preload, dict):
                             media_emotions_preload = {k: float(v) if isinstance(v,(int,float)) else 0.0 for k,v in media_emotions_preload.items()}
                        else: # If not a dict after parse, treat as invalid
                            media_emotions_preload = None
                    except:
                        media_emotions_preload = None
                elif not isinstance(media_emotions_json_str, str):
                     logging.warning(f"media_emotions_json for {row['url']} is not a string: {type(media_emotions_json_str)}. Skipping.")
                     media_emotions_preload = None

            if media_result_preload and media_emotions_preload is not None:
                media_result_preload['emotions'] = media_emotions_preload
            elif media_result_preload: # If media result exists but emotions don't
                media_result_preload['emotions'] = None # Explicitly set to None

            # Reconstruct comment results (simplified for preloading)
            comment_results_preload = []
            # Use a simpler approach instead of pd.notna
            if row['comment_label'] is not None and str(row['comment_label']).lower() != 'nan' and row['comment_avg_score'] is not None and str(row['comment_avg_score']).lower() != 'nan':
                comment_results_preload.append({'pred_label': str(row['comment_label']), 'score': float(row['comment_avg_score'])})

            # This is the JSON blob that mimics a live analysis
            sentiment_data_preload = {
                'text': text_result_preload,
                'media': media_result_preload,
                'comments': comment_results_preload,
                'overall_score': float(row['overall_score']),
                'overall_sentiment': str(row['overall_sentiment'])
            }

            # --- Prepare entry for database insertion (matches save_analysis_to_db structure) ---
            entry_preload = {
                'url': str(row['url']),
                'platform': str(row['platform']),
                # Assign a slightly different timestamp to each preloaded entry
                'timestamp': datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S'),
                'sentiment': str(row['overall_sentiment']),
                'score': float(row['overall_score']),
                'accuracy': float(row['accuracy_est']),
                'title': str(row['title']) if (row['title'] is not None and str(row['title']).lower() != 'nan') else None,
                'sentiment_data': sentiment_data_preload,
                'weights': None, # Not available in CSV
                'timings': None, # Not available in CSV
                'true_sentiment': str(row['true_sentiment']) if (row['true_sentiment'] is not None and str(row['true_sentiment']).lower() != 'nan') else None
            }



            # Call save function with specific data source
            if save_analysis_to_db(entry_preload, data_source='preloaded_sample'):
                inserted_count += 1
            else:
                # save_analysis_to_db logs errors (likely duplicate URL)
                skipped_count += 1



        # Commit all inserts at the end
        conn.commit()
        logging.info(f"Finished preloading: {inserted_count} entries added, {skipped_count} skipped (likely duplicates).")
        if inserted_count > 0:
            st.success(f"Preloaded {inserted_count} sample analyses into history database.")
        # Removed redundant message if already preloaded, handled by initial count check



    except pd.errors.EmptyDataError:
        logging.warning(f"Sample data file {csv_filepath} is empty. Skipping preload.")
    except FileNotFoundError:
        logging.warning(f"Sample data file not found: {csv_filepath}. Skipping preload.")
        st.warning(f"Sample data file '{csv_filepath}' not found. History will start empty.")
    except Exception as e:
        logging.error(f"Error preloading data from CSV: {e}\n{traceback.format_exc()}")
        st.error(f"Error preloading sample data: {e}")
        if conn: # Rollback if error during loop
            conn.rollback()
    finally:
        if conn:
            conn.close() # Close connection after preloading





def save_analysis_to_db(entry, data_source='live_analysis'): # Added data_source parameter
    """Saves a single analysis entry (including detailed components) to the SQLite database."""
    conn = None
    required_keys = ['url', 'timestamp']
    if not all(key in entry for key in required_keys):
        logging.error(f"Analysis entry missing required keys: {entry}")
        return False



    def safe_json_dumps(data):
        """Safely convert Python object to JSON string, handling numpy types and ensuring valid floats."""
        try:
            def default_serializer(obj):
                # Convert numpy float types to standard Python float, handle NaN/inf
                if hasattr(obj, 'dtype'):
                    if np.issubdtype(obj.dtype, np.floating):
                        if np.isnan(obj) or np.isinf(obj):
                            return None # Represent NaN/inf as null in JSON
                        return float(obj)
                    # Convert numpy int types to standard Python int
                    if np.issubdtype(obj.dtype, np.integer):
                        return int(obj)
                # Handle other non-serializable types if necessary
                # Example: Convert datetime objects
                # if isinstance(obj, datetime):
                #     return obj.isoformat()
                raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")

            # Handle None case for the entire data structure
            if data is None:
                return None

            # Convert data, using default serializer
            return json.dumps(data, default=default_serializer, allow_nan=False) # Important: disallow NaN/inf in final JSON
        except TypeError as e:
            logging.error(f"JSON serialization error: {e} - Data snippet: {str(data)[:200]}")
            return None # Return None if conversion fails
        except ValueError as e: # Catch NaN/inf errors if allow_nan=False
             logging.error(f"JSON encoding error (likely NaN/inf): {e} - Data snippet: {str(data)[:200]}")
             return None


    # Prepare data for insertion (extract from entry dict)
    url = entry.get('url')
    platform = entry.get('platform')
    timestamp = entry.get('timestamp')
    overall_sentiment = entry.get('sentiment')
    overall_score = entry.get('score')
    accuracy_est = entry.get('accuracy')
    title = entry.get('title')
    weights = entry.get('weights')
    timings = entry.get('timings')
    sentiment_data = entry.get('sentiment_data', {}) # Get the nested results
    true_sentiment = entry.get('true_sentiment') # Get ground truth if present



    # Extract detailed component results safely from sentiment_data
    text_result = (sentiment_data or {}).get('text') or {}
    media_result = (sentiment_data or {}).get('media') or {}
    comment_results = (sentiment_data or {}).get('comments') or []



    text_label = text_result.get('pred_label')
    text_score = text_result.get('score')
    media_label = media_result.get('pred_label')
    media_score = media_result.get('score')
    media_dominant_emotion = media_result.get('dominant_emotion')
    media_emotions = media_result.get('emotions') # Get the dict directly



    # Derive comment metrics safely, handle potential non-dict items and NaN/inf scores
    valid_comment_scores = []
    if isinstance(comment_results, list):
        for c in comment_results:
             if isinstance(c, dict) and 'score' in c:
                  score_val = c['score']
                  if isinstance(score_val, (int, float)) and not (np.isnan(score_val) or np.isinf(score_val)):
                       valid_comment_scores.append(float(score_val))

    comment_avg_score = float(np.mean(valid_comment_scores)) if valid_comment_scores else None # Ensure float
    comment_label = score_to_label(comment_avg_score) if comment_avg_score is not None else None



    # Convert complex objects to JSON
    weights_json = safe_json_dumps(weights)
    timings_json = safe_json_dumps(timings)
    media_emotions_json = safe_json_dumps(media_emotions)
    analysis_components_json = safe_json_dumps(sentiment_data) # Save raw structure too

    # Helper to convert potential numpy floats/ints to standard Python types for DB
    def to_db_type(value):
        # Handle numpy types
        if hasattr(value, 'dtype'):
            if np.issubdtype(value.dtype, np.floating):
                if np.isnan(value) or np.isinf(value): return None
                return float(value)
            if np.issubdtype(value.dtype, np.integer):
                return int(value)
        # Handle standard types, check for NaN/inf
        if isinstance(value, float):
             if np.isnan(value) or np.isinf(value): return None
             return value
        if isinstance(value, int):
             return value
        # Handle None explicitly
        if value is None:
             return None
        # Convert other types to string or handle as needed
        return str(value)


    sql = '''
        INSERT INTO analysis_history (
            url, platform, timestamp, overall_sentiment, overall_score,
            accuracy_est, title, analysis_components_json, weights_json, timings_json,
            text_label, text_score, media_label, media_score,
            media_dominant_emotion, media_emotions_json, comment_label, comment_avg_score,
            true_sentiment, data_source
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(url) DO UPDATE SET
            platform=excluded.platform,
            timestamp=excluded.timestamp,
            overall_sentiment=excluded.overall_sentiment,
            overall_score=excluded.overall_score,
            accuracy_est=excluded.accuracy_est,
            title=excluded.title,
            analysis_components_json=excluded.analysis_components_json,
            weights_json=excluded.weights_json,
            timings_json=excluded.timings_json,
            text_label=excluded.text_label,
            text_score=excluded.text_score,
            media_label=excluded.media_label,
            media_score=excluded.media_score,
            media_dominant_emotion=excluded.media_dominant_emotion,
            media_emotions_json=excluded.media_emotions_json,
            comment_label=excluded.comment_label,
            comment_avg_score=excluded.comment_avg_score,
            true_sentiment=COALESCE(excluded.true_sentiment, true_sentiment), -- Only update true_sentiment if new value is not NULL
            data_source=excluded.data_source -- Update source (e.g., if re-analyzing)
        '''



    # Apply type conversion for numeric fields before passing to execute
    params = (
            url, platform, timestamp, overall_sentiment,
            to_db_type(overall_score), # Convert score
            to_db_type(accuracy_est), # Convert accuracy
            title, analysis_components_json, weights_json, timings_json,
            text_label, to_db_type(text_score), # Convert text score
            media_label, to_db_type(media_score), # Convert media score
            media_dominant_emotion, media_emotions_json, comment_label,
            to_db_type(comment_avg_score), # Convert comment avg score
            true_sentiment, data_source
        )
    try:
        conn = sqlite3.connect(DB_NAME, timeout=10)
        cursor = conn.cursor()
        cursor.execute(sql, params)
        conn.commit()

        row_id = cursor.lastrowid # Get ID of inserted row
        # Check if an update occurred (ON CONFLICT DO UPDATE doesn't set lastrowid)
        # Use rowcount to confirm if INSERT or UPDATE happened
        if cursor.rowcount > 0:
             if row_id and row_id > 0:
                  logging.info(f"Saved new analysis (ID: {row_id}) for {url} to database (Source: {data_source}).")
             else:
                  logging.info(f"Updated analysis for {url} in database (Source: {data_source}).")
             return True # Indicate success
        else:
             # This might happen if ON CONFLICT clause didn't execute (e.g., data identical)
             # Or potentially if ON CONFLICT DO NOTHING was used (not the case here)
             logging.warning(f"DB operation completed but rowcount is 0 for {url}. Might indicate no change.")
             return True # Still technically not an error

    except sqlite3.Error as e:
        logging.error(f"Database save error for {url}: {e}\n{traceback.format_exc()}")
        st.warning(f"Warning: Could not save analysis to history database: {e}")
        if conn:
            conn.rollback() # Rollback changes on error
        return False
    finally:
        if conn:
            conn.close()



def load_history_from_db(limit=100): # Increased default limit
    """Loads the last N analysis entries from the SQLite database, prioritizing new columns."""
    conn = None
    history = []
    try:
        conn = sqlite3.connect(DB_NAME, timeout=10)
        conn.row_factory = sqlite3.Row # Return rows as dict-like objects
        cursor = conn.cursor()
        # Select all columns
        cursor.execute("""
            SELECT id, url, platform, timestamp, overall_sentiment, overall_score,
                    accuracy_est, title,
                    text_label, text_score,
                    media_label, media_score, media_dominant_emotion, media_emotions_json,
                    comment_label, comment_avg_score,
                    weights_json, timings_json,
                    data_source, true_sentiment,
                    analysis_components_json -- Load old blob just in case
            FROM analysis_history
            ORDER BY timestamp DESC
            LIMIT ?
            """, (limit,))
        rows = cursor.fetchall()



        for row in rows:
            entry = dict(row) # Convert row object to dict



            # Safely parse JSON strings back to dicts/lists
            def safe_json_loads(json_str):
                # Check if input is already None or not a string
                if json_str is None or not isinstance(json_str, str):
                     return None
                try:
                    # Handle NULL strings explicitly if they appear
                    if json_str.upper() == 'NULL':
                        return None
                    # Replace NaN/inf representations if they exist BEFORE parsing
                    json_str_cleaned = json_str.replace('NaN', 'null').replace('Infinity', 'null').replace('-Infinity', 'null')
                    return json.loads(json_str_cleaned)
                except json.JSONDecodeError as e:
                    logging.error(f"JSON decode error loading history for url {entry.get('url', 'N/A')}: {e} - String snippet: {json_str[:200]}")
                    return None



            # Parse general JSON fields
            entry['weights'] = safe_json_loads(entry.get('weights_json'))
            entry['timings'] = safe_json_loads(entry.get('timings_json'))
            media_emotions = safe_json_loads(entry.get('media_emotions_json')) # Get parsed emotions



            # --- Reconstruct sentiment_data partially for plotting compatibility ---
            # Priority is using the direct columns for metrics calculation

            # Reconstruct text_result
            text_result_hist = None
            if entry.get('text_label') is not None or entry.get('text_score') is not None:
                text_result_hist = {
                    'pred_label': entry.get('text_label'),
                    'score': entry.get('text_score')
                }

            # Reconstruct media_result
            media_result_hist = None
            if entry.get('media_label') is not None or entry.get('media_score') is not None:
                media_result_hist = {
                    'pred_label': entry.get('media_label'),
                    'score': entry.get('media_score'),
                    'dominant_emotion': entry.get('media_dominant_emotion'),
                    'emotions': media_emotions # Add parsed emotions
                }



            # Reconstruct comments (simplified list with just the average)
            comment_results_hist = []
            if entry.get('comment_label') is not None or entry.get('comment_avg_score') is not None:
                comment_results_hist.append({
                    'pred_label': entry.get('comment_label'),
                    'score': entry.get('comment_avg_score')
                })





            entry['sentiment_data'] = {
                'text': text_result_hist,
                'media': media_result_hist,
                'comments': comment_results_hist, # Use reconstructed simplified list
                'overall_score': entry.get('overall_score'),
                'overall_sentiment': entry.get('overall_sentiment')
            }

            # Add other top-level keys expected by display/metrics
            entry['accuracy'] = entry.get('accuracy_est') # Use alias
            entry['sentiment'] = entry.get('overall_sentiment')
            entry['score'] = entry.get('overall_score')

            # Note: comments_analyzed (raw comments) is NOT stored/loaded
            # We will adapt the History tab display to handle this
            entry['comments_analyzed_count'] = len(comment_results_hist) if comment_results_hist else 0





            history.append(entry)



        logging.info(f"Loaded {len(history)} entries from database.")
    except sqlite3.Error as e:
        logging.error(f"Database load error: {e}")
        st.warning(f"Warning: Could not load analysis history from database: {e}")
    finally:
        if conn:
            conn.close()
    # Return newest first (already ordered by DESC in SQL)
    return history

# --- 4. MODEL INITIALIZATION ---
# Use st.cache_resource for models that should persist across reruns
@st.cache_resource
def initialize_models():
    """Loads all AI models into memory FORCE ON CPU."""
    with st.spinner('🚀 Initializing AI engines... (CPU Mode)'):
        logging.info("Initializing models on CPU...")
        device = "cpu"
        # Check GPU availability for EasyOCR (though we force CPU)
        try:
             import torch
             gpu_available_easyocr = torch.cuda.is_available()
             if gpu_available_easyocr:
                  logging.info("GPU detected by PyTorch, but forcing EasyOCR to CPU.")
                  gpu_available_easyocr = False # Force CPU
        except ImportError:
             gpu_available_easyocr = False
             logging.info("PyTorch not found, EasyOCR will use CPU.")

        logging.info(f"Forcing all models to load onto CPU.")

        models_dict = {}
        try:
            logging.info("Loading EasyOCR...")
            models_dict['reader'] = easyocr.Reader(['en'], gpu=gpu_available_easyocr)
            logging.info("EasyOCR loaded.")

            logging.info("Loading BLIP...")
            models_dict['blip_processor'] = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
            blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")
            # Simplify device assignment for BLIP model
            models_dict['blip_model'] = blip_model
            logging.info("BLIP loaded.")

            models_dict['text_analyzer'] = initialize_text_ensemble()
            models_dict['emotion_ensemble'] = initialize_emotion_ensemble()
            models_dict['audio_analyzer'] = initialize_audio_ensemble()
            models_dict['vlm_ensemble'] = initialize_vlm_ensemble()

            logging.info("All models initialized successfully on CPU.")
            return models_dict

        except Exception as e:
            st.error(f"Fatal Model Load Error: {e}")
            logging.error(f"Model init failed: {traceback.format_exc()}")
            st.stop() # Stop execution if models can't load


@st.cache_resource
def initialize_text_ensemble():
    """Initializes text sentiment analysis ensemble on CPU."""
    device = "cpu"
    logging.info("Init text ensemble (CPU)...")
    try:
        roberta_pipeline = pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment-latest", device=device)
        distilbert_pipeline = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english", device=device)
        deberta_pipeline = pipeline("sentiment-analysis", model="siebert/sentiment-roberta-large-english", device=device)
        logging.info("Text ensemble loaded.")
        return {
            'roberta': roberta_pipeline,
            'distilbert': distilbert_pipeline,
            'deberta': deberta_pipeline
        }
    except Exception as e:
         logging.error(f"Failed to load text ensemble: {e}\n{traceback.format_exc()}")
         st.error(f"Error loading text models: {e}")
         st.stop()


@st.cache_resource
def initialize_vlm_ensemble():
    """Initializes vision-language model ensemble (Blip only) on CPU."""
    device = "cpu"
    logging.info("Init VLM ensemble (Blip-only CPU)...")
    try:
        # Re-using the already loaded BLIP model components if available
        # This function primarily wraps the pipeline creation
        # Note: If initialize_models fails before this, it won't be called.
        # If it succeeds, blip_model components are already loaded.
        blip_pipeline = pipeline("image-to-text", model="Salesforce/blip-image-captioning-large", device=device)
        ensemble = {'blip': blip_pipeline} # Renamed key for clarity
        logging.info("BLIP VLM Pipeline ready.")
        return ensemble
    except Exception as e:
        logging.error(f"BLIP VLM pipeline creation failed: {e}\n{traceback.format_exc()}")
        st.error(f"Fatal BLIP VLM Pipeline Error: {e}")
        st.stop()


@st.cache_resource
def initialize_emotion_ensemble():
    """Initializes facial emotion detection ensemble (FER models) on CPU."""
    logging.info("Init emotion ensemble (FER CPU)...")
    try:
        fer_mtcnn_model = FER(mtcnn=True)
        # Haar cascades might require specific file paths depending on install
        # FER library usually handles this internally, but good to be aware
        fer_haar_model = FER(mtcnn=False)
        logging.info("FER models loaded.")
        return {
            'fer_mtcnn': fer_mtcnn_model,
            'fer_haar': fer_haar_model
        }
    except Exception as e:
        logging.error(f"Failed to load FER models: {e}\n{traceback.format_exc()}")
        st.error(f"Error loading emotion models: {e}")
        st.stop()


@st.cache_resource
def initialize_audio_ensemble():
    """Initializes audio analysis ensemble on CPU."""
    device = "cpu"
    logging.info("Init audio ensemble (CPU)...")
    try:
        # Suppress specific Hugging Face warnings if needed, but errors should still show
        logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)
        wav2vec_pipeline = pipeline("audio-classification", model="ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition", device=device)
        whisper_pipeline = pipeline("automatic-speech-recognition", model="openai/whisper-tiny", device=device)
        logging.getLogger("transformers.modeling_utils").setLevel(logging.WARNING) # Restore warnings
        logging.info("Audio ensemble loaded.")
        return {
            'wav2vec': wav2vec_pipeline,
            'whisper': whisper_pipeline
        }
    except Exception as e:
        logging.error(f"Failed to load audio ensemble: {e}\n{traceback.format_exc()}")
        st.error(f"Error loading audio models: {e}")
        st.stop()





# --- 5. DATA FETCHING ---
def fetch_social_media_data(url, session):
    """
    Fetches metadata (title, comments, and media URL) for a given social media post URL.

    ✅ FIXES:
    - Proper Reddit handling using PRAW (no yt-dlp for Reddit images/videos).
    - Handles Reddit galleries and image-only posts.
    - Uses RapidAPI for Instagram/TikTok/Twitter/Facebook/Pinterest/YouTube.
    - Returns direct media_url (image/video/audio) for downstream analysis.
    - Added robust handling for lists/dicts and image-only URLs from RapidAPI.
    """

    logging.info(f"Fetching data: {url}")

    SUPPORTED_MEDIA_EXTS = (
        '.mp4', '.mov', '.avi', '.webm', '.mkv',    # video
        '.mp3', '.wav', '.m4a', '.aac', '.ogg',     # audio
        '.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp'  # image
    )

    try:
        # --- REDDIT ---
        if "reddit.com" in url or "redd.it" in url:
            logging.info("Handling as Reddit URL...")

            # Ensure Reddit credentials exist
            if not all([REDDIT_CLIENT_ID, REDDIT_CLIENT_SECRET, REDDIT_USER_AGENT]):
                st.error("Reddit API credentials missing in environment variables.")
                logging.error("Reddit API credentials missing.")
                return None

            try:
                # Initialize PRAW Reddit client (explicit, only for Reddit)
                reddit = praw.Reddit(
                    client_id=REDDIT_CLIENT_ID,
                    client_secret=REDDIT_CLIENT_SECRET,
                    user_agent=REDDIT_USER_AGENT
                )
                submission = reddit.submission(url=url)
                title = submission.title if hasattr(submission, "title") else ""
                author = submission.author
                if author is None:
                    st.error("Reddit post appears deleted or removed.")
                    logging.error(f"Reddit post {url} has no author.")
                    return None
            except Exception as e:
                st.error(f"Failed to access Reddit API: {e}")
                logging.error(f"Reddit API failure: {traceback.format_exc()}")
                return None

            try:
                # --- Comments (top-level) ---
                submission.comments.replace_more(limit=0)
                max_comments = st.session_state.get('max_comments', 5)
                comments_data = [getattr(c, "body", "") for c in submission.comments.list()[:max_comments] if hasattr(c, "body")]

                # --- Media Extraction ---
                media_urls = []

                # Case 1: Reddit gallery (multi-image)
                if getattr(submission, "is_gallery", False):
                    for item in submission.gallery_data["items"]:
                        media_id = item["media_id"]
                        img_info = submission.media_metadata.get(media_id, {})
                        if "s" in img_info and "u" in img_info["s"]:
                            img_url = img_info["s"]["u"].split("?")[0].replace("amp;", "")
                            if validators.url(img_url):
                                media_urls.append(img_url)
                    logging.info(f"Extracted {len(media_urls)} images from Reddit gallery.")

                # Case 2: Reddit video
                if hasattr(submission, "media") and submission.media:
                    reddit_video = submission.media.get("reddit_video", {})
                    if "fallback_url" in reddit_video:
                        video_url = reddit_video["fallback_url"].split("?")[0]
                        if validators.url(video_url):
                            media_urls.append(video_url)
                            logging.info(f"Reddit video found: {video_url}")

                # Case 3: Direct image post
                elif hasattr(submission, "url") and validators.url(submission.url):
                    temp_url = submission.url.split("?")[0]
                    if any(temp_url.lower().endswith(ext) for ext in SUPPORTED_MEDIA_EXTS):
                        media_urls.append(temp_url)
                    elif "v.redd.it" in temp_url:
                        media_urls.append(temp_url)

                if not media_urls:
                    media_urls.append(submission.url)

                formatted_comments = [{"text": clean_text(c)} for c in comments_data if c]

                return {
                    "platform": "reddit",
                    "title": clean_text(title),
                    "comments": formatted_comments,
                    "media_url": media_urls[0]
                }

            except Exception as e:
                st.error(f"Error processing Reddit data: {e}")
                logging.error(f"Reddit processing error: {traceback.format_exc()}")
                return None

        # --- OTHER PLATFORMS (YouTube, Instagram, TikTok, etc.) ---
        else:
            logging.info(f"Fetching via RapidAPI for non-Reddit URL: {url}")
            platform = "unknown"
            title = ""
            media_url = None

            if not RAPIDAPI_KEY or RAPIDAPI_KEY == "api key here":
                st.warning("RapidAPI key missing. Skipping external fetch.")
                return {"platform": "unknown", "title": "", "comments": [], "media_url": url}

            headers = {
                "x-rapidapi-key": RAPIDAPI_KEY,
                "x-rapidapi-host": "social-media-master.p.rapidapi.com"
            }
            querystring = {"url": url}
            api_url = "https://social-media-master.p.rapidapi.com/universal-download"

            try:
                response = session.get(api_url, headers=headers, params=querystring, timeout=40)
                response.raise_for_status()
                data = response.json()

                title = data.get("caption") or data.get("title", "")
                platform = data.get("platform", "unknown")
                possible_media = data.get("media_url") or data.get("media") or url

                candidate = None

                if isinstance(possible_media, list) and possible_media:
                    for item in possible_media:
                        if isinstance(item, dict):
                            for k in ("url", "src", "link"):
                                val = item.get(k)
                                if val and validators.url(val):
                                    candidate = val
                                    break
                        elif isinstance(item, str) and validators.url(item):
                            candidate = item
                            break

                elif isinstance(possible_media, dict):
                    for k in ("url", "src", "link"):
                        val = possible_media.get(k)
                        if val and validators.url(val):
                            candidate = val
                            break

                elif isinstance(possible_media, str) and validators.url(possible_media):
                    candidate = possible_media

                if not candidate:
                    candidate = url

                media_url = candidate.split("?")[0]

            except Exception as e:
                logging.error(f"RapidAPI error: {traceback.format_exc()}")
                media_url = url

            return {
                "platform": platform,
                "title": clean_text(title),
                "comments": [],
                "media_url": media_url
            }

    except Exception as e:
        st.error(f"Unexpected fetch error: {e}")
        logging.error(traceback.format_exc())
        return {"platform": "unknown", "title": "", "comments": [], "media_url": url}
# --- END fetch_social_media_data ---

def download_media(url, session):
    """
    Downloads media (image, video, or audio) from any social post URL.

    ✅ FINAL RULES:
    - Reddit: handled only by PRAW elsewhere — this function never uses yt-dlp for Reddit.
    - Non-Reddit platforms:
        • If media URL is an image (HEAD or extension), download via requests.
        • If media URL is a video/audio link, use yt-dlp.
        • Added clear logging for yt-dlp usage.
    """

    import yt_dlp, shutil, urllib.parse

    TEMP_DIR = "temp_media"
    os.makedirs(TEMP_DIR, exist_ok=True)

    IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp", ".tiff")
    VIDEO_EXTS = (".mp4", ".mov", ".avi", ".mkv", ".webm")
    AUDIO_EXTS = (".mp3", ".wav", ".m4a", ".aac", ".ogg", ".flac")

    if not url or not validators.url(url):
        logging.warning(f"Invalid or missing URL for download: {url}")
        return None, None

    # --- Reddit check ---
    if any(x in url for x in ["reddit.com", "redd.it", "v.redd.it", "preview.redd.it", "i.redd.it"]):
        logging.info(f"Skipping yt-dlp for Reddit URL (handled by PRAW): {url}")
        headers = {"User-Agent": REDDIT_USER_AGENT or "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
        try:
            head = session.head(url, allow_redirects=True, headers=headers, timeout=100)
            ctype = (head.headers.get("content-type") or "").lower()
            ext = os.path.splitext(urllib.parse.urlparse(url).path)[1]
            if "image/" in ctype or ext.lower() in IMAGE_EXTS:
                local = os.path.join(TEMP_DIR, f"media_{int(time.time())}{ext or '.jpg'}")
                with session.get(url, stream=True, headers=headers, timeout=600) as r:
                    r.raise_for_status()
                    with open(local, "wb") as f:
                        shutil.copyfileobj(r.raw, f)
                return local, "image"
            elif "video/" in ctype or ext.lower() in VIDEO_EXTS:
                local = os.path.join(TEMP_DIR, f"media_{int(time.time())}{ext or '.mp4'}")
                with session.get(url, stream=True, headers=headers, timeout=600) as r:
                    r.raise_for_status()
                    with open(local, "wb") as f:
                        shutil.copyfileobj(r.raw, f)
                return local, "video"
        except Exception as e:
            logging.error(f"Reddit download failed: {traceback.format_exc()}")
            return None, None
        return None, None

    logging.info(f"Attempting download (non-Reddit): {url}")

    try:
        head = session.head(url, allow_redirects=True, timeout=10)
        ctype = (head.headers.get("content-type") or "").lower()
    except Exception:
        ctype = ""

    # --- Direct image ---
    if "image/" in ctype or any(url.lower().endswith(ext) for ext in IMAGE_EXTS):
        logging.info("Detected image via HEAD or extension; downloading with requests.")
        ext = os.path.splitext(url.split("?")[0])[1] or ".jpg"
        local = os.path.join(TEMP_DIR, f"media_{int(time.time())}{ext}")
        with session.get(url, stream=True, timeout=60) as r:
            r.raise_for_status()
            with open(local, "wb") as f:
                shutil.copyfileobj(r.raw, f)
        return local, "image"

    # --- Direct video/audio ---
    if "video/" in ctype or any(url.lower().endswith(ext) for ext in VIDEO_EXTS + AUDIO_EXTS):
        logging.info("Detected video/audio file — Using yt-dlp for non-Reddit video download.")
        try:
            ydl_opts = {
                "quiet": True,
                "no_warnings": True,
                "noplaylist": True,
                "retries": 3,
                "socket_timeout": 45,
                "format": "bestvideo+bestaudio/best",
                "merge_output_format": "mp4",
                "outtmpl": os.path.join(TEMP_DIR, "media_%(id)s.%(ext)s")
            }
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=True)
                dl_path = ydl.prepare_filename(info)
            if os.path.exists(dl_path):
                ext = os.path.splitext(dl_path)[1].lower()
                if ext in VIDEO_EXTS:
                    return dl_path, "video"
                if ext in AUDIO_EXTS:
                    return dl_path, "audio"
                return dl_path, "unknown"
        except Exception as e:
            logging.error(f"yt-dlp failed for non-Reddit video: {traceback.format_exc()}")
            return None, None

    # --- Fallback ---
    logging.warning("Unknown or unsupported content-type; attempting yt-dlp as fallback.")
    try:
        ydl_opts = {
            "quiet": True,
            "no_warnings": True,
            "noplaylist": True,
            "retries": 3,
            "socket_timeout": 45,
            "format": "bestvideo+bestaudio/best",
            "merge_output_format": "mp4",
            "outtmpl": os.path.join(TEMP_DIR, "media_%(id)s.%(ext)s")
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            dl_path = ydl.prepare_filename(info)
        if os.path.exists(dl_path):
            ext = os.path.splitext(dl_path)[1].lower()
            if ext in VIDEO_EXTS:
                return dl_path, "video"
            if ext in AUDIO_EXTS:
                return dl_path, "audio"
            return dl_path, "unknown"
    except Exception as e:
        logging.error(f"Final yt-dlp fallback failed: {traceback.format_exc()}")

    logging.error("Media download failed or was skipped.")
    return None, None
# --- END download_media ---


# --- 7. ANALYSIS FUNCTIONS ---
def analyze_text_ensemble(text, ensemble):
    """Analyzes text sentiment using an ensemble of models."""
    # --- FIX: Handle empty/invalid text ---
    if not text or not isinstance(text, str) or not text.strip():
        logging.info("Empty or invalid text provided for analysis. Returning NEUTRAL.")
        return {'pred_label': 'NEUTRAL', 'score': 0.5}

    text_to_analyze = text[:1024] # Limit length for performance
    logging.info(f"Analyze text: {text_to_analyze[:50]}...")
    results = []

    for name, analyzer in ensemble.items():
        try:
            # Add timeout to pipeline call if possible/needed, though less common for local pipelines
            # result = analyzer(text_to_analyze[:512], truncation=True, timeout=10)[0] # Example timeout
            result = analyzer(text_to_analyze[:512], truncation=True)[0] # Truncate for model input limits
            score = 0.0
            label_raw = result.get('label', 'neutral').lower() # Safely get label

            # Map various possible labels to scores (-1 to 1)
            # Add more mappings if other models return different labels
            if label_raw in ['positive', 'pos', 'label_2', '4 stars', '5 stars']: # Added examples
                score = result.get('score', 0.0) # Safely get score
            elif label_raw in ['negative', 'neg', 'label_0', '1 star', '2 stars']: # Added examples
                score = -result.get('score', 0.0)
            elif label_raw in ['neutral', 'neu', 'label_1', '3 stars']: # Added examples
                score = 0.0 # Neutral maps to 0 in [-1, 1] range
            else:
                logging.warning(f"Unknown label from model {name}: {result.get('label')}")
                continue # Skip unknown labels

            # Ensure score is a float before appending
            if isinstance(score, (int, float)):
                 results.append(float(score))
                 logging.info(f"{name}: Label='{result.get('label', 'N/A')}', RawScore={result.get('score', 0.0):.2f} -> MappedScore={score:.2f}")
            else:
                 logging.warning(f"Model {name} returned non-numeric score: {score}")

        except Exception as e:
            logging.error(f"Text model {name} failed for text '{text_to_analyze[:30]}...': {e}\n{traceback.format_exc()}")
            # Optionally add a placeholder score (e.g., 0.0 for neutral) on failure
            # results.append(0.0)
            continue # Continue with other models if one fails

    if not results:
        logging.warning("No text models succeeded or returned valid scores.")
        return {'pred_label': 'NEUTRAL', 'score': 0.5}

    # Average score (-1 to 1) and convert to 0-1 range
    final_avg_neg_pos = np.mean(results)
    final_score_0_1 = (final_avg_neg_pos + 1) / 2 # Convert [-1, 1] to [0, 1]
    final_label = score_to_label(final_score_0_1) # Use the consistent label function

    return {'pred_label': final_label, 'score': final_score_0_1}
# --- END analyze_text_ensemble ---



def analyze_image_advanced(image_path_or_img, models):
    """Analyzes image using OCR, VLM (Blip), and Emotion Detection."""
    logging.info("Analyze img...")
    img_obj_context = None # To manage PIL Image object lifecycle
    img_array = None

    try:
        # Load and prepare image
        if isinstance(image_path_or_img, str): # Input is a file path
            if not os.path.exists(image_path_or_img):
                raise FileNotFoundError(f"Image file not found: {image_path_or_img}")
            # Use context manager for file opening
            with Image.open(image_path_or_img) as img_opened:
                 img_converted = img_opened.convert("RGB")
                 img_resized = resize_image(img_converted) # Resize after converting
                 img_array = np.array(img_resized)
                 img_obj_context = img_resized # Keep resized object for VLM
        elif isinstance(image_path_or_img, Image.Image): # Input is a PIL Image object
            img_converted = image_path_or_img.convert("RGB")
            # Assume already resized if PIL object passed directly (e.g., from video frame)
            img_resized = img_converted
            img_array = np.array(img_resized)
            img_obj_context = img_resized # Keep for VLM
        else:
            raise ValueError(f"Invalid image input type: {type(image_path_or_img)}")

        if img_array is None:
            raise ValueError("Image array creation failed")

        captions = [] # List to hold text extracted from image (OCR + VLM)

        # 1. OCR
        try:
            logging.info("Running OCR...")
            # Ensure array is uint8 for easyocr
            img_arr_ocr = None
            if img_array.dtype == np.uint8:
                img_arr_ocr = img_array
            elif np.issubdtype(img_array.dtype, np.floating) and img_array.max() <= 1.0: # Check if float normalized 0-1
                img_arr_ocr = (img_array * 255).astype(np.uint8)
            elif np.issubdtype(img_array.dtype, np.integer): # Handle other integer types
                 img_arr_ocr = img_array.astype(np.uint8)
            else:
                 logging.warning(f"Unsupported image array dtype for OCR: {img_array.dtype}. Attempting conversion.")
                 try:
                      # Attempt conversion assuming 0-255 range if not float 0-1
                      img_arr_ocr = img_array.astype(np.uint8)
                 except ValueError:
                      logging.error("Could not convert image array to uint8 for OCR.")
                      img_arr_ocr = None # Signal failure

            if img_arr_ocr is not None:
                ocr_results = models['reader'].readtext(img_arr_ocr, detail=0, paragraph=True) # Use paragraph mode
                ocr_text = clean_text(' '.join(ocr_results))
                if ocr_text:
                    captions.append(ocr_text)
                    logging.info(f"OCR found: {ocr_text[:50]}...")
                else:
                    logging.info("No text found by OCR.")
            else:
                 logging.error("OCR skipped due to image array conversion failure.")

        except Exception as ocr_e:
            logging.error(f"OCR failed: {ocr_e}\n{traceback.format_exc()}")


        # 2. VLM Captioning (using Blip pipeline from vlm_ensemble)
        # --- Use the img_obj_context directly ---
        if img_obj_context and 'blip' in models['vlm_ensemble']: # Check key exists
            try:
                vlm_pipeline = models['vlm_ensemble']['blip']
                # Generate caption with adjusted parameters if needed
                # Example: max_length=50, num_beams=4
                vlm_results = vlm_pipeline(img_obj_context, generate_kwargs={"max_new_tokens": 75, "num_beams": 5})
                if vlm_results and isinstance(vlm_results, list) and vlm_results[0]:
                    caption_raw = vlm_results[0].get('generated_text', '')
                    caption_cleaned = clean_text(caption_raw)
                    if caption_cleaned:
                        captions.append(caption_cleaned)
                        logging.info(f"VLM (blip) caption: {caption_cleaned[:50]}...")
                else:
                    logging.warning("VLM (blip) returned empty or invalid result.")
            except Exception as vlm_e:
                logging.error(f"VLM (blip) failed: {vlm_e}\n{traceback.format_exc()}")
        else:
             logging.warning("VLM skipped: No valid image object or 'blip' model not found in ensemble.")


        # 3. Analyze Text Sentiment (OCR + Captions)
        avg_caption_score = 0.5 # Default to neutral
        if captions:
            caption_analysis_results = []
            for cap_text in captions:
                if cap_text: # Ensure text is not empty
                    analysis_res = analyze_text_ensemble(cap_text, models['text_analyzer'])
                    if analysis_res and 'score' in analysis_res: # Check if analysis returned a valid score
                        caption_analysis_results.append(analysis_res['score'])

            if caption_analysis_results: # If any analysis succeeded
                avg_caption_score = np.mean(caption_analysis_results)
                logging.info(f"Average sentiment score from image text/captions: {avg_caption_score:.2f}")
        else:
            logging.info("No text extracted from image (OCR or VLM) to analyze.")


        # 4. Emotion Detection
        detected_emotions = {'neutral': 1.0}; dominant_emotion = 'neutral' # Defaults
        try:
            # Pass the original numpy array, ensure it's suitable (e.g., BGR for DeepFace if needed, RGB for FER)
            # FER expects RGB, DeepFace might default to BGR but often handles RGB too
            emotions_from_ensemble = detect_emotions_ensemble(img_array, models['emotion_ensemble'])
            if emotions_from_ensemble: # Check if detection returned a valid dict
                detected_emotions = emotions_from_ensemble
                # Find dominant emotion (handle potential empty dict)
                if detected_emotions:
                     dominant_emotion = max(detected_emotions.items(), key=lambda item: item[1])[0]
                else: # Reset to neutral if ensemble returned empty
                     detected_emotions = {'neutral': 1.0}; dominant_emotion = 'neutral'
                logging.info(f"Detected emotions: {detected_emotions}, Dominant: {dominant_emotion}")
            else:
                 logging.info("No emotions detected by ensemble.")

        except Exception as emotion_e:
            logging.error(f"Emotion detection failed: {emotion_e}\n{traceback.format_exc()}")
            # Keep default neutral emotions


        # 5. Fuse Scores (Text/Caption Sentiment + Emotion Sentiment)
        # Map dominant emotion to a sentiment score [0, 1]
        emotion_to_sentiment_map = {'happy':0.9, 'surprise':0.6, 'neutral':0.5, 'sad':0.2, 'angry':0.1, 'fear':0.2, 'disgust':0.1}
        emotion_sentiment_score = emotion_to_sentiment_map.get(dominant_emotion, 0.5) # Default neutral

        # Weighted average: Adjust weights if needed (e.g., 70% text, 30% emotion)
        text_weight = 0.7
        emotion_weight = 0.3
        final_fused_score = (avg_caption_score * text_weight) + (emotion_sentiment_score * emotion_weight)

        # Determine final label based on the fused score
        final_label = score_to_label(final_fused_score)

        logging.info(f"Image Final: Label={final_label}, FusedScore={final_fused_score:.2f} (TextScore={avg_caption_score:.2f}, EmoScore={emotion_sentiment_score:.2f}), Dominant Emotion: {dominant_emotion}")
        return {'pred_label': final_label, 'score': final_fused_score, 'dominant_emotion': dominant_emotion, 'emotions': detected_emotions}



    except FileNotFoundError as fnf_err:
         st.error(f"Image analysis skipped: {fnf_err}")
         logging.error(f"Image analysis FileNotFoundError: {fnf_err}")
         return {'pred_label':'NEUTRAL', 'score':0.5, 'dominant_emotion':'neutral', 'emotions':{'neutral':1.0}}
    except Exception as e: # Catch-all for unexpected errors during image analysis
        st.error(f"Image analysis failed: {e}")
        logging.error(f"Image analysis general failure: {traceback.format_exc()}")
        return {'pred_label':'NEUTRAL', 'score':0.5, 'dominant_emotion':'neutral', 'emotions':{'neutral':1.0}} # Return neutral default
    # 'finally' block removed as context manager handles file closing
# --- END analyze_image_advanced ---



def detect_emotions_ensemble(image_array_rgb, ensemble):
    """Detects facial emotions using FER (MTCNN, Haar) and DeepFace (retinaface, opencv). Expects RGB numpy array."""
    if image_array_rgb is None or not isinstance(image_array_rgb, np.ndarray):
        logging.warning("Invalid image array provided to detect_emotions_ensemble.")
        return {'neutral': 1.0}

    logging.info("Detecting emotions...")
    # Dictionary to store lists of emotion dicts {emotion: score} per face, per model
    model_face_emotions = defaultdict(list)

    # --- Run FER models (expect RGB) ---
    for model_name, detector in ensemble.items():
        # Filter for FER models in the ensemble
        if 'fer_' in model_name:
            try:
                # FER detect_emotions returns a list of dicts, one per face
                # Each dict has 'box' and 'emotions' keys
                fer_results = detector.detect_emotions(image_array_rgb)
                if fer_results:
                    for face_data in fer_results:
                        if isinstance(face_data, dict) and 'emotions' in face_data and isinstance(face_data['emotions'], dict):
                            # Ensure scores are floats
                            valid_emotions = {emo: float(score) for emo, score in face_data['emotions'].items() if isinstance(score, (int, float))}
                            if valid_emotions: # Only add if valid emotions found
                                 model_face_emotions[model_name].append(valid_emotions)
                    logging.info(f"{model_name} found {len(model_face_emotions[model_name])} faces with valid emotions.")
                else:
                     logging.info(f"{model_name} found no faces.")
            except Exception as e:
                logging.error(f"{model_name} failed: {e}\n{traceback.format_exc()}")


    # --- Run DeepFace models (Try multiple backends, handles BGR/RGB conversion) ---
    deepface_backends = ['retinaface', 'opencv', 'ssd', 'mtcnn'] # Add more if needed
    deepface_success = False
    for backend in deepface_backends:
         if deepface_success: break # Stop if one backend succeeds
         try:
              logging.info(f"Trying DeepFace with backend: {backend}")
              # DeepFace analyze returns list of dicts, one per face detected by the backend
              # It handles BGR/RGB conversion internally based on the model.
              df_results = DeepFace.analyze(
                   img_path=image_array_rgb, # Pass numpy array directly
                   actions=['emotion'],
                   enforce_detection=False, # Don't error if no face found
                   detector_backend=backend,
                   silent=True # Suppress progress bars
              )

              # Process results if detection occurred
              if isinstance(df_results, list) and df_results:
                   num_faces_df = 0
                   for face_data in df_results:
                        # DeepFace result structure: list of dicts, each has 'emotion', 'dominant_emotion', 'region'
                        if isinstance(face_data, dict) and 'emotion' in face_data and isinstance(face_data['emotion'], dict):
                             # DeepFace returns percentages (0-100), convert to 0-1
                             emotion_scores_normalized = {emo: float(score / 100.0) for emo, score in face_data['emotion'].items() if isinstance(score, (int, float))}
                             if emotion_scores_normalized: # Check if valid scores exist
                                  model_face_emotions['deepface'].append(emotion_scores_normalized)
                                  num_faces_df += 1
                   if num_faces_df > 0:
                        logging.info(f"DeepFace ({backend}) found {num_faces_df} faces with emotions.")
                        deepface_success = True # Mark success
                   else:
                        logging.info(f"DeepFace ({backend}) ran but found no faces with valid emotions.")

              else: # Handle case where DeepFace returns None or empty list
                   logging.info(f"DeepFace ({backend}) returned no results (likely no faces detected).")

         except Exception as e_df:
              logging.warning(f"DeepFace backend '{backend}' failed: {e_df}")
              # Don't log full traceback for expected backend failures unless debugging
              if st.session_state.get('debug_mode', False):
                   logging.debug(f"DeepFace backend '{backend}' traceback:\n{traceback.format_exc()}")

    # --- Combine Results ---
    if not model_face_emotions:
        logging.warning("No faces detected by any model."); return {'neutral': 1.0}

    # Average emotions across *all detected faces* from *all successful models*
    # This gives equal weight to each detected face, regardless of the model that found it
    all_detected_face_emotions = []
    for model_name, face_list in model_face_emotions.items():
        all_detected_face_emotions.extend(face_list)

    if not all_detected_face_emotions:
         logging.warning("No valid emotions detected across all faces and models."); return {'neutral': 1.0}

    # Calculate the average score for each emotion across all faces
    final_avg_emotions = defaultdict(float)
    num_total_faces = len(all_detected_face_emotions)

    for face_emotion_dict in all_detected_face_emotions:
        for emotion, score in face_emotion_dict.items():
             # Standardize emotion names to lower case
             final_avg_emotions[emotion.lower()] += score

    # Divide sum by the number of faces to get the average
    if num_total_faces > 0:
        final_avg_emotions = {emo: total_score / num_total_faces for emo, total_score in final_avg_emotions.items()}
    else: # Should not happen if all_detected_face_emotions is not empty, but safeguard
         return {'neutral': 1.0}


    # Final normalization to ensure the distribution sums to approx 1 (due to potential float precision)
    total_score_sum = sum(final_avg_emotions.values())
    if total_score_sum <= 1e-6: # Check for near-zero sum
         logging.warning("Sum of averaged emotion scores is near zero. Returning neutral.")
         return {'neutral': 1.0}

    normalized_final_emotions = {k: v / total_score_sum for k, v in final_avg_emotions.items()}

    # Log the final averaged and normalized emotions
    logging.info(f"Final combined emotions (averaged across {num_total_faces} faces): {normalized_final_emotions}")

    return normalized_final_emotions
# --- END detect_emotions_ensemble ---



def analyze_media(media_url, models, platform, session):
    """Downloads and analyzes media (image, video, audio). Returns (result_dict, media_type, downloaded_path)."""
    media_path, media_type = (None, None)
    media_result = None  # Initialize analysis result

    try:
        # --- 1. Download ---
        st.info(f"Downloading media from {media_url}...")
        media_path, media_type = download_media(media_url, session)

        if not media_path or not media_type:
            logging.warning("Media download failed or was skipped.")
            st.warning("Media download failed or no media found at URL.")
            return None, None, None

        logging.info(f"Media downloaded: Path='{media_path}', Type='{media_type}'")
        st.info(f"Downloaded {media_type} successfully. Analyzing...")

        # --- 2. Analyze based on type ---
        analysis_prog_bar = st.progress(0, text=f"Analyzing {media_type}...")

        # --- IMAGE ANALYSIS ---
        if media_type == 'image':
            try:
                media_result = analyze_image_advanced(media_path, models)
                analysis_prog_bar.progress(1.0, text="✅ Image analysis complete.")
            except Exception as img_err:
                st.error(f"Image analysis failed: {img_err}")
                logging.error(f"Image analysis error: {traceback.format_exc()}")
                media_result = {'pred_label': 'NEUTRAL', 'score': 0.5}

        # --- VIDEO ANALYSIS ---
        elif media_type == 'video':
            video_clip = None
            try:
                video_clip = VideoFileClip(media_path)
                duration = float(video_clip.duration) if video_clip.duration else 0.0
                fps = video_clip.fps if (video_clip.fps and video_clip.fps > 0) else 20
                logging.info(f"Video loaded: Duration={duration:.2f}s, FPS={fps}")

                # --- Duration limit ---
                max_dur = st.session_state.get('max_video_duration', 30)
                if duration > max_dur:
                    st.info(f"Video duration ({duration:.0f}s) > limit ({max_dur}s). Trimming.")
                    try:
                        subclip = VideoFileClip(media_path).subclip(0, max_dur)
                        video_clip.close()
                        video_clip = subclip
                        duration = max_dur
                        logging.info(f"Trimmed video to {max_dur}s.")
                    except Exception as subclip_err:
                        st.warning(f"Subclip failed: {subclip_err}")
                        logging.error(f"Subclip error: {traceback.format_exc()}")

                # --- Frame sampling ---
                analysis_prog_bar.progress(0.1, text="Analyzing video frames...")
                total_frames = max(1, int(duration * fps))
                target_frames = 5 if duration < 10 else (10 if duration < 30 else 15)

                # Low RAM adjustment
                try:
                    available_ram_gb = psutil.virtual_memory().available / (1024**3)
                    if available_ram_gb < 4:
                        target_frames = min(target_frames, 5)
                        logging.warning(f"Low RAM ({available_ram_gb:.1f}GB). Reducing target frames.")
                except Exception:
                    pass

                n_skip = max(1, total_frames // target_frames)
                frame_scores, all_emotions = [], defaultdict(float)
                processed = 0

                # Try iter_frames first
                iter_failed = False
                try:
                    if hasattr(video_clip, "iter_frames") and callable(video_clip.iter_frames):
                        for i, frame_array in enumerate(video_clip.iter_frames(fps=fps, dtype="uint8")):
                            if i % n_skip != 0:
                                continue
                            if processed >= target_frames:
                                break
                            processed += 1
                            analysis_prog_bar.progress(0.1 + 0.6 * (processed / target_frames),
                                                       text=f"Analyzing frame {processed}/{target_frames}")
                            try:
                                if isinstance(frame_array, np.ndarray):
                                    frame_pil = Image.fromarray(frame_array)
                                    frame_result = analyze_image_advanced(frame_pil, models)
                                    if frame_result and 'score' in frame_result:
                                        frame_scores.append(frame_result['score'])
                                        if 'emotions' in frame_result:
                                            for emo, val in frame_result['emotions'].items():
                                                if isinstance(val, (int, float)):
                                                    all_emotions[emo] += val
                                    frame_pil.close()
                            except Exception as fe:
                                logging.error(f"Frame error: {fe}")
                    else:
                        iter_failed = True
                except Exception as iter_err:
                    logging.error(f"iter_frames failed: {iter_err}")
                    iter_failed = True

                # --- Fallback via timestamps ---
                if iter_failed or processed < target_frames:
                    logging.info("Falling back to get_frame sampling...")
                    processed = 0
                    for j in np.linspace(0.1, duration - 0.1, target_frames):
                        try:
                            frame = video_clip.get_frame(j)
                            if isinstance(frame, np.ndarray):
                                frame_pil = Image.fromarray(frame)
                                frame_result = analyze_image_advanced(frame_pil, models)
                                if frame_result and 'score' in frame_result:
                                    frame_scores.append(frame_result['score'])
                                    if 'emotions' in frame_result:
                                        for emo, val in frame_result['emotions'].items():
                                            if isinstance(val, (int, float)):
                                                all_emotions[emo] += val
                                frame_pil.close()
                                processed += 1
                        except Exception as gf_err:
                            logging.error(f"get_frame error: {gf_err}")
                            continue

                avg_frame_sentiment = np.mean(frame_scores) if frame_scores else 0.5
                total_emo_score = sum(all_emotions.values())
                avg_emotions = {k: v / total_emo_score for k, v in all_emotions.items()} if total_emo_score else {'neutral': 1.0}
                dominant_emo = max(avg_emotions.items(), key=lambda x: x[1])[0] if avg_emotions else 'neutral'

                # --- Skip audio extraction safely ---
                logging.info("Skipping audio extraction for simplicity (neutral audio score 0.5)")
                audio_sentiment_score = 0.5

                # Combine
                final_score = (avg_frame_sentiment * 0.7) + (audio_sentiment_score * 0.3)
                media_result = {
                    'pred_label': score_to_label(final_score),
                    'score': final_score,
                    'dominant_emotion': dominant_emo,
                    'emotions': avg_emotions
                }
                analysis_prog_bar.progress(1.0, text="✅ Video analysis complete.")
            except Exception as ve:
                st.error(f"Video processing failed: {ve}")
                logging.error(f"Video analysis error: {traceback.format_exc()}")
                return None, media_type, media_path
            finally:
                if video_clip:
                    try:
                        video_clip.close()
                    except Exception:
                        pass
                gc.collect()

        # --- AUDIO ANALYSIS ---
        elif media_type == 'audio':
            try:
                if not os.path.exists(media_path) or os.path.getsize(media_path) < 1024:
                    st.error("Audio file invalid or missing.")
                    raise FileNotFoundError("Invalid audio file.")

                analysis_prog_bar.progress(0.2, text="Reading audio...")
                audio_input, sr = sf.read(media_path, dtype='float32')
                if audio_input.ndim > 1:
                    audio_input = np.mean(audio_input, axis=1)

                audio_input = normalize_audio(audio_input, sr)
                target_sr = 16000
                if sr != target_sr:
                    audio_input = librosa.resample(y=audio_input, orig_sr=sr, target_sr=target_sr)
                    sr = target_sr
                    logging.info(f"Resampled audio to {target_sr} Hz.")

                analysis_prog_bar.progress(0.5, text="Running audio models...")
                scores = []
                for name, analyzer in models['audio_analyzer'].items():
                    try:
                        if name == 'whisper':
                            transcript_res = analyzer(audio_input.copy(), chunk_length_s=30, batch_size=4)
                            text = transcript_res.get('text', '') if isinstance(transcript_res, dict) else ''
                            if text:
                                txt_res = analyze_text_ensemble(clean_text(text), models['text_analyzer'])
                                if txt_res and 'score' in txt_res:
                                    scores.append(txt_res['score'])
                        elif name == 'wav2vec':
                            emo_res = analyzer(audio_input.copy(), sampling_rate=sr, top_k=None)
                            if emo_res and isinstance(emo_res, list):
                                emo = max(emo_res, key=lambda x: x['score'])
                                emo_map = {'happy':0.9,'neutral':0.5,'angry':0.1,'sad':0.2,'fear':0.2,'disgust':0.1,'calm':0.6,'surprised':0.7}
                                scores.append(emo_map.get(emo['label'].lower(), 0.5))
                    except Exception as am:
                        logging.error(f"Audio model {name} failed: {am}")

                final_score = np.mean(scores) if scores else 0.5
                media_result = {'pred_label': score_to_label(final_score), 'score': final_score}
                analysis_prog_bar.progress(1.0, text="✅ Audio analysis complete.")
            except Exception as ae:
                st.error(f"Audio processing failed: {ae}")
                logging.error(f"Audio analysis error: {traceback.format_exc()}")
                return None, media_type, media_path

        else:
            st.warning(f"Unsupported media type for analysis: {media_type}")
            logging.warning(f"Unsupported media type: {media_type} at path {media_path}")
            return None, media_type, media_path

        # --- Return results ---
        return media_result, media_type, media_path

    except Exception as e:
        st.error(f"Unexpected error during media analysis pipeline: {e}")
        logging.error(f"Media analysis general failure: {traceback.format_exc()}")
        return None, media_type, media_path
    finally:
        gc.collect()
# --- END analyze_media ---

def fuse_sentiments_adaptive(text_result, media_result, comment_results, platform, media_type):
    """Combines individual analysis results into an overall score."""
    logging.info("Fusing sentiments...")
    components = []
    component_details = {} # Store individual results for logging/debug

    # Add text component if analysis was successful and valid
    if text_result and isinstance(text_result.get('score'), (int, float)):
        text_score = text_result['score']
        text_conf = abs(text_score - 0.5) * 2 # Confidence based on distance from neutral
        components.append({'name': 'text', 'score': text_score, 'confidence': text_conf})
        component_details['text'] = {'score': text_score, 'confidence': text_conf, 'label': text_result.get('pred_label')}
        logging.info(f"Text component added: Score={text_score:.2f}, Confidence={text_conf:.2f}")
    else:
        logging.info("Text component skipped (no result or invalid score).")


    # Add media component if analysis was successful and valid
    if media_result and isinstance(media_result.get('score'), (int, float)):
        media_score = media_result['score']
        media_conf = abs(media_score - 0.5) * 2
        # Apply boost based on media type (more weight to richer media)
        boost = 1.3 if media_type == 'video' else 1.1 if media_type == 'image' else 1.0
        media_conf *= boost
        components.append({'name': 'media', 'score': media_score, 'confidence': media_conf})
        component_details['media'] = {'score': media_score, 'confidence': media_conf, 'label': media_result.get('pred_label')}
        logging.info(f"Media component added: Score={media_score:.2f}, Confidence={media_conf:.2f} (Type: {media_type}, Boost: {boost:.1f})")
    else:
         logging.info(f"Media component skipped (Type: {media_type}, No result or invalid score).")


    # Add comments component if analysis was successful and valid
    # --- FIX: Ensure comment results are valid floats and not N/A placeholders ---
    valid_comments = []
    if isinstance(comment_results, list):
         valid_comments = [c for c in comment_results if isinstance(c, dict) and isinstance(c.get('score'), (int, float)) and c.get('pred_label') != 'N/A']

    if valid_comments:
        comment_scores = [c['score'] for c in valid_comments]
        avg_comment_score = np.mean(comment_scores)
        # Confidence increases with number of comments, max boost at ~5 comments, scaled by avg deviation
        num_valid_comments = len(valid_comments)
        comment_conf = abs(avg_comment_score - 0.5) * 2 * min(1.2, 1.0 + (num_valid_comments -1) * 0.05) # Gentle boost for count
        components.append({'name': 'comments', 'score': avg_comment_score, 'confidence': comment_conf})
        component_details['comments'] = {'score': avg_comment_score, 'confidence': comment_conf, 'label': score_to_label(avg_comment_score), 'count': num_valid_comments}
        logging.info(f"Comments component added: Avg Score={avg_comment_score:.2f}, Confidence={comment_conf:.2f} ({num_valid_comments} valid comments)")
    else:
         logging.info("Comments component skipped (no valid comments analyzed).")


    # Handle case where no components could be analyzed
    if not components:
        logging.warning("No valid components to fuse. Returning neutral default.");
        return 0.5, "NEUTRAL", 0.5, {} # Score, Label, Accuracy, Weights


    # --- Discordance Check (Potential Sarcasm/Conflict) ---
    text_label = component_details.get('text', {}).get('label', 'NEUTRAL')
    media_label = component_details.get('media', {}).get('label', 'NEUTRAL')
    comment_label = component_details.get('comments', {}).get('label', 'NEUTRAL')
    media_present = 'media' in component_details
    text_present = 'text' in component_details
    comments_present = 'comments' in component_details

    # Check for strong conflict (Positive vs Negative) between major components
    text_media_conflict = text_present and media_present and text_label != 'NEUTRAL' and media_label != 'NEUTRAL' and text_label != media_label
    text_comment_conflict = text_present and comments_present and text_label != 'NEUTRAL' and comment_label != 'NEUTRAL' and text_label != comment_label
    media_comment_conflict = media_present and comments_present and media_label != 'NEUTRAL' and comment_label != 'NEUTRAL' and media_label != comment_label

    if text_media_conflict or text_comment_conflict or media_comment_conflict:
        logging.info("Potential conflict detected between component sentiments. Adjusting weights slightly.")
        # Apply a modest penalty to confidence of conflicting components, or boost most confident?
        # Simple approach: Slightly reduce confidence of all components in conflict
        conflict_penalty = 0.8 # Reduce confidence to 80%
        for comp in components:
            # Apply penalty if this component is involved in a conflict
            is_conflicting = False
            if comp['name'] == 'text' and (text_media_conflict or text_comment_conflict): is_conflicting = True
            if comp['name'] == 'media' and (text_media_conflict or media_comment_conflict): is_conflicting = True
            if comp['name'] == 'comments' and (text_comment_conflict or media_comment_conflict): is_conflicting = True

            if is_conflicting:
                 original_conf = comp['confidence']
                 comp['confidence'] *= conflict_penalty
                 logging.debug(f"Reduced confidence for '{comp['name']}' due to conflict: {original_conf:.2f} -> {comp['confidence']:.2f}")


    # --- Calculate Final Weights ---
    total_confidence = sum(c['confidence'] for c in components)

    if total_confidence <= 1e-6: # Handle near-zero confidence sum
        logging.warning("Total confidence near zero. Using equal weights.")
        num_components = len(components)
        weights = {c['name']: 1.0 / num_components for c in components} if num_components > 0 else {}
    else:
        weights = {c['name']: c['confidence'] / total_confidence for c in components}

    # --- Platform-specific adjustments (Optional) ---
    # Example: Slightly boost comments on discussion platforms
    if platform in ['reddit', 'youtube'] and 'comments' in weights and comments_present:
        comment_weight = weights['comments']
        other_weight_sum = 1.0 - comment_weight
        # Increase comment weight slightly, but don't exceed a cap (e.g., 60%)
        new_comment_weight = min(comment_weight * 1.2, 0.6) # Boost by 20%, capped at 0.6

        if new_comment_weight > comment_weight and other_weight_sum > 0:
            logging.info(f"Boosting comment weight for platform '{platform}'.")
            weights['comments'] = new_comment_weight
            # Renormalize other weights to sum to (1 - new_comment_weight)
            scale_factor = (1.0 - new_comment_weight) / other_weight_sum
            for name in weights:
                if name != 'comments':
                    weights[name] *= scale_factor
            # Verify weights sum close to 1 after adjustment
            if abs(sum(weights.values()) - 1.0) > 1e-6:
                 logging.warning("Weight normalization after boost failed. Weights might not sum to 1.")


    logging.info(f"Final Adaptive Weights: { {k: f'{v:.2f}' for k, v in weights.items()} }") # Log formatted weights


    # --- Calculate Final Weighted Score ---
    final_score = sum(weights.get(c['name'], 0) * c['score'] for c in components)
    # Ensure score is within [0, 1] bounds
    final_score = max(0.0, min(1.0, final_score))

    final_label = score_to_label(final_score)

    # --- Estimate Accuracy ---
    # Base accuracy + bonus based on max confidence component (how sure is the most sure component?)
    max_confidence = max(c['confidence'] for c in components) if components else 0.0
    # Consider number of components as well (more components might increase reliability)
    num_components_factor = 1.0 + min(0.1, (len(components) - 1) * 0.05) # Small bonus for more components
    estimated_accuracy = min(0.95, (0.6 + (max_confidence * 0.35)) * num_components_factor) # Base 60%, scales up

    logging.info(f"Fused Result: {final_label} (Score: {final_score:.3f}), Est. Accuracy: {estimated_accuracy:.3f}")
    return final_score, final_label, estimated_accuracy, weights
# --- END fuse_sentiments_adaptive ---



# --- 8. VISUALIZATION & METRICS ---
def calculate_pseudo_metrics(history):
    """
    Calculates pseudo-evaluation metrics (Precision, Recall, F1) and
    Confusion Matrices by comparing component predictions against the
    final overall sentiment label (stored in DB), using analysis history.
    Returns:
        tuple: (metrics_dict, conf_matrices_dict) or (None, None) if insufficient data.
    """
    if not history or len(history) < 2: # Need at least 2 entries for comparison
        logging.warning("Insufficient history (<2) to calculate pseudo-metrics.")
        return None, None



    y_true_overall = [] # List to store 'pseudo-truth' labels (final fused sentiment)
    y_pred_text = []    # List to store text component predictions
    y_pred_media = []   # List to store media component predictions
    y_pred_comments = [] # List to store comment component predictions (derived)



    # Iterate through each analysis run in the history loaded from DB
    for entry in history:
        # Use the stored overall sentiment as the pseudo-truth
        overall_sentiment_truth = entry.get('overall_sentiment')

        # Skip entry if the overall sentiment is missing or invalid
        if not overall_sentiment_truth or overall_sentiment_truth not in METRIC_LABELS:
            logging.warning(f"Skipping history entry {entry.get('id', 'N/A')} with missing/invalid overall sentiment: {entry.get('url')}")
            continue



        y_true_overall.append(overall_sentiment_truth)



        # --- Extract component predictions from the loaded entry ---
        # Get text prediction (default to NEUTRAL if missing/invalid)
        text_label = entry.get('text_label', 'NEUTRAL') or 'NEUTRAL'
        text_label = text_label if text_label in METRIC_LABELS else 'NEUTRAL'

        # Get media prediction (default to NEUTRAL if missing/invalid)
        media_label = entry.get('media_label', 'NEUTRAL') or 'NEUTRAL'
        media_label = media_label if media_label in METRIC_LABELS else 'NEUTRAL'

        # Get comment prediction (default to NEUTRAL if missing/invalid)
        comment_label = entry.get('comment_label', 'NEUTRAL') or 'NEUTRAL'
        comment_label = comment_label if comment_label in METRIC_LABELS else 'NEUTRAL'


        y_pred_text.append(text_label)
        y_pred_media.append(media_label)
        y_pred_comments.append(comment_label)
    # --- End extraction loop ---





    # Check if we collected enough valid data points for comparison
    if not y_true_overall or len(y_true_overall) < 2:
        logging.warning("Not enough valid history entries for metric calculation after filtering labels.")
        return None, None



    # Calculate metrics using sklearn
    metrics = {} # To store Precision, Recall, F1 for each component
    conf_matrices = {} # To store Confusion Matrix for each component
    try:
        # --- Expanded Calculation Logic ---
        # Text vs Overall
        p_t, r_t, f1_t, _ = precision_recall_fscore_support(
            y_true_overall, y_pred_text, average='weighted', labels=METRIC_LABELS, zero_division='warn'
        )
        metrics['Text'] = {'precision': p_t, 'recall': r_t, 'f1': f1_t}
        conf_matrices['Text'] = confusion_matrix(y_true_overall, y_pred_text, labels=METRIC_LABELS)
        logging.info(f"Text Metrics (vs Overall): P={p_t:.2f}, R={r_t:.2f}, F1={f1_t:.2f} (n={len(y_true_overall)})")



        # Media vs Overall
        p_m, r_m, f1_m, _ = precision_recall_fscore_support(
            y_true_overall, y_pred_media, average='weighted', labels=METRIC_LABELS, zero_division='warn'
        )
        metrics['Media'] = {'precision': p_m, 'recall': r_m, 'f1': f1_m}
        conf_matrices['Media'] = confusion_matrix(y_true_overall, y_pred_media, labels=METRIC_LABELS)
        logging.info(f"Media Metrics (vs Overall): P={p_m:.2f}, R={r_m:.2f}, F1={f1_m:.2f} (n={len(y_true_overall)})")



        # Comments vs Overall
        p_c, r_c, f1_c, _ = precision_recall_fscore_support(
            y_true_overall, y_pred_comments, average='weighted', labels=METRIC_LABELS, zero_division='warn'
        )
        metrics['Comments'] = {'precision': p_c, 'recall': r_c, 'f1': f1_c}
        conf_matrices['Comments'] = confusion_matrix(y_true_overall, y_pred_comments, labels=METRIC_LABELS)
        logging.info(f"Comments Metrics (vs Overall): P={p_c:.2f}, R={r_c:.2f}, F1={f1_c:.2f} (n={len(y_true_overall)})")
        # --- End Expanded ---



        return metrics, conf_matrices



    except ValueError as ve:
        # Handle potential ValueError if labels in y_pred don't match METRIC_LABELS
        # or if y_true_overall contains labels not in METRIC_LABELS (shouldn't happen with checks)
        logging.error(f"Error calculating sklearn metrics (likely label mismatch or insufficient data per class): {ve}")
        return None, None
    except Exception as e:
        logging.error(f"Unexpected error during sklearn metric calculation: {e}\n{traceback.format_exc()}")
        return None, None # Return None if calculation fails
# --- END calculate_pseudo_metrics ---



def plot_visualizations(sentiment_data, accuracy, weights, timings, metrics, conf_matrix):
    """Generates Matplotlib plots including pseudo-evaluation metrics."""
    if not sentiment_data or not isinstance(sentiment_data, dict):
        logging.warning("Invalid or missing sentiment_data for plotting."); return None

    # Apply seaborn style if desired, or use matplotlib default
    try:
         plt.style.use('seaborn-v0_8-darkgrid') # Example style
    except OSError:
         plt.style.use('dark_background') # Fallback style
    except Exception as style_err:
         logging.warning(f"Could not apply seaborn style: {style_err}. Using default.")


    fig, axes = plt.subplots(4, 2, figsize=(16, 24)) # Adjust figsize if needed
    # Set background to transparent for better theme integration
    fig.patch.set_alpha(0.0)

    # Determine text color based on Streamlit theme
    text_color = 'white' if st.session_state.get('theme', 'light') == 'dark' else 'black'
    # Update matplotlib parameters for text colors
    plt.rcParams.update({
        'text.color': text_color,
        'axes.labelcolor': text_color,
        'xtick.color': text_color,
        'ytick.color': text_color,
        'axes.titlecolor': text_color,
        'axes.edgecolor': text_color, # Color axes lines
        'figure.facecolor': 'none', # Transparent figure background
        'axes.facecolor': 'none' # Transparent axes background
    })

    ax = axes.flatten() # Flatten axes array for easy indexing

    # --- Plot 1: Sentiment Scores ---
    try:
        labels = ['Text', 'Media', 'Comments', 'Overall']
        # Safely get scores, default to 0.5 (neutral) if component missing/invalid
        text_result = sentiment_data.get('text') or {}
        media_result = sentiment_data.get('media') or {}
        
        text_s = text_result.get('score', 0.5)
        media_s = media_result.get('score', 0.5)

        # Get comm_s_list from sentiment_data
        comm_list = sentiment_data.get('comments', [])
        comm_s_list = [c['score'] for c in comm_list if isinstance(c, dict) and isinstance(c.get('score'), (int, float)) and c.get('pred_label') != 'N/A'] if isinstance(comm_list, list) else []
        comm_s = np.mean(comm_s_list) if comm_s_list else 0.5

        overall_s = sentiment_data.get('overall_score', 0.5)

        # Ensure scores are valid numbers between 0 and 1
        scores = [text_s, media_s, comm_s, overall_s]
        scores = [max(0.0, min(1.0, s)) if isinstance(s, (int, float)) else 0.5 for s in scores]

        # Determine bar colors based on score
        colors = ['#2196F3', '#FFC107', '#9C27B0', '#4CAF50' if scores[3] > 0.6 else ('#F44336' if scores[3] < 0.4 else '#757575')]

        bars = ax[0].bar(labels, scores, color=colors, alpha=0.85, zorder=3) # Increase alpha, set zorder
        ax[0].set_ylim(0, 1)
        ax[0].set_title('Sentiment Scores (Current Run)', fontsize=14, fontweight='bold')
        ax[0].set_ylabel('Score (0=Neg, 0.5=Neu, 1=Pos)')
        # Add grid lines for reference
        ax[0].grid(axis='y', linestyle='--', alpha=0.5, zorder=0)
        ax[0].axhline(0.5, color='grey', linestyle=':', linewidth=1.5, zorder=1) # Neutral line

        # Add score labels to bars, handling missing components
        component_present = {
            'Text': sentiment_data.get('text') is not None,
            'Media': sentiment_data.get('media') is not None,
            'Comments': bool(comm_s_list), # True if there were valid analyzed comments
            'Overall': sentiment_data.get('overall_score') is not None
        }

        for i, bar in enumerate(bars):
            label_name = labels[i]
            height = bar.get_height()
            if component_present[label_name]:
                 # Position text inside bar if high enough, else slightly above
                 y_pos_text = height - 0.05 if height > 0.15 else height + 0.02
                 va_text = 'top' if height > 0.15 else 'bottom'
                 ax[0].text(bar.get_x() + bar.get_width() / 2, y_pos_text, f'{height:.2f}',
                            ha='center', va=va_text, color='white' if height > 0.15 else text_color,
                            fontweight='bold', fontsize=10, zorder=4)
            elif label_name != 'Overall': # Don't label 'N/A' for Overall if missing
                 ax[0].text(bar.get_x() + bar.get_width() / 2, 0.25, 'N/A',
                            ha='center', va='center', color='grey', style='italic', fontsize=10, zorder=4)

    except Exception as e:
         logging.error(f"Error plotting Scores: {e}\n{traceback.format_exc()}")
         ax[0].text(0.5, 0.5, 'Plot Error', ha='center', va='center', color='red', transform=ax[0].transAxes)
         ax[0].set_title('Sentiment Scores (Current Run)', fontsize=14, fontweight='bold')


    # --- Plot 2: Media Emotion Distribution ---
    try:
        emos = (sentiment_data.get('media') or {}).get('emotions', None)
        if emos and isinstance(emos, dict) and sum(v for v in emos.values() if isinstance(v, (int, float))) > 0.01:
            # Filter emotions with significant scores and ensure scores are valid floats
            emos_filtered = {k: float(v) for k, v in emos.items() if isinstance(v, (int, float)) and v > 0.02}

            if emos_filtered:
                labs = list(emos_filtered.keys())
                vals = list(emos_filtered.values())
                # Normalize values if they don't sum to 1 (can happen due to filtering/precision)
                total_v = sum(vals)
                if total_v > 0 and abs(total_v - 1.0) > 1e-5:
                     vals = [v / total_v for v in vals]

                # Use a specific color palette
                emo_colors = sns.color_palette("Spectral", len(labs))
                # Explode the largest slice slightly for emphasis
                explode = [0.05 if v == max(vals) else 0 for v in vals]

                ax[1].pie(vals, labels=[l.capitalize() for l in labs], autopct='%1.1f%%',
                          startangle=140, colors=emo_colors, pctdistance=0.85, explode=explode,
                          wedgeprops={'edgecolor': 'grey', 'linewidth': 0.5}) # Add edge color

                # Add a center circle for donut chart effect (optional)
                # circ = plt.Circle((0, 0), 0.70, fc='none', edgecolor=text_color, linewidth=0.5)
                # ax[1].add_artist(circ)
            else:
                 # Case where emotions exist but all below threshold
                 dom_e = max(emos.items(), key=lambda x:x[1])[0] if emos else 'N/A'
                 ax[1].text(0.5, 0.5, f'Low Intensity Emotions\n(Dominant: {dom_e.capitalize()})',
                            ha='center', va='center', transform=ax[1].transAxes, fontsize=10, linespacing=1.5)
        else:
            # Case where no emotions were detected or media wasn't analyzed
            ax[1].text(0.5, 0.5, 'No Emotions Detected /\nNo Media Analyzed',
                       ha='center', va='center', transform=ax[1].transAxes, fontsize=10, linespacing=1.5)

        ax[1].set_title('Media Emotion Distribution', fontsize=14, fontweight='bold')

    except Exception as e:
         logging.error(f"Error plotting Emotions: {e}\n{traceback.format_exc()}")
         ax[1].text(0.5, 0.5, 'Plot Error', ha='center', va='center', color='red', transform=ax[1].transAxes)
         ax[1].set_title('Media Emotion Distribution', fontsize=14, fontweight='bold')


    # --- Plot 3: Comment Sentiment Distribution ---
    try:
        # Get comm_s_list from sentiment_data
        comm_list = sentiment_data.get('comments', [])
        comm_s_list = [c['score'] for c in comm_list if isinstance(c, dict) and isinstance(c.get('score'), (int, float)) and c.get('pred_label') != 'N/A'] if isinstance(comm_list, list) else []
        if comm_s_list: # Use the list of valid scores from Plot 1
            sns.histplot(comm_s_list, bins=np.linspace(0, 1, 11), kde=True, color='#9C27B0', ax=ax[2], alpha=0.7)
            ax[2].set_title(f'Comment Sentiment Dist. (n={len(comm_s_list)})', fontsize=14, fontweight='bold')
            ax[2].set_xlabel('Sentiment Score (0=Neg, 1=Pos)')
            ax[2].set_ylabel('Number of Comments')

            ax[2].axvline(0.5, color='grey', linestyle=':', linewidth=1.5)
            ax[2].set_xlim(0, 1)
            # Add mean score line
            mean_score = np.mean(comm_s_list)
            ax[2].axvline(mean_score, color='yellow', linestyle='--', linewidth=1.5, label=f'Mean: {mean_score:.2f}')
            ax[2].legend(fontsize=9)
        else:
            ax[2].text(0.5, 0.5, 'No Valid Comments Analyzed', ha='center', va='center', transform=ax[2].transAxes, fontsize=10)
            ax[2].set_title('Comment Sentiment Dist.', fontsize=14, fontweight='bold')
        ax[2].grid(axis='y', linestyle='--', alpha=0.5)

    except Exception as e:
         logging.error(f"Error plotting Comment Dist: {e}\n{traceback.format_exc()}")
         ax[2].text(0.5, 0.5, 'Plot Error', ha='center', va='center', color='red', transform=ax[2].transAxes)
         ax[2].set_title('Comment Sentiment Dist.', fontsize=14, fontweight='bold')


    # --- Plot 4: Contribution Weights ---
    try:
        w_dict = weights if isinstance(weights, dict) else {}
        # Filter weights > 0 and map names
        weight_map = {'text': 'Text', 'media': 'Media', 'comments': 'Comments'}
        weights_to_plot = {weight_map.get(k, k.capitalize()): v for k, v in w_dict.items() if isinstance(v, (int, float)) and v > 0.001}

        if weights_to_plot:
            w_labs = list(weights_to_plot.keys())
            w_vals = list(weights_to_plot.values())
            # Ensure normalization if needed (shouldn't be necessary if fuse function is correct)
            total_w = sum(w_vals)
            if abs(total_w - 1.0) > 1e-5:
                 logging.warning(f"Weights do not sum to 1 ({total_w}). Normalizing for plot.")
                 w_vals = [v / total_w for v in w_vals]

            # Define colors for components
            comp_colors = {'Text': '#2196F3', 'Media': '#FFC107', 'Comments': '#9C27B0'}
            pie_colors = [comp_colors.get(l, 'grey') for l in w_labs]
            explode = [0.05] * len(w_labs) # Small explode for all slices

            ax[3].pie(w_vals, labels=w_labs, autopct='%1.1f%%', startangle=90, colors=pie_colors,
                      pctdistance=0.85, explode=explode, wedgeprops={'edgecolor': 'grey', 'linewidth': 0.5})
            ax[3].set_title('Component Contribution Weight', fontsize=14, fontweight='bold')
        else:
            ax[3].text(0.5, 0.5, 'N/A (Only one component?)', ha='center', va='center', transform=ax[3].transAxes, fontsize=10)
            ax[3].set_title('Component Contribution Weight', fontsize=14, fontweight='bold')

    except Exception as e:
         logging.error(f"Error plotting Weights: {e}\n{traceback.format_exc()}")
         ax[3].text(0.5, 0.5, 'Plot Error', ha='center', va='center', color='red', transform=ax[3].transAxes)
         ax[3].set_title('Component Contribution Weight', fontsize=14, fontweight='bold')


    # --- Plot 5: Accuracy Trend ---
    try:
        hist = st.session_state.get('history_display_cache', []) # Use cached history from DB
        # Extract valid accuracy estimates from history
        acc_hist = [e.get('accuracy', 0.5) for e in hist if isinstance(e.get('accuracy'), (int, float))]
        # Reverse to show oldest to newest for plotting trend
        accs_trend = list(reversed(acc_hist))

        # Limit trend to last N points for readability (e.g., 20)
        max_trend_points = 20
        if len(accs_trend) > max_trend_points:
            accs_trend = accs_trend[-max_trend_points:]

        if len(accs_trend) > 1: # Need at least 2 points for a trend line
            x_vals = range(1, len(accs_trend) + 1)
            ax[4].plot(x_vals, accs_trend, marker='o', linestyle='-', color='darkorange', linewidth=2, markersize=5, zorder=3)
            ax[4].fill_between(x_vals, accs_trend, alpha=0.15, color='orange', zorder=2)
            ax[4].set_ylim(0, 1)
            ax[4].set_title(f'Est. Accuracy Trend (Last {len(accs_trend)} Runs)', fontsize=14, fontweight='bold')
            ax[4].set_xlabel('Analysis Run Index (Oldest -> Newest)')
            ax[4].set_ylabel('Estimated Accuracy')
            ax[4].grid(True, linestyle='--', alpha=0.5, zorder=0)

            # Simplify x-axis ticks for many points
            if len(x_vals) > 10:
                tick_indices = np.linspace(0, len(x_vals) - 1, num=5, dtype=int)
                ax[4].set_xticks(np.array(x_vals)[tick_indices])
            else:
                 ax[4].set_xticks(x_vals)
        elif len(accs_trend) == 1: # Show single point if only one history item
             ax[4].plot([1], accs_trend, marker='o', color='darkorange', markersize=6)
             ax[4].set_ylim(0, 1)
             ax[4].set_title('Est. Accuracy (Last Run)', fontsize=14, fontweight='bold')
             ax[4].set_xticks([1])
             ax[4].set_ylabel('Estimated Accuracy')
             ax[4].grid(True, linestyle='--', alpha=0.5)
        else:
            ax[4].text(0.5, 0.5, 'Need >1 History Item\nfor Trend', ha='center', va='center', transform=ax[4].transAxes, fontsize=10)
            ax[4].set_title('Est. Accuracy Trend', fontsize=14, fontweight='bold')

    except Exception as e:
         logging.error(f"Error plotting Accuracy Trend: {e}\n{traceback.format_exc()}")
         ax[4].text(0.5, 0.5, 'Plot Error', ha='center', va='center', color='red', transform=ax[4].transAxes)
         ax[4].set_title('Est. Accuracy Trend', fontsize=14, fontweight='bold')


    # --- Plot 6: Performance Breakdown ---
    try:
        t_dict = timings if isinstance(timings, dict) else {}
        # Define expected order and filter for valid timings > 0.01s
        t_order = ['Fetch', 'Text', 'Comments', 'Media', 'Fusion', 'Total']
        perf_data = {label: t_dict.get(label, 0.0) for label in t_order if isinstance(t_dict.get(label), (int, float)) and t_dict.get(label, 0.0) > 0.01}

        if perf_data:
            t_labs = list(perf_data.keys())
            t_vals = list(perf_data.values())
            perf_colors = sns.color_palette("viridis_r", len(t_labs)) # Use reversed viridis

            bars_perf = ax[5].bar(t_labs, t_vals, color=perf_colors, alpha=0.85, zorder=3)
            ax[5].set_title('Performance Breakdown (Current Run)', fontsize=14, fontweight='bold')
            ax[5].set_ylabel('Time (seconds)')
            ax[5].grid(axis='y', linestyle='--', alpha=0.5, zorder=0)

            # Add time labels above bars
            for bar in bars_perf:
                height = bar.get_height()
                ax[5].text(bar.get_x() + bar.get_width() / 2., height, f'{height:.1f}s',
                           ha='center', va='bottom', fontsize=9, color=text_color, zorder=4, fontweight='medium')
        else:
            ax[5].text(0.5, 0.5, 'N/A', ha='center', va='center', transform=ax[5].transAxes, fontsize=10)
            ax[5].set_title('Performance Breakdown (Current Run)', fontsize=14, fontweight='bold')

    except Exception as e:
         logging.error(f"Error plotting Performance: {e}\n{traceback.format_exc()}")
         ax[5].text(0.5, 0.5, 'Plot Error', ha='center', va='center', color='red', transform=ax[5].transAxes)
         ax[5].set_title('Performance Breakdown (Current Run)', fontsize=14, fontweight='bold')


    # --- Plot 7: Pseudo-Metrics (History) ---
    try:
        if metrics and isinstance(metrics, dict) and metrics.get('Text'): # Check if metrics were calculated
            metric_types = ['precision', 'recall', 'f1']
            components_with_metrics = ['Text', 'Media', 'Comments'] # Fixed order
            # Prepare data, handling missing components gracefully
            data_for_plot = {m_type: [metrics.get(comp, {}).get(m_type, 0.0) for comp in components_with_metrics] for m_type in metric_types}

            df_metrics = pd.DataFrame(data_for_plot, index=components_with_metrics)

            df_metrics.plot(kind='bar', ax=ax[6], colormap='Spectral', alpha=0.85, rot=0, zorder=3)
            ax[6].set_title('Internal Consistency Metrics (All History)', fontsize=14, fontweight='bold')
            ax[6].set_ylabel('Score (Weighted Avg)')
            ax[6].set_xlabel('Component Prediction (vs Overall Sentiment)')
            ax[6].set_ylim(0, 1)
            ax[6].legend(title='Metric', fontsize=9)
            ax[6].grid(axis='y', linestyle='--', alpha=0.5, zorder=0)

            # Add bar labels
            for container in ax[6].containers:
                 if container: # Check if container is valid
                      # Use a try-except block for bar_label as it might fail in some matplotlib versions
                      try:
                           ax[6].bar_label(container, fmt='%.2f', label_type='edge', fontsize=8, color=text_color, padding=2, zorder=4, fontweight='medium')
                      except AttributeError:
                           logging.warning("ax.bar_label not available or failed. Skipping metric bar labels.")
                      except Exception as label_err:
                            logging.warning(f"Error adding metric bar labels: {label_err}")
        else:
            ax[6].text(0.5, 0.5, 'Need >1 History Item\nfor Metrics', ha='center', va='center', transform=ax[6].transAxes, fontsize=10)
            ax[6].set_title('Internal Consistency Metrics (All History)', fontsize=14, fontweight='bold')

    except ImportError:
        logging.error("Pandas or Sklearn needed for metrics plot.")
        ax[6].text(0.5, 0.5, 'Error: Install Pandas & Sklearn', ha='center', va='center', color='red', transform=ax[6].transAxes)
        ax[6].set_title('Internal Consistency Metrics (All History)', fontsize=14, fontweight='bold')
    except Exception as e:
        logging.error(f"Error plotting Metrics: {e}\n{traceback.format_exc()}")
        ax[6].text(0.5, 0.5, 'Plot Error', ha='center', va='center', color='red', transform=ax[6].transAxes)
        ax[6].set_title('Internal Consistency Metrics (All History)', fontsize=14, fontweight='bold')


    # --- Plot 8: Confusion Matrix (History - Text vs Overall) ---
    try:
        cm_data = None
        if conf_matrix and isinstance(conf_matrix, dict):
             cm_data = conf_matrix.get('Text') # Focus on Text vs Overall consistency

        if cm_data is not None and isinstance(cm_data, np.ndarray) and cm_data.sum() > 0: # Check if CM has data
            try:
                disp = ConfusionMatrixDisplay(confusion_matrix=cm_data, display_labels=METRIC_LABELS)
                # Plot without colorbar for cleaner look in subplot
                disp.plot(ax=ax[7], cmap=plt.cm.Blues, colorbar=False, text_kw={'color': 'auto'}) # Use auto text color initially

                ax[7].set_title('Confusion Matrix (Text vs Overall, History)', fontsize=14, fontweight='bold')
                ax[7].grid(False) # Turn off grid for CM

                # Adjust text color for better contrast (needed if 'auto' doesn't work well)
                text_color_cm_threshold = cm_data.max() / 2.
                text_color_light = 'black' if st.session_state.theme == 'light' else 'white' # Light background text
                text_color_dark = 'white' if st.session_state.theme == 'light' else 'black' # Dark background text

                if hasattr(disp, 'text_') and disp.text_ is not None:
                     for i in range(len(METRIC_LABELS)):
                          for j in range(len(METRIC_LABELS)):
                               if disp.text_[i, j] is not None:
                                    cell_value = cm_data[i, j]
                                    color_to_use = text_color_dark if cell_value > text_color_cm_threshold else text_color_light
                                    disp.text_[i, j].set_color(color_to_use)
                                    disp.text_[i, j].set_fontweight('medium') # Make numbers clearer

            except Exception as disp_err:
                 logging.error(f"Error displaying Confusion Matrix: {disp_err}\n{traceback.format_exc()}")
                 ax[7].text(0.5, 0.5, 'CM Display Error', ha='center', va='center', color='red', transform=ax[7].transAxes)
                 ax[7].set_title('Confusion Matrix (Text vs Overall, History)', fontsize=14, fontweight='bold')

        else:
            ax[7].text(0.5, 0.5, 'Need >1 History Item\nfor Confusion Matrix', ha='center', va='center', transform=ax[7].transAxes, fontsize=10)
            ax[7].set_title('Confusion Matrix (Text vs Overall, History)', fontsize=14, fontweight='bold')

    except Exception as e:
         logging.error(f"Error plotting Confusion Matrix: {e}\n{traceback.format_exc()}")
         ax[7].text(0.5, 0.5, 'Plot Error', ha='center', va='center', color='red', transform=ax[7].transAxes)
         ax[7].set_title('Confusion Matrix (Text vs Overall, History)', fontsize=14, fontweight='bold')



    # --- Final Touches ---
    # Hide any unused axes if subplot layout changes
    # for i in range(len(ax)):
    #     if not ax[i].has_data(): # Check if axis has plotted data
    #          ax[i].set_visible(False)

    plt.tight_layout(pad=2.5, h_pad=3.5, w_pad=2.5) # Adjust padding
    return fig
# --- END plot_visualizations ---



# --- 9. MAIN APP ---
def main():
    """Main Streamlit application function."""
    # Run DB init and preload only once at the start
    if 'db_initialized' not in st.session_state:
        table_was_created = init_db()
        # Preload data only if the table was newly created to avoid duplicates
        if table_was_created:
            preload_data_from_csv(SAMPLE_CSV_PATH)
        st.session_state.db_initialized = True # Mark DB as initialized

    # Clean temp dir on each run (safer if previous run crashed)
    clean_temp_dir()

    # --- Theme Setup ---
    if 'theme' not in st.session_state:
        st.session_state.theme = 'light' # Default theme
    st.sidebar.header("⚙️ Settings & Evaluation")
    theme_options = ["Light", "Dark"]
    current_theme_index = 0 if st.session_state.theme == 'light' else 1
    theme_selection = st.sidebar.selectbox("Theme", theme_options, index=current_theme_index)
    selected_theme_lower = theme_selection.lower()
    if selected_theme_lower != st.session_state.theme:
        st.session_state.theme = selected_theme_lower
        st.rerun() # Rerun immediately to apply theme changes

    # --- Dynamic CSS Injection ---
    css = """
<style>
    /* Base theme colors */
    :root {{
        --bg-start: {bg_start};
        --bg-end: {bg_end};
        --text-color: {text_color};
        --input-text-color: {input_text_color};
        --input-bg: {input_bg};
        --metric-bg: {metric_bg};
        --tab-bg: {tab_bg};
        --expander-bg: {expander_bg};
        --border-color: {border_color};
        --scrollbar-thumb: {scrollbar_thumb};
        --primary-color: #4CAF50; /* Green accent */
        --primary-hover: #45a049;
        --secondary-color: #6c757d; /* Grey */
    }}

    /* Apply background gradient and base text color */
    [data-testid="stAppViewContainer"] > .main {{
        background: linear-gradient(135deg, var(--bg-start), var(--bg-end));
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; /* Cleaner font */
        color: var(--text-color);
    }}

    /* Buttons */
    .stButton > button {{
        background-color: var(--primary-color);
        color: white;
        border-radius: 8px;
        padding: 10px 24px; /* Slightly larger padding */
        font-weight: 600; /* Medium weight */
        border: none;
        transition: background-color 0.2s ease-in-out, transform 0.1s ease;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }}
    .stButton > button:hover {{
        background-color: var(--primary-hover);
        transform: translateY(-1px); /* Subtle lift */
        box-shadow: 0 4px 8px rgba(0,0,0,0.15);
    }}
     .stButton > button:active {{
         transform: translateY(0px);
         box-shadow: 0 1px 2px rgba(0,0,0,0.1);
     }}


    /* Text Input */
    .stTextInput > div > div > input {{
        color: var(--input-text-color) !important;
        background-color: var(--input-bg);
        border: 1px solid var(--primary-color);
        border-radius: 8px;
        padding: 12px; /* Increased padding */
        font-size: 1rem;
    }}
     .stTextInput > div > div > input:focus {{
         box-shadow: 0 0 0 2px rgba(76, 175, 80, 0.3); /* Focus ring */
         border-color: var(--primary-hover);
     }}

    /* Spinner */
    .stSpinner > div {{
        border-top-color: var(--primary-color); /* Color the spinner */
    }}

    /* Metrics */
    .stMetric {{
        background-color: var(--metric-bg);
        color: var(--text-color);
        border-radius: 10px; /* Slightly more rounded */
        padding: 18px; /* Increased padding */
        box-shadow: 0 4px 12px rgba(0,0,0,0.08); /* Softer shadow */
        margin-bottom: 12px;
        border-left: 5px solid var(--primary-color); /* Accent border */
    }}
    .stMetric > label {{ /* Style metric label */
        font-weight: 500;
        color: var(--secondary-color);
    }}

    /* Progress Bar */
    .stProgress > div > div > div {{
        background-color: var(--primary-color);
        border-radius: 4px;
    }}
    .stProgress > div {{ /* Container */
         background-color: var(--input-bg);
         border-radius: 4px;
    }}


    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {{
        background-color: transparent; /* Remove default background */
        border-bottom: 1px solid var(--border-color); /* Subtle line */
        gap: 10px; /* Space between tabs */
    }}
    .stTabs [data-baseweb="tab-list"] button {{
        background-color: transparent; /* Use transparent background */
        color: var(--secondary-color); /* Default inactive color */
        border-radius: 8px 8px 0 0;
        padding: 12px 18px; /* Adjust padding */
        font-weight: 600;
        border: none; /* Remove default borders */
        border-bottom: 3px solid transparent; /* Placeholder for active state */
        transition: color 0.2s ease, border-color 0.2s ease;
    }}
    .stTabs [data-baseweb="tab-list"] button:hover {{
         background-color: var(--input-bg); /* Subtle hover background */
         color: var(--text-color);
    }}
    .stTabs [data-baseweb="tab-list"] button[aria-selected="true"] {{
        color: var(--primary-color); /* Active color */
        border-bottom: 3px solid var(--primary-color); /* Active underline */
        background-color: transparent; /* Keep background transparent */
    }}

    /* Expander */
    .stExpander {{
        background-color: var(--expander-bg);
        border: 1px solid var(--border-color);
        border-radius: 8px;
        box-shadow: 0 2px 6px rgba(0,0,0,0.06);
        margin-bottom: 15px; /* More spacing */
    }}
    .stExpander > div:first-child {{ /* Header */
        font-weight: 600; /* Medium weight */
        padding: 10px 15px; /* Adjust padding */
        border-radius: 8px 8px 0 0; /* Match container */
    }}
     .stExpander > div:first-child:hover {{
          background-color: var(--input-bg); /* Subtle hover */
     }}
     .stExpander > div:last-child {{ /* Content */
          padding: 15px;
     }}


    /* Sidebar Styles */
    [data-testid="stSidebar"] {{
         background-color: var(--metric-bg); /* Match metric background */
         border-right: 1px solid var(--border-color);
    }}
    .sidebar .stSlider > label {{
        color: var(--primary-color);
        font-weight: 600;
    }}
    /* Style slider thumb */
    .sidebar .stSlider [data-baseweb="slider"] > div:nth-child(3) {{
         background-color: var(--primary-color) !important;
    }}


    /* Scrollbar */
    ::-webkit-scrollbar {{
        width: 10px; /* Slightly wider */
    }}
    ::-webkit-scrollbar-track {{
        background: var(--input-bg);
        border-radius: 5px;
    }}
    ::-webkit-scrollbar-thumb {{
        background: var(--scrollbar-thumb);
        border-radius: 5px;
        border: 2px solid var(--input-bg); /* Creates padding effect */
    }}
    ::-webkit-scrollbar-thumb:hover {{
        background: var(--secondary-color);
    }}
</style>
"""

    theme_params_dict = {
        'light': {
            'bg_start': '#e9eef6', 'bg_end': '#ffffff', 'text_color': '#212529',
            'metric_bg': '#ffffff', 'tab_bg': '#f8f9fa', 'expander_bg': '#ffffff',
            'border_color': '#dee2e6', 'input_text_color': '#343a40', 'input_bg': '#ffffff',
            'scrollbar_thumb': '#adb5bd'
        },
        'dark': {
            'bg_start': '#161b22', 'bg_end': '#0d1117', 'text_color': '#c9d1d9',
            'metric_bg': '#1c2128', 'tab_bg': '#21262d', 'expander_bg': '#1c2128',
            'border_color': '#30363d', 'input_text_color': '#c9d1d9', 'input_bg': '#0d1117',
            'scrollbar_thumb': '#484f58'
        }
    }
    st.markdown(css.format(**theme_params_dict[st.session_state.theme]), unsafe_allow_html=True)


    # --- UI Layout ---
    st.title("🌐 Social Media Sentiment Analyzer")
    st.markdown("Analyze sentiments across text, images, videos, and comments from various platforms. Enter a URL below.")

    # Sidebar controls bound to session state
    st.session_state.max_comments = st.sidebar.slider(
        "Max Comments to Analyze", min_value=1, max_value=20, # Increased max
        value=st.session_state.get('max_comments', 5), step=1,
        help="Maximum number of comments to fetch and analyze per post."
        )
    st.session_state.max_video_duration = st.sidebar.slider(
        "Max Video Duration (seconds)", min_value=5, max_value=120, # Increased range
        value=st.session_state.get('max_video_duration', 30), step=5,
        help="Maximum duration of video to analyze. Longer videos will be clipped."
        )
    st.session_state.debug_mode = st.sidebar.checkbox(
        "Enable Debug Mode", value=st.session_state.get('debug_mode', False),
        help="Show detailed JSON outputs and extra logs in the results tab."
        )


    # --- File Uploader ---
    st.sidebar.subheader("Upload Evaluation Data (Optional)")
    uploaded_file = st.sidebar.file_uploader(
        "Upload Ground Truth CSV", type="csv",
        help="Upload a CSV with 'url' and 'true_sentiment' (POSITIVE/NEGATIVE/NEUTRAL) columns to compare analysis results."
        )
    if uploaded_file is not None:
        try:
            # Read and validate uploaded data
            df_upload = pd.read_csv(uploaded_file)
            # Check for required columns, case-insensitive
            df_upload.columns = df_upload.columns.str.lower()
            if 'url' in df_upload.columns and 'true_sentiment' in df_upload.columns:
                # Validate labels
                valid_labels = set(METRIC_LABELS)
                # Convert to uppercase and handle potential NaN/None before check
                df_upload['true_sentiment'] = df_upload['true_sentiment'].fillna('N/A').astype(str).str.upper()
                invalid_rows = df_upload[~df_upload['true_sentiment'].isin(valid_labels)]
                valid_rows = df_upload[df_upload['true_sentiment'].isin(valid_labels)]

                if not invalid_rows.empty:
                    st.sidebar.warning(f"Uploaded data contains {len(invalid_rows)} rows with invalid labels (must be POSITIVE, NEGATIVE, or NEUTRAL). These rows were ignored.")

                if not valid_rows.empty:
                    st.session_state['uploaded_data'] = valid_rows # Store only valid rows
                    st.sidebar.success(f"Loaded {len(valid_rows)} evaluation entries.")
                    # Optionally, trigger re-analysis or update DB here if desired
                else:
                    st.sidebar.error("No valid entries found after label validation.")
                    if 'uploaded_data' in st.session_state: del st.session_state['uploaded_data'] # Clear previous

            else:
                st.sidebar.error("CSV must contain 'url' and 'true_sentiment' columns.")
                if 'uploaded_data' in st.session_state:
                    del st.session_state['uploaded_data'] # Clear invalid data
        except Exception as e:
            st.sidebar.error(f"Error reading CSV: {e}")
            logging.error(f"Error reading uploaded CSV: {e}\n{traceback.format_exc()}")
            if 'uploaded_data' in st.session_state:
                del st.session_state['uploaded_data']
    # --- End Evaluation Data ---





    # --- Model Loading (once per session via cache_resource) ---
    try:
        models = initialize_models()
        if not models: # Check if model loading failed inside the function
             st.error("Model initialization failed. Please check logs.")
             st.stop()
    except Exception as e:
        # This catches errors even if st.stop() was called inside initialize_models
        st.error(f"Critical Error during Model Initialization: {e}")
        logging.critical(f"Model initialization failed critically in main: {traceback.format_exc()}")
        st.stop() # Ensure execution stops


    # --- Tab Interface ---
    tab_titles = ["📥 Analysis Input", "📊 Analysis Results", "📜 History"]
    try:
         tab1, tab2, tab3 = st.tabs(tab_titles)
    except Exception as tab_err:
         st.error(f"Failed to create tabs: {tab_err}")
         logging.error(f"Streamlit tab creation failed: {tab_err}")
         st.stop()


    # --- Tab 1: Analysis Input ---
    with tab1:
        st.subheader("Enter Social Media Post URL")
        # Use a form to group input and button
        with st.form(key='analysis_form'):
            url = st.text_input("URL", placeholder="e.g., https://www.youtube.com/watch?v=...", key="url_input", label_visibility="collapsed")
            submitted = st.form_submit_button("Analyze Now ✨")

            if submitted:
                if not url or not validators.url(url):
                    st.error("❌ Please enter a valid URL.")
                else:
                    # Clear previous results before starting new analysis
                    st.session_state.analysis_results = None
                    st.session_state.analysis_error = None # Clear previous errors
                    # Trigger analysis directly (no need for separate state variable)
                    # Analysis logic moved outside the form block to run after submission

        # Analysis logic runs here if 'submitted' is true and URL is valid
        if submitted and url and validators.url(url):
             with st.spinner("⏳ Performing analysis... This may take a moment."):
                total_start_time = time.perf_counter()
                timings = {}
                analysis_entry = {} # Dict to store all results of this run
                analysis_successful = False # Flag to track success
                try:
                    session = create_secure_session() # Get requests session

                    # Rate Limit Warning (moved inside analysis block)
                    history_len_db = 0
                    try:
                        conn_hist = sqlite3.connect(DB_NAME, timeout=5)
                        cursor_hist = conn_hist.cursor()
                        cursor_hist.execute("SELECT COUNT(*) FROM analysis_history WHERE data_source = 'live_analysis'")
                        history_len_db = cursor_hist.fetchone()[0]
                        conn_hist.close()
                    except sqlite3.Error as db_err:
                        logging.error(f"Could not get history count from DB: {db_err}")

                    if history_len_db > 10 and (history_len_db % 10 == 0): # Warning every 10 runs after 10
                        st.warning(f"High API usage noted ({history_len_db} live runs in history). Please monitor your API quotas.")


                    # --- Core Analysis Pipeline ---
                    st.info("🔎 Fetching post metadata...")
                    fetch_start = time.perf_counter()
                    api_data = fetch_social_media_data(url, session)
                    timings['Fetch'] = time.perf_counter() - fetch_start

                    # Check if fetch_social_media_data returned None (critical error like PRAW/YT API fail)
                    if api_data is None:
                         # Error already shown by fetch function
                         raise ValueError("Failed to fetch essential post data. Cannot proceed.")


                    # Extract fetched data (even if API failed for non-Reddit/YT, we get defaults)
                    platform = api_data.get('platform', 'unknown')
                    title = api_data.get('title', '')
                    comments = api_data.get('comments', []) # List of dicts {'text': ...}
                    media_url_for_download = api_data.get('media_url', None) # URL to pass to downloader

                    st.info("📝 Analyzing text content...")
                    text_start = time.perf_counter()
                    text_result = analyze_text_ensemble(title, models['text_analyzer']) if title else None
                    timings['Text'] = time.perf_counter() - text_start

                    st.info(f"💬 Analyzing {len(comments)} comments...")
                    comments_start = time.perf_counter()
                    comment_results = [] # Stores analysis results for each comment
                    # --- FIX 3: Robust Comment Analysis Loop ---
                    if isinstance(comments, list):
                        for i, c_data in enumerate(comments):
                             comment_text = None
                             if isinstance(c_data, dict) and c_data.get('text'):
                                  comment_text = c_data.get('text', '') # Get the text safely

                             if comment_text: # Check if text is not None and not empty
                                  result = analyze_text_ensemble(comment_text, models['text_analyzer'])
                                  comment_results.append(result)
                             else:
                                  # Append placeholder if comment text is missing/empty
                                  logging.warning(f"Comment {i} has empty/missing text, appending N/A placeholder.")
                                  comment_results.append({'pred_label': 'N/A', 'score': 0.5})
                    else:
                         logging.warning("Comments data is not a list as expected.")
                    # --- END FIX 3 ---
                    timings['Comments'] = time.perf_counter() - comments_start

                    st.info("🖼️ Downloading & analyzing media (if present)...")
                    media_start = time.perf_counter()
                    media_result, media_type, downloaded_media_path = (None, None, None)
                    if media_url_for_download:
                        media_result, media_type, downloaded_media_path = analyze_media(
                            media_url_for_download, models, platform, session
                            )
                        # Add a message if download succeeded but analysis failed
                        if downloaded_media_path and media_result is None:
                             st.warning("Media downloaded, but analysis failed (e.g., corrupt file, unsupported format).")
                    else:
                        st.info("No media URL found to download or analyze.")
                    timings['Media'] = time.perf_counter() - media_start


                    st.info("⚙️ Fusing results...")
                    fuse_start = time.perf_counter()
                    overall_score, overall_sentiment, accuracy, weights = fuse_sentiments_adaptive(
                        text_result, media_result, comment_results, platform, media_type
                        )
                    timings['Fusion'] = time.perf_counter() - fuse_start


                    # --- Prepare and Store Results ---
                    sentiment_data_to_save = {
                        'text': text_result,
                        'media': media_result,
                        'comments': comment_results, # Save the list of result dicts
                        'overall_score': overall_score,
                        'overall_sentiment': overall_sentiment
                    }
                    total_time = time.perf_counter() - total_start_time
                    timings['Total'] = total_time

                    # Check against uploaded ground truth data
                    true_sentiment_label = None
                    if 'uploaded_data' in st.session_state:
                        match = st.session_state['uploaded_data'][st.session_state['uploaded_data']['url'] == url]
                        if not match.empty:
                            true_sentiment_label = match.iloc[0]['true_sentiment']
                            logging.info(f"Found matching URL in uploaded data. True Sentiment: {true_sentiment_label}")


                    analysis_entry_to_save_for_db = {
                        'url': url, 'platform': platform, 'timestamp': datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S'),
                        'sentiment': overall_sentiment, 'score': overall_score, 'accuracy': accuracy,
                        'title': title,
                        'sentiment_data': sentiment_data_to_save, # Contains nested results
                        'weights': weights, 'timings': timings,
                        'true_sentiment': true_sentiment_label
                    }

                    save_success = save_analysis_to_db(analysis_entry_to_save_for_db, data_source='live_analysis')

                    # Prepare entry for session state display (add raw comments, path, etc.)
                    analysis_entry_for_display = analysis_entry_to_save_for_db.copy()
                    analysis_entry_for_display['comments_analyzed'] = comments # Raw comments list [{text:..}]
                    analysis_entry_for_display['comment_results'] = comment_results # Analysis results list [{pred_label:.., score:..}]
                    analysis_entry_for_display['media_type'] = media_type
                    analysis_entry_for_display['downloaded_media_path'] = downloaded_media_path
                    analysis_entry_for_display['media_url'] = media_url_for_download # Keep original download URL

                    st.session_state.analysis_results = analysis_entry_for_display
                    analysis_successful = True # Mark as successful

                    # Invalidate history cache if save was successful
                    if save_success and 'history_display_cache' in st.session_state:
                        del st.session_state.history_display_cache

                    st.success(f"✅ Analysis Complete ({total_time:.1f}s)! View results in the next tab.")
                    logging.info(f"Analysis complete {url} ({total_time:.1f}s)")

                except ValueError as ve: # Catch specific errors raised in the pipeline
                    st.error(f"Analysis Error: {ve}")
                    st.session_state.analysis_error = str(ve)
                    logging.error(f"Analysis ValueError: {ve}")
                except Exception as e:
                    st.error(f"An unexpected error occurred: {e}")
                    st.session_state.analysis_error = f"Unexpected error: {e}"
                    logging.error(f"Analysis main loop failed: {traceback.format_exc()}")
                finally:
                    gc.collect() # Force garbage collection after analysis run

                # Rerun to switch to the results tab automatically after successful analysis
                if analysis_successful:
                     # This might cause issues if run inside the 'with spinner' or 'form'
                     # Consider using st.experimental_set_query_params or managing active tab state
                     # For simplicity, we just show the success message and user clicks tab.
                     pass # st.rerun() # Might cause issues here


    # --- Tab 2: Display Latest Results ---
    with tab2:
        st.header("📊 Latest Analysis Results")
        current_results_display = st.session_state.get('analysis_results', None)
        analysis_error_display = st.session_state.get('analysis_error', None)

        if analysis_error_display:
            st.error(f"Analysis failed: {analysis_error_display}")
            st.info("Please try a different URL or check the logs.")
        elif current_results_display is None:
            st.info("Run an analysis from the 'Analysis Input' tab first.")
        else:
            # --- Display Overall Assessment ---
            st.subheader("Overall Assessment")
            cols_metrics = st.columns(3)
            sentiment = current_results_display.get('sentiment', 'N/A')
            score = current_results_display.get('score', 0.5)
            # Ensure score is float for formatting
            score_f = float(score) if isinstance(score, (int, float, np.number)) else 0.5
            accuracy_disp = current_results_display.get('accuracy', 0.0)
            accuracy_f = float(accuracy_disp) if isinstance(accuracy_disp, (int, float, np.number)) else 0.0
            platform = current_results_display.get('platform', 'unknown').capitalize()

            # Determine delta color based on sentiment
            delta_color_val = "off" if sentiment == 'NEUTRAL' else ("normal" if sentiment == 'POSITIVE' else "inverse")

            cols_metrics[0].metric("Overall Sentiment", sentiment, delta=f"{score_f:.2f} Score", delta_color=delta_color_val)
            cols_metrics[1].metric("Est. Confidence", f"{accuracy_f:.1%}") # Renamed for clarity
            cols_metrics[2].metric("Platform", platform)

            # Display True Sentiment if available
            true_sent = current_results_display.get('true_sentiment')
            if true_sent:
                 st.success(f"**Ground Truth:** {true_sent} (from uploaded data)")


            # --- Detailed Breakdown Expander ---
            st.subheader("Detailed Breakdown")
            with st.expander("📝 Post Content Analysis", expanded=True):
                sentiment_data_disp = current_results_display.get('sentiment_data', {})
                text_res_disp = sentiment_data_disp.get('text') or {}
                title_text_disp = current_results_display.get('title') or ''

                # Display text analysis
                if title_text_disp:
                    text_label = text_res_disp.get('pred_label', 'N/A')
                    text_score_val = text_res_disp.get('score', 0.5)
                    text_score_f = float(text_score_val) if isinstance(text_score_val, (int, float, np.number)) else 0.5
                    st.markdown(f"**Title/Caption Sentiment**: {text_label} (Score: {text_score_f:.2f})")
                    # Use a blockquote for the text
                    st.markdown(f"> _{title_text_disp}_")
                else:
                    st.markdown("**(No Title/Caption Available or Analyzed)**")

                st.markdown("---") # Separator

                # --- FIX 2: Revised Media Display Logic ---
                media_path_disp = current_results_display.get('downloaded_media_path')
                media_type_disp = current_results_display.get('media_type')
                media_res_disp = sentiment_data_disp.get('media') # Analysis results dict
                original_media_url_ref = current_results_display.get('media_url') # Original URL used for download

                # 1. Display Media if Downloaded
                media_displayed = False
                if media_path_disp and os.path.exists(media_path_disp):
                    st.markdown(f"**Media ({media_type_disp.capitalize()})**:")
                    try:
                        if media_type_disp == 'image':
                            st.image(media_path_disp, caption="Downloaded Media", use_container_width=True)
                            media_displayed = True
                        elif media_type_disp == 'video':
                            try:
                                # Reading bytes is often more reliable with st.video
                                with open(media_path_disp, 'rb') as vf:
                                     video_bytes = vf.read()
                                st.video(video_bytes)
                                media_displayed = True
                            except Exception as video_display_err:
                                logging.error(f"Failed to display video from path/bytes {media_path_disp}: {video_display_err}")
                                st.warning("⚠️ Could not display downloaded video file.")
                        elif media_type_disp == 'audio':
                             try:
                                  with open(media_path_disp, 'rb') as af:
                                       audio_bytes = af.read()
                                  st.audio(audio_bytes)
                                  media_displayed = True
                             except Exception as audio_display_err:
                                  logging.error(f"Failed to display audio from path/bytes {media_path_disp}: {audio_display_err}")
                                  st.warning("⚠️ Could not display downloaded audio file.")

                    except Exception as display_err:
                        logging.error(f"Error displaying media from path {media_path_disp}: {display_err}")
                        st.warning(f"⚠️ Could not display downloaded media file.")
                elif original_media_url_ref: # If URL existed but download failed
                     st.markdown("**Media**: Download failed or no media found at URL.")
                else: # No media URL was present in the first place
                     st.markdown("**(No Media Associated with Post)**")

                # 2. Display Media Analysis Results (if analysis occurred)
                if media_res_disp: # Check if analysis result dict exists
                    media_label = media_res_disp.get('pred_label', 'N/A')
                    media_score_val = media_res_disp.get('score', 0.0)
                    media_score_f = float(media_score_val) if isinstance(media_score_val, (int, float, np.number)) else 0.0
                    st.markdown(f"**Media Sentiment Analysis**: {media_label} ({media_score_f:.2f})")
                    if 'dominant_emotion' in media_res_disp:
                        dom_emo = media_res_disp.get('dominant_emotion','N/A')
                        st.markdown(f"**Media Dominant Emotion**: {dom_emo.capitalize()}")
                elif media_displayed: # If media was displayed but analysis failed
                     st.warning("⚠️ Media downloaded, but sentiment/emotion analysis failed.")
                # No message needed if media wasn't downloaded/displayed anyway


            # --- Comment Analysis Expander ---
            with st.expander("💬 Comment Analysis", expanded=False):
                # Use raw comments and analysis results stored for display
                comments_analyzed_raw_disp = current_results_display.get('comments_analyzed', [])
                comment_results_list_disp = current_results_display.get('comment_results', [])

                if comments_analyzed_raw_disp: # Check if comments were fetched
                    num_fetched = len(comments_analyzed_raw_disp)
                    # Use analysis results for summary, filtering N/A placeholders
                    valid_results = [r for r in comment_results_list_disp if r and r.get('pred_label') != 'N/A']
                    pos_count = sum(1 for r in valid_results if r['pred_label'] == 'POSITIVE')
                    neg_count = sum(1 for r in valid_results if r['pred_label'] == 'NEGATIVE')
                    neu_count = len(valid_results) - pos_count - neg_count
                    na_count = num_fetched - len(valid_results) # Comments fetched but empty/failed analysis

                    st.markdown(f"**Analyzed {num_fetched} Comments**: <span style='color:green;'>**{pos_count} Pos**</span>, <span style='color:red;'>**{neg_count} Neg**</span>, <span style='color:grey;'>**{neu_count} Neu**</span> ({na_count} Skipped/N/A)", unsafe_allow_html=True)
                    st.markdown("---")

                    # Display individual comments and their results (ensure lists are aligned)
                    for i, c_data in enumerate(comments_analyzed_raw_disp):
                        # Safely get the corresponding result
                        c_res = comment_results_list_disp[i] if i < len(comment_results_list_disp) else None
                        label = (c_res or {}).get('pred_label', 'Error') # Default to Error if mismatch
                        score_val = (c_res or {}).get('score', 0.0)
                        score_f = float(score_val) if isinstance(score_val, (int, float, np.number)) else 0.0
                        text = c_data.get('text', '*Comment text missing*') if isinstance(c_data, dict) else str(c_data) # Handle if raw comment isn't dict

                        # Determine color based on label
                        color = "green" if label == "POSITIVE" else "red" if label == "NEGATIVE" else "grey" if label == "NEUTRAL" else "orange" # Orange for N/A or Error

                        st.markdown(f"**Comment {i+1}**: <span style='color:{color};'>{label}</span> ({score_f:.2f})", unsafe_allow_html=True)
                        st.caption(f"> {text[:300]}" + ("..." if len(text) > 300 else ""))
                        if i < num_fetched - 1:
                            st.markdown("---") # Separator unless last comment
                else:
                    st.info("No comments were found or fetched for this post.")


            # --- Visualizations & Metrics ---
            st.subheader("📈 Visual Summary & Performance")
            try:
                # Calculate metrics based on FULL history from DB before plotting
                # Use cached history if available, otherwise load
                if 'history_display_cache' not in st.session_state:
                    st.session_state.history_display_cache = load_history_from_db()
                history_for_metrics = st.session_state.history_display_cache

                metrics_results, conf_matrix_results = calculate_pseudo_metrics(history_for_metrics)

                # Pass data from the *current* analysis run for plotting components
                fig = plot_visualizations(
                    current_results_display.get('sentiment_data'),
                    current_results_display.get('accuracy'),
                    current_results_display.get('weights', {}),
                    current_results_display.get('timings', {}),
                    metrics_results, # Pass metrics calculated from history
                    conf_matrix_results # Pass confusion matrices calculated from history
                )
                if fig:
                    st.pyplot(fig, use_container_width=True) # Use container width
                    plt.close(fig) # Close figure to free memory
                else:
                    st.warning("Could not generate visualizations for the current run.")
            except Exception as plot_err:
                st.error(f"Error generating plots: {plot_err}")
                logging.error(f"Plotting failed: {traceback.format_exc()}")


            # --- Debug Output ---
            if st.session_state.debug_mode:
                st.subheader("🛠️ Debug Information")
                with st.expander("Show Current Analysis Data (JSON)"):
                    # Use safe_json_dumps to handle potential numpy types before display
                    try:
                         debug_json_str = safe_json_dumps(current_results_display)
                         st.json(debug_json_str if debug_json_str else "{}") # Show empty if dump fails
                    except Exception as json_dump_err:
                         st.error(f"Could not serialize debug data: {json_dump_err}")
                         st.write(current_results_display) # Fallback to raw write

                if metrics_results:
                    with st.expander("Show Calculated Pseudo-Metrics (Based on DB History)"):
                        st.json(metrics_results)
                if conf_matrix_results:
                    with st.expander("Show Confusion Matrix Data (Text vs Overall, Based on DB History)"):
                        cm_text_data = conf_matrix_results.get("Text")
                        if cm_text_data is not None:
                            st.write("Labels:", METRIC_LABELS)
                            # Display CM using pandas for better formatting
                            try:
                                 cm_df = pd.DataFrame(cm_text_data,
                                                      index=[f"True {l}" for l in METRIC_LABELS],
                                                      columns=[f"Pred {l}" for l in METRIC_LABELS])
                                 st.dataframe(cm_df)
                            except ImportError:
                                 st.write("Pandas not available to display Confusion Matrix table.")
                                 st.write(cm_text_data) # Show raw numpy array
                        else:
                            st.write("Not Available")
    # --- Tab 3: History ---
    with tab3:
        st.header("📜 Analysis History (from Database)")
        # Load history from DB for display, potentially cache it
        # Initialize cache if needed
        if 'history_display_cache' not in st.session_state:
            st.session_state.history_display_cache = load_history_from_db(limit=50) # Load 50 for display

        # Display history from cache
        history_display = st.session_state.history_display_cache

        # Add a refresh button
        if st.button("Refresh History from DB"):
             st.session_state.history_display_cache = load_history_from_db(limit=50) # Reload data
             history_display = st.session_state.history_display_cache # Update local var
             st.rerun() # Rerun to update the display immediately


        if not history_display:
            st.info("No analysis history found in the database. (Preloaded data might not have loaded?)")
        else:
            st.markdown(f"Displaying last {len(history_display)} runs from database (max 50).")
            # Iterate through history (newest first due to DB query)
            for i, entry in enumerate(history_display):
                # Use DB id if available, else calculate sequential number
                run_num = entry.get('id', len(history_display) - i)
                data_src_label = entry.get('data_source', 'live')[:4].upper() # e.g., 'LIVE' or 'PREL'
                # Create expander title
                exp_title = f"#{run_num} [{data_src_label}]: {entry.get('timestamp','N/A')} ({entry.get('platform','N/A').capitalize()}) - {entry.get('overall_sentiment','N/A')}"
                with st.expander(exp_title, expanded=False):
                    cols = st.columns([2, 3]) # Column layout
                    # Column 1: URL, Scores, Time
                    cols[0].markdown(f"**URL**: `{entry.get('url','N/A')}`")
                    cols[0].markdown(f"**Overall Score**: {entry.get('score', 0.0):.2f}")
                    cols[0].markdown(f"**Est. Accuracy**: {entry.get('accuracy', 0.0):.1%}") # Use 'accuracy' key
                    timings_hist = entry.get('timings', {}) # Parsed from JSON
                    if timings_hist:
                        cols[0].markdown(f"**Total Time**: {timings_hist.get('Total', 0.0):.1f}s")
                    if entry.get('true_sentiment'): # Display true sentiment if available
                         cols[0].markdown(f"**TRUE Sentiment**: {entry.get('true_sentiment')}")


                    # Column 2: Title, Media Summary, Comment Count (Reconstruct if needed)
                    # --- FIX: Handle NoneType title ---
                    title_text = entry.get('title') or '' # Default to empty string if None
                    cols[1].markdown(f"**Title**: {title_text[:100]}" + ("..." if len(title_text) > 100 else ""))
                    # --- End Fix ---

                    # Reconstruct media info from direct DB columns
                    media_label_hist = entry.get('media_label')
                    media_score_hist = entry.get('media_score')
                    media_dom_emo_hist = entry.get('media_dominant_emotion')
                    # Infer media type
                    media_type_hist = "Media" # Placeholder
                    if media_label_hist:
                         if media_dom_emo_hist:
                             media_type_hist = 'Image/Video'
                         else:
                             media_type_hist = 'Audio'

                    if media_label_hist: # Check if media was analyzed
                        info = f"**{media_type_hist}**: {media_label_hist} ({media_score_hist:.2f})"
                        if media_dom_emo_hist:
                            info += f", Emotion: {media_dom_emo_hist.capitalize()}"
                        cols[1].markdown(info)
                    # We don't store media URL in DB, so cannot indicate 'URL present'
                    # else:
                    #     cols[1].markdown(f"**Media**: No analysis data.")


                    # Comment count based on comment label stored (proxy for analyzed comments)
                    comment_label_hist = entry.get('comment_label')
                    # Could refine this if raw comment list was stored or count was stored
                    cols[1].markdown(f"**Comments Analyzed**: {'Yes' if comment_label_hist else 'No/Failed'}")


    # --- Footer ---
    st.markdown("---")
    st.markdown("Social Media Sentiment Analyzer v2.9 (Error Fixes)") # Version bump

# --- Entry Point ---
if __name__ == "__main__":
    main()