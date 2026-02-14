import re
import json
import os
import time
import logging
import praw
from transformers import pipeline

# Firebase imports
import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore

# --- Logging Setup ---
# Configure logging to display INFO level messages and higher to the console
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Configuration ---
# IMPORTANT: Replace with the actual path to your downloaded Firebase Service Account Key JSON file
# This file is used to authenticate your script with your Firebase project.
# Example: FIREBASE_SERVICE_ACCOUNT_KEY_PATH = "my-firebase-project-abcdef-firebase-adminsdk-xxxxx-xxxxx.json"
FIREBASE_SERVICE_ACCOUNT_KEY_PATH = "your-project-name-firebase-adminsdk-xxxxx-xxxxx.json" # <--- REPLACE THIS

# User ID and App ID for structuring data in Firestore.
# USER_ID allows for user-specific data paths. APP_ID helps organize data for this specific application.
USER_ID = "car_scanner_user_123"
APP_ID = "reddit_car_scanner_app"

# List of Reddit subreddits to monitor for car-related discussions.
# These are chosen to be relevant to the car market, especially with an Indian context.
SUBREDDITS_TO_SCAN = [
    "CarsIndia", "IndianGaming", "india", "AskIndia", "cars",
    "whatcarshouldibuy", "usedcars", "electricvehicles", "ev",
    "Autos", "mechanicadvice", "carbuyingsharing"
]

# Keywords that indicate a post is potentially relevant to car buying/advice.
# Posts must contain at least one of these to pass the initial filter.
INCLUSION_KEYWORDS = [
    "buy car", "buying advice", "which car", "recommend car", "suggest car",
    "new car", "used car", "pre-owned", "lease", "loan", "EMI", "down payment",
    "suv", "sedan", "hatchback", "ev", "electric vehicle", "compact suv", "mpv",
    "maruti", "hyundai", "tata", "mahindra", "kia", "mg", "honda", "toyota", "skoda",
    "india", "indian roads", "delhi", "mumbai", "bangalore", "pune", "chennai", "hyderabad",
    "budget", "price", "review", "ownership", "variant", "ex-showroom", "on-road",
    "resale", "maintenance", "reliability", "features", "safety", "mileage", "test drive"
]

# Keywords that, if present, indicate a post is irrelevant and should be excluded.
# These help filter out noise like accident reports or general repair discussions.
EXCLUSION_KEYWORDS = [
    "accident", "crash", "insurance claim", "racing", "tuning", "modding",
    "service issue", "breakdown", "repair", "nft", "crypto", "scrap"
]

# Candidate labels for the Zero-Shot Text Classification ML model.
# These are the intents/topics the ML model will try to classify text into.
CANDIDATE_LABELS = [
    "car buying advice",
    "new car recommendation",
    "used car inquiry",
    "car ownership review",
    "car problem / troubleshooting",
    "general car discussion",
    "unrelated topic"
]

# The minimum confidence score (0.0 to 1.0) the ML model needs to assign
# to a "car seeker" intent for the post/comment to be considered relevant.
ML_CONFIDENCE_THRESHOLD = 0.75

# How long the script waits between full scanning cycles (in seconds).
# Adjust to balance responsiveness with Reddit API rate limits.
SCAN_INTERVAL_SECONDS = 3600 # 1 hour

# Number of new posts to attempt to fetch from each subreddit per scan cycle.
POST_FETCH_LIMIT = 50

# --- New Configuration for JSON Export ---
# The filename for the JSON file where relevant comments will be exported locally.
RELEVANT_COMMENTS_JSON_FILE = "relevant_comments_export.json"

# --- Text Preprocessing Function ---
def preprocess_text(text):
    """
    Cleans and normalizes text for consistent keyword matching and ML analysis.
    Converts to lowercase, removes most punctuation, and normalizes whitespace.
    """
    if not text:
        return ""
    text = text.lower() # Convert all text to lowercase
    text = re.sub(r'[^\w\s]', '', text) # Remove non-alphanumeric characters (keep words and spaces)
    text = re.sub(r'\s+', ' ', text).strip() # Replace multiple spaces with single space and strip leading/trailing whitespace
    return text

# --- Keyword Filtering Logic (Phase 2) ---
def passes_keyword_filter(text, inclusion_keywords, exclusion_keywords):
    """
    Checks if a given text contains any inclusion keywords AND no exclusion keywords.
    Uses regex with word boundaries for accurate whole-word matching.
    """
    if not text:
        return False

    # Check for presence of at least one inclusion keyword
    found_inclusion = False
    for keyword in inclusion_keywords:
        # re.escape handles special characters in keywords if they exist
        if re.search(r'\b' + re.escape(keyword) + r'\b', text):
            found_inclusion = True
            break # Found one, no need to check further

    if not found_inclusion:
        return False # No inclusion keywords found, so it fails the filter

    # Check for absence of any exclusion keywords
    for keyword in exclusion_keywords:
        if re.search(r'\b' + re.escape(keyword) + r'\b', text):
            return False # An exclusion keyword was found, so it fails the filter

    return True # Passed both inclusion and exclusion checks

# --- Firebase Initialization and Database Helper Functions ---
# Initialize Firestore client globally. It will be None if initialization fails.
db = None
try:
    # Initialize Firebase Admin SDK only once per script run
    if not firebase_admin._apps:
        cred = credentials.Certificate(FIREBASE_SERVICE_ACCOUNT_KEY_PATH)
        firebase_admin.initialize_app(cred)
    db = firestore.client() # Get the Firestore client
    logger.info("Firebase initialized successfully.")
except Exception as e:
    logger.error(f"Error initializing Firebase: {e}. Please check your service account key path.")
    db = None # Ensure db remains None if initialization failed

def get_processed_posts_collection():
    """Returns a reference to the Firestore collection for tracking processed Reddit post IDs."""
    if db:
        # Path: /artifacts/{APP_ID}/users/{USER_ID}/processed_car_posts
        return db.collection('artifacts').document(APP_ID).collection('users').document(USER_ID).collection('processed_car_posts')
    return None

def get_relevant_posts_collection():
    """Returns a reference to the Firestore collection for storing relevant Reddit posts."""
    if db:
        # Path: /artifacts/{APP_ID}/users/{USER_ID}/relevant_car_posts
        return db.collection('artifacts').document(APP_ID).collection('users').document(USER_ID).collection('relevant_car_posts')
    return None

def is_post_processed(post_id):
    """Checks if a Reddit post ID has already been marked as processed in Firestore."""
    if not db:
        logger.warning("Firestore not initialized. Cannot check if post is processed.")
        return False # If DB is not available, proceed as if not processed
    try:
        doc_ref = get_processed_posts_collection().document(post_id)
        doc = doc_ref.get()
        return doc.exists
    except Exception as e:
        logger.error(f"Error checking processed status for post {post_id}: {e}")
        return False

def add_processed_post(post_id):
    """Adds a Reddit post ID to the processed posts collection in Firestore."""
    if not db:
        logger.warning("Firestore not initialized. Cannot add post as processed.")
        return
    try:
        # Set a document with the post_id and a timestamp
        get_processed_posts_collection().document(post_id).set({'processed_at': firestore.SERVER_TIMESTAMP})
        logger.debug(f"Added post ID {post_id} to processed list.")
    except Exception as e:
        logger.error(f"Error adding post ID {post_id} to processed list: {e}")

def save_relevant_post(post_data):
    """
    Saves a relevant Reddit post's data (including author and ML results) to Firestore.
    Uses the Reddit post's unique ID as the Firestore document ID.
    """
    if not db:
        logger.warning("Firestore not initialized. Cannot save relevant post.")
        return
    try:
        # Use Reddit's unique submission ID directly as the Firestore document ID
        doc_id = post_data['id']
        relevant_posts_col = get_relevant_posts_collection()
        relevant_posts_col.document(doc_id).set({
            'title': post_data['title'],
            'selftext': post_data['selftext'],
            'subreddit': post_data['subreddit'],
            'url': f"https://www.reddit.com{post_data['url']}", # Ensure full URL for direct access
            'author_username': post_data.get('author_username', '[deleted]'), # Safely get author, default to [deleted]
            'ml_score': post_data.get('ml_score'),
            'ml_label': post_data.get('ml_label'),
            'found_at': firestore.SERVER_TIMESTAMP # Server timestamp for when it was saved
        }, merge=True) # merge=True allows updating existing documents or creating new ones
        logger.info(f"Saved relevant post to Firestore: {post_data['title']}")
    except Exception as e:
        logger.error(f"Error saving relevant post to Firestore: {post_data.get('title', 'N/A')}: {e}")

def save_relevant_comment(parent_post_id, comment_data):
    """
    Saves a relevant Reddit comment's data to a subcollection under its parent post in Firestore.
    """
    if not db:
        logger.warning("Firestore not initialized. Cannot save relevant comment.")
        return
    try:
        comment_doc_id = comment_data['id']
        # Path: /artifacts/{APP_ID}/users/{USER_ID}/relevant_car_posts/{parent_post_id}/comments/{comment_id}
        comments_subcollection = get_relevant_posts_collection().document(parent_post_id).collection('comments')
        comments_subcollection.document(comment_doc_id).set({
            'body': comment_data['body'],
            'author_username': comment_data.get('author_username', '[deleted]'),
            'ml_score': comment_data.get('ml_score'),
            'ml_label': comment_data.get('ml_label'),
            'permalink': f"https://www.reddit.com{comment_data.get('permalink', '')}", # Full permalink to comment
            'created_at': firestore.SERVER_TIMESTAMP
        }, merge=True)
        logger.info(f"  Saved relevant comment to Firestore (Post ID: {parent_post_id}): {comment_data['body'][:50]}...")
    except Exception as e:
        logger.error(f"Error saving relevant comment to Firestore (Post ID: {parent_post_id}): {comment_data.get('body', 'N/A')}: {e}")

# --- JSON Export Function ---
def export_relevant_comments_to_json(new_comments_list):
    """
    Exports a list of new relevant comments to a JSON file.
    It reads existing data from the file, appends new comments, and overwrites the file.
    This ensures the JSON file always contains a valid array of comment objects.
    """
    all_comments = []
    # Check if the file exists and has valid JSON data
    if os.path.exists(RELEVANT_COMMENTS_JSON_FILE):
        try:
            with open(RELEVANT_COMMENTS_JSON_FILE, 'r', encoding='utf-8') as f:
                existing_data = json.load(f)
                # Ensure existing data is a list before extending
                if isinstance(existing_data, list):
                    all_comments.extend(existing_data)
                else:
                    logger.warning(f"Existing JSON file '{RELEVANT_COMMENTS_JSON_FILE}' is not a list. Overwriting with new data.")
        except json.JSONDecodeError as e:
            logger.error(f"Error decoding existing JSON file '{RELEVANT_COMMENTS_JSON_FILE}': {e}. Starting with an empty list for new data.")
        except Exception as e:
            logger.error(f"Error reading existing JSON file '{RELEVANT_COMMENTS_JSON_FILE}': {e}. Starting with an empty list for new data.")

    # Add the new comments to the collection
    all_comments.extend(new_comments_list)

    # Write the entire (updated) list back to the JSON file
    try:
        with open(RELEVANT_COMMENTS_JSON_FILE, 'w', encoding='utf-8') as f:
            # ensure_ascii=False allows non-ASCII characters (like emojis) to be written directly
            # indent=4 makes the JSON file human-readable with pretty printing
            json.dump(all_comments, f, ensure_ascii=False, indent=4)
        logger.info(f"Exported {len(new_comments_list)} new relevant comments to '{RELEVANT_COMMENTS_JSON_FILE}'. Total comments in file: {len(all_comments)}")
    except Exception as e:
        logger.error(f"Error writing relevant comments to JSON file '{RELEVANT_COMMENTS_JSON_FILE}': {e}")


# --- ML Model Initialization ---
# Initialize the Hugging Face zero-shot classification pipeline globally.
# It will be None if initialization fails (e.g., PyTorch/TensorFlow not installed).
classifier = None
try:
    logger.info("Loading pre-trained ML model (facebook/bart-large-mnli)... This may take a moment on first run.")
    # The pipeline automatically handles downloading the model weights if not cached
    classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
    logger.info("ML model loaded successfully.")
except Exception as e:
    logger.error(f"Error loading ML model: {e}. ML filtering will be skipped.")
    classifier = None # Ensure classifier remains None if loading failed

# --- REAL REDDIT API FETCHING ---
# Initialize the PRAW Reddit instance globally.
# It will be None if initialization fails (e.g., incorrect credentials).
reddit = None
try:
    # Your Reddit API credentials. REPLACE THE PLACEHOLDERS with your actual keys.
    reddit = praw.Reddit(
        client_id="YOUR_REDDIT_CLIENT_ID", # <--- REPLACE WITH YOUR REDDIT APP CLIENT ID (WITH QUOTES)
        client_secret="YOUR_REDDIT_CLIENT_SECRET", # <--- REPLACE WITH YOUR REDDIT APP CLIENT SECRET (WITH QUOTES)
        user_agent="CarScannerML/1.0 (by /u/No-Reveal9910)" # Your unique user agent (WITH QUOTES)
    )
    logger.info("PRAW initialized successfully.")
except Exception as e:
    logger.error(f"Error initializing PRAW: {e}. Please check your credentials.")
    reddit = None # Ensure reddit remains None if initialization failed

def fetch_reddit_posts(subreddit_list, num_posts_per_subreddit):
    """
    Fetches new Reddit posts and their comments from specified subreddits using PRAW.
    It checks Firestore to avoid reprocessing already seen posts.
    """
    if reddit is None:
        logger.warning("PRAW not initialized. Skipping real data fetch.")
        return [] # Return empty list if PRAW isn't configured/failed

    all_fetched_posts = []
    logger.info(f"Attempting to fetch {num_posts_per_subreddit} 'new' posts from each of {len(subreddit_list)} subreddits...")

    for sub_name in subreddit_list:
        try:
            subreddit = reddit.subreddit(sub_name)
            for submission in subreddit.new(limit=num_posts_per_subreddit):
                post_id = submission.id
                # Check if this post has already been processed in a previous run/cycle
                if not is_post_processed(post_id):
                    # Filter out sticky posts (announcements) and locked posts (no new comments)
                    if not submission.stickied and not submission.locked:
                        post_data = {
                            "id": post_id,
                            "title": submission.title,
                            "selftext": submission.selftext,
                            "subreddit": submission.subreddit.display_name,
                            "url": submission.permalink,
                            "author_username": submission.author.name if submission.author else '[deleted]', # Get post author
                            "comments": [] # Initialize an empty list to store fetched comments
                        }

                        # --- Fetch Comments for the current submission ---
                        try:
                            # replace_more(limit=None) fetches all comments, even nested "More Comments" links.
                            # Be cautious: this can make many API calls for popular posts.
                            submission.comments.replace_more(limit=None)
                            # Iterate through the flattened list of comments
                            for comment in submission.comments.list():
                                if isinstance(comment.author, praw.models.Redditor): # Ensure author is a real Redditor
                                    post_data["comments"].append({
                                        "id": comment.id,
                                        "body": comment.body,
                                        "author_username": comment.author.name,
                                        "permalink": comment.permalink # Store comment's permalink
                                    })
                                else:
                                    # Handle cases where comment author might be deleted or a special type
                                    post_data["comments"].append({
                                        "id": comment.id,
                                        "body": comment.body,
                                        "author_username": '[deleted]' if comment.author is None else str(comment.author),
                                        "permalink": comment.permalink
                                    })
                        except Exception as e:
                            logger.error(f"    Error fetching comments for post {post_id}: {e}")
                            # Continue processing the post even if comment fetching failed
                        # --- End Fetch Comments ---

                        all_fetched_posts.append(post_data)
                    # Mark the post as processed *after* attempting to fetch its data (including comments)
                    add_processed_post(post_id)
                else:
                    logger.debug(f"Post {post_id} from r/{sub_name} already processed.")
            logger.info(f"  Processed new posts from r/{sub_name}.")
            time.sleep(1) # Add a small delay between subreddits to be polite to Reddit's API
        except Exception as e:
            logger.error(f"  Error fetching posts from r/{sub_name}: {e}")
            continue # Continue to the next subreddit even if one fails

    logger.info(f"Finished fetching. Total new candidate posts found: {len(all_fetched_posts)}")
    return all_fetched_posts


# --- ML Filtering Logic for Posts and Comments (Phase 3) ---
def apply_ml_filter_to_text(text_content):
    """
    Applies the ML zero-shot classification to a given text (post body or comment body).
    Returns a dictionary with 'label', 'score', and 'relevant' status.
    """
    if classifier is None:
        # If ML model failed to load, return a default indicating it's skipped
        return {'label': 'ML_Skipped', 'score': 0.0, 'relevant': False}

    processed_text = preprocess_text(text_content)
    if not processed_text: # Handle cases where text might be empty after preprocessing
        return {'label': 'Empty_Text', 'score': 0.0, 'relevant': False}

    try:
        # Perform zero-shot classification against the predefined candidate labels
        result = classifier(processed_text, CANDIDATE_LABELS, multi_label=True)
        top_label = result['labels'][0] # Get the label with the highest score
        top_score = result['scores'][0] # Get the confidence score for that label

        # Define which labels indicate a "car seeker" intent for our project
        car_seeker_intents = [
            "car buying advice",
            "new car recommendation",
            "used car inquiry"
        ]

        # Determine relevance based on top label and confidence threshold
        is_relevant = (top_label in car_seeker_intents and top_score >= ML_CONFIDENCE_THRESHOLD)

        # Return a structured dictionary of ML results
        return {'label': top_label, 'score': float(top_score), 'relevant': is_relevant}
    except Exception as e:
        logger.error(f"Error during ML inference for text: '{text_content[:50]}...': {e}")
        return {'label': 'ML_Error', 'score': 0.0, 'relevant': False}

# --- Main Continuous Scanning Logic ---
def run_continuous_car_scanner():
    """
    The main function that orchestrates the continuous scanning, filtering, and saving.
    Runs in an infinite loop, pausing between cycles.
    """
    logger.info("Starting Reddit Car Market Scanner in continuous mode.")
    logger.info(f"Scanning every {SCAN_INTERVAL_SECONDS / 60} minutes.")

    while True: # Infinite loop for continuous operation
        try:
            logger.info(f"\n--- Initiating new scan cycle ({time.ctime()}) ---")

            # Phase 2: Data Collection & Initial Keyword Filtering for Posts
            all_raw_posts = fetch_reddit_posts(SUBREDDITS_TO_SCAN, num_posts_per_subreddit=POST_FETCH_LIMIT)
            logger.info(f"Total new posts fetched in this cycle: {len(all_raw_posts)}")

            keyword_filtered_posts = []
            logger.info("Applying keyword filter (Phase 2) to posts...")
            for post in all_raw_posts:
                title_text = post.get('title', '')
                self_text = post.get('selftext', '')
                combined_text = f"{title_text} {self_text}"
                processed_text = preprocess_text(combined_text)

                if passes_keyword_filter(processed_text, INCLUSION_KEYWORDS, EXCLUSION_KEYWORDS):
                    keyword_filtered_posts.append(post)
                else:
                    logger.debug(f"  [KEYWORD SKIPPED POST]: r/{post['subreddit']} - {post['title']}")

            logger.info(f"Found {len(keyword_filtered_posts)} posts passing initial keyword filter.")

            # Phase 3: Advanced ML Filtering for Posts and Comments
            current_cycle_relevant_posts_count = 0
            # List to collect all relevant comments found in this cycle for JSON export
            relevant_comments_for_json_export = []
            logger.info("Starting Advanced ML Filtering (Phase 3) for posts and comments...")

            for post in keyword_filtered_posts:
                # --- Apply ML filter to the main post body ---
                post_ml_result = apply_ml_filter_to_text(f"{post['title']} {post['selftext']}")
                post['ml_label'] = post_ml_result['label']
                post['ml_score'] = post_ml_result['score']

                post_is_relevant = post_ml_result['relevant']
                if post_is_relevant:
                    logger.info(f"  [ML PASSED POST - {post_ml_result['label']} ({post_ml_result['score']:.2f})]: {post['title']}")
                    save_relevant_post(post) # Save the relevant post to Firestore
                    current_cycle_relevant_posts_count += 1
                else:
                    logger.debug(f"  [ML SKIPPED POST - {post_ml_result['label']} ({post_ml_result['score']:.2f})]: {post['title']}")

                # --- Process Comments for this post (regardless of post's relevance) ---
                # We process comments for all keyword-filtered posts, as a comment might be relevant even if the post isn't
                for comment_data in post.get('comments', []):
                    comment_ml_result = apply_ml_filter_to_text(comment_data['body'])
                    comment_data['ml_label'] = comment_ml_result['label']
                    comment_data['ml_score'] = comment_ml_result['score']
                    comment_data['parent_post_id'] = post['id'] # Add parent post ID for context in JSON

                    if comment_ml_result['relevant']:
                        relevant_comments_for_json_export.append(comment_data) # Add to list for JSON export
                        save_relevant_comment(post['id'], comment_data) # Save to Firestore subcollection
                        logger.info(f"    [ML PASSED COMMENT - {comment_ml_result['label']} ({comment_ml_result['score']:.2f})]: {comment_data['body'][:50]}...")
                    else:
                        logger.debug(f"    [ML SKIPPED COMMENT - {comment_ml_result['label']} ({comment_ml_result['score']:.2f})]: {comment_data['body'][:50]}...")
                time.sleep(00.5) # Small delay after processing all comments for a post

            logger.info(f"Found {current_cycle_relevant_posts_count} genuinely relevant posts (top-level) in this cycle after all filters.")

            # --- Export relevant comments to JSON file ---
            if relevant_comments_for_json_export:
                export_relevant_comments_to_json(relevant_comments_for_json_export)
            else:
                logger.info("No new relevant comments found to export in this cycle.")

            logger.info(f"Scan cycle complete. Waiting {SCAN_INTERVAL_SECONDS} seconds for next cycle...")
            time.sleep(SCAN_INTERVAL_SECONDS)

        except KeyboardInterrupt:
            logger.info("Scanner stopped by user (KeyboardInterrupt). Exiting.")
            break # Exit the infinite loop gracefully on Ctrl+C
        except Exception as e:
            # Catch any unhandled errors, log them, and retry after a short delay
            logger.critical(f"An unhandled error occurred during scan cycle: {e}. Retrying after delay.", exc_info=True)
            time.sleep(60) # Wait 1 minute before retrying on critical error

# --- Main Entry Point for the Script ---
if __name__ == "__main__":
    # Perform a critical check to ensure the Firebase service account key exists before starting
    if not os.path.exists(FIREBASE_SERVICE_ACCOUNT_KEY_PATH):
        logger.critical(f"Firebase Service Account Key not found at: {FIREBASE_SERVICE_ACCOUNT_KEY_PATH}")
        logger.critical("Please download it from Firebase Console (Project settings -> Service accounts) and place it in your project folder.")
        logger.critical("Update FIREBASE_SERVICE_ACCOUNT_KEY_PATH in the script accordingly.")
    else:
        # If the key is found, start the continuous scanning process
        run_continuous_car_scanner()

    logger.info("Script finished execution.")
    