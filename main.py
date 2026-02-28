import re
import os
import time
import logging
import praw
from transformers import pipeline
from dotenv import load_dotenv

# Firebase imports
import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore

# --- Load Environment Variables ---
load_dotenv()

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Configuration ---
FIREBASE_SERVICE_ACCOUNT_KEY_PATH = os.getenv("FIREBASE_KEY_PATH")
USER_ID = "car_scanner_user_123"
APP_ID = "reddit_car_scanner_app"

SUBREDDITS_TO_SCAN = [
    "CarsIndia", "IndianGaming", "india", "AskIndia", "cars",
    "whatcarshouldibuy", "usedcars", "electricvehicles",
    "Autos", "mechanicadvice"
]

INCLUSION_KEYWORDS = [
    "buy car", "buying advice", "which car", "recommend car", "suggest car",
    "new car", "used car", "pre-owned", "lease", "loan", "EMI", "down payment",
    "suv", "sedan", "hatchback", "ev", "electric vehicle", "compact suv", "mpv",
    "maruti", "hyundai", "tata", "mahindra", "kia", "mg", "honda", "toyota", "skoda",
    "india", "indian roads", "delhi", "mumbai", "bangalore", "pune", "chennai", "hyderabad",
    "budget", "price", "review", "ownership", "variant", "ex-showroom", "on-road",
    "resale", "maintenance", "reliability", "features", "safety", "mileage", "test drive"
]

EXCLUSION_KEYWORDS = [
    "accident", "crash", "insurance claim", "racing", "tuning", "modding",
    "service issue", "breakdown", "repair", "nft", "crypto", "scrap"
]

CANDIDATE_LABELS = [
    "car buying advice",
    "new car recommendation",
    "used car inquiry",
    "car ownership review",
    "car problem / troubleshooting",
    "general car discussion",
    "unrelated topic"
]

ML_CONFIDENCE_THRESHOLD = 0.75
SCAN_INTERVAL_SECONDS = 3600
POST_FETCH_LIMIT = 50

# --- Text Preprocessing ---
def preprocess_text(text):
    if not text:
        return ""
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# --- Keyword Filtering Logic ---
def passes_keyword_filter(text, inclusion_keywords, exclusion_keywords):
    if not text:
        return False

    found_inclusion = False
    for keyword in inclusion_keywords:
        if re.search(r'\b' + re.escape(keyword) + r'\b', text):
            found_inclusion = True
            break

    if not found_inclusion:
        return False

    for keyword in exclusion_keywords:
        if re.search(r'\b' + re.escape(keyword) + r'\b', text):
            return False

    return True

# --- Firebase Initialization ---
db = None
try:
    if not FIREBASE_SERVICE_ACCOUNT_KEY_PATH:
        raise ValueError("FIREBASE_KEY_PATH not set in .env file")
    if not os.path.exists(FIREBASE_SERVICE_ACCOUNT_KEY_PATH):
        raise FileNotFoundError(f"Firebase key file not found at: {FIREBASE_SERVICE_ACCOUNT_KEY_PATH}")
    if not firebase_admin._apps:
        cred = credentials.Certificate(FIREBASE_SERVICE_ACCOUNT_KEY_PATH)
        firebase_admin.initialize_app(cred)
    db = firestore.client()
    logger.info("Firebase initialized successfully.")
except Exception as e:
    logger.error(f"Error initializing Firebase: {e}")
    db = None

# --- Firebase Helper Functions ---
def get_processed_posts_collection():
    if db:
        return db.collection('artifacts').document(APP_ID).collection('users').document(USER_ID).collection('processed_car_posts')
    return None

def get_relevant_posts_collection():
    if db:
        return db.collection('artifacts').document(APP_ID).collection('users').document(USER_ID).collection('relevant_car_posts')
    return None

def is_post_processed(post_id):
    if not db:
        logger.warning("Firestore not initialized. Cannot check if post is processed.")
        return False
    try:
        doc_ref = get_processed_posts_collection().document(post_id)
        doc = doc_ref.get()
        return doc.exists
    except Exception as e:
        logger.error(f"Error checking processed status for post {post_id}: {e}")
        return False

def add_processed_post(post_id):
    if not db:
        logger.warning("Firestore not initialized. Cannot add post as processed.")
        return
    try:
        get_processed_posts_collection().document(post_id).set({'processed_at': firestore.SERVER_TIMESTAMP})
        logger.debug(f"Added post ID {post_id} to processed list.")
    except Exception as e:
        logger.error(f"Error adding post ID {post_id} to processed list: {e}")

def save_relevant_post(post_data):
    if not db:
        logger.warning("Firestore not initialized. Cannot save relevant post.")
        return
    try:
        doc_id = post_data['id']
        relevant_posts_col = get_relevant_posts_collection()
        relevant_posts_col.document(doc_id).set({
            'title': post_data['title'],
            'selftext': post_data['selftext'],
            'subreddit': post_data['subreddit'],
            'url': f"https://www.reddit.com{post_data['url']}",
            'author_username': post_data.get('author_username', '[deleted]'),
            'ml_score': post_data.get('ml_score'),
            'ml_label': post_data.get('ml_label'),
            'found_at': firestore.SERVER_TIMESTAMP
        }, merge=True)
        logger.info(f"Saved relevant post to Firestore: {post_data['title']}")
    except Exception as e:
        logger.error(f"Error saving relevant post to Firestore: {post_data.get('title', 'N/A')}: {e}")

def save_relevant_comment(parent_post_id, comment_data):
    if not db:
        logger.warning("Firestore not initialized. Cannot save relevant comment.")
        return
    try:
        comment_doc_id = comment_data['id']
        comments_subcollection = get_relevant_posts_collection().document(parent_post_id).collection('comments')
        comments_subcollection.document(comment_doc_id).set({
            'body': comment_data['body'],
            'author_username': comment_data.get('author_username', '[deleted]'),
            'ml_score': comment_data.get('ml_score'),
            'ml_label': comment_data.get('ml_label'),
            'permalink': f"https://www.reddit.com{comment_data.get('permalink', '')}",
            'created_at': firestore.SERVER_TIMESTAMP
        }, merge=True)
        logger.info(f"  Saved relevant comment to Firestore (Post ID: {parent_post_id}): {comment_data['body'][:50]}...")
    except Exception as e:
        logger.error(f"Error saving relevant comment to Firestore (Post ID: {parent_post_id}): {e}")

# --- ML Model Initialization ---
classifier = None
try:
    logger.info("Loading pre-trained ML model (facebook/bart-large-mnli)... This may take a moment on first run.")
    classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
    logger.info("ML model loaded successfully.")
except Exception as e:
    logger.error(f"Error loading ML model: {e}. ML filtering will be skipped.")
    classifier = None

# --- Reddit Initialization ---
reddit = None
try:
    client_id = os.getenv("REDDIT_CLIENT_ID")
    client_secret = os.getenv("REDDIT_CLIENT_SECRET")
    user_agent = os.getenv("REDDIT_USER_AGENT")

    if not all([client_id, client_secret, user_agent]):
        raise ValueError("Reddit credentials not fully set in .env file. Check REDDIT_CLIENT_ID, REDDIT_CLIENT_SECRET, REDDIT_USER_AGENT.")

    reddit = praw.Reddit(
        client_id=client_id,
        client_secret=client_secret,
        user_agent=user_agent
    )
    logger.info("PRAW initialized successfully.")
except Exception as e:
    logger.error(f"Error initializing PRAW: {e}")
    reddit = None

# --- Reddit Post Fetching ---
def fetch_reddit_posts(subreddit_list, num_posts_per_subreddit):
    if reddit is None:
        logger.warning("PRAW not initialized. Skipping real data fetch.")
        return []

    all_fetched_posts = []
    logger.info(f"Attempting to fetch {num_posts_per_subreddit} 'new' posts from each of {len(subreddit_list)} subreddits...")

    for sub_name in subreddit_list:
        try:
            subreddit = reddit.subreddit(sub_name)
            for submission in subreddit.new(limit=num_posts_per_subreddit):
                post_id = submission.id
                if not is_post_processed(post_id):
                    if not submission.stickied and not submission.locked:
                        post_data = {
                            "id": post_id,
                            "title": submission.title,
                            "selftext": submission.selftext,
                            "subreddit": submission.subreddit.display_name,
                            "url": submission.permalink,
                            "author_username": submission.author.name if submission.author else '[deleted]',
                            "comments": []
                        }

                        try:
                            submission.comments.replace_more(limit=None)
                            for comment in submission.comments.list():
                                if comment.author:
                                    post_data["comments"].append({
                                        "id": comment.id,
                                        "body": comment.body,
                                        "author_username": comment.author.name,
                                        "permalink": comment.permalink
                                    })
                        except Exception as e:
                            logger.error(f"    Error fetching comments for post {post_id}: {e}")

                        all_fetched_posts.append(post_data)
                    add_processed_post(post_id)
                else:
                    logger.debug(f"Post {post_id} from r/{sub_name} already processed.")
            logger.info(f"  Processed new posts from r/{sub_name}.")
            time.sleep(1)
        except Exception as e:
            logger.error(f"  Error fetching posts from r/{sub_name}: {e}")
            continue

    logger.info(f"Finished fetching. Total new candidate posts found: {len(all_fetched_posts)}")
    return all_fetched_posts

# --- ML Filtering ---
def apply_ml_filter_to_text(text):
    if classifier is None:
        return {'label': 'ML_Skipped', 'score': 0.0, 'relevant': False}

    processed_text = preprocess_text(text)
    if not processed_text:
        return {'label': 'Empty_Text', 'score': 0.0, 'relevant': False}

    try:
        result = classifier(processed_text, CANDIDATE_LABELS, multi_label=True)
        top_label = result['labels'][0]
        top_score = result['scores'][0]

        car_seeker_intents = [
            "car buying advice",
            "new car recommendation",
            "used car inquiry"
        ]

        is_relevant = (top_label in car_seeker_intents and top_score >= ML_CONFIDENCE_THRESHOLD)

        return {'label': top_label, 'score': float(top_score), 'relevant': is_relevant}
    except Exception as e:
        logger.error(f"Error during ML inference: {e}")
        return {'label': 'ML_Error', 'score': 0.0, 'relevant': False}

# --- Main Scanning Loop ---
def run_continuous_car_scanner():
    logger.info("Starting Reddit Car Market Scanner in continuous mode.")
    logger.info(f"Scanning every {SCAN_INTERVAL_SECONDS / 60} minutes.")

    while True:
        try:
            logger.info(f"\n--- Initiating new scan cycle ({time.ctime()}) ---")

            all_raw_posts = fetch_reddit_posts(SUBREDDITS_TO_SCAN, num_posts_per_subreddit=POST_FETCH_LIMIT)
            logger.info(f"Total new posts fetched in this cycle: {len(all_raw_posts)}")

            keyword_filtered_posts = []
            logger.info("Applying keyword filter (Phase 2) to posts...")
            for post in all_raw_posts:
                combined_text = f"{post.get('title', '')} {post.get('selftext', '')}"
                processed_text = preprocess_text(combined_text)

                if passes_keyword_filter(processed_text, INCLUSION_KEYWORDS, EXCLUSION_KEYWORDS):
                    keyword_filtered_posts.append(post)
                else:
                    logger.debug(f"  [KEYWORD SKIPPED POST]: r/{post['subreddit']} - {post['title']}")

            logger.info(f"Found {len(keyword_filtered_posts)} posts passing initial keyword filter.")

            current_cycle_relevant_posts_count = 0
            logger.info("Starting Advanced ML Filtering (Phase 3) for posts and comments...")

            for post in keyword_filtered_posts:
                post_ml_result = apply_ml_filter_to_text(f"{post['title']} {post['selftext']}")
                post['ml_label'] = post_ml_result['label']
                post['ml_score'] = post_ml_result['score']

                post_is_relevant = post_ml_result['relevant']
                if post_is_relevant:
                    logger.info(f"  [ML PASSED POST - {post_ml_result['label']} ({post_ml_result['score']:.2f})]: {post['title']}")
                    save_relevant_post(post)
                    current_cycle_relevant_posts_count += 1
                else:
                    logger.debug(f"  [ML SKIPPED POST - {post_ml_result['label']} ({post_ml_result['score']:.2f})]: {post['title']}")

                relevant_comments_for_post = []
                for comment_data in post.get('comments', []):
                    comment_ml_result = apply_ml_filter_to_text(comment_data['body'])
                    comment_data['ml_label'] = comment_ml_result['label']
                    comment_data['ml_score'] = comment_ml_result['score']

                    if comment_ml_result['relevant']:
                        relevant_comments_for_post.append(comment_data)
                        save_relevant_comment(post['id'], comment_data)
                        logger.info(f"    [ML PASSED COMMENT - {comment_ml_result['label']} ({comment_ml_result['score']:.2f})]: {comment_data['body'][:50]}...")
                    else:
                        logger.debug(f"    [ML SKIPPED COMMENT - {comment_ml_result['label']} ({comment_ml_result['score']:.2f})]: {comment_data['body'][:50]}...")

                if relevant_comments_for_post:
                    logger.info(f"  Found {len(relevant_comments_for_post)} relevant comments for post: {post['title']}")

                time.sleep(0.5)

            logger.info(f"Found {current_cycle_relevant_posts_count} genuinely relevant posts in this cycle.")
            logger.info(f"Scan cycle complete. Waiting {SCAN_INTERVAL_SECONDS} seconds for next cycle...")
            time.sleep(SCAN_INTERVAL_SECONDS)

        except KeyboardInterrupt:
            logger.info("Scanner stopped by user (KeyboardInterrupt). Exiting.")
            break
        except Exception as e:
            logger.critical(f"An unhandled error occurred during scan cycle: {e}. Retrying after delay.", exc_info=True)
            time.sleep(60)

if __name__ == "__main__":
    if not FIREBASE_SERVICE_ACCOUNT_KEY_PATH:
        logger.critical("FIREBASE_KEY_PATH not found in .env file. Please set it and try again.")
    elif not os.path.exists(FIREBASE_SERVICE_ACCOUNT_KEY_PATH):
        logger.critical(f"Firebase Service Account Key not found at: {FIREBASE_SERVICE_ACCOUNT_KEY_PATH}")
        logger.critical("Please download it from Firebase Console and update your .env file.")
    else:
        run_continuous_car_scanner()

    logger.info("Script finished execution.")
