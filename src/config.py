# All shared constants for training and inference.
# Change values here — they propagate everywhere automatically.

# Data 
MOVIE_TABLE = "default.wiki_movie_plots_deduped"

# Tokenizer 
VOCAB_SIZE = 20_000

# Model 
SEQ_LEN    = 128
EMBED_DIM  = 256
LSTM_UNITS = 512

# Training 
BATCH_SIZE       = 128
STRIDE           = 64
VAL_EXAMPLES     = 5_000
STEPS_PER_EPOCH  = 2_000
VAL_STEPS        = 200
EPOCHS           = 30
LEARNING_RATE    = 1e-4
EARLY_STOP_PATIENCE = 3
CHECKPOINT_PATH  = "/dbfs/tmp/movie_lm_best.keras"

# MLflow 
MLFLOW_EXPERIMENT = "/Users/desiborisovab@gmail.com/movie_chatbot_experiment"
MLFLOW_RUN_NAME   = "lstm_attention_movie_lm"

# Inference 
DEFAULT_MAX_NEW_TOKENS = 120
DEFAULT_TEMPERATURE    = 0.7
DEFAULT_TOP_K          = 50
RETRIEVE_K             = 3
PLOT_TRUNCATE_CHARS    = 300

STOPWORDS = {
    "the", "and", "for", "with", "was", "are", "this", "that",
    "have", "from", "not", "who", "what", "when", "where", "about",
    "movie", "film", "show", "did", "does", "can", "tell", "give",
}