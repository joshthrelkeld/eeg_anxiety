import os

DATA_DIR     = "/Users/joshuathrelkeld/Desktop/Resting State EEG and Trait Anxiety/data/derivatives/preprocessed"
STAI_PATH    = "/Users/joshuathrelkeld/Desktop/Resting State EEG and Trait Anxiety/data/phenotype/STAI.tsv"
FEATURES_DIR = os.path.join(os.path.dirname(__file__), "features")
RESULTS_DIR  = os.path.join(os.path.dirname(__file__), "results")

os.makedirs(FEATURES_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# EGI 256-channel net mapped to 10-20 labels; only frontal, occipital, and
# parietal sites are used — see FEATURE_COLS in classify.py for rationale
CHANNELS = {
    'F3': 'E37',
    'F4': 'E18',
    'Fz': 'E26',
    'O1': 'E123',
    'O2': 'E158',
    'Pz': 'E101',
}

# Bandpass and sampling rate were applied by the dataset authors; listed here
# for reproducibility. EPOCH_LENGTH and ARTIFACT_THRESH are our decisions.
BANDPASS_LOW    = 0.1
BANDPASS_HIGH   = 50.0
SFREQ           = 500
EPOCH_LENGTH    = 2.0    # seconds
ARTIFACT_THRESH = 150e-6 # 150 µV — final safety net after authors' preprocessing

BANDS = {
    'theta': (4, 8),
    'alpha': (8, 13),
    'beta':  (13, 30),
}

BLOCK_ORDER        = ['EO', 'EC', 'EO', 'EC']
BLOCK_DURATION     = 60.0
ANNOTATION_TRIGGER = 'bgin'