import pandas as pd
import os
from preprocess import get_subject_ids, process_subject
from features import extract_features_subject
from config import FEATURES_DIR, STAI_PATH

def build_full_dataset():
    subject_ids = get_subject_ids()
    print(f"Found {len(subject_ids)} subjects")

    stai = pd.read_csv(STAI_PATH, sep='\t')
    print(f"STAI data loaded: {stai.shape}")
    print(stai.head(3))

    all_rows = []
    failed = []

    for i, subject_id in enumerate(subject_ids):
        print(f"[{i+1}/{len(subject_ids)}] Processing {subject_id}...", end=' ')
        try:
            result = process_subject(subject_id)
            if result is None:
                print("SKIPPED")
                failed.append(subject_id)
                continue
            rows = extract_features_subject(result)
            all_rows.extend(rows)
            print(f"OK ({len(rows)} epochs)")
        except Exception as e:
            print(f"ERROR: {e}")
            failed.append(subject_id)

    df = pd.DataFrame(all_rows)
    print(f"\nFull feature matrix: {df.shape}")

    # STAI uses 'participant_id'; align to the subject_id key used throughout
    stai = stai.rename(columns={'participant_id': 'subject_id'})
    df = df.merge(stai[['subject_id', 'stai_trait', 'stai_state']],
                  on='subject_id', how='left')

    # Median split rather than a fixed cutoff keeps class balance equal regardless
    # of sample composition, which matters for macro-F1 under small N
    median_trait = df['stai_trait'].median()
    median_state = df['stai_state'].median()

    df['anxiety_label'] = (df['stai_trait'] >= median_trait).astype(int)
    df['state_label'] = (df['stai_state'] >= median_state).astype(int)

    print(f"Median STAI-trait: {median_trait}")
    print(f"Median STAI-state: {median_state}")
    print(f"Trait label distribution:\n{df['anxiety_label'].value_counts()}")
    print(f"State label distribution:\n{df['state_label'].value_counts()}")

    eo_ec_path = os.path.join(FEATURES_DIR, "features_eo_ec.csv")
    df.to_csv(eo_ec_path, index=False)
    print(f"\nSaved to {eo_ec_path}")

    if failed:
        print(f"\nFailed subjects: {failed}")

    return df

if __name__ == "__main__":
    build_full_dataset()