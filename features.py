import numpy as np
from scipy.signal import welch
from config import BANDS, EPOCH_LENGTH

def compute_band_power(epoch, sfreq, band):
    freqs, psd = welch(epoch, fs=sfreq, nperseg=int(sfreq * EPOCH_LENGTH))
    freq_mask = (freqs >= band[0]) & (freqs <= band[1])
    return np.mean(psd[freq_mask])


def extract_features_single_epoch(epoch_data, ch_indices, sfreq):
    features = {}

    # Log-transform stabilizes variance across subjects
    for ch_name, ch_idx in ch_indices.items():
        signal = epoch_data[ch_idx]
        for band_name, band_range in BANDS.items():
            power = compute_band_power(signal, sfreq, band_range)
            features[f"{ch_name}_{band_name}"] = np.log(power + 1e-10)

    # F4 - F3 asymmetry: positive values indicate relatively greater right frontal alpha,
    # associated with withdrawal motivation and anxiety in the Davidson model
    if 'F3' in ch_indices and 'F4' in ch_indices:
        f3_alpha = compute_band_power(
            epoch_data[ch_indices['F3']], sfreq, BANDS['alpha'])
        f4_alpha = compute_band_power(
            epoch_data[ch_indices['F4']], sfreq, BANDS['alpha'])
        features['frontal_alpha_asymmetry'] = (
            np.log(f4_alpha + 1e-10) - np.log(f3_alpha + 1e-10)
        )

    # Pz alpha/beta ratio is a marker of posterior cortical relaxation vs engagement
    if 'Pz' in ch_indices:
        pz_alpha = compute_band_power(
            epoch_data[ch_indices['Pz']], sfreq, BANDS['alpha'])
        pz_beta = compute_band_power(
            epoch_data[ch_indices['Pz']], sfreq, BANDS['beta'])
        features['Pz_alpha_beta_ratio'] = pz_alpha / (pz_beta + 1e-10)

    return features

def extract_features_subject(subject_result):
    subject_id = subject_result['subject_id']
    sfreq = subject_result['sfreq']
    ch_indices = subject_result['ch_indices']
    rows = []

    for block in subject_result['blocks']:
        label = block['label']
        epochs = block['epochs']

        for epoch_idx in range(epochs.shape[0]):
            features = extract_features_single_epoch(
                epochs[epoch_idx], ch_indices, sfreq)
            features['label'] = label
            features['subject_id'] = subject_id
            rows.append(features)

    return rows

if __name__ == "__main__":
    from preprocess import process_subject
    import pandas as pd
    result = process_subject("sub-1006")
    rows = extract_features_subject(result)
    df = pd.DataFrame(rows)
    print(f"sub-1006 features: {df.shape}")
    print(f"Columns: {list(df.columns)}")