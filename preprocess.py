import mne
import numpy as np
import glob
import os
from config import (DATA_DIR, CHANNELS, BLOCK_ORDER,
                    BLOCK_DURATION, ANNOTATION_TRIGGER, EPOCH_LENGTH)

mne.set_log_level('WARNING')

def get_subject_ids():
    pattern = os.path.join(DATA_DIR, "sub-*", "eeg", "*desc-preproc_eeg.set")
    files = sorted(glob.glob(pattern))
    subject_ids = []
    for f in files:
        parts = f.split(os.sep)
        for part in parts:
            if part.startswith("sub-"):
                subject_ids.append(part)
                break
    return subject_ids

def load_subject(subject_id):
    path = os.path.join(
        DATA_DIR, subject_id, "eeg",
        f"{subject_id}_task-rest_desc-preproc_eeg.set"
    )
    if not os.path.exists(path):
        print(f"  [SKIP] File not found: {path}")
        return None
    raw = mne.io.read_raw_eeglab(path, preload=True)
    return raw

def get_block_onsets(raw):
    onsets = [ann['onset'] for ann in raw.annotations
              if ann['description'] == ANNOTATION_TRIGGER]
    if len(onsets) != 4:
        print(f"  [WARN] Expected 4 bgin markers, found {len(onsets)}")
        return None
    return onsets

def segment_blocks(raw):
    onsets = get_block_onsets(raw)
    if onsets is None:
        return None

    blocks = []
    for onset, label in zip(onsets, BLOCK_ORDER):
        tmin = onset
        tmax = onset + BLOCK_DURATION
        # Clamp to recording length in case the final block was cut short
        tmax = min(tmax, raw.times[-1])
        segment = raw.copy().crop(tmin=tmin, tmax=tmax, include_tmax=False)
        blocks.append({'label': label, 'raw': segment})
    return blocks

def epoch_block(block_raw):
    """
    Epoch a block into fixed-length non-overlapping windows.

    Trailing samples that don't fill a complete epoch are discarded rather
    than zero-padded, to avoid introducing artificial low-frequency content
    at epoch boundaries that would contaminate band power estimates.
    """
    sfreq = block_raw.info['sfreq']
    n_samples = int(EPOCH_LENGTH * sfreq)
    data = block_raw.get_data()
    n_channels, n_times = data.shape

    n_epochs = n_times // n_samples
    data = data[:, :n_epochs * n_samples]
    epochs = data.reshape(n_channels, n_epochs, n_samples)
    epochs = np.transpose(epochs, (1, 0, 2))
    return epochs

def get_channel_indices(raw):
    ch_names = raw.info['ch_names']
    indices = {}
    for label, egi_name in CHANNELS.items():
        if egi_name in ch_names:
            indices[label] = ch_names.index(egi_name)
        else:
            print(f"  [WARN] Channel {egi_name} ({label}) not found")
    return indices

def process_subject(subject_id):
    raw = load_subject(subject_id)
    if raw is None:
        return None

    blocks = segment_blocks(raw)
    if blocks is None:
        return None

    ch_indices = get_channel_indices(raw)

    result = {
        'subject_id': subject_id,
        'sfreq': raw.info['sfreq'],
        'ch_indices': ch_indices,
        'blocks': []
    }

    for block in blocks:
        epochs = epoch_block(block['raw'])
        result['blocks'].append({
            'label': block['label'],
            'epochs': epochs  # (n_epochs, n_channels, n_times)
        })

    return result

if __name__ == "__main__":
    result = process_subject("sub-1006")
    if result:
        print(f"sub-1006: {len(result['blocks'])} blocks")
        for block in result['blocks']:
            print(f"  {block['label']}: {block['epochs'].shape}")