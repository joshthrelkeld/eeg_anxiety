import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import (classification_report, confusion_matrix,
                             f1_score, accuracy_score)
from config import FEATURES_DIR, RESULTS_DIR

# Frontal (F3/F4/Fz), occipital (O1/O2), and parietal (Pz) coverage targets
# the canonical EEG correlates of anxiety and attentional state.
# Alpha asymmetry and Pz alpha/beta ratio are included as theory-driven composites.
FEATURE_COLS = [
    'F3_theta', 'F3_alpha', 'F3_beta',
    'F4_theta', 'F4_alpha', 'F4_beta',
    'Fz_theta', 'Fz_alpha', 'Fz_beta',
    'O1_theta', 'O1_alpha', 'O1_beta',
    'O2_theta', 'O2_alpha', 'O2_beta',
    'Pz_theta', 'Pz_alpha', 'Pz_beta',
    'frontal_alpha_asymmetry',
    'Pz_alpha_beta_ratio'
]

def load_data(task='eo_ec'):
    path = os.path.join(FEATURES_DIR, "features_eo_ec.csv")
    df = pd.read_csv(path)

    if task == 'eo_ec':
        df['y'] = (df['label'] == 'EC').astype(int)
        print(f"Task: EO vs EC | EC=1, EO=0")
    elif task == 'anxiety':
        df['y'] = df['anxiety_label']
        print(f"Task: Anxiety | High=1, Low=0")

    X = df[FEATURE_COLS].values
    y = df['y'].values
    # Passed to StratifiedGroupKFold to enforce subject-level train/test splits
    groups = df['subject_id'].values

    print(f"X shape: {X.shape} | y distribution: {np.bincount(y)}")
    return X, y, groups, df

def run_models(X, y, groups, task_name):
    """
    Evaluate LR, RF, and SVM under StratifiedGroupKFold.

    Three model families are included to span the bias-variance tradeoff:
    LR as a linear baseline, RF for non-linear feature interactions, and
    SVM-RBF as a kernel method robust to small N. class_weight='balanced'
    compensates for any label imbalance without resampling.

    StratifiedGroupKFold is preferred over standard k-fold because EEG epochs
    from the same subject are highly correlated; leaking subjects across folds
    would inflate performance estimates.
    """
    models = {
        'Logistic Regression': LogisticRegression(
            max_iter=1000, class_weight='balanced'),
        'Random Forest': RandomForestClassifier(
            n_estimators=100, class_weight='balanced', random_state=42),
        'SVM': SVC(kernel='rbf', class_weight='balanced', random_state=42),
    }

    cv = StratifiedGroupKFold(n_splits=5)
    scaler = StandardScaler()
    results = {}

    for model_name, model in models.items():
        fold_accs, fold_f1s = [], []
        all_y_true, all_y_pred = [], []

        for fold, (train_idx, test_idx) in enumerate(
                cv.split(X, y, groups=groups)):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            # Fit scaler on train only to avoid leaking test-set statistics
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            fold_accs.append(accuracy_score(y_test, y_pred))
            fold_f1s.append(f1_score(y_test, y_pred, average='macro'))
            all_y_true.extend(y_test)
            all_y_pred.extend(y_pred)

        results[model_name] = {
            'accuracy': np.mean(fold_accs),
            'f1': np.mean(fold_f1s),
            'y_true': all_y_true,
            'y_pred': all_y_pred,
        }

        print(f"\n{model_name}:")
        print(f"  Accuracy: {np.mean(fold_accs):.3f} ± {np.std(fold_accs):.3f}")
        print(f"  F1 (macro): {np.mean(fold_f1s):.3f} ± {np.std(fold_f1s):.3f}")

    return results

def plot_confusion_matrices(results, task_name, labels):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle(f'Confusion Matrices — {task_name}', fontsize=13)

    for ax, (model_name, res) in zip(axes, results.items()):
        cm = confusion_matrix(res['y_true'], res['y_pred'])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=labels, yticklabels=labels, ax=ax)
        ax.set_title(f"{model_name}\nAcc={res['accuracy']:.3f} F1={res['f1']:.3f}")
        ax.set_ylabel('True')
        ax.set_xlabel('Predicted')

    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, f"confusion_{task_name}.png")
    plt.savefig(path, dpi=150)
    print(f"Saved: {path}")
    plt.show()

def plot_feature_importance(X, y, df):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    rf = RandomForestClassifier(
        n_estimators=100, class_weight='balanced', random_state=42)
    rf.fit(X_scaled, y)

    importances = pd.Series(rf.feature_importances_, index=FEATURE_COLS)
    importances = importances.sort_values(ascending=True)

    fig, ax = plt.subplots(figsize=(8, 7))
    importances.plot(kind='barh', ax=ax, color='steelblue')
    ax.set_title('Random Forest Feature Importances')
    ax.set_xlabel('Importance')
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, "feature_importance.png")
    plt.savefig(path, dpi=150)
    print(f"Saved: {path}")
    plt.show()

def run_subject_level_anxiety():
    """
    Classify trait anxiety at the subject level using LOO-CV.

    Trait anxiety is a stable individual characteristic, so it is more
    appropriate to classify at the subject level than the epoch level.
    Averaging epochs collapses within-subject variability and produces a
    single representative feature vector per person, which is the correct
    unit of analysis for a trait measure.

    LOO-CV is used because N=51 subjects is too small for held-out folds
    to be reliably stratified; LOO maximises training data at each split.
    """
    from sklearn.model_selection import LeaveOneOut

    path = os.path.join(FEATURES_DIR, "features_eo_ec.csv")
    df = pd.read_csv(path)

    subject_df = df.groupby('subject_id')[FEATURE_COLS].mean().reset_index()

    anxiety_labels = df.groupby('subject_id')['anxiety_label'].first().reset_index()
    subject_df = subject_df.merge(anxiety_labels, on='subject_id')

    X = subject_df[FEATURE_COLS].values
    y = subject_df['anxiety_label'].values

    print(f"\nSubject-level matrix: {X.shape}")
    print(f"y distribution: {np.bincount(y)}")

    models = {
        'Logistic Regression': LogisticRegression(
            max_iter=1000, class_weight='balanced'),
        'Random Forest': RandomForestClassifier(
            n_estimators=100, class_weight='balanced', random_state=42),
        'SVM': SVC(kernel='rbf', class_weight='balanced', random_state=42),
    }

    scaler = StandardScaler()
    loo = LeaveOneOut()
    results = {}

    for model_name, model in models.items():
        all_y_true, all_y_pred = [], []

        for train_idx, test_idx in loo.split(X):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            all_y_true.extend(y_test)
            all_y_pred.extend(y_pred)

        acc = accuracy_score(all_y_true, all_y_pred)
        f1 = f1_score(all_y_true, all_y_pred, average='macro')
        results[model_name] = {
            'accuracy': acc, 'f1': f1,
            'y_true': all_y_true, 'y_pred': all_y_pred
        }
        print(f"\n{model_name}:")
        print(f"  Accuracy: {acc:.3f}")
        print(f"  F1 (macro): {f1:.3f}")

    plot_confusion_matrices(results, 'Anxiety_Subject_Level', ['Low', 'High'])

    scaler2 = StandardScaler()
    X_scaled = scaler2.fit_transform(X)
    rf = RandomForestClassifier(
        n_estimators=100, class_weight='balanced', random_state=42)
    rf.fit(X_scaled, y)
    importances = pd.Series(rf.feature_importances_, index=FEATURE_COLS)
    importances = importances.sort_values(ascending=True)

    fig, ax = plt.subplots(figsize=(8, 7))
    importances.plot(kind='barh', ax=ax, color='steelblue')
    ax.set_title('Feature Importances — Subject-Level Anxiety')
    ax.set_xlabel('Importance')
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, "feature_importance_anxiety.png")
    plt.savefig(path, dpi=150)
    print(f"Saved: {path}")
    plt.show()

    return results

def plot_feature_correlations():
    """
    Correlate subject-level features with continuous STAI-trait scores.

    Pearson r against the raw STAI score complements binary classification:
    if classification accuracy is near chance but some r values are meaningful,
    that suggests a graded rather than categorical relationship with anxiety,
    which is consistent with STAI being a continuous measure.
    """
    from scipy import stats

    path = os.path.join(FEATURES_DIR, "features_eo_ec.csv")
    df = pd.read_csv(path)

    subject_df = df.groupby('subject_id')[FEATURE_COLS].mean().reset_index()
    stai = df.groupby('subject_id')['stai_trait'].first().reset_index()
    subject_df = subject_df.merge(stai, on='subject_id')

    correlations = []
    for col in FEATURE_COLS:
        r, p = stats.pearsonr(subject_df[col], subject_df['stai_trait'])
        correlations.append({'feature': col, 'r': r, 'p': p})

    corr_df = pd.DataFrame(correlations).sort_values('r', ascending=True)

    # Red encodes statistical significance to make the threshold salient at a glance
    colors = ['#d62728' if p < 0.05 else 'steelblue'
              for p in corr_df['p']]

    fig, ax = plt.subplots(figsize=(8, 7))
    bars = ax.barh(corr_df['feature'], corr_df['r'], color=colors)
    ax.axvline(x=0, color='black', linewidth=0.8, linestyle='--')
    ax.set_title('Feature Correlations with STAI-Trait Score\n'
                 '(red = p < 0.05, blue = n.s.)', fontsize=12)
    ax.set_xlabel('Pearson r')
    plt.tight_layout()

    path = os.path.join(RESULTS_DIR, "feature_stai_correlations.png")
    plt.savefig(path, dpi=150)
    print(f"Saved: {path}")
    plt.show()

    print("\nFeature correlations with STAI-trait:")
    print(corr_df.to_string(index=False))
    sig = corr_df[corr_df['p'] < 0.05]
    if len(sig) > 0:
        print(f"\nSignificant correlations (p<0.05): {len(sig)}")
        print(sig.to_string(index=False))
    else:
        print("\nNo correlations reach p<0.05")

    return corr_df

def plot_state_correlations():
    """
    Correlate subject-level features with STAI-state scores and compare to trait.

    State and trait anxiety have distinct temporal profiles: state reflects
    momentary stimulation while trait is dispositional. If features correlate
    more strongly with state than trait, it suggests the EEG signal is tracking
    acute arousal during the recording rather than stable individual differences —
    an important confound to rule out when making trait-level claims.
    """
    from scipy import stats

    path = os.path.join(FEATURES_DIR, "features_eo_ec.csv")
    df = pd.read_csv(path)

    subject_df = df.groupby('subject_id')[FEATURE_COLS].mean().reset_index()
    stai = df.groupby('subject_id')[['stai_state', 'stai_trait']].first().reset_index()
    subject_df = subject_df.merge(stai, on='subject_id')

    state_corrs, trait_corrs = [], []
    for col in FEATURE_COLS:
        r_state, p_state = stats.pearsonr(
            subject_df[col], subject_df['stai_state'])
        r_trait, p_trait = stats.pearsonr(
            subject_df[col], subject_df['stai_trait'])
        state_corrs.append({'feature': col, 'r': r_state, 'p': p_state})
        trait_corrs.append({'feature': col, 'r': r_trait, 'p': p_trait})

    state_df = pd.DataFrame(state_corrs).sort_values('r', ascending=True)
    trait_df = pd.DataFrame(trait_corrs).set_index('feature')

    colors = ['#d62728' if p < 0.05 else 'steelblue'
              for p in state_df['p']]

    fig, ax = plt.subplots(figsize=(8, 7))
    ax.barh(state_df['feature'], state_df['r'], color=colors)
    ax.axvline(x=0, color='black', linewidth=0.8, linestyle='--')
    ax.set_title('Feature Correlations with STAI-State Score\n'
                 '(red = p < 0.05, blue = n.s.)', fontsize=12)
    ax.set_xlabel('Pearson r')
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, "feature_state_correlations.png")
    plt.savefig(path, dpi=150)
    print(f"Saved: {path}")
    plt.show()

    fig, axes = plt.subplots(1, 2, figsize=(14, 7), sharey=False)

    # Shared feature ordering makes the state/trait comparison visually legible
    feature_order = state_df['feature'].tolist()
    state_r = state_df.set_index('feature').loc[feature_order, 'r']
    trait_r = trait_df.loc[feature_order, 'r']

    state_colors = ['#d62728' if state_df.set_index('feature').loc[f, 'p'] < 0.05
                    else 'steelblue' for f in feature_order]
    trait_colors = ['#d62728' if trait_df.loc[f, 'p'] < 0.05
                    else 'steelblue' for f in feature_order]

    axes[0].barh(feature_order, state_r, color=state_colors)
    axes[0].axvline(x=0, color='black', linewidth=0.8, linestyle='--')
    axes[0].set_title('STAI-State Correlations\n(red = p<0.05)', fontsize=11)
    axes[0].set_xlabel('Pearson r')

    axes[1].barh(feature_order, trait_r, color=trait_colors)
    axes[1].axvline(x=0, color='black', linewidth=0.8, linestyle='--')
    axes[1].set_title('STAI-Trait Correlations\n(red = p<0.05)', fontsize=11)
    axes[1].set_xlabel('Pearson r')

    plt.suptitle('Feature Correlations: State vs Trait Anxiety',
                 fontsize=13, y=1.02)
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, "feature_correlations_comparison.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    print(f"Saved: {path}")
    plt.show()

    print("\nFeature correlations with STAI-state:")
    print(state_df.to_string(index=False))
    sig_state = state_df[state_df['p'] < 0.05]
    if len(sig_state) > 0:
        print(f"\nSignificant state correlations (p<0.05): {len(sig_state)}")
        print(sig_state.to_string(index=False))
    else:
        print("\nNo state correlations reach p<0.05")

    print("\n--- State vs Trait r comparison ---")
    print(f"{'Feature':<30} {'r_state':>10} {'r_trait':>10} {'difference':>12}")
    print("-" * 65)
    for feat in feature_order:
        r_s = state_df.set_index('feature').loc[feat, 'r']
        r_t = trait_df.loc[feat, 'r']
        diff = abs(r_s) - abs(r_t)
        print(f"{feat:<30} {r_s:>10.4f} {r_t:>10.4f} {diff:>+12.4f}")

    return state_df

if __name__ == "__main__":
    print("Run main.py to execute the full pipeline.")
    print("Individual functions can be imported and called directly.")