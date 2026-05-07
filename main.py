import argparse
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import LeaveOneOut

from config import FEATURES_DIR
from classify import (
    load_data,
    run_models,
    plot_confusion_matrices,
    plot_feature_importance,
    run_subject_level_anxiety,
    plot_feature_correlations,
    plot_state_correlations,
    FEATURE_COLS
)

def main(rebuild=False):

    # Feature extraction is expensive across 51 subjects. The cached CSV is
    # used by default. --rebuild forces re-extraction if the pipeline or
    # raw data changes.
    features_path = os.path.join(FEATURES_DIR, "features_eo_ec.csv")

    if rebuild or not os.path.exists(features_path):
        print("Building feature dataset from EEG files...")
        from build_dataset import build_full_dataset
        build_full_dataset()
    else:
        print(f"Feature dataset found at {features_path}")
        print("Skipping feature extraction. Use --rebuild to re-extract.\n")

    # Phase 1: EO vs EC
    # A perceptual state classification with a well-established neural
    # signature — occipital alpha suppression during eyes open. Serves as
    # a pipeline validation task and provides a meaningful performance
    # ceiling before moving to psychological constructs.
    # In theory, this should display ample differentiation.
    print("=" * 50)
    print("PHASE 1: Eyes Open vs Eyes Closed")
    print("=" * 50)
    X, y, groups, df = load_data(task='eo_ec')
    results = run_models(X, y, groups, task_name='EO_vs_EC')
    plot_confusion_matrices(results, 'EO vs EC', ['EO', 'EC'])
    plot_feature_importance(X, y, df)

    # Phase 2: Trait anxiety — epoch level
    # Trait anxiety is a stable dispositional construct measured prior to
    # any recording. Classifying it from a two-minute EEG snapshot tests
    # whether spectral features carry information about personality rather
    # than momentary state.
    print("\n" + "=" * 50)
    print("PHASE 2: Trait Anxiety (High vs Low)")
    print("=" * 50)
    X, y, groups, df = load_data(task='anxiety')
    results = run_models(X, y, groups, task_name='Anxiety')
    plot_confusion_matrices(results, 'Trait Anxiety', ['Low', 'High'])

    # Phase 2b: Trait anxiety — subject level (LOO-CV)
    # Epoch-level classification treats each 2-second window as independent,
    # which inflates sample size but misrepresents the prediction target.
    # Aggregating to subject-level means before classification is the
    # methodologically honest approach, at the cost of reducing N to 51.
    print("\n" + "=" * 50)
    print("PHASE 2b: Subject-Level Trait Anxiety (LOO-CV)")
    print("=" * 50)
    run_subject_level_anxiety()

    # Phase 3: State anxiety — epoch and subject level
    # State anxiety reflects how a participant felt at the moment of
    # assessment or immediately before the EEG recording. Unlike trait,
    # it is temporally aligned with the neural signal. If the temporal
    # distance hypothesis holds, state should classify better than trait
    # at both aggregation levels.
    print("\n" + "=" * 50)
    print("PHASE 3: State Anxiety (High vs Low)")
    print("=" * 50)
    df_state = pd.read_csv(os.path.join(FEATURES_DIR, "features_eo_ec.csv"))
    X_state = df_state[FEATURE_COLS].values
    y_state = df_state['state_label'].values
    groups_state = df_state['subject_id'].values

    print(f"Task: State Anxiety | High=1, Low=0")
    print(f"X shape: {X_state.shape} | "
          f"y distribution: {np.bincount(y_state)}")

    results_state = run_models(
        X_state, y_state, groups_state, task_name='State_Anxiety')
    plot_confusion_matrices(
        results_state, 'State Anxiety', ['Low', 'High'])

    # Subject-level state anxiety mirrors Phase 2b for direct comparison.
    # The gap between state and trait performance at the subject level is
    # the cleanest test of the temporal distance hypothesis.
    df_sub = df_state.groupby('subject_id')[FEATURE_COLS].mean().reset_index()
    state_labels = (df_state.groupby('subject_id')['state_label']
                    .first().reset_index())
    df_sub = df_sub.merge(state_labels, on='subject_id')

    X_sub = df_sub[FEATURE_COLS].values
    y_sub = df_sub['state_label'].values

    scaler = StandardScaler()
    loo = LeaveOneOut()
    models = {
        'Logistic Regression': LogisticRegression(
            max_iter=1000, class_weight='balanced'),
        'Random Forest': RandomForestClassifier(
            n_estimators=100, class_weight='balanced', random_state=42),
        'SVM': SVC(
            kernel='rbf', class_weight='balanced', random_state=42),
    }

    print(f"\nSubject-level state anxiety:")
    print(f"Matrix: {X_sub.shape} | y: {np.bincount(y_sub)}")

    sub_results = {}
    for model_name, model in models.items():
        all_y_true, all_y_pred = [], []
        for train_idx, test_idx in loo.split(X_sub):
            X_train = scaler.fit_transform(X_sub[train_idx])
            X_test = scaler.transform(X_sub[test_idx])
            model.fit(X_train, y_sub[train_idx])
            y_pred = model.predict(X_test)
            all_y_true.extend(y_sub[test_idx])
            all_y_pred.extend(y_pred)

        acc = accuracy_score(all_y_true, all_y_pred)
        f1 = f1_score(all_y_true, all_y_pred, average='macro')
        sub_results[model_name] = {
            'accuracy': acc, 'f1': f1,
            'y_true': all_y_true, 'y_pred': all_y_pred
        }
        print(f"\n{model_name}:")
        print(f"  Accuracy: {acc:.3f} | F1: {f1:.3f}")

    plot_confusion_matrices(
        sub_results, 'State_Anxiety_Subject_Level', ['Low', 'High'])

    # Binary classification can fail for reasons unrelated to signal quality,
    # such as threshold sensitivity, class imbalance, small N. Pearson correlations
    # between subject-level features and continuous STAI scores provide a
    # complementary, threshold-free view of which features carry linear
    # information about anxiety, and whether state and trait engage the
    # feature space in different directions.
    print("\n" + "=" * 50)
    print("FEATURE CORRELATIONS")
    print("=" * 50)
    plot_feature_correlations()
    plot_state_correlations()

    print("\n" + "=" * 50)
    print("ALL ANALYSES COMPLETE")
    print(f"Results saved to: {os.path.abspath('results/')}")
    print("=" * 50)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="EEG Anxiety Classification Pipeline")
    parser.add_argument(
        '--rebuild', action='store_true',
        help='Re-extract features from raw EEG files')
    args = parser.parse_args()
    main(rebuild=args.rebuild)