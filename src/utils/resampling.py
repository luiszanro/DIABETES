from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline

def apply_smote(X, y, target_samples={1: 5000, 2: 37000}):
    """
    Apply SMOTE to oversample minority classes.
    Args:
        X (pd.DataFrame): Feature set.
        y (pd.Series): Target variable.
        target_samples (dict): Target sample sizes for each class.
    Returns:
        tuple: Resampled feature and target sets.
    """
    smote = SMOTE(sampling_strategy=target_samples)
    X_resampled, y_resampled = smote.fit_resample(X, y)
    return X_resampled, y_resampled

def apply_random_undersampling(X, y, target_samples={0: 50000}):
    """
    Apply random undersampling to majority class.
    Args:
        X (pd.DataFrame): Feature set.
        y (pd.Series): Target variable.
        target_samples (dict): Target sample sizes for each class.
    Returns:
        tuple: Resampled feature and target sets.
    """
    undersample = RandomUnderSampler(sampling_strategy=target_samples)
    X_resampled, y_resampled = undersample.fit_resample(X, y)
    return X_resampled, y_resampled

def combine_resampling(X, y, oversample_target={1: 5000, 2: 37000}, undersample_target={0: 50000}):
    """
    Combine oversampling and undersampling.
    Args:
        X (pd.DataFrame): Feature set.
        y (pd.Series): Target variable.
        oversample_target (dict): Target sample sizes for oversampling.
        undersample_target (dict): Target sample sizes for undersampling.
    Returns:
        tuple: Resampled feature and target sets.
    """
    over = SMOTE(sampling_strategy=oversample_target)
    under = RandomUnderSampler(sampling_strategy=undersample_target)
    pipeline = Pipeline([('under', under), ('over', over)])
    X_resampled, y_resampled = pipeline.fit_resample(X, y)
    return X_resampled, y_resampled
