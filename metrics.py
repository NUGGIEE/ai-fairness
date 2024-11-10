import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score

def calculate_demographic_parity(y_true, y_pred, protected_attribute):
    df = pd.DataFrame({'y_true': y_true, 'y_pred': y_pred, 'protected': protected_attribute})
    positive_rate = df.groupby('protected')['y_pred'].mean()
    disparity = abs(positive_rate.iloc[0] - positive_rate.iloc[1])
    return disparity

def calculate_equalized_odds(y_true, y_pred, protected_attribute):
    df = pd.DataFrame({'y_true': y_true, 'y_pred': y_pred, 'protected': protected_attribute})
    odds_metrics = {}
    for group in df['protected'].unique():
        group_df = df[df['protected'] == group]
        tn, fp, fn, tp = confusion_matrix(group_df['y_true'], group_df['y_pred']).ravel()
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
        odds_metrics[group] = {'FPR': fpr, 'FNR': fnr}
    return odds_metrics
