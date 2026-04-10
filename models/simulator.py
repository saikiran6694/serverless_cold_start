import pandas as pd
import numpy as np
from models.adaptive_threshold_controller import AdaptiveThresholdController


def simulate(test_data: pd.DataFrame, hybrid_probs: np.ndarray,
             y_test: np.ndarray):
    """
    Compare No Warming / Fixed (0.5) / Adaptive on the test day.
    For each minute, we have:
      - actual: whether a cold start occurred (1) or not (0)
      - prob: model's predicted probability of a cold start
    Strategies:
    1. No Warming (Baseline): never warm; cold starts = actual cold starts
    2. Fixed Threshold: warm if prob >= 0.5; cold starts = 0 if (warmed and actual) else actual
    3. Adaptive: use controller to decide; cold starts = 0 if (warmed and actual) else actual
    
    The controller updates its threshold after each prediction based on the outcome, aiming to minimize cold starts while avoiding unnecessary warmings.
    Returns a summary DataFrame comparing the strategies, and detailed DataFrames for the adaptive and fixed strategies.
    """
    controller = AdaptiveThresholdController()
    rows_adap, rows_fixed, rows_none = [], [], []

    for prob, actual, row in zip(hybrid_probs, y_test, test_data.itertuples()):
        pred_a = controller.decide(prob)
        cold_a = 0 if (pred_a == 1 and actual == 1) else int(actual == 1)
        controller.update(pred_a, actual)

        pred_f = int(prob >= 0.5)
        cold_f = 0 if (pred_f == 1 and actual == 1) else int(actual == 1)

        rows_adap.append({'day': row.day, 'minute': row.minute,
                          'threshold': controller.threshold,
                          'warmed': pred_a, 'cold_start': cold_a, 'prob': prob})
        rows_fixed.append({'warmed': pred_f, 'cold_start': cold_f})
        rows_none.append({'cold_start': int(actual == 1)})

    adap  = pd.DataFrame(rows_adap)
    fixed = pd.DataFrame(rows_fixed)
    none_ = pd.DataFrame(rows_none)

    summary = pd.DataFrame({
        'Strategy': ['No Warming (Baseline)',
                     'Fixed Threshold (0.5)',
                     'Adaptive (Proposed)'],
        'Cold Start Rate': [none_['cold_start'].mean(),
                            fixed['cold_start'].mean(),
                            adap['cold_start'].mean()],
        'Warmings': [0, int(fixed['warmed'].sum()), int(adap['warmed'].sum())],
    })
    summary['Reduction vs Baseline'] = (
        1 - summary['Cold Start Rate'] / summary['Cold Start Rate'].iloc[0]
    )
    return summary, adap, fixed, none_