class AdaptiveThresholdController:
    """
    Online controller: adjusts decision threshold every step based on
    rolling false-positive / false-negative rates.
    - FP rate too high → raise threshold (reduce resource waste)
    - FN rate too high → lower threshold (catch more cold starts)
    """
    def __init__(self, base_threshold=0.5, window=30, alpha=0.03,
                 fp_limit=0.25, fn_limit=0.25):
        self.threshold = base_threshold
        self.window    = window
        self.alpha     = alpha
        self.fp_limit  = fp_limit
        self.fn_limit  = fn_limit
        self._history  = []

    def decide(self, prob: float) -> int:
        return int(prob >= self.threshold)

    def update(self, pred: int, actual: int):
        self._history.append((pred, actual))
        if len(self._history) > self.window:
            self._history.pop(0)
        if len(self._history) < 10:
            return
        preds, actuals = zip(*self._history)
        n_neg   = max(1, sum(a == 0 for a in actuals))
        n_pos   = max(1, sum(a == 1 for a in actuals))
        fp_rate = sum(p==1 and a==0 for p,a in zip(preds,actuals)) / n_neg
        fn_rate = sum(p==0 and a==1 for p,a in zip(preds,actuals)) / n_pos
        if fp_rate > self.fp_limit:
            self.threshold = min(0.85, self.threshold + self.alpha)
        elif fn_rate > self.fn_limit:
            self.threshold = max(0.15, self.threshold - self.alpha)
