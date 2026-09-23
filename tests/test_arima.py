import numpy as np
import pandas as pd

from eval.baseline_arima import validation_order_histories
from utils.dataset import DataConfig


def test_arima_auto_order_samples_validation_history_only():
    values = np.arange(100.0)
    index = pd.date_range('2024-01-01', periods=100, freq='h')
    config = DataConfig(seq_len=3, pred_len=2, dedup_policy='keep',
                        zero_run_clean=False, clip_sigma=0.0)

    histories = validation_order_histories(values, index, config, history=5, limit=2)

    # Validation starts at row 60. Input context must precede each forecast origin;
    # no order-selection history may include a test row (row 80 onward).
    assert len(histories) == 2
    np.testing.assert_array_equal(histories[0], values[58:63])
    np.testing.assert_array_equal(histories[1], values[59:64])
