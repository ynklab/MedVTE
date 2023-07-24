from typing import Dict

import datasets
import evaluate
import numpy as np
from sklearn.metrics import confusion_matrix


class ClassWiseAccuracy(datasets.metric.Metric):
    def _info(self):
        return evaluate.MetricInfo(
            description="",
            citation="",
            inputs_description="",
            features=datasets.Features(
                {
                    "predictions": datasets.Sequence(datasets.Value("int32")),
                    "references": datasets.Sequence(datasets.Value("int32")),
                }
                if self.config_name == "multilabel"
                else {
                    "predictions": datasets.Value("int32"),
                    "references": datasets.Value("int32"),
                }
            ),
            reference_urls=[],
        )

    def _compute(self, predictions, references, labels=None) -> Dict:
        matrix = confusion_matrix(references, predictions, labels=labels)
        n_labels = len(labels) if labels else matrix.shape[0]
        result = []
        for i in range(n_labels):
            tp = matrix[i, i]
            tn = matrix.sum() - matrix[:, i].sum() - matrix[i, :].sum() + matrix[i, i]
            acc = (tp + tn) / matrix.sum()
            result.append(acc)
        return {"classwise_accuracy": np.array(result)}
