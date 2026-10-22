"""Runtime repair for the RF-DETR 1.3.0 zero-AP checkpoint edge case.

Vendored upstream stays byte-identical. Accept the first metric observation;
all subsequent comparisons retain upstream behavior.
"""
from pathlib import Path
from common import activate_upstream, sha256


def apply_checkpoint_safety():
    activate_upstream()
    from rfdetr.util.utils import BestMetricSingle
    if not getattr(BestMetricSingle, "_first_observation_patch", False):
        original_update = BestMetricSingle.update

        def update(self, new_res, ep):
            if self.best_ep < 0:
                self.best_res, self.best_ep = new_res, ep
                return True
            return original_update(self, new_res, ep)

        BestMetricSingle.update = update
        BestMetricSingle._first_observation_patch = True
    return {"id": "rfdetr-1.3.0-first-validation-checkpoint", "file_sha256": sha256(Path(__file__)),
            "scope": "accept first metric even when AP=0; preserve later upstream comparisons"}
