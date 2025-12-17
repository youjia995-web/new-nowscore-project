from typing import Dict, Any, Optional
from ..services.calibration import CalibrationService


class BayesParamsStore:
    """存取分层贝叶斯泊松参数（团队攻防、主场优势、基础率、分布参数等）。
    参数以键 'bayes_hier_params_v1' 存于 calibration_params 表。
    """

    DEFAULT_KEY = "bayes_hier_params_v1"

    def __init__(self, key: Optional[str] = None):
        self.key = key or self.DEFAULT_KEY
        self.calib = CalibrationService()

    def load(self) -> Optional[Dict[str, Any]]:
        params = self.calib.get(self.key)
        if not params:
            return None
        return params

    def save(self, params: Dict[str, Any]) -> None:
        self.calib.set(self.key, params)