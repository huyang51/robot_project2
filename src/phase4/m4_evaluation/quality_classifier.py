"""
M4 质量分级器

根据 EvalResult 判定 H/M/L 等级，检查否决条件。
"""

from typing import Dict, Any, Optional

from ...config import M4_EVALUATION_PARAMS
from .eval_schema import EvalResult, DimensionScore


def classify_quality(
    eval_result: EvalResult,
    params: Optional[Dict[str, Any]] = None,
) -> str:
    """根据评估结果判定质量等级

    规则:
    1. 先检查否决条件 → L
    2. 任何单一维度 < 4.0 → L
    3. H: Q >= 8.0 且 军事可行性 >= 7.0
    4. M: 6.0 <= Q < 8.0
    5. L: Q < 6.0
    """
    params = params or M4_EVALUATION_PARAMS
    thresholds = params.get("quality_thresholds", {"H": 8.0, "M": 6.0})
    military_min = params.get("military_min_for_H", 7.0)
    single_min = params.get("single_dimension_min", 4.0)
    vetoes = params.get("veto_thresholds", {})

    # 否决条件
    if eval_result.veto_triggered:
        return "L"

    # 单项最低分检查
    for dim_name, dim_data in eval_result.scores.items():
        dim_score = dim_data.score if isinstance(dim_data, DimensionScore) else float(dim_data)
        if dim_score < single_min:
            return "L"

    # 否决阈值检查
    def _get_score(obj) -> float:
        return obj.score if isinstance(obj, DimensionScore) else float(obj)

    military = eval_result.scores.get("military_feasibility")
    if military and _get_score(military) < vetoes.get("V1_military", 3.0):
        return "L"

    scene = eval_result.scores.get("scene_adaptation")
    if scene and _get_score(scene) < vetoes.get("V2_scene_adaptation", 3.0):
        return "L"

    granularity = eval_result.scores.get("granularity_compliance")
    if granularity and _get_score(granularity) < vetoes.get("V3_granularity", 3.0):
        return "L"

    q = eval_result.overall_score

    if q >= thresholds["H"]:
        if military and _get_score(military) >= military_min:
            return "H"
        return "M"  # 综合分够高但军事分不够

    if q >= thresholds["M"]:
        return "M"

    return "L"
