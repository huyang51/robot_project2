"""
Phase 3 输出自动校验

desc.json 基础 Schema 验证 + scene.usda 交叉校验。
从 robot_project core/phase3c_schema.py 适配。
"""

from typing import List, Dict, Any

from .schemas import COVER_QUALITY_ENUM, THREAT_SEVERITY_GRADES


def validate_desc_json_basic(desc: Dict[str, Any]) -> List[Dict[str, Any]]:
    """desc.json 基础 Schema 校验"""
    warnings = []

    required_fields = [
        "sub_scene_id", "tactical_role", "task_hint",
        "zones", "openings", "cover_assessment", "inferred_threats",
        "movement_analysis", "exposure_assessment", "tactical_boundary",
        "spatial_description", "inferred_tags",
    ]
    for field in required_fields:
        if field not in desc:
            warnings.append({
                "level": "error",
                "field": field,
                "message": f"缺少必填字段 '{field}'",
            })

    if warnings:
        return warnings

    # zones 约束
    zones = desc.get("zones", [])
    if len(zones) < 1:
        warnings.append({
            "level": "error", "field": "zones",
            "message": "zones 至少需要1个区域",
        })

    zone_ids = {z.get("zone_id") for z in zones}

    # openings 引用检查
    for op in desc.get("openings", []):
        for conn in op.get("connects", []):
            if conn not in zone_ids and not conn.startswith("EXTERIOR") and not conn.startswith("ROOM_"):
                warnings.append({
                    "level": "warning",
                    "field": f"openings.{op.get('id', '?')}.connects",
                    "message": f"引用的 zone_id '{conn}' 不在 zones 列表中",
                })

    # cover_assessment 检查
    for cv in desc.get("cover_assessment", []):
        quality = cv.get("quality", "")
        if quality not in COVER_QUALITY_ENUM:
            warnings.append({
                "level": "warning",
                "field": f"cover_assessment.{cv.get('cube_id', '?')}.quality",
                "message": f"quality '{quality}' 不在标准枚举中",
            })

    # inferred_threats 约束
    threats = desc.get("inferred_threats", [])
    if len(threats) < 1:
        warnings.append({
            "level": "error", "field": "inferred_threats",
            "message": "inferred_threats 至少需要1条",
        })
    for t in threats:
        if t.get("severity", "") not in THREAT_SEVERITY_GRADES:
            warnings.append({
                "level": "warning",
                "field": f"inferred_threats.severity",
                "message": f"无效 severity '{t.get('severity')}'",
            })

    # movement_analysis 检查
    ma = desc.get("movement_analysis", {})
    kcps = ma.get("key_control_points", [])
    if len(kcps) < 1:
        warnings.append({
            "level": "error",
            "field": "movement_analysis.key_control_points",
            "message": "至少需要1个关键控制点",
        })

    # spatial_description 长度检查
    sd = desc.get("spatial_description", "")
    if len(sd) < 50:
        warnings.append({
            "level": "warning",
            "field": "spatial_description",
            "message": f"spatial_description 过短（{len(sd)}字）",
        })

    # effective_positions 检查
    for cv in desc.get("cover_assessment", []):
        ep = cv.get("effective_positions", [])
        if not ep:
            warnings.append({
                "level": "warning",
                "field": f"cover_assessment.{cv.get('cube_id', '?')}.effective_positions",
                "message": "effective_positions 为空，每个掩体至少需要1个有效位置",
            })

    # exposure_assessment 结构检查
    for ea in desc.get("exposure_assessment", []):
        for key in ["from_position", "to_position", "exposed_to", "exposure_time_category"]:
            if key not in ea:
                warnings.append({
                    "level": "warning",
                    "field": f"exposure_assessment.{ea.get('from_position', '?')}.{key}",
                    "message": f"缺少字段 '{key}'",
                })
        etc = ea.get("exposure_time_category", "")
        if etc and etc not in ("brief", "medium", "prolonged"):
            warnings.append({
                "level": "warning",
                "field": f"exposure_assessment.exposure_time_category",
                "message": f"无效值 '{etc}'，应为 brief/medium/prolonged",
            })

    # tactical_boundary 结构检查
    tb = desc.get("tactical_boundary", {})
    if isinstance(tb, dict):
        if not tb.get("entry_points"):
            warnings.append({
                "level": "warning",
                "field": "tactical_boundary.entry_points",
                "message": "entry_points 为空",
            })
        if not tb.get("objective_criteria"):
            warnings.append({
                "level": "warning",
                "field": "tactical_boundary.objective_criteria",
                "message": "objective_criteria 为空",
            })
        for ep in tb.get("entry_points", []):
            for key in ["id", "zone_id", "opening_id"]:
                if key not in ep:
                    warnings.append({
                        "level": "warning",
                        "field": f"tactical_boundary.entry_points.{key}",
                        "message": f"缺少字段 '{key}'",
                    })

    # inferred_tags 数量检查
    tags = desc.get("inferred_tags", [])
    if len(tags) < 3:
        warnings.append({
            "level": "warning",
            "field": "inferred_tags",
            "message": f"inferred_tags 只有{len(tags)}个，建议3-8个",
        })

    return warnings


def validate_sub_scene_completeness(
    desc: Dict[str, Any],
    scene_cubes: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    """子场景完整性综合校验

    scene_cubes 可以是 Dict[str, Dict]（以 cube_id 为 key 的索引）
    或 List[Dict]（cube 列表）。同时接受 int 表示 cube 数量。
    """
    # 兼容多种输入格式
    cube_count = _get_cube_count(scene_cubes)

    # 硬性检查：空子场景直接失败
    precheck_errors = []
    if cube_count == 0:
        precheck_errors.append({
            "level": "error",
            "field": "scene_cubes",
            "message": "子场景无任何几何数据（cube_count=0），裁切可能失败。"
                       "检查 spatial_bounds 坐标轴是否与 cube 数据匹配。",
        })

    basic_warnings = validate_desc_json_basic(desc)

    # 交叉校验仅在 scene_cubes 为可迭代对象时执行
    if isinstance(scene_cubes, (dict, list)) and len(scene_cubes) > 0:
        scene_warnings = _validate_with_scene(desc, scene_cubes)
    else:
        scene_warnings = []

    all_issues = precheck_errors + basic_warnings + scene_warnings
    errors = [w for w in all_issues if w.get("level") == "error"]

    return {
        "passed": len(errors) == 0,
        "cube_count": cube_count,
        "errors": errors,
        "warnings": [w for w in all_issues if w.get("level") == "warning"],
        "validation_failed": len(errors) > 0,
    }


def _get_cube_count(scene_cubes) -> int:
    """从多种输入格式中提取 cube 数量"""
    if isinstance(scene_cubes, dict):
        # Dict 索引格式
        return len(scene_cubes)
    elif isinstance(scene_cubes, list):
        return len(scene_cubes)
    elif isinstance(scene_cubes, int):
        return scene_cubes
    return 0


def _validate_with_scene(
    desc: Dict[str, Any],
    scene_cubes: Dict[str, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """desc.json 与 scene_cubes 交叉校验"""
    warnings = []

    # zone 内 cube_id 坐标范围检查
    for z in desc.get("zones", []):
        for cube_id in z.get("contained_cube_ids", []):
            if cube_id not in scene_cubes:
                warnings.append({
                    "level": "warning",
                    "field": f"zones.{z.get('zone_id')}.contained_cube_ids",
                    "message": f"cube_id '{cube_id}' 不在 scene_cubes 中",
                })

    # cover_assessment 中 cube_id 存在性
    for cv in desc.get("cover_assessment", []):
        cube_id = cv.get("cube_id", "")
        if cube_id and cube_id not in scene_cubes:
            warnings.append({
                "level": "warning",
                "field": f"cover_assessment.{cube_id}",
                "message": f"cube_id '{cube_id}' 不在 scene_cubes 中",
            })

    return warnings
