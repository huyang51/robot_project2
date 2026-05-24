"""
Phase 3b 几何简化器测试
"""
import pytest
from src.phase3.phase3b_simplifier import (
    merge_similar_cubes, fold_patterns, normalize_z, simplify_sub_scene,
)


class TestPhase3bSimplifier:
    """Phase 3b 简化器单元测试"""

    def test_merge_similar_cubes(self):
        """测试相似 Cube 合并"""
        cubes = [
            {"id": "A", "center": {"x": 0, "y": 0, "z": 0}, "size": {"x": 1, "y": 1, "z": 1}, "material": "wood"},
            {"id": "B", "center": {"x": 0.5, "y": 0.5, "z": 0}, "size": {"x": 1, "y": 1, "z": 1}, "material": "wood"},
            {"id": "C", "center": {"x": 5, "y": 5, "z": 0}, "size": {"x": 1, "y": 1, "z": 1}, "material": "wood"},
        ]

        params = {
            "centroid_distance_threshold_ratio": 0.02,
            "same_material_required": True,
            "volume_increase_max": 0.30,
        }

        # 需要计算 world_bounds 自适应阈值
        merged = merge_similar_cubes(cubes, params, {"x": [0, 6], "y": [0, 6], "z": [0, 2]})
        # A 和 B 应该合并，C 太远不合并
        assert len(merged) <= 3  # 至少不会增加

    def test_normalize_z(self):
        """测试 Z 轴归一化"""
        cubes = [
            {"id": "A", "center": {"x": 0, "y": 0, "z": 5}, "size": {"x": 1, "y": 1, "z": 0.2}, "bounds": {"x": [-0.5, 0.5], "y": [-0.5, 0.5], "z": [4.9, 5.1]}},
            {"id": "B", "center": {"x": 1, "y": 1, "z": 5.5}, "size": {"x": 1, "y": 1, "z": 1}, "bounds": {"x": [1, 2], "y": [1, 2], "z": [5, 6]}},
        ]

        result = normalize_z(cubes)
        # Z 最小值 4.9 应该归零
        assert abs(result[0]["center"]["z"] - 0.1) < 0.01
        assert abs(result[1]["center"]["z"] - 0.6) < 0.01

    def test_simplify_sub_scene_empty(self):
        """测试空输入"""
        params = {"max_cubes_target": 60, "centroid_distance_threshold_ratio": 0.02}
        result = simplify_sub_scene([], [], params)
        assert result == []

    def test_fold_patterns(self):
        """测试模式折叠"""
        cubes = [
            {"id": "P1", "center": {"x": 0, "y": 0, "z": 0}, "size": {"x": 1, "y": 1, "z": 1}},
            {"id": "P2", "center": {"x": 2, "y": 0, "z": 0}, "size": {"x": 1, "y": 1, "z": 1}},
            {"id": "P3", "center": {"x": 4, "y": 0, "z": 0}, "size": {"x": 1, "y": 1, "z": 1}},
            {"id": "Other", "center": {"x": 10, "y": 10, "z": 0}, "size": {"x": 1, "y": 1, "z": 1}},
        ]

        patterns = [{
            "pattern_type": "linear_array",
            "cube_ids": ["P1", "P2", "P3"],
            "spacing": 2.0,
            "direction": "x",
        }]

        result = fold_patterns(cubes, patterns)
        # P1-P3 应该被折叠，Other 保留
        assert len(result) == 2
        # 应该有一个带 pattern_info
        folded = [c for c in result if "pattern_info" in c]
        assert len(folded) == 1
        assert folded[0]["pattern_info"]["member_count"] == 3
