"""
Phase 0 模式检测器测试
"""
import pytest
from src.core.types import PrimRecord, BBox, TacticalType, Vec3
from src.phase0.pattern_detector import (
    detect_linear_arrays, detect_symmetric_pairs, detect_dense_clusters,
    detect_all_patterns, PatternMatch,
)


def make_prim(path: str, x: float, y: float, z: float,
              sx: float = 1.0, sy: float = 1.0, sz: float = 1.0,
              ttype: TacticalType = TacticalType.UNKNOWN) -> PrimRecord:
    """创建测试用 PrimRecord"""
    return PrimRecord(
        prim_type="Cube",
        prim_path=path,
        parent_path="/World",
        world_bbox=BBox(
            x - sx / 2, x + sx / 2,
            y - sy / 2, y + sy / 2,
            z - sz / 2, z + sz / 2,
        ),
        tactical_type=ttype,
    )


class TestPatternDetector:
    """模式检测器单元测试"""

    def test_linear_array_detection(self):
        """测试等距线性排列检测"""
        prims = [
            make_prim("/World/Pillar_1", 0, 0, 0, 0.3, 0.3, 3.0, TacticalType.PILLAR),
            make_prim("/World/Pillar_2", 2, 0, 0, 0.3, 0.3, 3.0, TacticalType.PILLAR),
            make_prim("/World/Pillar_3", 4, 0, 0, 0.3, 0.3, 3.0, TacticalType.PILLAR),
            make_prim("/World/Pillar_4", 6, 0, 0, 0.3, 0.3, 3.0, TacticalType.PILLAR),
        ]

        results = detect_linear_arrays(prims, max_spacing_deviation=0.2)
        assert len(results) >= 1
        assert results[0].pattern_type == "linear_array"
        assert results[0].direction == "x"
        assert len(results[0].cube_ids) >= 3

    def test_symmetric_pair_detection(self):
        """测试对称对检测"""
        prims = [
            make_prim("/World/Win_L", -1.5, 0, 1.5, 0.8, 0.1, 1.2, TacticalType.WINDOW),
            make_prim("/World/Win_R", 1.5, 0, 1.5, 0.8, 0.1, 1.2, TacticalType.WINDOW),
        ]

        results = detect_symmetric_pairs(prims, axis="x", tolerance=0.5)
        assert len(results) >= 1
        assert results[0].pattern_type == "symmetric_pair"

    def test_dense_cluster_detection(self):
        """测试密集簇检测"""
        prims = [
            make_prim("/World/Obj_1", 0, 0, 0, 0.5, 0.5, 0.5),
            make_prim("/World/Obj_2", 0.3, 0.3, 0, 0.5, 0.5, 0.5),
            make_prim("/World/Obj_3", -0.3, -0.3, 0, 0.5, 0.5, 0.5),
            make_prim("/World/Obj_4", 0, 0.3, 0, 0.5, 0.5, 0.5),
        ]

        results = detect_dense_clusters(prims, distance_threshold=1.0, min_cluster_size=3)
        assert len(results) >= 1
        assert results[0].pattern_type == "dense_cluster"

    def test_no_pattern_on_sparse_data(self):
        """稀疏数据不产生误检 — 使用不同类型和不规则位置"""
        prims = [
            make_prim("/World/A", 0, 0, 0, 1, 1, 1, TacticalType.WALL),
            make_prim("/World/B", 7, 12, 1.7, 1, 1, 1, TacticalType.COVER),
            make_prim("/World/C", -8, -5, 4.2, 1, 1, 1, TacticalType.PILLAR),
        ]

        results = detect_all_patterns(prims)
        assert len(results["linear_arrays"]) == 0
        # 密集簇需要至少 4 个
        assert len(results["dense_clusters"]) == 0
