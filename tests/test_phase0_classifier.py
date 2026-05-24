"""
Phase 0 启发式分类器测试
"""
import pytest
from src.core.types import PrimRecord, BBox, TacticalType, Vec3
from src.phase0.heuristic_classifier import HeuristicClassifier


def make_prim(path: str, bbox: BBox, ttype: TacticalType = TacticalType.UNKNOWN) -> PrimRecord:
    """创建测试用 PrimRecord"""
    return PrimRecord(
        prim_type="Cube",
        prim_path=path,
        parent_path="/World",
        world_bbox=bbox,
        tactical_type=ttype,
    )


class TestHeuristicClassifier:
    """启发式分类器单元测试"""

    def test_wall_detection(self):
        """R1: 墙体检测"""
        classifier = HeuristicClassifier()

        # 典型墙体: 高2.5m x 宽4m x 厚0.2m
        wall_bbox = BBox(-2, 2, -0.1, 0.1, 0, 2.5)
        wall = make_prim("/World/Wall", wall_bbox)
        assert classifier._is_wall(wall)

        # 太矮不是墙
        short_bbox = BBox(-2, 2, -0.1, 0.1, 0, 1.0)
        short = make_prim("/World/Short", short_bbox)
        assert not classifier._is_wall(short)

    def test_floor_detection(self):
        """R2: 楼板检测"""
        classifier = HeuristicClassifier()

        # 典型楼板: 10x10m 面积, 0.2m 厚
        floor_bbox = BBox(-5, 5, -5, 5, -0.1, 0.1)
        floor = make_prim("/World/Floor", floor_bbox)
        assert classifier._is_floor(floor)

    def test_door_detection(self):
        """R3: 门检测"""
        classifier = HeuristicClassifier()
        classifier.floor_z_levels = [0.0]  # 模拟地面在 Z=0

        # 典型门: 高2.1m x 宽0.9m x 厚0.1m, 底在 Z=0
        door_bbox = BBox(-0.45, 0.45, -0.05, 0.05, 0, 2.1)
        door = make_prim("/World/Door", door_bbox)
        assert classifier._is_door(door)

    def test_decor_filter(self):
        """R8: 装饰过滤"""
        classifier = HeuristicClassifier()

        # 极小物体
        tiny_bbox = BBox(-0.01, 0.01, -0.01, 0.01, -0.01, 0.01)
        tiny = make_prim("/World/Tiny", tiny_bbox)
        assert classifier._is_decor(tiny)

        # 正常大小的物体不应被判为装饰
        normal_bbox = BBox(-0.5, 0.5, -0.5, 0.5, 0, 1.0)
        normal = make_prim("/World/Normal", normal_bbox)
        assert not classifier._is_decor(normal)

    def test_classify_all(self):
        """集成测试：分类多个 Prim"""
        classifier = HeuristicClassifier()

        prims = [
            make_prim("/World/Wall_1", BBox(-3, 3, -0.15, 0.15, 0, 3.0)),
            make_prim("/World/Wall_2", BBox(-3, 3, -0.15, 0.15, 0, 3.0)),
            make_prim("/World/Floor_1", BBox(-5, 5, -5, 5, -0.1, 0.1)),
            make_prim("/World/Tiny", BBox(-0.01, 0.01, -0.01, 0.01, 0.5, 0.52)),
        ]

        classifier.classify_all(prims)

        types = [p.tactical_type for p in prims]
        assert TacticalType.WALL in types
        assert TacticalType.FLOOR in types
        assert TacticalType.DECOR in types

        stats = classifier.get_statistics(prims)
        assert stats["total_cubes"] == 4
