"""
Phase 0 流式解析器测试
"""
import pytest
from pathlib import Path
import tempfile

from src.phase0.stream_parser import parse_usda_stream, scan_usda
from src.core.types import PrimRecord


SAMPLE_USDA = '''#usda 1.0

def Xform "World"
{
    def Cube "Wall_01"
    {
        double3 extent = [(-1.0, -0.1, -1.5), (1.0, 0.1, 1.5)]
        matrix4d xformOp:transform = (1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1)
    }

    def Cube "Floor_01"
    {
        double3 extent = [(-5.0, -5.0, -0.1), (5.0, 5.0, 0.1)]
        matrix4d xformOp:transform = (1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1)
    }
}
'''


class TestStreamParser:
    """流式解析器单元测试"""

    def test_parse_simple_usda(self):
        """测试解析简单 USDA"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.usda', delete=False, encoding='utf-8') as f:
            f.write(SAMPLE_USDA)
            tmp_path = f.name

        try:
            prims = parse_usda_stream(tmp_path)
            # 解析器返回所有 Prim（含 Xform），筛选出 Cube
            cubes = [p for p in prims if p.prim_type == "Cube"]
            assert len(cubes) == 2
            assert any("Wall_01" in c.prim_path for c in cubes)
            assert any("Floor_01" in c.prim_path for c in cubes)
        finally:
            Path(tmp_path).unlink()

    def test_extent_parsing(self):
        """测试 extent 属性解析"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.usda', delete=False, encoding='utf-8') as f:
            f.write(SAMPLE_USDA)
            tmp_path = f.name

        try:
            prims = parse_usda_stream(tmp_path)
            wall = [p for p in prims if "Wall" in p.prim_path][0]
            assert wall.extent is not None
            assert len(wall.extent) == 6
        finally:
            Path(tmp_path).unlink()

    def test_scan_empty_file(self):
        """测试扫描空文件"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.usda', delete=False, encoding='utf-8') as f:
            f.write('')
            tmp_path = f.name

        try:
            lines = list(scan_usda(tmp_path))
            assert len(lines) == 0
        finally:
            Path(tmp_path).unlink()
