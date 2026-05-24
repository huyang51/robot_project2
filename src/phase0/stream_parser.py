"""
Phase 0 流式 USDA 解析器

逐行扫描 USDA 文件，通过 brace counting 和 Prim 堆栈构建 Prim 树结构。
不一次性将整个文件加载到内存，支持 928MB+ 级别大文件。
"""

import logging
from typing import Iterator, List, Optional, Tuple

from ..core.usda_utils import (
    count_braces_close, net_braces, net_parens,
    parse_def_line,
    parse_extent, parse_vec3, parse_color3f, parse_matrix4d,
    build_transform_from_ops,
)
from ..core.types import PrimRecord, Vec3

logger = logging.getLogger(__name__)


class PrimStackEntry:
    """Prim 堆栈条目"""
    def __init__(self, prim_type: str, prim_path: str, parent_path: str, depth: int,
                 line_start: int):
        self.prim_type = prim_type
        self.prim_path = prim_path
        self.parent_path = parent_path
        self.depth = depth
        self.line_start = line_start
        self.transform: List[float] = []
        self.extent: Optional[List[float]] = None
        self.size: Optional[List[float]] = None
        self.properties: dict = {}
        self.material_binding: Optional[str] = None
        self.material_color: Optional[List[float]] = None
        # xformOp 分解（当没有 matrix4d 时使用）
        self._xform_translate: Optional[Tuple[float, float, float]] = None
        self._xform_rotate: Optional[Tuple[float, float, float]] = None
        self._xform_scale: Optional[Tuple[float, float, float]] = None
        self._lines: List[str] = []

    def add_line(self, line: str):
        self._lines.append(line)

    def feed_line(self, line: str):
        """处理一行内容，提取属性"""
        stripped = line.strip()
        self._lines.append(line)

        # 检查 material:binding
        if "material:binding" in stripped:
            # 格式: rel material:binding = </path/to/material>
            if "=" in stripped:
                rhs = stripped.split("=", 1)[1].strip()
                rhs = rhs.strip("<>/ ")
                self.material_binding = rhs

        # 检查 extent
        if "extent" in stripped and "=" in stripped:
            extent = parse_extent(stripped)
            if extent:
                self.extent = extent

        # 检查 size (double3)
        if "double3 size" in stripped or "float3 size" in stripped:
            size = parse_vec3(stripped)
            if size:
                self.size = list(size)

        # 检查变换：matrix4d 优先，否则收集 xformOp
        if "matrix4d" in stripped and "xformOp:transform" in stripped:
            self.transform = parse_matrix4d(stripped)
        elif "xformOp:translate" in stripped:
            v = parse_vec3(stripped)
            if v:
                self._xform_translate = v
        elif "xformOp:rotateXYZ" in stripped:
            v = parse_vec3(stripped)
            if v:
                self._xform_rotate = v
        elif "xformOp:scale" in stripped:
            v = parse_vec3(stripped)
            if v:
                self._xform_scale = v

        # 检查颜色
        if "color3f" in stripped and "diffuseColor" in stripped:
            color = parse_color3f(stripped)
            if color:
                self.material_color = list(color)

    def to_prim_record(self) -> PrimRecord:
        """转换为 PrimRecord"""
        identity = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]

        # 若没有 matrix4d 但有 xformOp，从 xformOp 构建矩阵
        if not self.transform and (self._xform_translate or self._xform_rotate or self._xform_scale):
            self.transform = build_transform_from_ops(
                translate=self._xform_translate,
                rotate_xyz=self._xform_rotate,
                scale=self._xform_scale,
            )

        return PrimRecord(
            prim_type=self.prim_type,
            prim_path=self.prim_path,
            parent_path=self.parent_path,
            extent=self.extent,
            size=Vec3(*self.size) if self.size else None,
            transform=self.transform if self.transform else identity,
            material_binding=self.material_binding,
            material_color=(
                None if self.material_color is None else
                Vec3(*self.material_color) if len(self.material_color) >= 3 else None
            ),
            depth=self.depth,
        )


def scan_usda(filepath: str) -> Iterator[Tuple[str, int]]:
    """逐行扫描 USDA 文件，yield (line, line_number)"""
    with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
        for i, line in enumerate(f, 1):
            yield line, i


def parse_usda_stream(filepath: str) -> List[PrimRecord]:
    """流式解析整个 USDA 文件，返回所有 PrimRecord

    算法：
    1. 逐行扫描
    2. brace counting 确定每个 def 块的边界
    3. Prim 堆栈跟踪层级关系
    4. 遇到 def Cube/Xform/Material 时创建 PrimStackEntry
    5. 块结束时弹出堆栈并生成 PrimRecord

    Returns:
        PrimRecord 列表（已解析属性，未计算 world_bbox）
    """
    prims: List[PrimRecord] = []
    stack: List[PrimStackEntry] = []
    brace_level = 0
    # 记录每个 def 块打开时的 brace_level，用于确定何时该块闭合
    stack_open_levels: List[int] = []
    # 元数据括号深度：def 行上的 ( customData = {...} ) 内的 { } 不计数
    paren_depth = 0

    for line, line_no in scan_usda(filepath):
        stripped = line.strip()

        # 跳过注释和空行
        if not stripped or stripped.startswith('#'):
            continue

        # 解析 def 声明（允许嵌套 def——检查是否以 def 开头即可）
        if stripped.startswith('def '):
            parsed = parse_def_line(stripped)
            if parsed:
                prim_type, prim_name = parsed
                # 计算路径
                if stack:
                    parent_path = stack[-1].prim_path
                else:
                    parent_path = "/"
                prim_path = f"{parent_path}/{prim_name}" if parent_path != "/" else f"/{prim_name}"

                depth = len(stack)
                entry = PrimStackEntry(
                    prim_type=prim_type,
                    prim_path=prim_path,
                    parent_path=parent_path,
                    depth=depth,
                    line_start=line_no,
                )
                stack.append(entry)
                stack_open_levels.append(brace_level)

        # 更新元数据括号深度（跟踪 ( ) 配对，不含字符串内和注释内的）
        paren_depth += net_parens(stripped)

        # 括号计数：仅在元数据括号外计数（paren_depth == 0 时 { } 才是结构括号）
        if paren_depth <= 0:
            net = net_braces(stripped)
        else:
            net = 0

        brace_level += net

        # 将当前行送入堆栈顶部的 entry
        if stack:
            stack[-1].feed_line(line)

        # 检查块是否结束：仅当本行有 } 时才检查（防止 def-only 行误触发闭合）
        # 使用 count_braces_close 而非 net < 0：单行自闭合 def（如 def Cube "x" {}）
        # 的 net==0 但确实需要弹出，因为 brace_level 已回到 open_level。
        brace_close_count = count_braces_close(stripped)
        if brace_close_count > 0:
            while stack and stack_open_levels and brace_level <= stack_open_levels[-1]:
                entry = stack.pop()
                stack_open_levels.pop()
                prims.append(entry.to_prim_record())

    logger.info(f"解析完成: {len(prims)} 个 Prim（{filepath}）")
    return prims
