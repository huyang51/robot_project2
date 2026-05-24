"""
USDA 文件解析工具函数

包含 brace counting, extent 解析, 变换矩阵累乘等底层操作。
"""

import math
import re
from typing import List, Optional, Tuple


def count_braces_open(line: str) -> int:
    """统计一行中 { 的数量（扣除字符串内的）"""
    return _count_outside_strings(line, '{')


def count_braces_close(line: str) -> int:
    """统计一行中 } 的数量（扣除字符串内的）"""
    return _count_outside_strings(line, '}')


def _count_outside_strings(line: str, char: str) -> int:
    """统计字符出现次数，忽略字符串字面量内以及 # 注释内的字符"""
    count = 0
    in_string = False
    string_char = None
    in_comment = False
    i = 0
    while i < len(line):
        c = line[i]
        if in_comment:
            # 注释持续到行尾，后续字符全部忽略
            break
        if not in_string:
            if c == '#':
                in_comment = True
                continue
            if c in ('"', "'"):
                in_string = True
                string_char = c
            elif c == char:
                count += 1
        else:
            if c == '\\':
                i += 1  # 跳过转义字符
            elif c == string_char:
                in_string = False
                string_char = None
        i += 1
    return count


def net_braces(line: str) -> int:
    """计算一行的净括号数: { 为正, } 为负"""
    return count_braces_open(line) - count_braces_close(line)


def count_paren_open(line: str) -> int:
    """统计一行中 ( 的数量（扣除字符串和注释内的）"""
    return _count_outside_strings(line, '(')


def count_paren_close(line: str) -> int:
    """统计一行中 ) 的数量（扣除字符串和注释内的）"""
    return _count_outside_strings(line, ')')


def net_parens(line: str) -> int:
    """计算一行的净括号数: ( 为正, ) 为负，用于跟踪元数据括号深度"""
    return count_paren_open(line) - count_paren_close(line)


def _identity_matrix() -> List[float]:
    return [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]


def parse_matrix4d(line: str) -> List[float]:
    """从 matrix4d 行解析 4x4 变换矩阵（16 个 float）

    仅提取括号内的数值——避免 "matrix4d" 中的 "4" 被误提取。
    """
    start = line.find("(")
    end = line.rfind(")")
    if start == -1 or end == -1:
        return _identity_matrix()
    paren_content = line[start:end + 1]
    values = extract_float_list(paren_content)
    if len(values) >= 16:
        return values[:16]
    return _identity_matrix()


def multiply_matrices(a: List[float], b: List[float]) -> List[float]:
    """两个 4x4 矩阵相乘（row-major）: a * b"""
    result = [0.0] * 16
    for row in range(4):
        for col in range(4):
            total = 0.0
            for k in range(4):
                total += a[row * 4 + k] * b[k * 4 + col]
            result[row * 4 + col] = total
    return result


def transform_point(matrix: List[float], point: Tuple[float, float, float]) -> Tuple[float, float, float]:
    """用 4x4 变换矩阵变换一个 3D 点"""
    x, y, z = point
    w = matrix[12] * x + matrix[13] * y + matrix[14] * z + matrix[15]
    if abs(w) < 1e-10:
        w = 1.0
    nx = (matrix[0] * x + matrix[1] * y + matrix[2] * z + matrix[3]) / w
    ny = (matrix[4] * x + matrix[5] * y + matrix[6] * z + matrix[7]) / w
    nz = (matrix[8] * x + matrix[9] * y + matrix[10] * z + matrix[11]) / w
    return (nx, ny, nz)


def extract_float_list(token_str: str) -> List[float]:
    """从 USDS 格式的浮点列表字符串中提取所有 float 值"""
    # 匹配浮点数（含负号、科学记数法）
    pattern = r'-?\d+\.?\d*(?:[eE][+-]?\d+)?'
    return [float(m) for m in re.findall(pattern, token_str)]


def parse_extent(line: str) -> Optional[List[float]]:
    """从一行 USDA 中解析 extent 属性

    USDA 格式: float3[] extent = [(x_min, y_min, z_min), (x_max, y_max, z_max)]
    返回 BBox 兼容格式: [x_min, x_max, y_min, y_max, z_min, z_max]

    注意: 不能直接用 extract_float_list——"float3" 中的 "3" 会被误提取。
    改为仅提取方括号内的数值。
    """
    # 定位 [( ... )] 部分
    start = line.find("[(")
    end = line.rfind(")]")
    if start == -1 or end == -1 or end <= start:
        return None
    bracket_content = line[start:end + 2]  # 取 [(...)]
    floats = extract_float_list(bracket_content)
    if len(floats) >= 6:
        x_min, y_min, z_min, x_max, y_max, z_max = floats[:6]
        return [x_min, x_max, y_min, y_max, z_min, z_max]
    return None


def parse_vec3(line: str) -> Optional[Tuple[float, float, float]]:
    """从一行中解析 float3 值

    仅提取括号内的数值——避免 "float3" 中的 "3" 被误提取。
    例如: float3 xformOp:translate = (-1384.4, 0.0, -264.0)
    """
    # 定位 (...) 部分
    start = line.find("(")
    end = line.rfind(")")
    if start == -1 or end == -1 or end <= start:
        return None
    paren_content = line[start:end + 1]
    floats = extract_float_list(paren_content)
    if len(floats) >= 3:
        return (floats[0], floats[1], floats[2])
    return None


def parse_color3f(line: str) -> Optional[Tuple[float, float, float]]:
    """从一行中解析 color3f 值"""
    return parse_vec3(line)


def extract_prim_name(prim_path: str) -> str:
    """从 Prim 路径中提取名称（最后一个 / 之后的部分）"""
    return prim_path.rstrip("/").split("/")[-1] if prim_path else ""


def extract_parent_path(prim_path: str) -> str:
    """从 Prim 路径中提取父路径"""
    if not prim_path:
        return "/"
    stripped = prim_path.rstrip("/")
    idx = stripped.rfind("/")
    if idx <= 0:
        return "/"
    return stripped[:idx]


PRIM_TYPE_PATTERN = re.compile(r'def\s+(\w+)\s+"([^"]*)"')


def parse_def_line(line: str) -> Optional[Tuple[str, str]]:
    """解析 def 行，返回 (prim_type, prim_name)"""
    m = PRIM_TYPE_PATTERN.search(line)
    if m:
        return (m.group(1), m.group(2))
    return None


def build_transform_from_ops(
    translate: Optional[Tuple[float, float, float]] = None,
    rotate_xyz: Optional[Tuple[float, float, float]] = None,
    scale: Optional[Tuple[float, float, float]] = None,
) -> List[float]:
    """从 xformOp 构建 4x4 行优先矩阵

    当前仅支持标准 xformOpOrder: [translate, rotateXYZ, scale]。
    非标准顺序（如 [scale, rotateXYZ, translate]）暂不支持，
    如遇此类 USDA 文件需扩展此函数。
    组合矩阵: M = S * R * T (translate 最先作用于点)
    """
    tx, ty, tz = translate if translate else (0.0, 0.0, 0.0)
    rx, ry, rz = rotate_xyz if rotate_xyz else (0.0, 0.0, 0.0)
    scl_x, scl_y, scl_z = scale if scale else (1.0, 1.0, 1.0)

    # 旋转矩阵 (XYZ Euler, 度→弧度)
    cx, sx = math.cos(math.radians(rx)), math.sin(math.radians(rx))
    cy, sy = math.cos(math.radians(ry)), math.sin(math.radians(ry))
    cz, sz = math.cos(math.radians(rz)), math.sin(math.radians(rz))

    # R = Rz * Ry * Rx
    r00 = cy * cz
    r01 = cz * sy * sx - cx * sz
    r02 = cx * cz * sy + sx * sz
    r10 = cy * sz
    r11 = cx * cz + sy * sx * sz
    r12 = -cz * sx + cx * sy * sz
    r20 = -sy
    r21 = cy * sx
    r22 = cx * cy

    # M = S * R * T (4x4 行优先)
    return [
        scl_x * r00, scl_x * r01, scl_x * r02, scl_x * (r00 * tx + r01 * ty + r02 * tz),
        scl_y * r10, scl_y * r11, scl_y * r12, scl_y * (r10 * tx + r11 * ty + r12 * tz),
        scl_z * r20, scl_z * r21, scl_z * r22, scl_z * (r20 * tx + r21 * ty + r22 * tz),
        0.0, 0.0, 0.0, 1.0,
    ]


def read_usda_header(filepath: str, max_lines: int = 50) -> dict:
    """从 USDA 文件头部读取元数据

    扫描文件头部（前 max_lines 行），提取 USD stage 元数据。
    关键字段：
    - metersPerUnit: 单位缩放（1=米, 0.01=厘米）
    - upAxis: 垂直轴方向（"Z" 或 "Y"）

    Returns:
        {"metersPerUnit": float, "upAxis": str}
    """
    import re
    result = {"metersPerUnit": 0.01, "upAxis": "Y"}  # USD 默认值

    with open(filepath, "r", encoding="utf-8", errors="replace") as f:
        for i, line in enumerate(f):
            if i >= max_lines:
                break
            # 匹配: metersPerUnit = <float>
            m = re.search(r'metersPerUnit\s*=\s*([\d.]+)', line)
            if m:
                result["metersPerUnit"] = float(m.group(1))
            # 匹配: upAxis = "Z" 或 "Y"
            m = re.search(r'upAxis\s*=\s*"([ZY])"', line)
            if m:
                result["upAxis"] = m.group(1)

    return result
