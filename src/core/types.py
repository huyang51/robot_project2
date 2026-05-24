"""
GTKG-CM 核心类型定义

包含 Vec3, BBox, TacticalType 枚举, PrimRecord 等基础 dataclass。
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any
from enum import Enum


class TacticalType(Enum):
    """启发式分类器输出的战术类型枚举"""
    WALL = "wall"
    FLOOR = "floor"
    DOOR = "door"
    WINDOW = "window"
    STAIR = "stair"
    COVER = "cover"
    PILLAR = "pillar"
    DECOR = "decor"
    UNKNOWN = "unknown"


# 9 规则优先级（数字越小优先级越高）
TACTICAL_TYPE_PRIORITY = {
    TacticalType.STAIR: 1,
    TacticalType.DOOR: 2,
    TacticalType.WINDOW: 3,
    TacticalType.WALL: 4,
    TacticalType.FLOOR: 5,
    TacticalType.PILLAR: 6,
    TacticalType.COVER: 7,
    TacticalType.DECOR: 8,
    TacticalType.UNKNOWN: 9,
}


@dataclass
class Vec3:
    """三维向量"""
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0

    def __add__(self, other: "Vec3") -> "Vec3":
        return Vec3(self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other: "Vec3") -> "Vec3":
        return Vec3(self.x - other.x, self.y - other.y, self.z - other.z)

    def __mul__(self, scalar: float) -> "Vec3":
        return Vec3(self.x * scalar, self.y * scalar, self.z * scalar)

    def __truediv__(self, scalar: float) -> "Vec3":
        return Vec3(self.x / scalar, self.y / scalar, self.z / scalar)

    def length(self) -> float:
        return (self.x ** 2 + self.y ** 2 + self.z ** 2) ** 0.5

    def to_tuple(self) -> Tuple[float, float, float]:
        return (self.x, self.y, self.z)

    def to_list(self) -> List[float]:
        return [self.x, self.y, self.z]


@dataclass
class BBox:
    """轴对齐包围盒"""
    x_min: float = 0.0
    x_max: float = 0.0
    y_min: float = 0.0
    y_max: float = 0.0
    z_min: float = 0.0
    z_max: float = 0.0

    @property
    def width(self) -> float:
        return self.x_max - self.x_min

    @property
    def depth(self) -> float:
        return self.y_max - self.y_min

    @property
    def height(self) -> float:
        return self.z_max - self.z_min

    @property
    def volume(self) -> float:
        return self.width * self.depth * self.height

    @property
    def center(self) -> Vec3:
        return Vec3(
            (self.x_min + self.x_max) / 2,
            (self.y_min + self.y_max) / 2,
            (self.z_min + self.z_max) / 2,
        )

    @property
    def size(self) -> Vec3:
        return Vec3(self.width, self.depth, self.height)

    def to_dict(self) -> Dict[str, List[float]]:
        return {
            "x": [self.x_min, self.x_max],
            "y": [self.y_min, self.y_max],
            "z": [self.z_min, self.z_max],
        }

    def intersects(self, other: "BBox") -> bool:
        """检查两个包围盒是否相交"""
        return (
            self.x_min < other.x_max and self.x_max > other.x_min and
            self.y_min < other.y_max and self.y_max > other.y_min and
            self.z_min < other.z_max and self.z_max > other.z_min
        )

    def intersection_volume(self, other: "BBox") -> float:
        """计算两个包围盒的交集体积"""
        if not self.intersects(other):
            return 0.0
        dx = min(self.x_max, other.x_max) - max(self.x_min, other.x_min)
        dy = min(self.y_max, other.y_max) - max(self.y_min, other.y_min)
        dz = min(self.z_max, other.z_max) - max(self.z_min, other.z_min)
        return dx * dy * dz

    def union(self, other: "BBox") -> "BBox":
        """合并两个包围盒"""
        return BBox(
            x_min=min(self.x_min, other.x_min),
            x_max=max(self.x_max, other.x_max),
            y_min=min(self.y_min, other.y_min),
            y_max=max(self.y_max, other.y_max),
            z_min=min(self.z_min, other.z_min),
            z_max=max(self.z_max, other.z_max),
        )

    @classmethod
    def from_center_size(cls, center: Vec3, size: Vec3) -> "BBox":
        """从中心点和尺寸创建包围盒"""
        return cls(
            x_min=center.x - size.x / 2,
            x_max=center.x + size.x / 2,
            y_min=center.y - size.y / 2,
            y_max=center.y + size.y / 2,
            z_min=center.z - size.z / 2,
            z_max=center.z + size.z / 2,
        )


@dataclass
class PrimRecord:
    """USDA Prim 解析记录"""
    prim_type: str  # "Cube", "Xform", "Material", 等
    prim_path: str
    parent_path: str
    # 几何属性 (仅 Cube)
    extent: Optional[List[float]] = None  # [x_min, x_max, y_min, y_max, z_min, z_max]
    size: Optional[Vec3] = None  # 语义上 size = extent 的范围
    # 变换
    transform: List[float] = field(default_factory=lambda: [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1])  # 4x4 matrix (row-major)
    world_bbox: Optional[BBox] = None  # 世界空间包围盒（变换后）
    # 材质
    material_binding: Optional[str] = None
    material_color: Optional[Vec3] = None
    # 分类
    tactical_type: TacticalType = TacticalType.UNKNOWN
    # 场景层级
    depth: int = 0

    @property
    def name(self) -> str:
        return self.prim_path.split("/")[-1] if self.prim_path else ""

    @property
    def local_bbox(self) -> Optional[BBox]:
        """从 extent 构建局部坐标包围盒"""
        if self.extent and len(self.extent) == 6:
            return BBox(*self.extent)
        if self.size:
            return BBox.from_center_size(Vec3(), self.size)
        return None


@dataclass
class CubeInfo:
    """简化 Cube 信息（用于楼梯检测和 Phase 3 处理）"""
    cube_id: str
    center: Tuple[float, float, float]
    size: Tuple[float, float, float]
    tactical_type: TacticalType = TacticalType.UNKNOWN
    material: str = ""
    material_color: Optional[Tuple[float, float, float]] = None

    @property
    def bbox(self) -> BBox:
        return BBox.from_center_size(
            Vec3(*self.center),
            Vec3(*self.size)
        )
