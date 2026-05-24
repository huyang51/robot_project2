"""
Phase 0 Z 层分析器

计算 Z 向直方图并推断楼层数量与边界。
集成了楼梯启发式检测器作为交叉验证。
"""

import logging
from collections import deque
from typing import List, Dict, Tuple, Optional

from ..core.types import PrimRecord
from ..core.geometry import axis_histogram, xy_overlap_ratio

logger = logging.getLogger(__name__)


class ZLayerAnalyzer:
    """Z 层分析器：推断楼层结构和楼梯位置

    支持自定义纵轴——Blender Z-up 经 90° X 旋转后纵轴变为 Y，
    需传入 up_axis='y'。
    """

    def __init__(self, params: Optional[Dict] = None, up_axis: str = "z",
                 floor_params: Optional[Dict] = None):
        self.up_axis = up_axis
        self.params = params or {}
        self.floor_params = floor_params or {}
        self.step_height_min = self.params.get("step_height_min", 0.12)
        self.step_height_max = self.params.get("step_height_max", 0.22)
        self.step_height_ideal = self.params.get("step_height_ideal", 0.18)
        self.min_xy_overlap_ratio = self.params.get("min_xy_overlap_ratio", 0.3)
        self.min_consecutive_steps = self.params.get("min_consecutive_steps", 5)
        self.step_width_min = self.params.get("step_width_min", 0.5)
        self.step_depth_min = self.params.get("step_depth_min", 0.2)
        self.stair_total_height_min = self.params.get("stair_total_height_min", 0.4)
        self.max_up_thickness_ratio = self.params.get("max_up_thickness_ratio", 0.3)
        self.max_up_thickness = self.params.get("max_up_thickness", 0.5)
        # 跨层密度检测参数
        self.density_xy_cell = self.params.get("xy_cell_size", 1.0)
        self.density_min_cubes = self.params.get("min_cubes_per_cell", 3)
        self.density_min_vert = self.params.get("min_vertical_density", 0.3)
        self.density_min_area = self.params.get("min_footprint_area", 1.0)
        self.density_max_area = self.params.get("max_footprint_area", 15.0)
        self.density_min_cells = self.params.get("min_adjacent_cluster_cells", 2)
        # 楼层检测参数
        self.histogram_bins = self.floor_params.get("histogram_bins", 80)
        self.adaptive_bin_resolution = self.floor_params.get("adaptive_bin_resolution", 0.5)
        self.weak_peak_min_ratio = self.floor_params.get("weak_peak_min_ratio", 0.05)
        self.peak_merge_distance = self.floor_params.get("peak_merge_distance", 3.0)
        self.peak_prominence_ratio = self.floor_params.get("peak_prominence_ratio", 0.10)
        self.boundary_method = self.floor_params.get("boundary_method", "valley")

    def analyze(self, cube_prims: List[PrimRecord]) -> Dict:
        """执行纵轴层分析

        支持 XZ 空间预聚类：先将 Cube 投影到 XZ 平面进行连通分量分析，
        分离空间上不相连的区域（如分离地形和建筑），再对每个空间聚类
        独立进行纵轴直方图分析。这解决了"户外地形和室内建筑在 Y 轴上
        混合导致楼层检测失败"的问题。
        """
        cubes_with_bbox = [p for p in cube_prims if p.world_bbox]
        if not cubes_with_bbox:
            return self._empty_result()

        # 水平面空间聚类：用非垂直轴确定水平面
        # up_axis=z → 水平面是 XY；up_axis=y → 水平面是 XZ
        if self.up_axis == "z":
            h1, h2 = "x", "y"
        else:
            h1, h2 = "x", "z"
        h1_vals = [getattr(p.world_bbox.center, h1) for p in cubes_with_bbox if p.world_bbox]
        h2_vals = [getattr(p.world_bbox.center, h2) for p in cubes_with_bbox if p.world_bbox]
        scene_h_span = max(max(h1_vals)-min(h1_vals), max(h2_vals)-min(h2_vals), 1.0) if h1_vals else 1.0
        adaptive_cell = max(0.5, min(10.0, scene_h_span / 50.0))
        xz_clusters = self._cluster_xz(cubes_with_bbox, cell_size=adaptive_cell,
                                       horiz_axes=(h1, h2))
        logger.info("Space cluster (%s%s plane): cell=%.1f, %d regions",
                    h1, h2, adaptive_cell, len(xz_clusters))

        if len(xz_clusters) <= 1:
            # 单一聚类，回退到全局分析
            return self._analyze_single(cubes_with_bbox)
        else:
            # 多聚类：每个聚类独立分析
            return self._analyze_multi(cubes_with_bbox, xz_clusters)

    def _empty_result(self) -> Dict:
        return {
            "analyzed_axis": self.up_axis,
            "vertical_histogram": {},
            "floor_count": 0,
            "floor_boundaries": [],
            "tentative_stairs": [],
            "vertical_complexity": "simple",
            "spatial_clusters": [],
        }

    def _analyze_single(self, cubes_with_bbox: List[PrimRecord]) -> Dict:
        """单一空间聚类的分析（原 analyze 逻辑）"""
        hist, floor_boundaries = self._detect_floors(cubes_with_bbox)
        tentative_stairs = self._detect_staircases(cubes_with_bbox)
        vertical_complexity = self._classify_complexity(hist)

        return {
            "analyzed_axis": self.up_axis,
            "vertical_histogram": hist,
            "floor_count": len(floor_boundaries),
            "floor_boundaries": floor_boundaries,
            "tentative_stairs": tentative_stairs,
            "vertical_complexity": vertical_complexity,
            "spatial_clusters": [],
        }

    def _analyze_multi(self, cubes_with_bbox: List[PrimRecord],
                       xz_clusters: List[List[PrimRecord]]) -> Dict:
        """多空间聚类的分析：每个聚类独立检测楼层

        修复要点:
        - 按 cube 密度和占比区分 building / exterior / terrain
        - 全局 floor_count 取主建筑聚类，不再跨聚类求和
        - 稀疏聚类 (< 5% 总 cube) 标记为 background，不参与楼层统计
        """
        all_boundaries = []
        all_stairs = []
        cluster_info = []
        merged_hist = {}
        total_cubes = sum(len(c) for c in xz_clusters)

        a1, a2 = ("x", "y") if self.up_axis == "z" else ("x", "z")

        for i, cluster_cubes in enumerate(xz_clusters):
            if len(cluster_cubes) < 10:
                continue

            hist, boundaries = self._detect_floors(cluster_cubes)
            stairs = self._detect_staircases(cluster_cubes)

            # 计算该聚类的空间范围
            a1_vals = [getattr(p.world_bbox.center, a1) for p in cluster_cubes if p.world_bbox]
            a2_vals = [getattr(p.world_bbox.center, a2) for p in cluster_cubes if p.world_bbox]
            up_vals = [getattr(p.world_bbox.center, self.up_axis) for p in cluster_cubes if p.world_bbox]
            z_range = max(up_vals) - min(up_vals) if up_vals else 1.0

            # 聚类分类启发式
            cube_ratio = len(cluster_cubes) / max(total_cubes, 1)
            z_density = len(cluster_cubes) / max(z_range, 1.0)  # cubes per meter

            if cube_ratio < 0.05:
                label = "background"       # 太稀疏，忽略
            elif cube_ratio >= 0.15 and z_density >= 50:
                label = "building"          # 高密度 + 显著占比 → 主建筑
            elif cube_ratio >= 0.10:
                label = "exterior_structure" # 中等占比 → 外部结构
            else:
                label = "outdoor_terrain"    # 低密度 → 地形

            cluster_info.append({
                "cluster_id": i,
                "cube_count": len(cluster_cubes),
                "cube_ratio": round(cube_ratio, 3),
                "z_density": round(z_density, 1),
                f"{a1}_range": [round(min(a1_vals), 1), round(max(a1_vals), 1)],
                f"{a2}_range": [round(min(a2_vals), 1), round(max(a2_vals), 1)],
                f"{self.up_axis}_range": [round(min(up_vals), 1), round(max(up_vals), 1)],
                "floor_count": len(boundaries),
                "floor_boundaries": boundaries,
                "label": label,
            })

            for b in boundaries:
                all_boundaries.append((i, b[0], b[1]))
            all_stairs.extend(stairs)
            if i == 0:
                merged_hist = hist

        # 全局 floor_count: 使用主建筑聚类的楼层数，而非跨聚类求和
        building_clusters = [c for c in cluster_info if c["label"] == "building"]
        if building_clusters:
            main_building = max(building_clusters, key=lambda c: c["cube_count"])
            total_floors = main_building["floor_count"]
        else:
            # 回退: 取最大的非 background 聚类
            non_bg = [c for c in cluster_info if c["label"] != "background"]
            total_floors = max((c["floor_count"] for c in non_bg), default=0)

        # 楼层边界: 仅来自 building 聚类
        building_ids = {c["cluster_id"] for c in cluster_info if c["label"] == "building"}
        building_boundaries = [(b[1], b[2]) for b in all_boundaries if b[0] in building_ids]
        if not building_boundaries:
            building_boundaries = [(b[1], b[2]) for b in all_boundaries]

        # 垂直复杂度基于建筑聚类
        if total_floors >= 3:
            complexity = "complex"
        elif total_floors >= 2:
            complexity = "moderate"
        else:
            complexity = "simple"

        return {
            "analyzed_axis": self.up_axis,
            "vertical_histogram": merged_hist,
            "floor_count": total_floors,
            "floor_boundaries": building_boundaries,
            "tentative_stairs": all_stairs,
            "vertical_complexity": complexity,
            "spatial_clusters": cluster_info,
        }

    def _detect_floors(self, cubes: List[PrimRecord]) -> tuple:
        """对一组 Cube 检测楼层（两步法：粗 bins 检测峰值，精细 bins 定边界）"""
        axis_idx = {"x": 0, "y": 1, "z": 2}[self.up_axis]
        if not cubes:
            return axis_histogram([], axis=self.up_axis, bins=80), []

        vals = [
            [p.world_bbox.center.x, p.world_bbox.center.y, p.world_bbox.center.z][axis_idx]
            for p in cubes if p.world_bbox
        ]
        if not vals:
            return axis_histogram([], axis=self.up_axis, bins=80), []

        axis_range_total = max(vals) - min(vals)

        # Step A: 粗 bins 检测峰值
        coarse_bins = min(self.histogram_bins, max(40, int(axis_range_total / 2.0)))
        coarse_hist = axis_histogram(cubes, axis=self.up_axis, bins=coarse_bins)
        coarse_hist = self._filter_weak_peaks(coarse_hist, min_ratio=self.weak_peak_min_ratio)
        detected_peaks = coarse_hist.get("peaks", [])

        # Step B: 精细 bins 确定边界
        fine_bins = max(self.histogram_bins,
                        int(axis_range_total / self.adaptive_bin_resolution))
        hist = axis_histogram(cubes, axis=self.up_axis, bins=fine_bins)
        hist["peaks"] = self._map_peaks_to_fine(detected_peaks, hist)
        hist = self._merge_nearby_peaks(hist, merge_distance=self.peak_merge_distance)

        if self.boundary_method == "expand":
            boundaries = self._infer_floors_expand(hist)
        elif self.boundary_method == "valley":
            boundaries = self._infer_floors_valley(hist)
        else:
            boundaries = self._infer_floors(hist)

        return hist, boundaries

    @staticmethod
    def _classify_complexity(hist: Dict) -> str:
        fc = hist.get("floor_count", 0)
        if fc >= 3: return "complex"
        if fc >= 2: return "moderate"
        return "simple"

    @staticmethod
    def _cluster_xz(cube_prims: List[PrimRecord], cell_size: float = 3.0,
                    horiz_axes: tuple = ("x", "z")) -> List[List[PrimRecord]]:
        """在水平面上对 Cube 进行网格连通分量聚类

        horiz_axes 指定水平面使用的两个轴。up_axis=z 时用 ("x","y")，
        up_axis=y 时用 ("x","z")。将平面划分为 cell_size 网格，
        标记有 Cube 的格子，对相邻非空格子做连通分量分析。
        """
        if not cube_prims:
            return []

        a1, a2 = horiz_axes
        grid = {}
        for p in cube_prims:
            if not p.world_bbox:
                continue
            c1 = getattr(p.world_bbox.center, a1)
            c2 = getattr(p.world_bbox.center, a2)
            g1 = int(c1 / cell_size)
            g2 = int(c2 / cell_size)
            key = (g1, g2)
            if key not in grid:
                grid[key] = []
            grid[key].append(p)

        if not grid:
            return []

        # BFS 找连通分量（4 邻域）
        visited = set()
        clusters = []
        for key in grid:
            if key in visited:
                continue
            stack = [key]
            visited.add(key)
            cluster_cubes = list(grid[key])
            while stack:
                gx, gz = stack.pop()
                for dx, dz in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nk = (gx + dx, gz + dz)
                    if nk in grid and nk not in visited:
                        visited.add(nk)
                        stack.append(nk)
                        cluster_cubes.extend(grid[nk])
            clusters.append(cluster_cubes)

        return clusters

    # ── 私有工具 ────────────────────────────────────────────

    @staticmethod
    def _map_peaks_to_fine(coarse_peaks: List[float], fine_hist: Dict) -> List[float]:
        """将粗 bins 检测到的峰值映射到精细 bins 直方图中的最近局部极大值

        对于粗 bins 中的每个峰值，在精细 bins 对应位置附近搜索最近的局部极大值。
        搜索范围 = ±fine_bin_width * 5（保证覆盖粗 bin 的范围）。
        """
        if not coarse_peaks:
            return []

        histogram = fine_hist.get("histogram", [])
        bin_edges = fine_hist.get("bin_edges", [])
        if not histogram or len(bin_edges) < 2:
            return list(coarse_peaks)

        bin_width = bin_edges[1] - bin_edges[0]
        search_radius = max(3, int(bin_width * 5 / bin_width))  # 5 bins or 3, whichever larger
        mapped = []

        for cp in coarse_peaks:
            center_bin = min(int((cp - bin_edges[0]) / bin_width), len(histogram) - 1)
            lo = max(1, center_bin - search_radius)
            hi = min(len(histogram) - 2, center_bin + search_radius)
            # 在搜索范围内找局部最大值
            best_bin = center_bin
            best_val = histogram[center_bin] if 0 <= center_bin < len(histogram) else 0
            for b in range(lo, hi + 1):
                if histogram[b] > histogram[b - 1] and histogram[b] > histogram[b + 1]:
                    if histogram[b] > best_val:
                        best_val = histogram[b]
                        best_bin = b
            mapped.append(bin_edges[best_bin] + bin_width / 2)

        return sorted(set(round(p, 2) for p in mapped))

    @staticmethod
    def _filter_weak_peaks(hist: Dict, min_ratio: float = 0.05) -> Dict:
        """过滤弱峰值：只保留 >= max_count * min_ratio 的局部极值"""
        peaks = hist.get("peaks", [])
        counts = hist.get("histogram", [])
        bin_edges = hist.get("bin_edges", [])
        if not peaks or not counts:
            return hist

        max_count = max(counts) if counts else 1
        threshold = max_count * min_ratio

        # 重建峰值列表
        filtered_peaks = []
        bin_width = bin_edges[1] - bin_edges[0] if len(bin_edges) >= 2 else 1
        for i in range(1, len(counts) - 1):
            if counts[i] > counts[i - 1] and counts[i] > counts[i + 1]:
                if counts[i] >= threshold:
                    filtered_peaks.append(bin_edges[i] + bin_width / 2)

        return {
            **hist,
            "peaks": filtered_peaks,
            "floor_count": max(1, len(filtered_peaks)),
        }

    def _infer_floors(self, hist: Dict) -> List[Tuple[float, float]]:
        """从纵轴直方图推断楼层边界

        使用峰值作为楼层中心，相邻峰值中点作为楼层边界。
        """
        peaks = hist.get("peaks", [])
        if not peaks:
            ax_range = hist.get("axis_range", [0, 3])
            return [(ax_range[0], ax_range[1])]

        boundaries = []
        ax_range = hist.get("axis_range", [0, 3])
        for i, peak in enumerate(peaks):
            # 下界：与前一峰值的中间点（或纵轴最小值）
            v_min = ax_range[0] if i == 0 else (peaks[i - 1] + peak) / 2
            # 上界：与后一峰值的中间点（或纵轴最大值）
            v_max = ax_range[1] if i == len(peaks) - 1 else (peak + peaks[i + 1]) / 2
            boundaries.append((round(v_min, 2), round(v_max, 2)))

        return boundaries

    @staticmethod
    def _filter_by_prominence(hist: Dict, min_prominence_ratio: float = 0.10) -> Dict:
        """峰突显度过滤：只保留比两侧波谷显著更高的峰值

        对于每个局部峰值，找到其左右最近的波谷位置，
        峰突显度 = peak_count - max(left_valley_count, right_valley_count)
        若突显度 < max_count * min_prominence_ratio，则过滤掉该峰值。

        这消除了因随机波动产生的"假峰"（峰值只比邻 bin 高一点点）。
        """
        peaks = hist.get("peaks", [])
        histogram = hist.get("histogram", [])
        bin_edges = hist.get("bin_edges", [])
        if not peaks or not histogram or len(bin_edges) < 2:
            return hist

        max_count = max(histogram) if histogram else 1
        bin_width = bin_edges[1] - bin_edges[0]
        prominence_threshold = max_count * min_prominence_ratio

        filtered_peaks = []
        for peak_val in peaks:
            peak_bin = min(int((peak_val - bin_edges[0]) / bin_width), len(histogram) - 1)

            # 找左侧波谷（peak 左边第一个比左边邻居低的 bin）
            left_valley = histogram[peak_bin]
            for b in range(peak_bin - 1, 0, -1):
                if histogram[b] < left_valley:
                    left_valley = histogram[b]
                else:
                    break

            # 找右侧波谷（peak 右边第一个比右边邻居低的 bin）
            right_valley = histogram[peak_bin]
            for b in range(peak_bin + 1, len(histogram)):
                if histogram[b] < right_valley:
                    right_valley = histogram[b]
                else:
                    break

            prominence = histogram[peak_bin] - max(left_valley, right_valley)
            if prominence >= prominence_threshold:
                filtered_peaks.append(peak_val)

        return {
            **hist,
            "peaks": filtered_peaks,
            "floor_count": max(1, len(filtered_peaks)),
        }

    @staticmethod
    def _merge_nearby_peaks(hist: Dict, merge_distance: float = 3.0) -> Dict:
        """合并邻近峰值：间距 < merge_distance 的相邻峰合并为一个

        合并策略：取合并组内峰值最高的 bin 作为代表。
        这解决了密集 bin 产生的"一个楼层被检测为多个峰"的问题。
        """
        peaks = hist.get("peaks", [])
        if len(peaks) <= 1:
            return hist

        histogram = hist.get("histogram", [])
        bin_edges = hist.get("bin_edges", [])
        if not histogram or len(bin_edges) < 2:
            return hist

        bin_width = bin_edges[1] - bin_edges[0]
        merged = []
        current_group = [peaks[0]]

        for i in range(1, len(peaks)):
            if abs(peaks[i] - current_group[-1]) < merge_distance:
                current_group.append(peaks[i])
            else:
                # 取组内峰值最高的
                best = max(current_group, key=lambda p:
                    histogram[min(int((p - bin_edges[0]) / bin_width), len(histogram) - 1)])
                merged.append(best)
                current_group = [peaks[i]]

        best = max(current_group, key=lambda p:
            histogram[min(int((p - bin_edges[0]) / bin_width), len(histogram) - 1)])
        merged.append(best)

        return {
            **hist,
            "peaks": merged,
            "floor_count": max(1, len(merged)),
        }

    def _infer_floors_expand(self, hist: Dict) -> List[Tuple[float, float]]:
        """从纵轴直方图使用聚类扩展法推断楼层边界

        对每个峰值，向两侧扩展直到 histogram count 降到峰值的一定比例以下，
        以确定该聚类的实际数据范围。相邻聚类的边界取两者扩展范围的中点。

        这解决了两个聚类差距巨大时（如 6→245 的空白区间），波谷法和中点法
        都找不到合理边界的问题。

        算法：
        1. 对每个峰值，向两侧扩展找 count < peak * 0.05 的位置（聚类边界）
        2. 聚类之间的楼层边界 = (下聚类上界 + 上聚类下界) / 2
        3. 最下层下界 = 轴最小值；最上层上界 = 轴最大值
        """
        peaks = hist.get("peaks", [])
        histogram = hist.get("histogram", [])
        bin_edges = hist.get("bin_edges", [])
        ax_range = hist.get("axis_range", [0, 3])

        if not peaks or not histogram or len(bin_edges) < 2:
            return [(ax_range[0], ax_range[1])]

        bin_width = bin_edges[1] - bin_edges[0]
        cluster_threshold_ratio = 0.05

        def _expand(peak_val, direction):
            """从峰值向 direction 方向扩展，直到 count < peak*5%"""
            peak_bin = min(int((peak_val - bin_edges[0]) / bin_width), len(histogram) - 1)
            peak_count = histogram[peak_bin]
            threshold = max(peak_count * cluster_threshold_ratio, 1.0)
            step = 1 if direction == "up" else -1
            current = peak_bin
            while 0 <= current < len(histogram):
                if histogram[current] < threshold:
                    break
                current += step
            current -= step  # 回退一步（回到有效 bin）
            current = max(0, min(current, len(histogram) - 1))
            return bin_edges[current] + bin_width / 2

        # 计算每个聚类的上界和下界
        cluster_extents = []
        for peak in peaks:
            lower = _expand(peak, "down")
            upper = _expand(peak, "up")
            cluster_extents.append((lower, upper))

        # 确定楼层边界：相邻聚类间的边界取中点
        boundaries = []
        for i in range(len(peaks)):
            if i == 0:
                v_min = ax_range[0]
            else:
                v_min = (cluster_extents[i - 1][1] + cluster_extents[i][0]) / 2

            if i == len(peaks) - 1:
                v_max = ax_range[1]
            else:
                v_max = (cluster_extents[i][1] + cluster_extents[i + 1][0]) / 2

            boundaries.append((round(v_min, 2), round(v_max, 2)))

        return boundaries

    def _infer_floors_valley(self, hist: Dict) -> List[Tuple[float, float]]:
        """从纵轴直方图使用波谷法推断楼层边界

        与中点法（_infer_floors）不同，波谷法将楼层边界放在直方图计数的
        局部最低点（波谷）处，而非相邻峰值的数学中点。

        这解决了两个峰值高度差异巨大时（例如地面 800 cube vs 建筑主体 28000 cube）
        中点法产生的荒谬边界问题。

        算法：
        1. 对每对相邻峰值，找到它们之间 histogram count 最低的 bin
        2. 该 bin 的中心即为楼层边界
        3. 最下层边界 = 轴最小值；最上层边界 = 轴最大值
        """
        peaks = hist.get("peaks", [])
        histogram = hist.get("histogram", [])
        bin_edges = hist.get("bin_edges", [])
        ax_range = hist.get("axis_range", [0, 3])

        if not peaks or not histogram or len(bin_edges) < 2:
            return [(ax_range[0], ax_range[1])]

        bin_width = bin_edges[1] - bin_edges[0]
        boundaries = []

        for i, peak in enumerate(peaks):
            if i == 0:
                v_min = ax_range[0]
            else:
                # 在前一个峰值和当前峰值之间找最低的 bin（波谷）
                prev_peak = peaks[i - 1]
                prev_bin = min(int((prev_peak - bin_edges[0]) / bin_width), len(histogram) - 1)
                curr_bin = min(int((peak - bin_edges[0]) / bin_width), len(histogram) - 1)
                lo, hi = min(prev_bin, curr_bin), max(prev_bin, curr_bin)
                if lo < hi:
                    valley_bin = min(range(lo, hi + 1), key=lambda b: histogram[b])
                    v_min = bin_edges[valley_bin] + bin_width / 2
                else:
                    v_min = (prev_peak + peak) / 2  # 退化回中点法

            if i == len(peaks) - 1:
                v_max = ax_range[1]
            else:
                # 在当前峰值和后一个峰值之间找最低的 bin（波谷）
                next_peak = peaks[i + 1]
                curr_bin = min(int((peak - bin_edges[0]) / bin_width), len(histogram) - 1)
                next_bin = min(int((next_peak - bin_edges[0]) / bin_width), len(histogram) - 1)
                lo, hi = min(curr_bin, next_bin), max(curr_bin, next_bin)
                if lo < hi:
                    valley_bin = min(range(lo, hi + 1), key=lambda b: histogram[b])
                    v_max = bin_edges[valley_bin] + bin_width / 2
                else:
                    v_max = (peak + next_peak) / 2  # 退化回中点法

            boundaries.append((round(v_min, 2), round(v_max, 2)))

        return boundaries

    def _detect_staircases(self, cube_prims: List[PrimRecord]) -> List[Dict]:
        """检测疑似楼梯结构

        沿纵轴 (self.up_axis) 寻找连续步进的水平面序列。
        """
        up = self.up_axis
        up_idx = {"x": 0, "y": 1, "z": 2}[up]

        def _up_center(p):
            return [p.world_bbox.center.x, p.world_bbox.center.y, p.world_bbox.center.z][up_idx]

        def _up_min(p):
            return [p.world_bbox.x_min, p.world_bbox.y_min, p.world_bbox.z_min][up_idx]

        def _up_max(p):
            return [p.world_bbox.x_max, p.world_bbox.y_max, p.world_bbox.z_max][up_idx]

        # 筛选候选台阶 prim
        candidates = []
        for p in cube_prims:
            if p.world_bbox is None:
                continue
            bbox = p.world_bbox

            # 必须是与纵轴垂直的薄平面（台阶）
            _thickness_map = {"x": bbox.width, "y": bbox.depth, "z": bbox.height}
            up_thickness = _thickness_map.get(up, bbox.height)
            h_span = [bbox.width, bbox.depth, bbox.height]
            del h_span[up_idx]  # 移除纵轴方向
            plane_area_dim1, plane_area_dim2 = h_span
            if min(plane_area_dim1, plane_area_dim2) <= 0:
                continue
            if up_thickness / min(plane_area_dim1, plane_area_dim2) >= self.max_up_thickness_ratio:
                continue  # 不是薄平面

            w, d, h = bbox.width, bbox.depth, bbox.height
            if w < self.step_width_min and d < self.step_width_min:
                continue
            if min(w, d) < self.step_depth_min:
                continue
            if up_thickness > self.max_up_thickness:
                continue

            candidates.append(p)

        if len(candidates) < self.min_consecutive_steps:
            return []

        # 按纵轴排序
        candidates.sort(key=_up_center)
        n = len(candidates)

        # 构建相邻关系 (沿纵轴步进 + 水平重叠)
        adjacency = [[] for _ in range(n)]
        for i in range(n):
            bi = candidates[i].world_bbox
            for j in range(i + 1, n):
                bj = candidates[j].world_bbox
                dv = _up_center(candidates[j]) - _up_center(candidates[i])
                if self.step_height_min <= dv <= self.step_height_max:
                    if xy_overlap_ratio(bi, bj, up_axis=self.up_axis) >= self.min_xy_overlap_ratio:
                        adjacency[i].append(j)
                        adjacency[j].append(i)

        # BFS
        visited = [False] * n
        results = []
        for i in range(n):
            if visited[i]:
                continue
            component = []
            queue = deque([i])
            visited[i] = True
            while queue:
                v = queue.popleft()
                component.append(v)
                for nb in adjacency[v]:
                    if not visited[nb]:
                        visited[nb] = True
                        queue.append(nb)

            if len(component) >= self.min_consecutive_steps:
                comp_sorted = sorted(component, key=lambda idx: _up_center(candidates[idx]))
                chain_cubes = [candidates[idx] for idx in comp_sorted]

                up_values = [_up_center(c) for c in chain_cubes]
                up_min_val = min(_up_min(c) for c in chain_cubes)
                up_max_val = max(_up_max(c) for c in chain_cubes)
                total_h = up_max_val - up_min_val

                if total_h < self.stair_total_height_min:
                    continue

                step_heights = [up_values[k + 1] - up_values[k] for k in range(len(up_values) - 1)]
                avg_step = sum(step_heights) / len(step_heights)
                regularity = 1.0 - min(1.0, max(
                    abs(h - avg_step) / (self.step_height_max - self.step_height_min)
                    for h in step_heights
                ))
                idealness = 1.0 - min(1.0, abs(avg_step - self.step_height_ideal) / 0.1)
                count_score = min(1.0, len(chain_cubes) / 8)
                confidence = regularity * 0.4 + idealness * 0.3 + count_score * 0.3

                results.append({
                    "stair_id": f"TENT_STAIR_{len(results) + 1:03d}",
                    "step_cube_ids": [c.prim_path for c in chain_cubes],
                    "step_count": len(chain_cubes),
                    "total_height": round(total_h, 3),
                    "avg_step_height": round(avg_step, 3),
                    "direction": "ascending" if up_values[-1] > up_values[0] else "descending",
                    "confidence": round(confidence, 3),
                })

        results.sort(key=lambda s: s["confidence"], reverse=True)
        return results

    def detect_stairwells_by_density(
        self, cube_prims: List, stories: List[Dict],
    ) -> List[Dict]:
        """基于跨层立方体密度检测楼梯间

        替代方法：不依赖台阶序列检测，而是在相邻楼层之间寻找
        XY 占地面积小而垂直贯穿范围广的高密度立方体聚集区。

        原理：楼梯间在 XY 平面上的投影面积小（2-15 m²），
        但在 Z 轴上跨越整层高度，且每层都有密集的立方体分布。
        """
        if not stories or len(stories) < 1:
            return []

        up = self.up_axis
        horiz_axes = [a for a in ("x", "y", "z") if a != up]
        h1, h2 = horiz_axes[0], horiz_axes[1]

        results = []
        seen_ids = set()

        # 对每对相邻楼层检测楼梯间
        for i in range(len(stories) - 1):
            lower = stories[i]
            upper = stories[i + 1]

            z_lo = lower.get("floor_slab_z", lower.get("z_range", [0, 0])[0])
            z_hi = upper.get("floor_slab_z", upper.get("z_range", [0, 0])[0])
            if z_hi <= z_lo:
                continue

            z_range_span = z_hi - z_lo
            if z_range_span < 1.5:
                continue

            # 筛选位于此楼层范围内的 cube
            story_cubes = [
                p for p in cube_prims
                if p.world_bbox is not None
                and z_lo <= getattr(p.world_bbox.center, up) <= z_hi
            ]
            if len(story_cubes) < 10:
                continue

            # XY 网格
            h1_vals = [getattr(p.world_bbox.center, h1) for p in story_cubes]
            h2_vals = [getattr(p.world_bbox.center, h2) for p in story_cubes]
            h1_min, h1_max = min(h1_vals), max(h1_vals)
            h2_min, h2_max = min(h2_vals), max(h2_vals)
            if h1_max <= h1_min or h2_max <= h2_min:
                continue

            cell_size = self.density_xy_cell
            h1_bins = max(1, int((h1_max - h1_min) / cell_size) + 1)
            h2_bins = max(1, int((h2_max - h2_min) / cell_size) + 1)

            # 每格统计：cube 计数 + Z 跨度
            cell_counts = {}
            cell_z_vals = {}
            for p in story_cubes:
                g1 = min(int((getattr(p.world_bbox.center, h1) - h1_min) / cell_size), h1_bins - 1)
                g2 = min(int((getattr(p.world_bbox.center, h2) - h2_min) / cell_size), h2_bins - 1)
                key = (g1, g2)
                cell_counts[key] = cell_counts.get(key, 0) + 1
                if key not in cell_z_vals:
                    cell_z_vals[key] = []
                cell_z_vals[key].append(getattr(p.world_bbox.center, up))

            # 筛选高密度格：cube 数量足够 AND 垂直分布广泛
            dense_cells = set()
            for key, count in cell_counts.items():
                if count < self.density_min_cubes:
                    continue
                z_vals = cell_z_vals.get(key, [])
                if not z_vals:
                    continue
                z_span = max(z_vals) - min(z_vals)
                vert_density = z_span / max(z_range_span, 0.1)
                if vert_density >= self.density_min_vert:
                    dense_cells.add(key)

            if not dense_cells:
                continue

            # BFS 聚类连通高密度格
            visited = set()
            clusters = []
            for cell in dense_cells:
                if cell in visited:
                    continue
                stack = [cell]
                visited.add(cell)
                cluster_cells = []
                while stack:
                    c1, c2 = stack.pop()
                    cluster_cells.append((c1, c2))
                    for d1, d2 in [(-1, -1), (-1, 0), (-1, 1), (0, -1),
                                    (0, 1), (1, -1), (1, 0), (1, 1)]:
                        nb = (c1 + d1, c2 + d2)
                        if nb in dense_cells and nb not in visited:
                            visited.add(nb)
                            stack.append(nb)
                if len(cluster_cells) >= self.density_min_cells:
                    clusters.append(cluster_cells)

            # 筛选占地面积在合理范围内的聚类
            for c_idx, cluster in enumerate(clusters):
                area = len(cluster) * cell_size * cell_size
                if area < self.density_min_area or area > self.density_max_area:
                    continue

                # 聚类中心
                h1_centers = [(c1 + 0.5) * cell_size + h1_min for c1, _ in cluster]
                h2_centers = [(c2 + 0.5) * cell_size + h2_min for _, c2 in cluster]
                center_h1 = sum(h1_centers) / len(h1_centers)
                center_h2 = sum(h2_centers) / len(h2_centers)

                # 聚类内总 cube 数
                total_cubes = sum(cell_counts.get(c, 0) for c in cluster)

                stair_id = f"DENS_STAIR_{i:02d}_{c_idx:02d}"
                results.append({
                    "stair_id": stair_id,
                    "detection_method": "cross_story_density",
                    "connected_stories": [lower.get("story_id"), upper.get("story_id")],
                    "z_range": [round(z_lo, 1), round(z_hi, 1)],
                    "center": {h1: round(center_h1, 1), h2: round(center_h2, 1)},
                    "footprint_area_m2": round(area, 1),
                    "cube_count": total_cubes,
                    "cell_count": len(cluster),
                    "confidence": round(min(1.0, area / 10.0 + total_cubes / 100), 2),
                })

        results.sort(key=lambda s: -s["confidence"])
        return results
