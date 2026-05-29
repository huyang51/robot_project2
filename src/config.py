"""
robot_project2 集中配置模块
"""
import os
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

# 项目根目录
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

# Phase 输出目录
PHASE0_DIR = PROCESSED_DIR / "phase0"
PHASE1_DIR = PROCESSED_DIR / "phase1"
PHASE2_DIR = PROCESSED_DIR / "phase2"
SUB_SCENES_DIR = PROCESSED_DIR / "sub_scenes"
TACTICS_TEXT_DIR = PROCESSED_DIR / "tactics" / "text_version"
TACTICS_STRUCT_DIR = PROCESSED_DIR / "tactics" / "struct_version"
CHROMA_DB_DIR = PROCESSED_DIR / "chroma_db"

# 加载 .env
load_dotenv()

# MiniMax API 配置
MINIMAX_API_KEY = os.environ.get("MINIMAX_API_KEY", "")
MINIMAX_BASE_URL = "https://api.minimaxi.com/v1"
MINIMAX_MODEL = "MiniMax-M2.7"

# LLM 配置
LLM_TEMPERATURE = 0.7
LLM_MAX_TOKENS = 160000
LLM_JSON_TEMPERATURE = 0.3

# 嵌入模型配置
# 默认使用本地 SentenceTransformer 模型（无需 API Key），
# 自动检测 CUDA 并使用 GPU 加速（RTX PRO 6000, 96GB VRAM）。
EMBEDDING_MODEL = "BAAI/bge-large-zh-v1.5"           # 本地中文嵌入模型（1024维, GPU）
EMBEDDING_DIMENSION = 1024

# 本地模型路径（留空则从 HuggingFace 自动下载）
# 设置此路径指向已下载的模型目录，例如:
#   EMBEDDING_MODEL_LOCAL_PATH = "/home/huyang/models/bge-large-zh-v1.5"
# 下载方式:
#   pip install huggingface_hub
#   huggingface-cli download BAAI/bge-large-zh-v1.5 --local-dir /path/to/save
EMBEDDING_MODEL_LOCAL_PATH = os.environ.get(
    "BGE_LOCAL_PATH",
    str(PROJECT_ROOT / "data" / "models" / "bge-large-zh-v1.5")
)

# 可选 API 嵌入后端（需显式切换）
EMBEDDING_MODEL_OPENAI = "text-embedding-3-large"    # OpenAI 嵌入（1024维）
EMBEDDING_DIMENSION_OPENAI = 1024
EMBEDDING_MODEL_LOCAL = "BAAI/bge-large-zh-v1.5"    # 本地 BGE 中文模型（1024维）

# MiniMax 嵌入配置（需 API Key 且有余额）
MINIMAX_EMBEDDING_MODEL = "embo-01"                  # MiniMax 嵌入模型
MINIMAX_EMBEDDING_DIMENSION = 1536                    # embo-01 向量维度

# ChromaDB 配置
CHROMA_PERSISTENCE_DIR = str(CHROMA_DB_DIR)
COLLECTION_TACTICS_TEXT = "tactics_text"
COLLECTION_TACTICS_STRUCT = "tactics_struct"
COLLECTION_PDF_CHAPTERS = "pdf_chapters"

# ============================================================
# Phase 0 启发式分类可配置参数
# ============================================================
PHASE0_HEURISTIC_PARAMS = {
    "wall_height_min": 2.0,
    "wall_width_min": 0.5,
    "wall_thickness_max": 0.5,
    "floor_thickness_max": 0.4,
    "floor_area_min": 4.0,
    "door_height_min": 1.8,
    "door_height_max": 2.5,
    "door_width_min": 0.6,
    "door_width_max": 1.5,
    "door_z_bottom_margin": 0.3,
    "cover_volume_min": 0.02,
    "cover_volume_max": 3.0,
    "cover_height_min": 0.4,
    "cover_height_max": 1.5,
    "cover_wall_distance_min": 0.3,
    "pillar_height_min": 2.0,
    "pillar_width_max": 0.6,
    "pillar_depth_max": 0.6,
    "decor_volume_max": 0.005,
    "decor_dimension_min": 0.03,
    "adaptive_scale_enabled": True,
    "adaptive_percentile": 95,
    "min_scenes_for_calibration": 2,
}

# ============================================================
# Phase 3b 聚合参数
# ============================================================
PHASE3B_AGGREGATION_PARAMS = {
    "centroid_distance_threshold_ratio": 0.02,
    "same_material_required": True,
    "volume_increase_max": 0.30,
    "pattern_spacing_deviation_max": 0.1,
    "max_cubes_target": 60,
    "downgrade_cube_threshold": 100,
    "downgrade_distance_multiplier": 1.5,
    "downgrade_volume_increase_max": 0.50,
}

# ============================================================
# 楼梯检测参数
# ============================================================
# 方法1: 台阶序列启发式（检测薄水平面步进序列）
STAIRCASE_DETECTION_PARAMS = {
    "step_height_min": 0.08,          # 放宽下限，适应非标准台阶（原 0.12）
    "step_height_max": 0.35,          # 放宽上限，适应非标准台阶（原 0.22）
    "step_height_ideal": 0.18,
    "min_xy_overlap_ratio": 0.15,     # 放宽重叠要求，适应体素化台阶偏移（原 0.30）
    "min_consecutive_steps": 3,       # 降低最小连续步数（原5），短楼梯也算
    "step_width_min": 0.3,            # 放宽最小宽度（原0.5）
    "step_depth_min": 0.1,            # 放宽最小深度（原0.2）
    "stair_total_height_min": 0.4,
    # 台阶候选筛选参数（已从硬编码移入配置）
    "max_up_thickness_ratio": 0.5,    # up_thickness / min(plane_dim) 上限（原硬编码 0.3）
    "max_up_thickness": 1.0,          # 绝对垂直厚度上限 m（原硬编码 0.5）
}

# 方法2: 跨层密度检测（在相邻楼层间寻找垂直高密度紧凑区）
STAIRCASE_DENSITY_PARAMS = {
    "xy_cell_size": 1.0,              # XY 网格单元大小 m
    "min_cubes_per_cell": 3,          # 每格最少 cube 数
    "min_vertical_density": 0.3,      # 垂直密度 (cube 覆盖率, 0-1)
    "min_footprint_area": 1.0,        # 最小 XY 占地面积 m²
    "max_footprint_area": 15.0,       # 最大 XY 占地面积 m²
    "min_adjacent_cluster_cells": 2,  # 最少连通格数
}

# ============================================================
# 楼层检测参数
# ============================================================
FLOOR_DETECTION_PARAMS = {
    # ── 楼板面证据收集（主方法：楼板面→N-1楼层逻辑）──
    "floor_candidate_z_tolerance": 2.0,       # ±m，围绕 floor_plane Z 层级收集证据的容差
    # ── 证据强度阈值 ──
    "strong_evidence_min_fragments": 1000,     # STRONG: 该层级至少 N 个碎片
    "strong_evidence_min_wall_intersections": 5,  # STRONG: 至少 N 条墙面线穿过此层级
    "strong_evidence_min_horiz_coverage": 0.40,   # STRONG: 水平覆盖至少 40%
    "medium_evidence_min_fragments": 200,      # MEDIUM: 该层级至少 N 个碎片
    "medium_evidence_min_wall_intersections": 2,   # MEDIUM: 至少 N 条墙面线穿过此层级
    "medium_evidence_min_horiz_coverage": 0.20,    # MEDIUM: 水平覆盖至少 20%
    # ── 楼层间距检查 ──
    "inter_floor_gap_min": 2.5,               # 最小合理楼层间距（m）
    "inter_floor_gap_max": 8.0,               # 最大合理楼层间距（m），超过需警告可能漏层
    "inter_floor_density_warning": 100,        # 空白区间 cube 密度超过此值需警告
    # ── 逻辑楼层合并 ──
    "logical_floor_merge_threshold": 6.0,     # 楼板面候选间距小于此值视为同一逻辑楼层（m）
    # 此值应在 inter_floor_gap_min(2.5) 和 inter_floor_gap_max(8.0) 之间。
    # 两个楼板面候选间距 < merge_threshold → 同一楼层的子结构（夹层/台阶平台/不同房间标高差异）→ 合并
    # 两个楼板面候选间距 ≥ merge_threshold → 不同逻辑楼层
}

# ============================================================
# M2 自适应策略双阈值
# ============================================================
M2_THETA_LOW = 0.3
M2_THETA_HIGH = 0.7

# ============================================================
# M3 迭代精炼参数
# ============================================================
M3_ITERATION_PARAMS = {
    "max_rounds": 3,
    "q_pass_threshold": 7.0,
    "early_discard_threshold": 3.0,
    "score_convergence_delta": 0.5,
}

# ============================================================
# M4 质量评价参数
# ============================================================
M4_EVALUATION_PARAMS = {
    "quality_thresholds": {
        "H": 8.0,
        "M": 6.0,
    },
    "military_min_for_H": 7.0,
    "granularity_min_for_H": 6.5,
    "single_dimension_min": 4.0,
    "veto_thresholds": {
        "V1_military": 3.0,
        "V2_scene_adaptation": 3.0,
        "V3_granularity": 3.0,
    },
}


# ============================================================
# Few-Shot 示例配置
# 注意: 实际 Few-Shot 选择逻辑在 phase4/m3_refinement/few_shot_examples.py
# 的 EXAMPLES_BY_MODE 中定义。此配置供未来统一管理。
# ============================================================
FEW_SHOT_CONFIG = {
    "fixed_example_count": 8,
    "examples_by_mode": {
        "RAG": ["EX-01", "EX-03", "EX-06", "EX-07", "EX-08"],
        "HYBRID": ["EX-01", "EX-02", "EX-03", "EX-06", "EX-07", "EX-08"],
        "GEN": ["EX-01", "EX-02", "EX-03", "EX-04", "EX-05", "EX-06", "EX-07", "EX-08"],
    },
    "max_example_tokens_estimate": 8000,
    "position_in_prompt": "before_scene_input",
}

# ============================================================
# PDF 预处理参数
# ============================================================
# PDF 文件扩展名
PDF_EXTENSIONS = {".pdf"}

# 文本分块参数
CHUNK_SIZE = 1000      # 每块最大字符数
CHUNK_OVERLAP = 200    # 块间重叠字符数

# PDF 嵌入与存储参数
PDF_EMBEDDING_BATCH_SIZE = 50   # 嵌入批处理大小

# PDF 章节拆分模式（正则）
PDF_CHAPTER_PATTERNS = [
    r"第[一二三四五六七八九十百千万\d]+章",     # 第X章
    r"Chapter\s+\d+",                           # Chapter X
    r"[一二三四五六七八九十百千万]+[、，,]",      # 一、二、
    r"第[一二三四五六七八九十百千万\d]+节",     # 第X节
    r"^\d+\.\d+",                               # 1.1
    r"^\d+[、，,]",                              # 1、
    r"^【[^】]+】",                               # 【标题】
    r"^##?\s+.+",                                # Markdown 标题
]


def get_api_key() -> Optional[str]:
    return MINIMAX_API_KEY if MINIMAX_API_KEY else None


def ensure_dirs():
    """确保所有输出目录存在"""
    for dir_path in [
        RAW_DATA_DIR, PHASE0_DIR, PHASE1_DIR, PHASE2_DIR,
        SUB_SCENES_DIR,
        TACTICS_TEXT_DIR / "H", TACTICS_TEXT_DIR / "M", TACTICS_TEXT_DIR / "L",
        TACTICS_STRUCT_DIR / "H", TACTICS_STRUCT_DIR / "M", TACTICS_STRUCT_DIR / "L",
        CHROMA_DB_DIR,
    ]:
        dir_path.mkdir(parents=True, exist_ok=True)
