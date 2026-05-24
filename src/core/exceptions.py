"""
GTKG-CM Pipeline 自定义异常层次结构
"""


class GTKGError(Exception):
    """GTKG-CM 流水线基础异常"""
    pass


class LLMError(GTKGError):
    """LLM 调用错误"""
    pass


class USDAParseError(GTKGError):
    """USDA 文件解析错误"""
    pass


class ValidationError(GTKGError):
    """数据验证错误"""
    pass


class GeometryError(GTKGError):
    """几何计算错误"""
    pass


class PhaseError(GTKGError):
    """Phase 执行错误"""
    pass


class CollectionError(GTKGError):
    """ChromaDB Collection 操作错误"""
    pass


class RetrievalError(GTKGError):
    """知识库检索错误"""
    pass


class TacticGenerateError(GTKGError):
    """战术生成错误"""
    pass


class TacticReviewError(GTKGError):
    """战术审查错误"""
    pass


class TacticEvalError(GTKGError):
    """战术评估错误"""
    pass


class EmbeddingError(GTKGError):
    """嵌入向量生成错误"""
    pass
