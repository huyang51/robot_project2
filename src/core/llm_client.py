"""
MiniMax API 客户端封装（从 robot_project 适配）

提供 chat_completion 和 generate_json 两个核心接口。
"""

import json
import logging
import re
import time
from typing import Any, Dict, List, Optional

import requests

from ..config import (
    get_api_key, LLM_MAX_TOKENS, LLM_TEMPERATURE,
    MINIMAX_BASE_URL, MINIMAX_MODEL
)
from .exceptions import LLMError

logger = logging.getLogger(__name__)

MAX_RETRIES = 3
INITIAL_RETRY_DELAY = 2
BACKOFF_FACTOR = 2
MIN_REQUEST_INTERVAL = 1.0


class MiniMaxClient:
    """MiniMax API 客户端"""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or get_api_key()
        self.base_url = MINIMAX_BASE_URL
        self.model = MINIMAX_MODEL
        self._last_request_time = 0.0

    def _wait_before_request(self):
        current_time = time.time()
        elapsed = current_time - self._last_request_time
        if elapsed < MIN_REQUEST_INTERVAL:
            time.sleep(MIN_REQUEST_INTERVAL - elapsed)
        self._last_request_time = time.time()

    def _request(self, messages: List[Dict[str, str]], temperature: float = LLM_TEMPERATURE,
                 max_tokens: int = LLM_MAX_TOKENS) -> Dict[str, Any]:
        if not self.api_key:
            raise LLMError("API 密钥未设置")

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens
        }

        last_error = None
        retry_count = 0

        while retry_count <= MAX_RETRIES:
            try:
                self._wait_before_request()

                response = requests.post(
                    f"{self.base_url}/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=600
                )

                if response.status_code >= 500 or response.status_code == 429:
                    last_error = f"服务器错误: {response.status_code}"
                    retry_count += 1
                    if retry_count <= MAX_RETRIES:
                        delay = INITIAL_RETRY_DELAY * (BACKOFF_FACTOR ** (retry_count - 1))
                        logger.warning(f"MiniMax API 返回 {response.status_code}，{delay}秒后重试 ({retry_count}/{MAX_RETRIES})")
                        time.sleep(delay)
                        continue
                    else:
                        break

                response.raise_for_status()
                return response.json()

            except requests.exceptions.Timeout:
                last_error = "API 请求超时"
                retry_count += 1
                if retry_count <= MAX_RETRIES:
                    delay = INITIAL_RETRY_DELAY * (BACKOFF_FACTOR ** (retry_count - 1))
                    logger.warning(f"请求超时，{delay}秒后重试 ({retry_count}/{MAX_RETRIES})")
                    time.sleep(delay)
                    continue
            except requests.exceptions.RequestException as e:
                last_error = f"API 请求失败: {str(e)}"
                if "Connection" in str(e) or "Timeout" in str(e):
                    retry_count += 1
                    if retry_count <= MAX_RETRIES:
                        delay = INITIAL_RETRY_DELAY * (BACKOFF_FACTOR ** (retry_count - 1))
                        logger.warning(f"连接错误，{delay}秒后重试 ({retry_count}/{MAX_RETRIES})")
                        time.sleep(delay)
                        continue
                break

        raise LLMError(f"{last_error}（已重试{retry_count}次）")

    def chat_completion(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = LLM_TEMPERATURE,
        max_tokens: int = LLM_MAX_TOKENS
    ) -> str:
        """调用聊天补全接口"""
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        try:
            result = self._request(messages, temperature, max_tokens)
            return result["choices"][0]["message"]["content"]
        except (KeyError, IndexError) as e:
            raise LLMError(f"解析响应失败: {str(e)}")

    def _extract_json(self, content: str) -> str:
        """从 LLM 输出中提取 JSON 内容"""
        content = content.strip()

        # 移除 <think> 标签（DeepSeek 系列模型思维链）
        content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL).strip()

        for tag in ["<think>", "</think>", "<think"]:
            if tag in content:
                content = content.split(tag)[-1].strip()

        # 提取 ```json 代码块
        if "```json" in content:
            parts = content.split("```json")
            if len(parts) > 1:
                inner = parts[1].split("```")
                if inner[0].strip():
                    return inner[0].strip()
        elif "```" in content:
            parts = content.split("```")
            if len(parts) >= 3:
                return parts[1].strip()

        # 查找 JSON 边界
        for open_char, close_char in [("{", "}"), ("[", "]")]:
            first = content.find(open_char)
            last = content.rfind(close_char)
            if first != -1 and last != -1 and last > first:
                extracted = content[first:last + 1].strip()
                if extracted:
                    return extracted

        return content.strip()

    def generate_json(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.3,
        max_tokens: int = LLM_MAX_TOKENS,
        max_retries: int = 2
    ) -> Dict[str, Any]:
        """调用 JSON 生成接口，带自动重试和修复"""
        last_error_content = None
        last_error_msg = None

        for retry_round in range(max_retries + 1):
            extra_instruction = ""
            if retry_round > 0 and last_error_msg:
                extra_instruction = (
                    f"\n\n【重要】上一轮返回的 JSON 解析失败，错误信息: {last_error_msg}。"
                    f"请检查 JSON 格式：确保所有字符串用双引号包裹、特殊字符正确转义、"
                    f"对象/数组正确闭合，不要返回截断内容。"
                )

            messages = [
                {"role": "system", "content": system_prompt + "\n\n请只返回 JSON，不要包含其他内容。" + extra_instruction},
                {"role": "user", "content": user_prompt}
            ]

            content = None  # 确保在 except 块中可安全引用
            try:
                # JSON 解析重试时降低 temperature 以获得更稳定的输出
                retry_temp = max(0.1, temperature - (retry_round * 0.1)) if retry_round > 0 else temperature
                result = self._request(messages, retry_temp, max_tokens)
                content = result["choices"][0]["message"]["content"]

                json_str = self._extract_json(content)
                last_error_content = json_str

                try:
                    return json.loads(json_str)
                except json.JSONDecodeError as parse_error:
                    raise json.JSONDecodeError(
                        f"无法解析 JSON: {parse_error.msg}",
                        parse_error.doc,
                        parse_error.pos
                    ) from parse_error

            except json.JSONDecodeError as e:
                last_error_msg = str(e)

                # 若因 "Extra data" 失败，包裹为数组，取第一个有效对象
                if "Extra data" in last_error_msg and last_error_content:
                    wrapped = "[" + last_error_content + "]"
                    try:
                        wrapped_result = json.loads(wrapped)
                        if isinstance(wrapped_result, list) and len(wrapped_result) > 0:
                            logger.info("JSON 解析修复：自动包裹多对象为数组，取第1个（共%s个）",
                                        len(wrapped_result))
                            return wrapped_result[0] if isinstance(wrapped_result[0], dict) else wrapped_result
                        logger.info("JSON 解析修复：包裹后返回非列表结果")
                        return wrapped_result if isinstance(wrapped_result, dict) else {}
                    except json.JSONDecodeError:
                        pass

                preview = (last_error_content or content or "N/A")
                logger.warning(
                    f"JSON 解析失败（第{retry_round + 1}次）: {last_error_msg}\n  内容预览: {preview}"
                )

                if retry_round < max_retries:
                    continue

                error_preview = last_error_content[:800] if last_error_content else "N/A"
                raise LLMError(f"JSON 解析失败（已重试{max_retries}次）: {last_error_msg}\n原始内容: {error_preview}")

            except (KeyError, IndexError) as e:
                raise LLMError(f"解析响应失败: {str(e)}")

    def batch_generate(
        self,
        system_prompt: str,
        user_prompts: List[str],
        temperature: float = LLM_TEMPERATURE
    ) -> List[str]:
        """批量生成，失败时返回 "" 并记录索引"""
        results = []
        failed_indices = []
        for i, user_prompt in enumerate(user_prompts):
            try:
                result = self.chat_completion(system_prompt, user_prompt, temperature)
                results.append(result)
            except LLMError as e:
                logger.warning(f"批量生成第{i + 1}个失败: {e}")
                results.append("")
                failed_indices.append(i)
        if failed_indices:
            logger.warning(f"批量生成完成: {len(user_prompts)} 个请求, "
                           f"失败 {len(failed_indices)} 个 (索引: {failed_indices})")
        return results
