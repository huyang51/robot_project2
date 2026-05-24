"""
Phase 2 CLI 人工审核工具

提供交互式命令行界面，供人工审核和修改 LLM 生成的子场景定义。
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

from .schemas import SubSceneDef

logger = logging.getLogger(__name__)


class SubSceneReviewer:
    """子场景人工审核 CLI"""

    def __init__(self, definitions_path: str):
        self._current_path = definitions_path
        with open(definitions_path, "r", encoding="utf-8") as f:
            self.data = json.load(f)

        self.sub_scenes: List[SubSceneDef] = [
            SubSceneDef.from_dict(s) for s in self.data.get("sub_scenes", [])
        ]
        self.graph = self.data.get("sub_scene_graph", {})

    def run(self) -> bool:
        """启动交互式审核会话

        Returns:
            True 如果用户确认所有定义
        """
        print(f"\n{'='*60}")
        print(f"Phase 2 子场景审核")
        print(f"{'='*60}")
        print(f"场景: {self.data.get('parent_scene_id', 'N/A')}")
        print(f"子场景数: {len(self.sub_scenes)}")
        print()

        # 显示摘要
        for i, ss in enumerate(self.sub_scenes):
            sp = ss.space_profile or {}
            shape = sp.get("shape", "?")
            floor = ss.floor if ss.floor is not None else "?"
            print(f"  [{ss.sub_scene_id}] shape={shape}, floor={floor}, priority={ss.priority}")
            print(f"    primary_role: {ss.primary_role}")
            if ss.suggested_roles:
                print(f"    suggested_roles: {', '.join(ss.suggested_roles)}")
            print(f"    task_hint: {ss.task_hint}")
            print(f"    zones: {', '.join(ss.zone_ids)}")
            print(f"    连接: {', '.join(ss.connected_sub_scenes) if ss.connected_sub_scenes else '(无)'}")
            print()

        # 显示入口选项
        entries = self.data.get("entry_options", [])
        if entries:
            print(f"--- 潜在入口选项 ({len(entries)} 个) ---")
            for e in entries:
                print(f"  {e.get('entry_id')}: {e.get('type')}, floor={e.get('floor')}, access={e.get('access_means')}")
            print()

        # 检查连通性
        self._check_connectivity()

        while True:
            cmd = input("\n操作 [a=接受, m=修改, r=重排序, d=删除, q=退出]: ").strip().lower()

            if cmd == 'a':
                return self._save_and_exit()
            elif cmd == 'm':
                self._modify_sub_scene()
            elif cmd == 'r':
                self._reorder()
            elif cmd == 'd':
                self._delete_sub_scene()
            elif cmd == 'q':
                return False
            else:
                print("无效命令")

    def _check_connectivity(self):
        """检查子场景连通性"""
        ids = {ss.sub_scene_id for ss in self.sub_scenes}
        referenced = set()
        for ss in self.sub_scenes:
            for conn in ss.connected_sub_scenes:
                referenced.add(conn)

        orphan = referenced - ids
        if orphan:
            print(f"\n[警告] 以下子场景被引用但不存在: {orphan}")

        # 检查是否有孤立节点（出度和入度都为 0）
        all_connections = set()
        for ss in self.sub_scenes:
            all_connections.update(ss.connected_sub_scenes)
        for ss in self.sub_scenes:
            if not ss.connected_sub_scenes and ss.sub_scene_id not in all_connections:
                if len(self.sub_scenes) > 1:
                    print(f"[警告] {ss.sub_scene_id} 是孤立节点")

    def _modify_sub_scene(self):
        ssid = input("输入要修改的子场景 ID: ").strip()
        ss = next((s for s in self.sub_scenes if s.sub_scene_id == ssid), None)
        if not ss:
            print(f"未找到子场景 {ssid}")
            return

        print(f"当前: primary_role={ss.primary_role}, suggested_roles={ss.suggested_roles}, priority={ss.priority}")
        new_role = input("新 primary_role (回车保持): ").strip()
        if new_role:
            ss.primary_role = new_role

        new_priority = input("新 priority (high/medium/low, 回车保持): ").strip()
        if new_priority in ("high", "medium", "low"):
            ss.priority = new_priority

        new_task = input("新 task_hint (回车保持): ").strip()
        if new_task:
            ss.task_hint = new_task

        print(f"已更新 {ssid}")

    def _reorder(self):
        print("当前子场景顺序:")
        for i, ss in enumerate(self.sub_scenes):
            print(f"  {i}: [{ss.sub_scene_id}] {ss.primary_role}")

        try:
            old_idx = int(input("要移动的子场景索引: ").strip())
            new_idx = int(input("目标位置索引: ").strip())
            if 0 <= old_idx < len(self.sub_scenes) and 0 <= new_idx < len(self.sub_scenes):
                ss = self.sub_scenes.pop(old_idx)
                self.sub_scenes.insert(new_idx, ss)
                print("已重新排序")
        except ValueError:
            print("无效索引")

    def _delete_sub_scene(self):
        ssid = input("输入要删除的子场景 ID: ").strip()
        self.sub_scenes = [s for s in self.sub_scenes if s.sub_scene_id != ssid]
        # 清理引用
        for ss in self.sub_scenes:
            ss.connected_sub_scenes = [c for c in ss.connected_sub_scenes if c != ssid]
        print(f"已删除 {ssid}")

    def _save_and_exit(self) -> bool:
        """保存修改并退出"""
        self.data["sub_scenes"] = [ss.to_dict() for ss in self.sub_scenes]
        # 更新图
        nodes = [ss.sub_scene_id for ss in self.sub_scenes]
        self.data["sub_scene_graph"]["nodes"] = nodes

        save = input("保存修改? [y/N]: ").strip().lower()
        if save == 'y':
            # 写回原文件
            with open(self._current_path, "w", encoding="utf-8") as f:
                json.dump(self.data, f, ensure_ascii=False, indent=2)
            print("已保存")
        return True
