import os
import json
from typing import Any, Dict, Optional

_DEFAULT_CONFIG_FILENAME = "config.json"


class Settings:
	"""
	Thin wrapper around a nested dict with dotted-path access.
	"""
	def __init__(self, data: Dict[str, Any]) -> None:
		self._data = data or {}

	def get(self, dotted_path: str, default: Optional[Any] = None) -> Any:
		node: Any = self._data
		for key in dotted_path.split("."):
			if not isinstance(node, dict) or key not in node:
				return default
			node = node[key]
		return node

	def section(self, key: str) -> Dict[str, Any]:
		value = self._data.get(key, {})
		return value if isinstance(value, dict) else {}

	def as_dict(self) -> Dict[str, Any]:
		return self._data


def _resolve_config_path() -> str:
	# Allow override via env var; fallback to repository root 'config.json'
	env_path = os.getenv("APP_CONFIG_PATH")
	if env_path:
		return env_path
	return os.path.join(os.path.dirname(os.path.abspath(__file__)), _DEFAULT_CONFIG_FILENAME)


def _load_settings(path: str) -> Settings:
	if not os.path.isfile(path):
		raise FileNotFoundError(
			f"Config file not found at '{path}'. "
			f"Create '{_DEFAULT_CONFIG_FILENAME}' at project root or set APP_CONFIG_PATH."
		)
	with open(path, "r", encoding="utf-8") as f:
		data = json.load(f)
	return Settings(data)


# Eagerly load settings on import
_CONFIG_PATH = _resolve_config_path()
settings = _load_settings(_CONFIG_PATH)


