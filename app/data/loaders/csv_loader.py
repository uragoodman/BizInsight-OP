"""CSV 文件发现工具。

该文件提供轻量级的数据文件扫描能力，适合在引导、调试或排查数据目录结构时使用。
"""

from pathlib import Path


def list_csv_files(root: str = "kagglehub") -> list[str]:
    """递归列出指定目录下的全部 CSV 文件路径。

    Args:
        root: 递归扫描的根目录，默认值为 `kagglehub`。

    Returns:
        list[str]: 扫描到的 CSV 文件完整路径字符串列表；若未找到则返回空列表。
    """

    return [str(path) for path in Path(root).rglob("*.csv")]
