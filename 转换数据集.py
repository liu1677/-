#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import io
import json
import random
import shutil
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import yaml
from PIL import Image


IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
LABEL_ALIASES = {
    "1": "block",
}


@dataclass
class Sample:
    image_path: Path
    json_path: Path
    label_lines: list[str]


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="将 Labelme 标注转换为 YOLO Segmentation 数据集。"
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=root / "labelme",
        help="Labelme 图片和 JSON 所在目录。",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=root / "datasets" ,
        help="导出的 YOLO Segmentation 数据集目录。",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.2,
        help="测试集占比，默认 0.2。",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机种子，默认 42。",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="若输出目录内有非脚本生成内容，则强制整体覆盖重建。",
    )
    return parser.parse_args()


def sorted_labels(labels: set[str]) -> list[str]:
    def label_key(value: str) -> tuple[int, int | str]:
        if value.isdigit():
            return (0, int(value))
        return (1, value)

    return sorted(labels, key=label_key)


def normalize_label_name(label: str) -> str:
    text = str(label).strip()
    return LABEL_ALIASES.get(text, text)


def resolve_image_path(input_dir: Path, json_path: Path, data: dict) -> Path:
    image_path = data.get("imagePath")
    if image_path:
        candidate = (input_dir / image_path).resolve()
        if candidate.exists():
            return candidate

        candidate = (json_path.parent / image_path).resolve()
        if candidate.exists():
            return candidate

    for suffix in IMAGE_SUFFIXES:
        candidate = json_path.with_suffix(suffix)
        if candidate.exists():
            return candidate

    raise FileNotFoundError(f"未找到与 {json_path.name} 对应的图片文件。")


def decode_mask(mask_b64: str) -> np.ndarray:
    mask_image = Image.open(io.BytesIO(base64.b64decode(mask_b64)))
    mask = np.array(mask_image)
    if mask.ndim == 3:
        mask = mask[..., 0]
    return (mask > 0).astype(np.uint8)


def contour_to_yolo_line(class_id: int, contour: np.ndarray, image_w: int, image_h: int) -> str | None:
    if contour.ndim != 2 or contour.shape[0] < 3:
        return None

    contour = contour.astype(np.float32)
    contour[:, 0] = np.clip(contour[:, 0], 0, image_w - 1)
    contour[:, 1] = np.clip(contour[:, 1], 0, image_h - 1)
    contour[:, 0] /= image_w
    contour[:, 1] /= image_h

    flat = contour.reshape(-1)
    if flat.size < 6:
        return None

    coords = " ".join(f"{value:.6f}" for value in flat)
    return f"{class_id} {coords}"


def mask_shape_to_lines(shape: dict, class_id: int, image_w: int, image_h: int) -> list[str]:
    if "mask" not in shape:
        raise ValueError("shape_type=mask 的标注缺少 mask 字段。")

    points = shape.get("points", [])
    if len(points) != 2:
        raise ValueError("当前脚本要求 mask 标注的 points 为左上/右下两个点。")

    (x1, y1), (x2, y2) = points
    x_min = int(round(min(x1, x2)))
    y_min = int(round(min(y1, y2)))
    x_max = int(round(max(x1, x2)))
    y_max = int(round(max(y1, y2)))

    mask = decode_mask(shape["mask"])
    expected_w = x_max - x_min + 1
    expected_h = y_max - y_min + 1
    if mask.shape[:2] != (expected_h, expected_w):
        raise ValueError(
            "mask 尺寸与 points 定义的包围框不一致："
            f"{mask.shape[:2]} != {(expected_h, expected_w)}"
        )

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    lines: list[str] = []
    for contour in contours:
        contour = contour.reshape(-1, 2)
        if contour.shape[0] < 3:
            continue
        if cv2.contourArea(contour.astype(np.float32)) < 2:
            continue

        contour = contour.copy()
        contour[:, 0] += x_min
        contour[:, 1] += y_min
        line = contour_to_yolo_line(class_id, contour, image_w, image_h)
        if line:
            lines.append(line)

    return lines


def polygon_shape_to_lines(shape: dict, class_id: int, image_w: int, image_h: int) -> list[str]:
    contour = np.array(shape.get("points", []), dtype=np.float32)
    if contour.shape[0] < 3:
        return []

    if np.allclose(contour[0], contour[-1]):
        contour = contour[:-1]

    line = contour_to_yolo_line(class_id, contour, image_w, image_h)
    return [line] if line else []


def rectangle_shape_to_lines(shape: dict, class_id: int, image_w: int, image_h: int) -> list[str]:
    points = shape.get("points", [])
    if len(points) != 2:
        return []

    (x1, y1), (x2, y2) = points
    contour = np.array(
        [
            [x1, y1],
            [x2, y1],
            [x2, y2],
            [x1, y2],
        ],
        dtype=np.float32,
    )
    line = contour_to_yolo_line(class_id, contour, image_w, image_h)
    return [line] if line else []


def shape_to_lines(shape: dict, class_id: int, image_w: int, image_h: int) -> list[str]:
    shape_type = shape.get("shape_type", "polygon")
    if shape_type == "mask":
        return mask_shape_to_lines(shape, class_id, image_w, image_h)
    if shape_type == "polygon":
        return polygon_shape_to_lines(shape, class_id, image_w, image_h)
    if shape_type == "rectangle":
        return rectangle_shape_to_lines(shape, class_id, image_w, image_h)
    raise ValueError(f"暂不支持的 shape_type: {shape_type}")


def collect_class_names(input_dir: Path) -> list[str]:
    labels: set[str] = set()
    for json_path in sorted(input_dir.glob("*.json")):
        data = json.loads(json_path.read_text(encoding="utf-8"))
        for shape in data.get("shapes", []):
            labels.add(normalize_label_name(shape["label"]))

    if not labels:
        raise ValueError("没有在 Labelme JSON 中发现任何类别。")
    return sorted_labels(labels)


def build_samples(input_dir: Path, class_to_id: dict[str, int]) -> list[Sample]:
    samples: list[Sample] = []
    for json_path in sorted(input_dir.glob("*.json")):
        data = json.loads(json_path.read_text(encoding="utf-8"))
        image_path = resolve_image_path(input_dir, json_path, data)
        image_w = int(data["imageWidth"])
        image_h = int(data["imageHeight"])

        lines: list[str] = []
        for shape in data.get("shapes", []):
            class_name = normalize_label_name(shape["label"])
            class_id = class_to_id[class_name]
            lines.extend(shape_to_lines(shape, class_id, image_w, image_h))

        samples.append(Sample(image_path=image_path, json_path=json_path, label_lines=lines))

    if not samples:
        raise ValueError(f"在 {input_dir} 中没有找到任何 JSON 标注文件。")
    return samples


def prepare_output_dir(output_dir: Path, overwrite: bool) -> None:
    managed_names = {"images", "labels", "data.yaml"}

    if output_dir.exists():
        if overwrite:
            shutil.rmtree(output_dir)
        else:
            unexpected_items = [
                child.name
                for child in output_dir.iterdir()
                if child.name not in managed_names and not child.name.startswith(".")
            ]
            if unexpected_items:
                raise FileExistsError(
                    f"输出目录中存在非脚本生成内容：{output_dir}\n"
                    f"检测到额外文件: {unexpected_items}\n"
                    "如确认整体覆盖，请加上 --overwrite。"
                )

            for name in managed_names:
                target = output_dir / name
                if not target.exists():
                    continue
                if target.is_dir():
                    shutil.rmtree(target)
                else:
                    target.unlink()

    for split in ("train", "test"):
        (output_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_dir / "labels" / split).mkdir(parents=True, exist_ok=True)


def split_samples(samples: list[Sample], test_ratio: float, seed: int) -> tuple[list[Sample], list[Sample]]:
    if not 0 < test_ratio < 1:
        raise ValueError("--test-ratio 必须在 0 和 1 之间。")

    shuffled = samples[:]
    random.Random(seed).shuffle(shuffled)

    total = len(shuffled)
    if total == 1:
        return shuffled, []

    test_count = int(round(total * test_ratio))
    test_count = max(1, min(test_count, total - 1))
    train_count = total - test_count
    return shuffled[:train_count], shuffled[train_count:]


def export_split(output_dir: Path, split: str, samples: list[Sample]) -> None:
    image_dir = output_dir / "images" / split
    label_dir = output_dir / "labels" / split

    for sample in samples:
        target_image = image_dir / sample.image_path.name
        target_label = label_dir / f"{sample.image_path.stem}.txt"
        shutil.copy2(sample.image_path, target_image)
        target_label.write_text("\n".join(sample.label_lines), encoding="utf-8")


def write_yaml(output_dir: Path, class_names: list[str]) -> Path:
    yaml_path = output_dir / "data.yaml"
    yaml_data = {
        "path": "datasets",
        "train": "images/train",
        "val": "images/test",
        "test": "images/test",
        "nc": len(class_names),
        "names": class_names,
    }
    yaml_path.write_text(
        yaml.safe_dump(yaml_data, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    return yaml_path


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()

    if not input_dir.exists():
        raise FileNotFoundError(f"输入目录不存在：{input_dir}")

    class_names = collect_class_names(input_dir)
    class_to_id = {name: idx for idx, name in enumerate(class_names)}
    samples = build_samples(input_dir, class_to_id)
    train_samples, test_samples = split_samples(samples, args.test_ratio, args.seed)

    prepare_output_dir(output_dir, args.overwrite)
    export_split(output_dir, "train", train_samples)
    export_split(output_dir, "test", test_samples)
    yaml_path = write_yaml(output_dir, class_names)

    total_instances = sum(len(sample.label_lines) for sample in samples)
    print(f"转换完成: {output_dir}")
    print(f"data.yaml: {yaml_path}")
    print(f"类别: {class_names}")
    print(f"样本总数: {len(samples)}")
    print(f"实例总数: {total_instances}")
    print(f"训练集: {len(train_samples)}")
    print(f"测试集: {len(test_samples)}")


if __name__ == "__main__":
    main()
