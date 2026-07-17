from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from skimage.measure import label, regionprops


@dataclass
class Component:
    x1: int
    y1: int
    x2: int
    y2: int
    area: int

    @property
    def width(self) -> int:
        return self.x2 - self.x1

    @property
    def height(self) -> int:
        return self.y2 - self.y1

    @property
    def center_y(self) -> float:
        return (self.y1 + self.y2) / 2

    @property
    def baseline(self) -> int:
        return self.y2


@dataclass
class TextLine:
    components: list[Component]

    @property
    def x1(self) -> int:
        return min(component.x1 for component in self.components)

    @property
    def y1(self) -> int:
        return min(component.y1 for component in self.components)

    @property
    def x2(self) -> int:
        return max(component.x2 for component in self.components)

    @property
    def y2(self) -> int:
        return max(component.y2 for component in self.components)

    @property
    def center_y(self) -> float:
        return np.median(
            [component.center_y for component in self.components]
        )

    @property
    def median_baseline(self) -> float:
        return np.median(
            [component.baseline for component in self.components]
        )

    @property
    def median_height(self) -> float:
        return np.median(
            [component.height for component in self.components]
        )


def load_image(image_path: str | Path) -> np.ndarray:
    image_path = Path(image_path)

    if not image_path.exists():
        raise FileNotFoundError(f"Không tìm thấy ảnh: {image_path}")

    image = cv2.imread(str(image_path))

    if image is None:
        raise ValueError(f"Không thể đọc ảnh: {image_path}")

    return image


def preprocess_image(
    image: np.ndarray,
    remove_horizontal_lines: bool = True,
) -> np.ndarray:
    """
    Chuyển ảnh sang binary:
    - chữ: màu trắng, giá trị 255
    - nền: màu đen, giá trị 0
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Khử nhiễu nhẹ nhưng vẫn giữ nét chữ.
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)

    binary = cv2.adaptiveThreshold(
        blurred,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        blockSize=31,
        C=15,
    )

    # Loại bỏ nhiễu nhỏ.
    open_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (2, 2),
    )

    binary = cv2.morphologyEx(
        binary,
        cv2.MORPH_OPEN,
        open_kernel,
        iterations=1,
    )

    if remove_horizontal_lines:
        binary = remove_form_lines(binary)

    return binary


def remove_form_lines(binary: np.ndarray) -> np.ndarray:
    """
    Tìm và loại bỏ các đường kẻ ngang dài trong biểu mẫu.
    """
    image_width = binary.shape[1]

    horizontal_kernel_width = max(20, image_width // 20)

    horizontal_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (horizontal_kernel_width, 1),
    )

    horizontal_lines = cv2.morphologyEx(
        binary,
        cv2.MORPH_OPEN,
        horizontal_kernel,
        iterations=1,
    )

    cleaned = cv2.subtract(binary, horizontal_lines)

    return cleaned


def extract_components(
    binary: np.ndarray,
    min_area: int = 8,
    min_height: int = 3,
    max_height_ratio: float = 0.3,
    max_width_ratio: float = 0.8,
) -> list[Component]:
    """
    Tìm connected components bằng scikit-image.
    """
    labeled_image = label(
        binary > 0,
        connectivity=2,
    )

    image_height, image_width = binary.shape

    components: list[Component] = []

    for region in regionprops(labeled_image):
        min_row, min_col, max_row, max_col = region.bbox

        component = Component(
            x1=min_col,
            y1=min_row,
            x2=max_col,
            y2=max_row,
            area=int(region.area),
        )

        if component.area < min_area:
            continue

        if component.height < min_height:
            continue

        # Loại các vùng quá cao, thường là viền hoặc nhiễu lớn.
        if component.height > image_height * max_height_ratio:
            continue

        # Loại các đường hoặc vùng quá rộng.
        if component.width > image_width * max_width_ratio:
            continue

        components.append(component)

    return components


def vertical_overlap_ratio(
    first: Component,
    second: Component,
) -> float:
    """
    Độ chồng theo chiều dọc, chuẩn hóa theo component thấp hơn.
    """
    overlap = max(
        0,
        min(first.y2, second.y2) - max(first.y1, second.y1),
    )

    minimum_height = max(
        1,
        min(first.height, second.height),
    )

    return overlap / minimum_height


def component_matches_line(
    component: Component,
    line: TextLine,
    median_global_height: float,
    min_vertical_overlap: float = 0.2,
    center_distance_ratio: float = 0.65,
    baseline_distance_ratio: float = 0.8,
) -> bool:
    """
    Kiểm tra component có thuộc một dòng hiện tại hay không.
    """
    line_box = Component(
        x1=line.x1,
        y1=line.y1,
        x2=line.x2,
        y2=line.y2,
        area=1,
    )

    overlap = vertical_overlap_ratio(component, line_box)

    normalized_center_distance = (
        abs(component.center_y - line.center_y)
        / max(median_global_height, 1)
    )

    normalized_baseline_distance = (
        abs(component.baseline - line.median_baseline)
        / max(median_global_height, 1)
    )

    return (
        overlap >= min_vertical_overlap
        or normalized_center_distance <= center_distance_ratio
        or normalized_baseline_distance <= baseline_distance_ratio
    )


def cluster_components_into_lines(
    components: list[Component],
) -> list[TextLine]:
    """
    Gom connected components thành các dòng chữ.

    Component được duyệt từ trên xuống dưới. Mỗi component được gán vào
    dòng phù hợp nhất dựa trên center_y, baseline và vertical overlap.
    """
    if not components:
        return []

    median_global_height = float(
        np.median([component.height for component in components])
    )

    # Duyệt từ trên xuống, sau đó từ trái sang phải.
    sorted_components = sorted(
        components,
        key=lambda component: (
            component.center_y,
            component.x1,
        ),
    )

    lines: list[TextLine] = []

    for component in sorted_components:
        candidate_lines: list[tuple[float, TextLine]] = []

        for current_line in lines:
            if not component_matches_line(
                component,
                current_line,
                median_global_height,
            ):
                continue

            distance = abs(
                component.center_y - current_line.center_y
            )

            candidate_lines.append(
                (distance, current_line)
            )

        if candidate_lines:
            _, closest_line = min(
                candidate_lines,
                key=lambda item: item[0],
            )

            closest_line.components.append(component)
        else:
            lines.append(
                TextLine(components=[component])
            )

    lines = merge_overlapping_lines(
        lines,
        median_global_height,
    )

    lines = remove_invalid_lines(
        lines,
        median_global_height,
    )

    return sorted(
        lines,
        key=lambda line: line.center_y,
    )


def line_vertical_overlap_ratio(
    first: TextLine,
    second: TextLine,
) -> float:
    overlap = max(
        0,
        min(first.y2, second.y2) - max(first.y1, second.y1),
    )

    minimum_height = max(
        1,
        min(
            first.y2 - first.y1,
            second.y2 - second.y1,
        ),
    )

    return overlap / minimum_height


def merge_overlapping_lines(
    lines: list[TextLine],
    median_global_height: float,
) -> list[TextLine]:
    """
    Hợp nhất các dòng bị tách nhầm, ví dụ dấu tiếng Việt bị xem thành
    một dòng riêng.
    """
    if not lines:
        return []

    sorted_lines = sorted(
        lines,
        key=lambda line: line.center_y,
    )

    merged_lines: list[TextLine] = []

    for current_line in sorted_lines:
        if not merged_lines:
            merged_lines.append(current_line)
            continue

        previous_line = merged_lines[-1]

        center_distance = abs(
            current_line.center_y - previous_line.center_y
        )

        baseline_distance = abs(
            current_line.median_baseline
            - previous_line.median_baseline
        )

        overlap = line_vertical_overlap_ratio(
            previous_line,
            current_line,
        )

        should_merge = (
            overlap >= 0.25
            or center_distance <= median_global_height * 0.55
            or baseline_distance <= median_global_height * 0.5
        )

        if should_merge:
            previous_line.components.extend(
                current_line.components
            )
        else:
            merged_lines.append(current_line)

    return merged_lines


def remove_invalid_lines(
    lines: list[TextLine],
    median_global_height: float,
) -> list[TextLine]:
    """
    Loại các dòng rất nhỏ chỉ gồm nhiễu.
    """
    valid_lines: list[TextLine] = []

    for line in lines:
        line_width = line.x2 - line.x1
        line_height = line.y2 - line.y1

        total_area = sum(
            component.area for component in line.components
        )

        if total_area < 15:
            continue

        if line_width < median_global_height * 0.4:
            continue

        if line_height < 2:
            continue

        valid_lines.append(line)

    return valid_lines


def add_padding_to_lines(
    lines: list[TextLine],
    image_shape: tuple[int, ...],
    horizontal_padding: int = 5,
    vertical_padding: int = 3,
) -> list[tuple[int, int, int, int]]:
    image_height, image_width = image_shape[:2]

    boxes: list[tuple[int, int, int, int]] = []

    for line in lines:
        x1 = max(0, line.x1 - horizontal_padding)
        y1 = max(0, line.y1 - vertical_padding)
        x2 = min(image_width, line.x2 + horizontal_padding)
        y2 = min(image_height, line.y2 + vertical_padding)

        boxes.append((x1, y1, x2, y2))

    return boxes


def show_detected_lines(
    image: np.ndarray,
    boxes: list[tuple[int, int, int, int]],
    binary: np.ndarray | None = None,
) -> None:
    """
    Hiển thị ảnh gốc cùng bounding box của từng dòng.
    """
    rgb_image = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB,
    )

    if binary is not None:
        figure, axes = plt.subplots(
            1,
            2,
            figsize=(16, 9),
        )

        original_axis = axes[0]
        binary_axis = axes[1]

        binary_axis.imshow(binary, cmap="gray")
        binary_axis.set_title("Ảnh sau tiền xử lý")
        binary_axis.axis("off")
    else:
        figure, original_axis = plt.subplots(
            figsize=(12, 9),
        )

    original_axis.imshow(rgb_image)
    original_axis.set_title(
        f"Phát hiện được {len(boxes)} dòng"
    )
    original_axis.axis("off")

    for line_index, (x1, y1, x2, y2) in enumerate(
        boxes,
        start=1,
    ):
        rectangle = Rectangle(
            (x1, y1),
            x2 - x1,
            y2 - y1,
            fill=False,
            linewidth=2,
        )

        original_axis.add_patch(rectangle)

        original_axis.text(
            x1,
            max(0, y1 - 4),
            f"Line {line_index}",
            fontsize=10,
            bbox={
                "facecolor": "white",
                "alpha": 0.75,
                "edgecolor": "none",
                "pad": 1,
            },
        )

    figure.tight_layout()
    plt.show()


def crop_lines(
    image: np.ndarray,
    boxes: list[tuple[int, int, int, int]],
) -> list[np.ndarray]:
    """
    Trả về ảnh crop của từng dòng để đưa vào recognizer.
    """
    crops: list[np.ndarray] = []

    for x1, y1, x2, y2 in boxes:
        crop = image[y1:y2, x1:x2]

        if crop.size > 0:
            crops.append(crop)

    return crops


def detect_text_lines(
    image: np.ndarray,
    remove_horizontal_lines: bool = True,
) -> tuple[
    list[tuple[int, int, int, int]],
    np.ndarray,
]:
    binary = preprocess_image(
        image,
        remove_horizontal_lines=remove_horizontal_lines,
    )

    components = extract_components(binary)

    lines = cluster_components_into_lines(
        components
    )

    boxes = add_padding_to_lines(
        lines,
        image.shape,
        horizontal_padding=6,
        vertical_padding=4,
    )

    return boxes, binary


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Phát hiện từng dòng chữ bằng OpenCV và scikit-image."
        )
    )

    parser.add_argument(
        "image",
        help="Đường dẫn tới ảnh đầu vào.",
    )

    parser.add_argument(
        "--keep-horizontal-lines",
        action="store_true",
        help="Không xóa các đường kẻ ngang của biểu mẫu.",
    )

    args = parser.parse_args()

    image = load_image(args.image)

    boxes, binary = detect_text_lines(
        image,
        remove_horizontal_lines=not args.keep_horizontal_lines,
    )

    print(f"Phát hiện được {len(boxes)} dòng:")

    for index, box in enumerate(boxes, start=1):
        print(f"  Line {index}: {box}")

    show_detected_lines(
        image,
        boxes,
        binary,
    )


if __name__ == "__main__":
    main()
