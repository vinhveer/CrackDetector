## 1. Giới thiệu

- Pipeline phát hiện và segment vết nứt (crack) trên bề mặt (bê tông, tường, đường, kim loại…) bằng GroundingDINO (detection) + SAM (segmentation).
- Zero-shot: chỉ cần checkpoint có sẵn, không bắt buộc train lại.
- Input: ảnh JPG/PNG. Output: mask crack + ảnh overlay tô màu vết nứt và các metrics định lượng.

Repository này hiện thực pipeline dưới dạng:

- `main.py`: CLI entrypoint.
- Thư mục `src/`: code chính (preprocess, detection, segmentation, postprocess, pipeline, utils, ui).
- Thư mục `tests/`: test cơ bản cho một số module.

---

## 2. Cấu trúc codebase

```text
CrackDetector/
├── main.py              # CLI entrypoint
├── requirements.txt
├── src/
│   ├── pipeline/        # CrackDetectionPipeline và luồng xử lý chính
│   ├── preprocess/      # Tiền xử lý ảnh
│   ├── detection/       # GroundingDINO wrapper, SAHI, lọc box
│   ├── segmentation/    # SAM wrapper, sinh mask từ box/point
│   ├── postprocess/     # Morphological ops, lọc nhiễu, refine biên
│   ├── models/          # Dataclass: BoxResult, CrackResult, CrackMetrics, ...
│   ├── utils/           # Config loader, model registry/cache, helper chung
│   └── ui/              # Thành phần phục vụ UI (nếu dùng web/desktop)
└── debug/               # Thư mục lưu output debug (có thể cấu hình)
```

Các thành phần chính:

- **`CrackDetectionPipeline`** (`src/pipeline/pipeline.py`): entrypoint điều phối toàn bộ.
- **Tiền xử lý (preprocess)**: resize, lọc noise, CLAHE, optional highpass/Gabor.
- **Prompt manager**: sinh prompt fine-grained theo `material_type`, hỗ trợ multi-prompt + adaptive prompt.
- **GroundingDINO model wrapper**: detect box với dynamic threshold, SAHI/multi-scale, shape filter.
- **SAM model wrapper**: segment mask từ box/point prompt (chọn variant qua config).
- **Post-processor**: lọc vùng nhỏ, morphological ops (opening/closing/dilate), edge refinement.
- **Model registry/cache**: cache model, tránh load nhiều lần.
- **Dataclass models**: `BoxResult`, `CrackResult`, `CrackMetrics`, …

Luồng tổng quát:

```text
Image -> Preprocessor -> GroundingDINO -> SAM -> PostProcessor -> CrackResult
```

---

## 3. Tính năng chính

- **Zero-shot crack detection** với GroundingDINO + SAM (không cần train lại).
- **Multi-prompt & adaptive prompt**: nhiều prompt cho từng loại bề mặt (concrete, asphalt, wall,…), tự điều chỉnh theo `material_type`.
- **SAHI/multi-scale** cho ảnh lớn, crack nhỏ, giúp không bỏ sót vết nứt mảnh.
- **Post-processing linh hoạt**: lọc nhiễu, mở/đóng/dilate, refine biên crack.
- **Debug mode chi tiết**: lưu từng bước (preprocess, box, mask, overlay) để dễ chẩn đoán.
- **Metrics định lượng**: số vùng crack, tỉ lệ diện tích, thời gian xử lý, avg confidence.
- **CLI & Python API**: dễ tích hợp vào pipeline khác hoặc UI.

---

## 4. Yêu cầu hệ thống & cài đặt

- Python **3.10+**.
- GPU (CUDA) **khuyến nghị**, nhưng có thể chạy CPU (sẽ chậm hơn).

**Cài đặt môi trường:**

```bash
python -m venv .venv && source .venv/bin/activate  # tùy chọn
pip install -r requirements.txt
```

**Checkpoint model:**

- Chuẩn bị checkpoint GroundingDINO + SAM tương ứng.
- Chỉnh đường dẫn trong phần loader/model registry (ví dụ qua `model_loader`, `sam_loader`) hoặc trong file config nếu bạn bổ sung.

---

## 5. Cách chạy nhanh (Quickstart)

### 5.1. CLI (`main.py`)

`main.py` là entrypoint CLI chính của project.

```bash
python main.py path/to/image.jpg --config path/to/config.yaml
```

- **`image` (positional)**: đường dẫn ảnh input.
- **`--config` (optional)**: đường dẫn file YAML/JSON config. Nếu bỏ trống, pipeline sẽ dùng config mặc định bên trong `CrackDetectionPipeline`.

Kết thúc run, CLI in ra một số metrics cơ bản:

- `num_regions`
- `area_ratio`
- `avg_conf`
- `time_ms`
- `fallback_used`

### 5.2. Python API

Ví dụ sử dụng trực tiếp trong Python:

```python
from src.pipeline.pipeline import CrackDetectionPipeline

pipeline = CrackDetectionPipeline(config_path_or_dict=None)  # hoặc truyền file config
result = pipeline.run("path/to/image.jpg")

# Lưu ảnh overlay
import cv2
cv2.imwrite("output_overlay.jpg", result.overlay_image)
```

`result` là một `CrackResult` (xem thêm ở phần *Kết quả & metrics*).

---

## 6. Cấu hình (config)

- Config mặc định được load trong `CrackDetectionPipeline` (thông qua util config loader trong `src/utils/`).
- Bạn có thể:
  - Truyền **đường dẫn file YAML/JSON** khi gọi CLI (`--config`).
  - Hoặc truyền **dict config** trực tiếp khi khởi tạo `CrackDetectionPipeline` trong Python.

Một số nhóm tham số chính (tên có thể thay đổi nhẹ tùy implement):

- **`material_type`**: loại bề mặt (concrete, asphalt, wall,…).
- **`prompts.*`**:
  - `prompts.primary`
  - `prompts.secondary`
  - `prompts.adaptive`
- **`threshold.*`**:
  - `threshold.base`, `threshold.low`
  - `threshold.use_dynamic`, `threshold.quantile`
- **`preprocess.*`**:
  - `preprocess.noise_filter`
  - `preprocess.clahe`
  - `preprocess.target_size`
- **`postprocess.*`**:
  - `postprocess.min_region_area`
  - `postprocess.dilate_iters`
  - `postprocess.edge_refine`
- **`sahi.*`**:
  - `sahi.enabled`
  - `sahi.tile_size`
  - `sahi.overlap`
- **`debug.*`**:
  - `debug.enabled`
  - `debug.output_dir`
  - `debug.log_csv`

Ví dụ snippet YAML:

```yaml
material_type: concrete
prompts:
  primary: "hairline crack on {material} surface"
  secondary:
    - "tiny surface fracture on {material}"
    - "narrow linear crack"
threshold:
  base: 0.35
  low: 0.2
  use_dynamic: true
preprocess:
  noise_filter: bilateral
  clahe: true
postprocess:
  min_region_area: 50
  dilate_iters: 1
sahi:
  enabled: true
  tile_size: 512
  overlap: 0.2
debug:
  enabled: true
  output_dir: debug
  log_csv: true
```

---

## 7. Debug mode & output

Khi **`debug.enabled: true`**, pipeline sẽ lưu các artifact trung gian (mặc định trong thư mục `debug/`, có thể đổi qua `debug.output_dir`):

- `step1_preprocessed.jpg`
- `step2_boxes.jpg`
- `step3_masks.jpg`
- `step4_refined_mask.jpg`
- `step5_overlay.jpg`

Ngoài ra, nếu bật **`debug.log_csv`**, pipeline sẽ ghi log CSV (ví dụ: `debug/results.csv`) với các cột:

- `image`
- `num_regions`
- `area_ratio`
- `avg_conf`
- `time_ms`
- `fallback_used`
- `prompts`

Những file này hữu ích để:

- Kiểm tra hiệu quả prompt & threshold.
- Kiểm tra box GroundingDINO, mask từ SAM.
- Tinh chỉnh post-process (lọc nhiễu, morphological ops,…).

---

## 8. Kết quả & metrics

`CrackResult` (định nghĩa trong `src/models/`) thường bao gồm:

- **Ảnh/mask**:
  - `final_mask`
  - `overlay_image`
- **Box & vùng**:
  - `boxes` (danh sách `BoxResult`)
- **Metrics** (`CrackMetrics`):
  - `num_regions`
  - `crack_area_ratio`
  - `avg_confidence`
  - `processing_time_ms`
- **Thông tin khác**:
  - `used_prompts`
  - `fallback_used`

---

## 9. Gợi ý prompt & mẹo sử dụng

**Prompt mẫu:**

- "hairline crack on concrete surface"
- "tiny surface fracture on asphalt road"
- "narrow linear crack on wall"
- "branching crack pattern on painted concrete"

**Mẹo:**

- Ảnh có **bóng đổ mạnh**: thêm material cụ thể trong prompt (`concrete wall`, `asphalt road`,…).
- Crack **mờ, khó thấy**: bật multi-prompt, giảm `threshold.low` để bắt thêm vùng nghi ngờ.
- Ảnh **nhiễu nhiều**: tăng `postprocess.min_region_area`, chỉnh prompt cụ thể hơn để tránh pick nhầm texture.

---

## 10. Roadmap / TODO

- Thêm web UI upload/preview overlay (sử dụng lại module `src/ui/`).
- Tạo HTML report tự động cho mỗi run (nhúng metrics + ảnh overlay).
- Semi-supervised training với pseudo-label sinh ra từ pipeline.
- Tích hợp thêm các model crack chuyên biệt (fine-tune hoặc lightweight) làm back-end cho detection/segmentation.

