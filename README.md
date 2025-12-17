## 1. Giới thiệu

- Pipeline phát hiện và segment vết nứt (crack) trên bề mặt (bê tông, tường, đường, kim loại…) bằng GroundingDINO (detection) + SAM (segmentation).
- Zero-shot (proposal): có thể chạy với checkpoint có sẵn mà không bắt buộc train lại cho detection/segmentation. Các trường hợp “vùng xám” (crack mờ/ngắn/đứt đoạn) là mục tiêu để bổ sung classifier nhẹ ở mức region (planned).
- Input: ảnh JPG/PNG. Output: mask crack + ảnh overlay tô màu vết nứt và các metrics định lượng.

**Cập nhật (12/2025):**

- Thêm **tight-box SAM** (segment trên ảnh crop theo box, trả mask về full image).
- Thêm **seed-first SAM** (seed mask -> sample điểm positive/negative -> gọi SAM với point prompts + optional `mask_input`).
- Thêm **geometry/topology hard-reject filter v3** (conservative, chỉ loại các vùng chắc chắn không phải crack) + xuất `features` và `dropped_reason` theo từng region.
- Thêm tab **PySide6: Region Inspector** để xem/browse từng region (kept/dropped), overlay/mask/skeleton ngay trong RAM (không save file mặc định).

Repository này hiện thực pipeline dưới dạng:

- `main.py`: entrypoint (CLI/UI).
- Thư mục `src/`: code chính (preprocess, detection, segmentation, postprocess, pipeline, utils, ui).
- Thư mục `tests/`: test cơ bản cho một số module.
- Thư mục `docs/`: tiêu chí validation + checklist regression.

---

## 2. Cấu trúc codebase

```text
CrackDetector/
├── main.py              # CLI/UI entrypoint
├── requirements.txt
├── docs/                 # Validation criteria / report template / regression checklist
├── src/
│   ├── pipeline/        # CrackDetectionPipeline và luồng xử lý chính
│   ├── preprocess/      # Tiền xử lý ảnh
│   ├── detection/       # GroundingDINO wrapper, SAHI, lọc box
│   ├── segmentation/    # SAM wrapper, sinh mask từ box/point
│   ├── postprocess/     # Morphological ops + geometry/topology filter
│   ├── models/          # Model loaders + weights (GroundingDINO/SAM)
│   ├── utils/           # Config loader, model registry/cache, helper chung
│   └── ui/              # PySide6 UI (desktop)
└── debug/               # (optional) output debug nếu bật debug.enabled
```

Các thành phần chính:

- **`CrackDetectionPipeline`** (`src/pipeline/pipeline.py`): entrypoint điều phối toàn bộ.
- **Tiền xử lý (preprocess)**: resize, lọc noise, CLAHE, optional highpass/Gabor.
- **Prompt manager**: sinh prompt fine-grained theo `material_type`, hỗ trợ multi-prompt + adaptive prompt.
- **GroundingDINO model wrapper**: detect box với dynamic threshold, SAHI-style tiling (sliding-window), shape filter.
- **`SAM model wrapper`**: segment mask theo **tight-box crop** và **seed-first prompting** (points + optional `mask_input`).
- **Post-processor**: combine masks, morphological ops, edge refinement, geometry filter.
- **Geometry + Topology filter v3** (`src/postprocess/geometry_filter_v3.py`): hard-reject ngoại cảnh rõ ràng (blob, border-following, too thick smooth edge, closed-loop, ornament-like, non-crack edge) theo feature/threshold; kèm **keep override** để giữ crack mảnh.
- **Overlap resolve + Crack merge**: planned (mục tiêu là xử lý overlap mask và nối crack dài bị cắt) — hiện chưa có stage/module dedicated.
- **Region-level Crack Classifier** (`src/classifier/` – planned): classifier nhẹ (MobileNet/ResNet18) để quyết định crack/non-crack cho các region còn lại sau hard-reject.
- **Model registry/cache**: cache model, tránh load nhiều lần.
- **Dataclass models**: `BoxResult`, `CrackMetrics`, `PipelineResult`, … (trong `src/pipeline/models.py`).
- **Model loaders**: load checkpoint + tạo predictor cho GroundingDINO/SAM (trong `src/models/loaders.py`).

Luồng tổng quát:

```text
CURRENT (đã implement):
Image
 -> Preprocessor
 -> GroundingDINO + SAM (recall-first proposal)
 -> PostProcess (morph + edge refine)
 -> Geometry/Topology (hard reject; hiện đang dùng để tạo mask cuối vì chưa có classifier)
 -> PipelineResult

TARGET (định hướng, planned):
Image
 -> Preprocessor
 -> GroundingDINO + SAM (recall-first)
 -> Overlap Resolve + Merge
 -> Geometry / Topology (HARD REJECT)
 -> Region-level ML Classifier (crack vs non-crack)
 -> PipelineResult
```

---

## 3. Tính năng chính

- **Recall-first proposal generation (zero-shot)**: GroundingDINO + SAM sinh nhiều proposal để tránh miss crack mảnh.
- **Multi-prompt & adaptive prompt**: nhiều prompt cho từng loại bề mặt (concrete, asphalt, wall,…), tự điều chỉnh theo `material_type`.
- **SAHI-style tiling (sliding-window)** cho ảnh lớn, crack nhỏ, giúp không bỏ sót vết nứt mảnh.
- **Tight-box SAM segmentation (damage-level)**: với mỗi box, crop đúng vùng box (không mở rộng), chạy SAM trên crop và paste mask về full image.
- **Seed-first SAM segmentation**:
  - Tạo seed mask trong crop (grayscale + black-hat + threshold).
  - Cleanup seed: loại component chạm border, giữ component “crack-like”, skeletonize bằng OpenCV (không dùng `skimage`).
  - Sample điểm positive dọc skeleton (farthest-point sampling) và negative từ vùng low-gradient.
  - Gọi SAM bằng `point_coords/point_labels` và **optional** `mask_input` (nếu predictor hỗ trợ).
  - Tighten `crop_box` theo bbox của seed (có guard theo tỷ lệ diện tích seed).
- **Post-processing linh hoạt**: lọc nhiễu, mở/đóng/dilate, refine biên crack.
- **Geometry + topology filter = HARD REJECT (v3)**: loại ngoại cảnh rõ ràng (blob/border-following/too thick smooth/closed-loop/ornament-like/non-crack edge). Bộ lọc này được thiết kế **conservative** để không làm rớt crack mảnh.
- **Region-level ML classifier (planned)**: quyết định crack/non-crack cho các region còn lại sau hard-reject để giảm miss crack mờ.
- **Topology filter (post-V2)**: loại closed-loop/ornament-like bằng endpoint/open_ratio và curvature_variance.
- **Edge-first fallback (khi DINO fail/conf thấp)**: Edge → Skeleton → Region Grow → Geometry Filter.
- **Confidence aggregation (final_conf)**: tổng hợp từ DINO + geometry + continuity.
- **UI (PySide6) = debug tool chính**: xem từng stage theo tab, metrics panel, region table + click highlight; toggle runtime.
- **UI: Region Inspector tab**: bảng region đầy đủ feature (kept/dropped/reason + geometry stats), click để xem `crop_overlay`, `crop_mask`, skeleton và highlight lên ảnh full.
- **Debug mode ra file là optional**: chỉ ghi ra `debug/` khi bật `debug.enabled` / `debug.log_csv`.
- **CLI & Python API**: dùng chung pipeline artifact bundle (`PipelineResult`).

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
- Checkpoint mặc định được kỳ vọng nằm trong `src/models/weights/` (xem `src/models/loaders.py`).
- Với inference thật, bạn cần cài thêm các dependency (ngoài `requirements.txt`): `groundingdino` và `segment-anything`.
- Có thể override loader qua `model_loader`, `sam_loader` khi khởi tạo pipeline.

---

## 5. Cách chạy nhanh (Quickstart)

### 5.1. CLI (`main.py`)

`main.py` là entrypoint CLI chính của project.

```bash
python main.py path/to/image.jpg --config path/to/config.yaml
```

- Nếu bạn **không truyền `image`** (hoặc dùng `--ui`) thì `main.py` sẽ **mở UI**.
- Nếu bạn **truyền `image`** thì `main.py` sẽ chạy pipeline và in metrics ra terminal.

Ví dụ:

```bash
python main.py
python main.py --ui
python main.py path/to/image.jpg --config path/to/config.yaml
```

- **`image` (positional)**: đường dẫn ảnh input (optional).
- **`--config` (optional)**: đường dẫn file YAML/JSON config. Nếu bỏ trống, pipeline sẽ dùng config mặc định bên trong `CrackDetectionPipeline`.

Kết thúc run, CLI in ra một số metrics cơ bản:

- `num_regions_before` / `num_regions_after`
- `avg_width` / `avg_length`
- `final_confidence`
- `fallback_used`

### 5.2. Python API

Ví dụ sử dụng trực tiếp trong Python:

```python
from src.pipeline.pipeline import CrackDetectionPipeline

pipeline = CrackDetectionPipeline(config=None)  # hoặc truyền file config
result = pipeline.run("path/to/image.jpg")

# Lưu ảnh overlay (BGR)
import cv2
cv2.imwrite("output_overlay.jpg", result.images["final_overlay"])

# Lấy mask sau geometry filter (ảnh visualize)
mask_img = result.images.get("geometry_filtered_mask")
```

`result` là một `PipelineResult` (artifact bundle cho UI/validation) (xem thêm ở phần *Kết quả & metrics*).

### 5.3. PySide6 UI (Debug)

Chạy UI:

```bash
python main.py
python main.py --ui
```

Trong UI:

- **Các tab ảnh**: Input / Preprocess / DINO Boxes / SAM Raw Mask / Geometry Filter / Final Overlay.
- **Region Inspector**:
  - Bảng region với cột kept/dropped_reason và các feature hình học.
  - Bộ lọc: show kept / show dropped / filter theo dropped_reason.
  - Click 1 region: xem `crop_overlay`, `crop_mask`, skeleton và highlight vùng đó trên ảnh full.

Lưu ý: mặc định UI **không ghi file**; mọi ảnh lấy trực tiếp từ `PipelineResult.images` và `PipelineResult.regions` trong bộ nhớ.

### 5.4. Batch runner (không cần UI) — `auto.py`

`auto.py` dùng để chạy hàng loạt ảnh trong một thư mục và xuất toàn bộ kết quả ra folder `result_<time>/`.

Chuẩn bị:

- Thả ảnh vào thư mục `img_input/` (đã tạo sẵn).

Chạy:

```bash
python auto.py --input-dir img_input
```

Một số tuỳ chọn thường dùng:

- `--recursive`: quét ảnh trong các thư mục con.
- `--disable-geometry`: tắt geometry filter.
- `--prompt-mode disabled|one_pass|multi_pass`: chế độ prompt damage.
- `--preprocess-variants base,blur_boost,ridge`: chọn các biến thể preprocess.

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

Lưu ý:

- `debug.enabled` hiện đang **tắt mặc định** trong `DEFAULT_CONFIG` (để không ghi file khi chạy UI/CLI).
- `geometry_filter.enabled` hiện đang **bật mặc định**.
- `sahi.enabled` hiện tương ứng với **tiling kiểu sliding-window** (SAHI-style) trong `src/detection/sahi_tiler.py`.

Geometry filter v3 được tinh chỉnh qua nhóm tham số:

- `geometry_filter.enabled`
- `geometry_filter.rules.*` (ví dụ: `min_region_area`, `blob_area_ratio_max`, `t_border`, `t_width_mean_high`, `t_width_var_low`, `t_ori_var_low`, `t_curv_var_low`, `thin_width_max`, ...)

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

Mặc định pipeline/UI **không lưu file**. Khi **`debug.enabled: true`**, pipeline sẽ lưu các artifact trung gian (mặc định trong thư mục `debug/`, có thể đổi qua `debug.output_dir`):

- `01_preprocess.jpg`
- `02_dino_boxes.jpg`
- `03_sam_raw_mask.jpg`
- `04_geometry_input.jpg`
- `05_geometry_filtered.jpg`
- `06_overlay.jpg`

Ngoài ra, nếu bật **`debug.log_csv`**, pipeline sẽ ghi log CSV (ví dụ: `debug/results.csv`) với các cột:

- `image`
- `num_regions`
- `area_ratio`
- `avg_conf`
- `final_conf`
- `time_ms`
- `fallback_used`
- `prompts`

Những file này hữu ích để:

- Kiểm tra hiệu quả prompt & threshold.
- Kiểm tra box GroundingDINO, mask từ SAM.
- Tinh chỉnh post-process (lọc nhiễu, morphological ops,…).

### 7.1. Output chuẩn khi chạy batch bằng `auto.py`

Khi chạy `auto.py`, tool sẽ tạo một thư mục theo timestamp:

```text
result_<time>/
├── result.csv                      # (TỔNG HỢP) 1 dòng / 1 ảnh
└── <img_name>/
    ├── result.csv                  # (SUMMARY) 1 dòng / ảnh
    ├── stages.csv                  # (STAGES) nhiều dòng: 1 dòng / stage
    ├── regions.csv                 # (REGIONS) nhiều dòng: 1 dòng / region
    └── step_img/
        ├── 01_input.jpg
        ├── 02_preprocess.jpg
        ├── 03_dino_boxes.jpg
        ├── 04_sam_raw_mask.jpg
        ├── 05_geometry_input.jpg
        ├── 06_geometry_kept.jpg
        ├── 07_final_mask.png
        ├── 08_final_overlay.jpg
        └── ... (các key khác nếu pipeline có)
```

Giải nghĩa:

- **`result_<time>/result.csv`**
  - File tổng hợp cho cả thư mục input.
  - Mỗi ảnh = 1 dòng.
  - Có cột `status` (`ok|failed`). Nếu `failed` sẽ có thêm cột `error`.

- **`<img_name>/result.csv`**
  - File tổng kết cho riêng 1 ảnh.
  - Chỉ có 1 dòng vì đây là metrics “tổng” sau khi pipeline chạy xong.
  - Các cột thường gặp:
    - `num_regions_before`: số vùng liên thông trước geometry filter.
    - `num_regions_after`: số vùng liên thông sau geometry filter.
    - `num_kept` / `num_dropped`: số region kept/dropped.
    - `avg_width`, `avg_length`: trung bình theo các region kept.
    - `final_confidence`: điểm tin cậy tổng hợp.
    - `fallback_used`: có dùng edge-first fallback hay không.
    - `dropped_reason_counts`: thống kê lý do drop (JSON string).
    - `time_ms_total`: thời gian xử lý.

- **`<img_name>/stages.csv`**
  - CSV “theo từng giai đoạn”. Mỗi dòng = 1 stage.
  - Có thứ tự `stage_index` để map sang file trong `step_img/`.
  - Các cột chính:
    - `stage_index`: thứ tự stage (01, 02, ...).
    - `stage_key`: key trong `PipelineResult.images`.
    - `stage_name`: tên rút gọn để đặt filename.
    - `filename`: tên file đã export.
    - `shape`, `dtype`, `min`, `max`, `sum`, `nonzero`: thống kê nhanh để debug.

- **`<img_name>/regions.csv`**
  - CSV “theo từng region”. Mỗi dòng = 1 region (được pipeline sinh ra để UI/validation).
  - Cột thường gặp:
    - `region_id`, `kept`, `dropped_reason`, `bbox`
    - geometry: `area`, `skeleton_length`, `endpoint_count`, `width_mean`, `width_var`, `length_area_ratio`, `curvature_var`
    - scores/meta: `score_dino_conf`, `meta_prompt_name`, `meta_variant_name`, ... (nếu có)
  - Dùng file này khi bạn muốn phân tích “nhiều vết nứt/region” thay vì chỉ nhìn 1 số tổng.

### 7.2. Lưu ý quan trọng về “số vết nứt” vs `num_regions_after`

`num_regions_before/after` được tính bằng **connected components** trên mask nhị phân.
Vì vậy:

- Nếu nhiều vết nứt **dính nhau/chạm nhau/nối bằng đoạn rất mảnh**, chúng có thể trở thành **1 vùng liên thông** → `num_regions_after` thấp.
- Nếu muốn “đếm vết” theo cảm nhận thị giác, thường cần metric nâng cao (skeleton graph, endpoints/branches, ...) thay vì chỉ connected components.

---

## 8. Kết quả & metrics

`PipelineResult` (định nghĩa trong `src/pipeline/models.py`) bao gồm:

- **`images: Dict[str, np.ndarray]`**: ảnh theo stage (ví dụ: `input`, `preprocessed`, `dino_boxes`, `sam_raw_mask`, `geometry_filtered_mask`, `final_overlay`, ...)
- **`metrics: Dict[str, Any]`**: metrics tổng hợp (regions before/after, avg width/length, final_confidence, fallback_used, validation counters, ...)
- **`regions: List[Dict[str, Any]]`**: region-level outputs để UI/validation:
  - **Core**: `region_id`, `bbox`, `kept`, `dropped_reason`
  - **Features**: `features: Dict[str, Any]` chứa các trường như:
    - `area`, `skeleton_length`, `len_area_ratio`
    - `width_mean`, `width_var` (và các thống kê width khác)
    - `touch_border_ratio`, `touch_border_sides`
    - `orientation_variance`, `straightness`
    - `endpoints_count`, `branchpoints_count`, `curvature_variance`
  - **Optional debug in-memory**: `crop_mask`, `crop_overlay`, `mask` (full image)

Lưu ý tương thích: một số đường chạy legacy có thể vẫn trả regions dạng “flat fields” (vd: `width_variance`, `length_area_ratio`) thay vì gói trong `features`. UI đã có fallback để đọc cả hai.

Danh sách `dropped_reason` được chuẩn hoá (enum):

- `closed_loop`
- `blob_like`
- `too_thick`
- `ornament_like`
- `border_drop`
- `merged`
- `unknown`

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

- Hoàn thiện checklist regression với bộ ảnh thực tế theo `docs/validation.md`, `docs/validation_report.md`, `docs/v2_regression_checklist.md`.
- Add overlap-resolve + crack-merge stage (để xử lý overlap mask và nối crack dài bị cắt).
- Add region-level crack classifier (sau geometry/topology hard reject) để xử lý vùng xám và giảm false negative.
- Tạo HTML report tự động cho mỗi run (nhúng metrics + ảnh overlay).
- Semi-supervised training với pseudo-label sinh ra từ pipeline.
- Tích hợp thêm các model crack chuyên biệt (fine-tune hoặc lightweight) làm back-end cho detection/segmentation.

