# Validation Criteria (PHASE 8)

## Group A — Crack rõ (Easy)

### Criteria

- Crack nhìn rõ bằng mắt
- Nền đơn giản
- Ít structure / ornament

### Expectation

- Geometry filter KHÔNG drop crack
- wrong_drop = 0

## Group B — Crack mờ / khó (Hard)

### Criteria

- Hairline crack
- Low contrast / blur / noise
- Crack có thể đứt đoạn

### Expectation

- Có false positive OK
- KHÔNG drop crack vì:

  - too_short
  - blob_like

- Nếu drop (được phép) → chỉ được vì: discontinuous

## Group C — Ngoại cảnh (Background-heavy)

### Criteria

- Tượng, viền kiến trúc, ornament
- Blob lớn, width lớn
- Có thể không có crack thật

### Expectation

- ≥ 90% region bị drop
- dropped_reason chủ yếu:

  - blob_like
  - too_thick

- Final overlay không còn blob lớn

## Acceptance Checklist

- [ ] Không còn pixel overlap sau overlap-resolve
- [ ] Crack dài không bị chia nhỏ sau merge
- [ ] Group A: wrong_drop = 0
- [ ] Group B: wrong_drop ≤ 1 / ảnh
- [ ] Group C: blob lớn không xuất hiện ở final overlay
- [ ] dropped_reason hợp lý, nhất quán
