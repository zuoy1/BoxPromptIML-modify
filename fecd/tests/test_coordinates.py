"""Small deterministic checks for the selected native-to-cache coordinate rules."""

import numpy as np

from fecd.coordinates import rasterize_box_xyxy, resize_probability, scale_box_xyxy, synchronized_rot90


def main() -> None:
    native = np.zeros((256, 384), dtype=np.float32)
    native[64:192, 96:288] = 1.0
    probability = resize_probability(native, (224, 224))
    assert probability.shape == (224, 224)
    assert probability.dtype == np.float32
    assert np.isfinite(probability).all() and 0 <= probability.min() <= probability.max() <= 1

    box = scale_box_xyxy((96, 64, 288, 192), (256, 384), (224, 224))
    assert np.allclose(box, [56, 56, 168, 168])
    box_mask = rasterize_box_xyxy(box, (224, 224))
    assert box_mask.shape == (224, 224) and box_mask.sum() == 112 * 112

    image = np.zeros((8, 8, 3), dtype=np.uint8)
    teacher = np.zeros((8, 8), dtype=np.float32)
    evidence = np.zeros((8, 8), dtype=np.float32)
    image[1, 2, :] = 255
    teacher[1, 2] = 0.7
    evidence[1, 2] = 0.9
    image2, teacher2, evidence2 = synchronized_rot90(image, teacher, evidence, k=1)
    assert np.array_equal(image2[5, 1], [255, 255, 255])
    assert np.isclose(teacher2[5, 1], 0.7) and np.isclose(evidence2[5, 1], 0.9)
    print({"status": "passed", "probability_shape": list(probability.shape), "scaled_box_xyxy": box.tolist(), "box_pixels": int(box_mask.sum()), "rot90_synchronized": True})


if __name__ == "__main__":
    main()
