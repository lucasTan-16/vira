"""从 PPE train 集固定随机抽取校准图，复制到 calib/images 并生成 dataset.txt。

校准只用 train，绝不用 val/test，避免量化校准泄漏到评估集。
固定随机种子保证可复现。
"""
import random
import shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent
TRAIN_IMG_DIR = HERE.parent / "datasets_ppe6" / "images" / "train"
CALIB_DIR = HERE / "calib"
CALIB_IMG_DIR = CALIB_DIR / "images"
LIST_PATH = CALIB_DIR / "dataset.txt"

N_CALIB = 300
SEED = 42
IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp")


def main():
    imgs = sorted(
        p for p in TRAIN_IMG_DIR.iterdir() if p.suffix.lower() in IMG_EXTS
    )
    assert len(imgs) >= N_CALIB, f"train images {len(imgs)} < {N_CALIB}"

    rng = random.Random(SEED)
    rng.shuffle(imgs)
    calib = imgs[:N_CALIB]

    CALIB_IMG_DIR.mkdir(parents=True, exist_ok=True)
    for old in CALIB_IMG_DIR.iterdir():
        if old.is_file():
            old.unlink()

    for src in calib:
        shutil.copy2(src, CALIB_IMG_DIR / src.name)

    lines = [f"images/{p.name}" for p in calib]
    LIST_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"train images: {len(imgs)}")
    print(f"copied {len(calib)} calib images -> {CALIB_IMG_DIR}")
    print(f"dataset list -> {LIST_PATH}")


if __name__ == "__main__":
    main()
