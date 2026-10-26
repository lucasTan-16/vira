from pathlib import Path
import shutil

SRC = Path("/home/lucas/vira/vision/datasets")
DST = Path("/home/lucas/vira/vision/datasets_ppe6")

# 原类别id -> 新类别id；不在表中的类删除
KEEP = {6: 0, 5: 0, 0: 1, 1: 2, 2: 3, 3: 4, 4: 5}
NEW_NAMES = ["Person", "helmet", "gloves", "vest", "boots", "goggles"]
SPLITS = ["train", "val", "test"]


def main():
    kept = {n: 0 for n in NEW_NAMES}
    dropped = {}
    empty_after = 0

    for split in SPLITS:
        src_img = SRC / "images" / split
        src_lbl = SRC / "labels" / split
        dst_img = DST / "images" / split
        dst_lbl = DST / "labels" / split
        dst_img.mkdir(parents=True, exist_ok=True)
        dst_lbl.mkdir(parents=True, exist_ok=True)

        n_img = 0
        for img in src_img.iterdir():
            if img.suffix.lower() in (".jpg", ".jpeg", ".png"):
                dst = dst_img / img.name
                if not dst.exists():
                    shutil.copy2(img, dst)
                n_img += 1

        n_lbl = 0
        for lbl in src_lbl.glob("*.txt"):
            new_lines = []
            for line in lbl.read_text().splitlines():
                p = line.split()
                if len(p) != 5:
                    continue
                old = int(p[0])
                if old in KEEP:
                    new = KEEP[old]
                    kept[NEW_NAMES[new]] += 1
                    new_lines.append(f"{new} {p[1]} {p[2]} {p[3]} {p[4]}")
                else:
                    dropped[old] = dropped.get(old, 0) + 1
            if not new_lines:
                empty_after += 1
            (dst_lbl / lbl.name).write_text(
                "\n".join(new_lines) + ("\n" if new_lines else "")
            )
            n_lbl += 1

        print(f"{split}: {n_img} images, {n_lbl} labels")

    yaml_text = "path: /home/lucas/vira/vision/datasets_ppe6\n"
    yaml_text += "train: images/train\nval: images/val\ntest: images/test\n"
    yaml_text += "names:\n"
    for i, n in enumerate(NEW_NAMES):
        yaml_text += f"  {i}: {n}\n"
    (DST / "data.yaml").write_text(yaml_text)

    print("\n=== kept boxes ===")
    for n in NEW_NAMES:
        print(f"  {n}: {kept[n]}")
    print("\n=== dropped boxes (old id: count) ===")
    for old in sorted(dropped):
        print(f"  old_id {old}: {dropped[old]}")
    print(f"\n=== labels empty after conversion: {empty_after} ===")
    print("data.yaml written.")


if __name__ == "__main__":
    main()
