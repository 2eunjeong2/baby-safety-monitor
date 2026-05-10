"""
데이터셋 다운로드 스크립트 v2
기존: SabiCare + VTAR (사선·측면 촬영 위주)
신규: Infant-Respiration + Baby-Sleeping + Baby-Monitoring (오버헤드 포함)

클래스 매핑:
  정면(0): back, supine, face-up 계열
  측면(1): side, lateral, sideways 계열
  후면(2): stomach, prone, tummy, crawl 계열

실행:
  python3 ai/download_dataset.py --api-key <YOUR_KEY>           # 신규 데이터셋만 추가
  python3 ai/download_dataset.py --api-key <YOUR_KEY> --fresh   # 전체 초기화 후 재다운로드
  python3 ai/download_dataset.py --api-key <YOUR_KEY> --remerge # 다운로드 유지, 병합만 재실행
"""

import argparse
import shutil
import yaml
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEST = ROOT / "ai" / "dataset_v4"

CLASSES = {
    0: "정면",
    1: "측면",
    2: "후면",
}

# ── 기존 데이터셋 명시적 매핑 ──────────────────────────────────
SABICARE_MAP = {
    "baby-lying-on-back":       0,
    "baby-lying-on-left-side":  1,
    "baby-lying-on-right-side": 1,
    "baby-lying-on-stomach":    2,
    "baby-crawling":            2,
}

VTAR_MAP = {
    "supine":   0,
    "sideways": 1,
    "prone":    2,
}

# ── 신규 데이터셋용 키워드 자동 매핑 ──────────────────────────
class KeywordMapper:
    """
    클래스 이름에 키워드가 포함되면 자동으로 정면/측면/후면으로 분류.
    dict처럼 동작하므로 기존 병합 로직과 호환된다.
    """
    _FRONT = ["back", "supine", "face up", "faceup", "on back", "lying back",
              "dorsal", "face-up"]
    _SIDE  = ["side", "lateral", "sideways", "left side", "right side",
              "lying side", "side lying"]
    _BACK  = ["stomach", "prone", "tummy", "belly", "crawl",
              "face down", "facedown", "face-down", "lying front",
              "ventral", "on stomach"]

    def _classify(self, raw_name: str):
        name = raw_name.lower().strip().replace("-", " ").replace("_", " ")
        if any(k in name for k in self._FRONT): return 0
        if any(k in name for k in self._SIDE):  return 1
        if any(k in name for k in self._BACK):  return 2
        return None

    def __contains__(self, key: str) -> bool:
        return self._classify(key) is not None

    def __getitem__(self, key: str) -> int:
        result = self._classify(key)
        if result is None:
            raise KeyError(key)
        return result

KEYWORD_MAP = KeywordMapper()

# ── 데이터셋 목록 ──────────────────────────────────────────────
DATASETS = [
    # 기존 (사선·측면 촬영 — 다양한 각도 기반)
    {
        "workspace": "sabicare",
        "project":   "sabicare-for-training",
        "version":   1,
        "name":      "sabicare",
        "cls_map":   SABICARE_MAP,
    },
    {
        "workspace": "vtar",
        "project":   "baby-object-detection-final",
        "version":   1,
        "name":      "vtar",
        "cls_map":   VTAR_MAP,
    },
    # 신규 (오버헤드/모니터링 특화 — 정면 오탐 해결 핵심)
    {
        "workspace": "infant-respiration",
        "project":   "sleeping-infant-body-detection",
        "version":   1,
        "name":      "infant_respiration",
        "cls_map":   KEYWORD_MAP,
    },
    {
        "workspace": "project-lbnl9",
        "project":   "baby-sleeping",
        "version":   1,
        "name":      "baby_sleeping",
        "cls_map":   KEYWORD_MAP,
    },
    {
        "workspace": "idp-dataset",
        "project":   "baby-monitoring-4",
        "version":   1,
        "name":      "baby_monitoring",
        "cls_map":   KEYWORD_MAP,
    },
]


def download_all(api_key: str, fresh: bool = False):
    from roboflow import Roboflow
    rf = Roboflow(api_key=api_key)

    for ds in DATASETS:
        out_dir = DEST / f"raw_{ds['name']}"

        if fresh and out_dir.exists():
            shutil.rmtree(out_dir)
            print(f"[Fresh] {ds['name']} 기존 다운로드 삭제")

        if out_dir.exists() and any(out_dir.iterdir()):
            print(f"[Skip] {ds['name']} 이미 있음 (--fresh 로 재다운로드 가능)")
            continue

        print(f"\n[Download] {ds['workspace']}/{ds['project']} v{ds['version']} ...")
        try:
            project = rf.workspace(ds["workspace"]).project(ds["project"])
            project.version(ds["version"]).download("yolov8", location=str(out_dir))
            print(f"[Done] → {out_dir}")
        except Exception as e:
            print(f"[Warning] {ds['name']} 다운로드 실패 (계속 진행): {e}")


def remap_and_merge(fresh: bool = False):
    # fresh 시 병합 결과만 초기화 (raw 다운로드는 유지)
    if fresh:
        for split in ("train", "valid", "test"):
            for sub in ("images", "labels"):
                d = DEST / split / sub
                if d.exists():
                    shutil.rmtree(d)
            print(f"[Fresh] {split}/ 초기화 완료")

    for split in ("train", "valid", "test"):
        (DEST / split / "images").mkdir(parents=True, exist_ok=True)
        (DEST / split / "labels").mkdir(parents=True, exist_ok=True)

    total_kept = total_dropped = total_imgs = 0

    for ds in DATASETS:
        raw_dir = DEST / f"raw_{ds['name']}"
        if not raw_dir.exists():
            print(f"[Skip] raw_{ds['name']} 없음")
            continue

        data_yaml = raw_dir / "data.yaml"
        if not data_yaml.exists():
            found = list(raw_dir.rglob("data.yaml"))
            if found:
                data_yaml = found[0]
                raw_dir = data_yaml.parent
            else:
                print(f"[Error] data.yaml 없음: {ds['name']}")
                continue

        with open(data_yaml) as f:
            meta = yaml.safe_load(f)
        orig_names = meta.get("names", [])
        if isinstance(orig_names, dict):
            orig_names = [orig_names[i] for i in sorted(orig_names)]

        cls_map = ds["cls_map"]
        idx_map = {}
        for orig_idx, name in enumerate(orig_names):
            key = name.lower().strip()
            if key in cls_map:
                idx_map[orig_idx] = cls_map[key]

        mapped_names = {orig_names[k]: CLASSES[v] for k, v in idx_map.items()}
        unmapped = [n for i, n in enumerate(orig_names) if i not in idx_map]
        print(f"\n[Merge] {ds['name']}")
        print(f"  매핑됨  : {mapped_names}")
        if unmapped:
            print(f"  무시됨  : {unmapped}")

        if not idx_map:
            print(f"  ⚠ 매핑된 클래스 없음 — 이 데이터셋은 건너뜁니다")
            continue

        kept = dropped = imgs = 0
        for split in ("train", "valid", "test"):
            lbl_dir = raw_dir / split / "labels"
            img_dir = raw_dir / split / "images"
            if not lbl_dir.exists():
                continue

            for lbl_file in lbl_dir.glob("*.txt"):
                new_lines = []
                for line in lbl_file.read_text().strip().splitlines():
                    if not line.strip():
                        continue
                    parts = line.split()
                    orig_cls = int(parts[0])
                    if orig_cls in idx_map:
                        new_lines.append(f"{idx_map[orig_cls]} " + " ".join(parts[1:]))
                        kept += 1
                    else:
                        dropped += 1

                if new_lines:
                    dst_lbl = DEST / split / "labels" / lbl_file.name
                    if not dst_lbl.exists():
                        dst_lbl.write_text("\n".join(new_lines) + "\n")
                        for ext in (".jpg", ".jpeg", ".png"):
                            src_img = img_dir / (lbl_file.stem + ext)
                            if src_img.exists():
                                shutil.copy2(src_img, DEST / split / "images" / src_img.name)
                                imgs += 1
                                break

        print(f"  이미지 {imgs}장  어노테이션 유지 {kept} / 제거 {dropped}")
        total_kept += kept
        total_dropped += dropped
        total_imgs += imgs

    print(f"\n[전체] 이미지 {total_imgs}장  유지 {total_kept} / 제거 {total_dropped}")


def write_yaml():
    yaml_path = DEST / "data.yaml"
    content = {
        "path":  str(DEST),
        "train": "train/images",
        "val":   "valid/images",
        "test":  "test/images",
        "nc":    3,
        "names": [CLASSES[i] for i in sorted(CLASSES)],
    }
    with open(yaml_path, "w") as f:
        yaml.dump(content, f, allow_unicode=True, sort_keys=False)
    print(f"\n[YAML] {yaml_path} 작성 완료")


def count():
    print("\n[데이터셋 현황]")
    cls_count = {0: 0, 1: 0, 2: 0}
    for split in ("train", "valid", "test"):
        imgs = len(list((DEST / split / "images").glob("*.*")))
        lbls_dir = DEST / split / "labels"
        for lbl in lbls_dir.glob("*.txt"):
            for line in lbl.read_text().strip().splitlines():
                if line.strip():
                    cls_count[int(line.split()[0])] = cls_count.get(int(line.split()[0]), 0) + 1
        print(f"  {split:6s}: {imgs}장")
    print(f"\n  클래스 분포 (어노테이션 기준)")
    total = sum(cls_count.values()) or 1
    for i, name in CLASSES.items():
        print(f"    {name}: {cls_count[i]}개 ({cls_count[i]/total*100:.1f}%)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--api-key",   required=True,        help="Roboflow API 키")
    parser.add_argument("--fresh",     action="store_true",  help="전체 초기화 후 재다운로드·재병합")
    parser.add_argument("--remerge",   action="store_true",  help="다운로드 유지, 병합만 재실행")
    args = parser.parse_args()

    if not args.remerge:
        download_all(args.api_key, fresh=args.fresh)

    remap_and_merge(fresh=args.fresh or args.remerge)
    write_yaml()
    count()
    print("\n완료! 다음 실행:")
    print("  python3 ai/train_v4.py --fresh")
