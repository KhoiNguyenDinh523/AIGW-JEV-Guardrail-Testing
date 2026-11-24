"""BƯỚC 1 - Data Ingestion: tải WildGuardMix, khám phá schema, làm sạch, chuẩn hóa nhãn nhị phân.

Chạy:
    python step1_ingest.py                    # tải tập test (1.725 mẫu) -> data/wildguardtest_en.parquet
    python step1_ingest.py --val-n 500        # lấy thêm 500 mẫu từ train làm VALIDATION (để chọn τ)
"""
import argparse
import re

import pandas as pd

import config as C

# Gom ~13 subcategory của WildGuardMix về 4 trụ cột (khớp Bảng 4.3 trong đề cương).
# Tên subcategory thật in ra ở phần khám phá bên dưới -> HÃY ĐỐI CHIẾU và sửa rule nếu có mục rơi vào "other".
PILLAR_RULES = [
    ("privacy", r"privacy|private|sensitive_information|personal|copyright"),
    ("misinformation", r"misinfo|false|disinfo|defamation|conspiracy|deceptive"),
    ("harmful_language", r"toxic|hate|stereotype|discrimination|violence|sexual|harassment|insult"),
    ("malicious_uses", r"cyber|fraud|illegal|malicious|weapon|terror|crime|mental_health|material_harm|unethical|unsafe"),
]


def to_pillar(sub) -> str:
    if not isinstance(sub, str) or sub.lower() in ("benign", "none", ""):
        return "benign"
    for name, pat in PILLAR_RULES:
        if re.search(pat, sub.lower()):
            return name
    return "other"


def load(config_name: str) -> pd.DataFrame:
    from datasets import load_dataset
    ds = load_dataset("allenai/wildguardmix", config_name, token=C.HF_TOKEN or None)
    split = list(ds.keys())[0]
    print(f"[load] config={config_name} splits={list(ds.keys())} -> dùng '{split}'")
    return ds[split].to_pandas()


def explore(df: pd.DataFrame, name: str):
    print(f"\n===== KHÁM PHÁ {name} =====")
    print("shape:", df.shape)
    print("cột:", list(df.columns))
    print("\nkiểu dữ liệu / số null:")
    print(pd.DataFrame({"dtype": df.dtypes.astype(str), "null": df.isna().sum()}))
    for col in ("prompt_harm_label", "adversarial", "subcategory"):
        if col in df.columns:
            print(f"\n--- phân bố {col} ---")
            print(df[col].value_counts(dropna=False).to_string())
    if {"prompt_harm_label", "adversarial"} <= set(df.columns):
        print("\n--- chéo harm_label x adversarial ---")
        print(pd.crosstab(df["prompt_harm_label"], df["adversarial"]))
    ln = df["prompt"].fillna("").str.len()
    print("\nđộ dài prompt (ký tự):", ln.describe().round(1).to_dict())


def clean(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    n0 = len(df)
    df = df.copy()
    df = df[df["prompt"].notna() & (df["prompt"].str.strip() != "")]
    df = df[df["prompt_harm_label"].notna()]
    n1 = len(df)
    df = df.drop_duplicates(subset="prompt")
    print(f"[clean] {n0} -> bỏ rỗng/thiếu nhãn: {n1} -> bỏ trùng prompt: {len(df)}")

    lab = df["prompt_harm_label"].astype(str).str.lower().str.strip()
    # Chuẩn hóa: 1 = harmful -> BLOCK ; 0 = unharmful -> ACCEPT
    df["y_true"] = lab.map({"harmful": 1, "unharmful": 0, "1": 1, "0": 0, "true": 1, "false": 0})
    bad = df["y_true"].isna().sum()
    if bad:
        print(f"[cảnh báo] {bad} dòng có nhãn lạ, bị loại: {lab[df['y_true'].isna()].unique()}")
        df = df[df["y_true"].notna()]
    df["y_true"] = df["y_true"].astype(int)
    df["label"] = df["y_true"].map({1: "BLOCK", 0: "ACCEPT"})
    df["adversarial"] = df["adversarial"].astype(bool)
    df["pillar"] = df["subcategory"].map(to_pillar)
    df = df.reset_index(drop=True)
    df.insert(0, "id", [f"{prefix}{i:05d}" for i in range(len(df))])
    df = df.rename(columns={"prompt": "prompt_en"})
    keep = ["id", "prompt_en", "y_true", "label", "adversarial", "subcategory", "pillar"]
    unmapped = df.loc[df["pillar"] == "other", "subcategory"].unique()
    if len(unmapped):
        print("[chú ý] subcategory chưa gán được trụ cột (sửa PILLAR_RULES):", unmapped)
    return df[keep]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val-n", type=int, default=0, help="số mẫu validation lấy từ train (0 = bỏ qua)")
    a = ap.parse_args()

    test = clean(load("wildguardtest"), "T")
    explore(test.rename(columns={"prompt_en": "prompt", "label": "prompt_harm_label"}), "WildGuardTest (sau chuẩn hóa)")
    out = C.DATA / "wildguardtest_en.parquet"
    test.to_parquet(out, index=False)
    print(f"\n[ok] lưu {len(test)} mẫu -> {out}")
    print("Tỷ lệ BLOCK/ACCEPT:", test["label"].value_counts().to_dict(),
          "| adversarial:", test["adversarial"].value_counts().to_dict())

    if a.val_n:
        tr = clean(load("wildguardtrain"), "V")
        tr = tr[~tr["prompt_en"].isin(test["prompt_en"])]          # chống rò rỉ test -> val
        # lấy mẫu phân tầng theo (nhãn, adversarial)
        val = (tr.groupby(["y_true", "adversarial"], group_keys=False)
                 .apply(lambda g: g.sample(min(len(g), max(1, round(a.val_n * len(g) / len(tr)))), random_state=C.SEED)))
        val = val.reset_index(drop=True)
        val["id"] = [f"V{i:05d}" for i in range(len(val))]
        out = C.DATA / "wildguardval_en.parquet"
        val.to_parquet(out, index=False)
        print(f"[ok] lưu validation {len(val)} mẫu -> {out}")


if __name__ == "__main__":
    main()
