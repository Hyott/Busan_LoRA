# scripts/build_regions_list.py
import pandas as pd

FILE_PATH = "scripts/KIKcd_H.20250513(말소코드포함).xlsx"
OUTPUT_PATH = "mapping/lexicon_region.py"

def load_valid_regions(path: str) -> pd.DataFrame:
    df = pd.read_excel(path, dtype=str)
    df = df.fillna("")
    return df[df["말소일자"] == ""]  # 현재 유효 지역만

def build_regions(df: pd.DataFrame):
    region_list = []

    for _, row in df.iterrows():
        sido = row["시도명"].strip()
        sigungu = row["시군구명"].strip()
        emd = row["읍면동명"].strip()
        code = int(row["행정동코드"])

        # 표준명 생성
        parts = [sido]
        if sigungu:
            parts.append(sigungu)
        if emd:
            parts.append(emd)
        full_name = " ".join(parts)

        # 별칭 생성
        aliases = []
        if emd:
            aliases.append(emd)                              # '일광읍'
            aliases.append(f"{sigungu} {emd}".strip())      # '기장군 일광읍'
            aliases.append(f"{sido.split()[0]} {sigungu} {emd}".strip())  # '부산 기장군 일광읍'
        elif sigungu:
            aliases.append(sigungu)                          # '기장군'
            aliases.append(f"{sido.split()[0]} {sigungu}".strip())        # '부산 기장군'

        region_list.append((full_name, code, sorted(set(aliases))))

    return region_list

def save_regions_list(regions: list, out_path: str):
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("# Auto-generated REGIONS list (풀 지역명, 코드, 별칭)\n\n")
        f.write("REGIONS = [\n")
        for name, code, aliases in sorted(regions, key=lambda x: x[1]):
            alias_str = ", ".join(f'"{a}"' for a in aliases)
            f.write(f'    ("{name}", {code}, [{alias_str}]),\n')
        f.write("]\n")

def main():
    df = load_valid_regions(FILE_PATH)
    regions = build_regions(df)
    save_regions_list(regions, OUTPUT_PATH)
    print(f"[✅ 완료] 지역 수: {len(regions)} → {OUTPUT_PATH}에 저장됨")

if __name__ == "__main__":
    main()