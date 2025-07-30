# scripts/build_region_map_full.py
import pandas as pd

FILE_PATH = "scripts/KIKcd_H.20250513(말소코드포함).xlsx"
OUTPUT_PATH = "mapping/lexicon_region.py"

def load_region_df(filepath: str):
    df = pd.read_excel(filepath, dtype=str)
    df = df.fillna("")
    df = df[df["말소일자"] == ""]  # 말소 안 된 지역만
    return df

def build_region_map(df: pd.DataFrame) -> dict:
    region_map = {}

    for _, row in df.iterrows():
        sido = row["시도명"].strip()
        sigungu = row["시군구명"].strip()
        eupmyeondong = row["읍면동명"].strip()
        code = row["행정동코드"].strip()

        # 지역 이름 조합
        if eupmyeondong:
            name = f"{sido} {sigungu} {eupmyeondong}"
        elif sigungu:
            name = f"{sido} {sigungu}"
        else:
            name = sido  # 서울특별시 단독 같은 경우

        region_map[name.strip()] = int(code)

    return region_map

def build_aliases(region_map: dict) -> dict:
    aliases = {}
    for full in region_map:
        parts = full.split()
        if len(parts) >= 2:
            aliases[parts[-1]] = full                      # ex) '일광읍': '부산광역시 기장군 일광읍'
            aliases[" ".join(parts[-2:])] = full           # ex) '기장군 일광읍'
        if len(parts) >= 3:
            aliases[" ".join(parts[-3:])] = full           # ex) '기장군 일광읍'
        if len(parts) >= 2:
            aliases[" ".join(parts[:2])] = " ".join(parts[:2])  # ex) '부산 기장군'
    return aliases

def save_dict_as_python(name: str, d: dict, out_path: str):
    with open(out_path, "a", encoding="utf-8") as f:
        f.write(f"{name} = {{\n")
        for k, v in sorted(d.items()):
            if isinstance(v, str):
                f.write(f'    "{k}": "{v}",\n')
            else:
                f.write(f'    "{k}": {v},\n')
        f.write("}\n\n")

def main():
    df = load_region_df(FILE_PATH)
    region_map = build_region_map(df)
    region_aliases = build_aliases(region_map)

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        f.write("# Auto-generated REGION_MAP with 읍면동 포함\n\n")

    save_dict_as_python("REGION_MAP", region_map, OUTPUT_PATH)
    save_dict_as_python("REGION_ALIASES", region_aliases, OUTPUT_PATH)

    print(f"[✅ 완료] 총 지역 수: {len(region_map)}개 → {OUTPUT_PATH}에 저장됨")

if __name__ == "__main__":
    main()