# utils/region_rules.py
from mapping.lexicon_region import REGION_MAP, REGION_ALIASES

def normalize_region(text: str) -> tuple[str | None, list[int] | None]:
    # 가장 긴 별칭부터 매칭(간단 구현)
    for alias, std in sorted(REGION_ALIASES.items(), key=lambda x: -len(x[0])):
        if alias in text:
            code = REGION_MAP.get(std)
            return std, [code] if code is not None else None
    # 미검출 시 None
    return None, None