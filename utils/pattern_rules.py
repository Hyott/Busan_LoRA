# utils/pattern_rules.py
from mapping.lexicon_region import PTRN_MAP

def parse_ptrn(text: str) -> list[int]:
    hits = []
    for k, code in PTRN_MAP.items():
        if k in text:
            hits.append(code)
    # 미지정 시 서비스 기본값(예: '방문'로 본다) 정책 여부 결정
    return sorted(set(hits)) or [2]  # 예: 기본 [2]=방문