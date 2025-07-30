# mapping/ptrn_normalizer.py
# -*- coding: utf-8 -*-
"""
Gen BI 파서용 패턴 정규화(SSOT)
- 최종 스펙: 거주=0, 직장=1, 방문=2, 생활인구=[0,1,2]
- 동의어를 텍스트에서 감지해 코드 리스트로 변환
- '출퇴근' 표현은 의도적으로 제외
- 과거/레거시 코드(예: 3=체류, 5=유입 등)는 신규 코드로 리매핑
"""

from __future__ import annotations
from typing import Dict, List, Optional
import re

try:
    # 서비스 측 정의: {"거주":0, "직장":1, "방문":2, "생활인구":[0,1,2]}
    from mapping.lexicon import PTRN_MAP as SERVICE_PTRN_MAP
except Exception:
    SERVICE_PTRN_MAP = {"거주": 0, "직장": 1, "방문": 2, "생활인구": [0, 1, 2]}

# -----------------------------
# 스펙/코드 유틸
# -----------------------------
_ALLOWED = {0, 1, 2}

def _as_list(v) -> List[int]:
    return v if isinstance(v, list) else [v]

# 서비스 정의로부터 스펙 테이블 구성(보정)
CANON_TO_CODES: Dict[str, List[int]] = {
    k: sorted({c for c in _as_list(v) if c in _ALLOWED})
    for k, v in SERVICE_PTRN_MAP.items()
    if k in ("거주", "직장", "방문", "생활인구")
}
# 누락 시 안전값
if "생활인구" not in CANON_TO_CODES:
    CANON_TO_CODES["생활인구"] = [0, 1, 2]

# -----------------------------
# 동의어 사전 (길이 긴 표현 우선 매칭)
#  - '출퇴근' 제외
# -----------------------------
SYN2CANON: Dict[str, str] = {
    # 직장 (명사구/표현) — '출퇴근'은 제외
    "직장 목적": "직장", "직장 인구": "직장", "근로자": "직장",
    "근무 목적": "직장", "회사 관련": "직장", "업무 목적": "직장",
    "근무 인구": "직장", "회사 인력": "직장", "직장 수요": "직장",
    "회사 업무": "직장", "회사 업무로": "직장",  # 문장형도 일부 수용
    "직장": "직장",

    # 방문(=유입)
    "유입 인구": "방문", "외부 유입": "방문", "일시 방문": "방문",
    "관광 방문": "방문", "외지 방문": "방문", "방문 규모": "방문",
    "방문자": "방문", "유입": "방문", "방문": "방문",

    # 거주(=체류/상주/정주)
    "정주 인구": "거주", "상주 인구": "거주", "거주 인구": "거주",
    "주소지 거주": "거주", "해당 지역 거주": "거주", "지역 거주": "거주",
    "거주": "거주",

    # 생활인구(=전체)
    "전체 생활인구": "생활인구", "생활 인구": "생활인구",
    "생활인구 기준으로": "생활인구", "생활인구": "생활인구",

    # 이유/용도형(ADV)도 일부 지원
    "일 때문에": "직장", "일 보러": "직장", "업무 보러": "직장",
    "회사 때문에": "직장", "근무 때문에": "직장",

    "관광하러": "방문", "놀러": "방문", "들르러": "방문",
    "잠깐 들르려고": "방문", "일시적으로 방문하려고": "방문",

    "살아서": "거주", "거주해서": "거주", "머물러서": "거주",
}

# -----------------------------
# 레거시 코드 → 신규 코드 리매핑
#   예) 3(체류) → 0(거주), 5(유입) → 2(방문)
# -----------------------------
LEGACY_REMAP = {
    3: 0,
    4: 2,
    5: 2,
    6: 2,
}

# -----------------------------
# 정규화 함수
# -----------------------------
def _norm_space(s: str) -> str:
    return re.sub(r"\s+", " ", s or "").strip()

def normalize_ptrn_to_codes(text: str) -> List[int]:
    """
    문장(text)에서 패턴 동의어를 감지하여 스펙에 맞는 코드 리스트를 반환.
    규칙:
      - '생활인구' 키워드가 있으면 → [0,1,2]
      - 아무 패턴이 언급되지 않으면 → [0,1,2]
      - 그 외엔 발견된 캐논 라벨들의 합집합 (예: 직장+방문 → [1,2])
    """
    t = _norm_space(text)

    # 1) 생활인구 먼저 체크
    living_keys = [k for k, v in SYN2CANON.items() if v == "생활인구"]
    if any(k in t for k in living_keys):
        return CANON_TO_CODES["생활인구"]

    # 2) 기타 동의어 수집 (긴-키워드 우선)
    buckets = set()
    for syn in sorted(SYN2CANON.keys(), key=len, reverse=True):
        if syn and syn in t:
            buckets.add(SYN2CANON[syn])

    # 3) 미언급 → [0,1,2]
    if not buckets:
        return CANON_TO_CODES["생활인구"]

    # 4) 캐논 라벨들의 코드 합집합
    out = set()
    for canon in buckets:
        for c in CANON_TO_CODES.get(canon, []):
            out.add(c)
    return sorted(out)

def remap_legacy_codes(codes: Optional[List[int]]) -> Optional[List[int]]:
    """
    과거 데이터/예측에 섞인 레거시 코드를 새 스펙으로 변환.
    """
    if not codes:
        return codes
    out = []
    for c in codes:
        c2 = LEGACY_REMAP.get(c, c)
        if c2 in _ALLOWED:
            out.append(c2)
    out = sorted(set(out))
    return out or None

def surface_synonyms(canon_label: str) -> List[str]:
    """
    NLG(데이터 생성)에서 사용할 수 있는 동의어 후보를 반환.
    """
    canon = (canon_label or "").strip()
    return [k for k, v in SYN2CANON.items() if v == canon]

__all__ = [
    "CANON_TO_CODES",
    "SYN2CANON",
    "LEGACY_REMAP",
    "normalize_ptrn_to_codes",
    "remap_legacy_codes",
    "surface_synonyms",
]