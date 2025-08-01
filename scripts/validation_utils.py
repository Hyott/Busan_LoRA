# scripts/validation_utils.py
# -*- coding: utf-8 -*-
from __future__ import annotations
import re
from typing import Dict, List, Tuple, Any

MARK_L = "[["
MARK_R = "]]"

GENDER_KEYS = [
    # 전체/무관류 (공백 없는 변형 포함)
    "남녀","남/녀","남·녀","남녀 모두","남녀모두","남녀 전원","남녀 구분 없이",
    "모든 성별","성별 무관","전체","남여","여남","여녀",
    # 단일
    "남성","남자","여성","여자",
]
PTRN_KEYS = [
    "생활인구","생활 인구","거주","직장","방문","방문자","유입","유입 인구","정주 인구","상주 인구",
]

RE_YM  = re.compile(r'\b(20\d{2})[.\s년/-]?\s?(\d{1,2})\b', re.U)
RE_AGE_RANGE  = re.compile(r'(\d{2})\s*([·~,\-–—])\s*(\d{2})\s*대', re.U)  # 60·70대/60~70대/60-70대/60,70대
RE_AGE_SINGLE = re.compile(r'(\d{2})\s*대', re.U)                           # 60대

def _wrap(tag: str) -> str:
    return f"{MARK_L}{tag}{MARK_R}"

def _protect_ages_once(orig: str, meta: Dict[str, Any]) -> str:
    """원본 문자열 기준으로 범위/단일 나이를 겹치지 않게 한 번만 태깅한다."""
    text = orig
    replacements: List[Tuple[int,int,str,List[str]]] = []  # (start, end, repl, tags)

    taken = [False] * (len(orig) + 1)

    # 1) 범위 먼저 수집
    for m in RE_AGE_RANGE.finditer(orig):
        a, sep, b = m.group(1), m.group(2), m.group(3)
        s, e = m.span(0)
        # 마킹되지 않은 구간만
        if any(taken[s:e]):
            continue
        tag_a = _wrap(f"AGE:{a}대")
        tag_b = _wrap(f"AGE:{b}대")
        repl = f"{tag_a}{sep}{tag_b}"
        replacements.append((s, e, repl, [tag_a, tag_b]))
        for i in range(s, e):
            taken[i] = True

    # 2) 단일 수집(범위와 겹치지 않는 것만)
    for m in RE_AGE_SINGLE.finditer(orig):
        s, e = m.span(0)
        if any(taken[s:e]):
            continue
        val = m.group(1) + "대"
        tag = _wrap(f"AGE:{val}")
        replacements.append((s, e, tag, [tag]))
        for i in range(s, e):
            taken[i] = True

    # 3) 뒤에서 앞으로 치환
    replacements.sort(key=lambda x: x[0], reverse=True)
    for s, e, repl, tags in replacements:
        text = text[:s] + repl + text[e:]
        meta["age"].extend(tags)

    return text

def protect_slots(text: str, gold: Dict[str,Any]) -> Tuple[str, Dict[str, Any]]:
    t = text
    meta = {"ym": [], "region": [], "gender": [], "ptrn": [], "age": []}

    # === A) 연령 태깅 (먼저) ===
    t = _protect_ages_once(t, meta)

    # === B) 연월 ===
    def _ym_sub(m):
        y, mm = m.group(1), m.group(2)
        tag = _wrap(f"YM:{y}.{mm}")
        meta["ym"].append(tag)
        return tag
    t = RE_YM.sub(_ym_sub, t)

    # === C) 지역 (긴 토큰 우선, 단어경계) ===
    reg = (gold or {}).get("region_nm", "") or ""
    tokens = sorted({w for w in reg.split() if len(w) >= 2}, key=len, reverse=True)
    for tok in tokens:
        pat = re.compile(rf'(?<!\w){re.escape(tok)}(?!\w)', re.U)
        if pat.search(t):
            tag = _wrap(f"REG:{tok}")
            t = pat.sub(tag, t)
            meta["region"].append(tag)

    # === D) 성별 ===
    for gk in sorted(GENDER_KEYS, key=len, reverse=True):
        pat = re.compile(rf'(?<!\w){re.escape(gk)}(?!\w)', re.U)
        if pat.search(t):
            tag = _wrap(f"G:{gk}")
            t = pat.sub(tag, t)
            meta["gender"].append(tag)

    # === E) 패턴 ===
    for pk in sorted(PTRN_KEYS, key=len, reverse=True):
        pat = re.compile(rf'(?<!\w){re.escape(pk)}(?!\w)', re.U)
        if pat.search(t):
            tag = _wrap(f"P:{pk}")
            t = pat.sub(tag, t)
            meta["ptrn"].append(tag)

    return t, meta

def unprotect_slots(text: str) -> str:
    return text.replace(MARK_L, "").replace(MARK_R, "")

def markers_preserved(src_protected: str, cand_protected: str, meta: Dict[str,Any]) -> bool:
    for group in ("ym","region","gender","ptrn","age"):
        for tag in meta.get(group, []):
            if tag not in cand_protected:
                return False
    def _count_all(s, tags):
        return sum(s.count(t) for t in tags)
    for group in ("ym","region","gender","ptrn","age"):
        if _count_all(src_protected, meta.get(group, [])) != _count_all(cand_protected, meta.get(group, [])):
            return False
    return True









# ---------- 의미 동등성 검사(옵션) ----------

def _extract_ym(text: str) -> int | None:
    m = RE_YM.search(text)
    if not m:
        return None
    y, mm = int(m.group(1)), int(m.group(2))
    if 1 <= mm <= 12:
        return y*100 + mm
    return None

def canon(d: Dict[str,Any]) -> Dict[str,Any]:
    return {
        "base_ym": int(d.get("base_ym")) if d.get("base_ym") is not None else None,
        "region_nm": (d.get("region_nm") or "").strip(),
        "ptrn": sorted(d.get("ptrn") or []),
        "gender_cd": sorted(d.get("gender_cd") or []),
        "age_cd": sorted(d.get("age_cd") or []),
    }

def semantic_equivalent(text: str, gold: Dict[str,Any]) -> bool:
    """
    (옵션) 정규화 기반 의미 동등성 검사.
    - 월: 텍스트에서 YYYY.MM 추출 → 골드와 동일?
    - 패턴/성별: 서비스 정규화기로 추출 → 골드와 동일?
    - 지역: normalize_region_nm 결과가 골드와 동일? (정규화기가 없으면 패스)
    """
    g = canon(gold)

    # 1) base_ym
    ym = _extract_ym(text)
    if ym is not None and g["base_ym"] is not None and ym != g["base_ym"]:
        return False

    # 2) 패턴
    if normalize_ptrn_to_codes is not None:
        pc = normalize_ptrn_to_codes(text) or []
        if sorted(pc) != g["ptrn"]:
            return False

    # 3) 성별
    if normalize_gender_to_codes is not None:
        gc = normalize_gender_to_codes(text) or []
        if sorted(gc) != g["gender_cd"]:
            return False

    # 4) 지역
    if normalize_region_nm is not None:
        rn = normalize_region_nm(text) or None
        # 정규화 실패시 패스, 성공하면 골드와 동일해야 통과
        if rn is not None and g["region_nm"] and rn != g["region_nm"]:
            return False

    return True