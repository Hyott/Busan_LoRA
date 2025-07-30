# mapping/region_normalizer.py
import re
from typing import Dict, List, Tuple
from mapping.lexicon_region import REGIONS  # [(std, code, [aliases...]), ...]

# 시/도 축약 표 (필요시 추가)
SIDO_SHORT = {
    "서울특별시":"서울","부산광역시":"부산","대구광역시":"대구","인천광역시":"인천","광주광역시":"광주",
    "대전광역시":"대전","울산광역시":"울산","세종특별자치시":"세종","제주특별자치도":"제주",
    "경기도":"경기","강원도":"강원","충청북도":"충북","충청남도":"충남","전라북도":"전북",
    "전라남도":"전남","경상북도":"경북","경상남도":"경남",
}

# 문장에서 제거할 접미/수식어
NOISE_TOKENS = ["일대","근처","인근","부근","쪽"]

def _norm_space(s:str)->str:
    return re.sub(r"\s+", " ", s).strip()

def _strip_noise(s:str)->str:
    s = _norm_space(s)
    # “사상구 일대/근처 …” → “사상구 …”
    for tok in NOISE_TOKENS:
        s = re.sub(rf"\b{tok}\b", "", s)
    return _norm_space(s)

def build_alias2std(regions=REGIONS) -> Dict[str,str]:
    """REGIONS로부터 alias→표준명 딕셔너리 생성(여러 변형 포함)"""
    alias2std = {}
    for std, code, aliases in regions:
        std = _norm_space(std)
        cand = set(aliases or [])
        cand.add(std)

        # 시/도 축약형 추가: "부산광역시 사상구" → "부산 사상구"
        m = re.match(r"^(\S+)\s+(.+)$", std)
        if m:
            sido, rest = m.group(1), m.group(2)
            short = SIDO_SHORT.get(sido)
            if short:
                cand.add(f"{short} {rest}")

        # 읍/면/동 단독, 구/군 + 읍면동 조합도 alias에 들어있다면 추가
        parts = std.split()
        if len(parts) >= 2:
            cand.add(parts[-1])                    # "무악동" / "일광읍"
            cand.add(" ".join(parts[-2:]))         # "종로구 무악동" / "기장군 일광읍"

        # 노이즈 토큰 붙은 변형도 등록: “사상구 일대”, “종로구 무악동 근처”
        more = set()
        for a in list(cand):
            for tok in NOISE_TOKENS:
                more.add(_norm_space(f"{a} {tok}"))
        cand |= more

        # 정규화해서 넣기
        for a in cand:
            alias2std[_strip_noise(a)] = std

    # 긴 문자열 우선 매칭을 위해 정렬 키로 사용 가능
    return alias2std

ALIAS2STD = build_alias2std()

def normalize_region_nm(text:str, alias2std:Dict[str,str]=ALIAS2STD) -> str | None:
    """문장 속에서 지역 alias를 찾아 표준명으로 반환 (긴-별칭 우선)"""
    t = _strip_noise(text)
    # 긴 별칭 우선 검색
    for a in sorted(alias2std.keys(), key=len, reverse=True):
        if a and a in t:
            return alias2std[a]
    return None