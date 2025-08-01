# mapping/gender_normalizer.py
# -*- coding: utf-8 -*-
"""
성별 정규화(SSOT)
- 코드: 남성=0, 여성=1
- 문장에 남/여/남녀/전체/성별 무관 등 다양한 표기를 수용하여 [0], [1], [0,1] 반환
- 미언급이면 [0,1]
"""

import re
from typing import List

MALE = [
    "남성만","남자만","남성 대상","남성 전용","남자 전용",
    "남성","남자","남","남/男"
]
FEMALE = [
    "여성만","여자만","여성 대상","여성 전용","여자 전용",
    "여성","여자","여","여/女"
]
BOTH = [
    "남녀","남/녀","남·녀","남녀 모두","남녀 전원","남녀 구분 없이",
    "모든 성별","전체","성별 무관","남여","여남","여녀"
]

def normalize_gender_to_codes(text: str) -> List[int]:
    t = re.sub(r"\s+", " ", text or "").strip()
    # 둘 다 존재 또는 '모두/전체/무관'류 표현
    if any(k in t for k in BOTH) or (any(m in t for m in MALE) and any(f in t for f in FEMALE)):
        return [0,1]
    # 단일
    if any(f in t for f in FEMALE):
        return [1]
    if any(m in t for m in MALE):
        return [0]
    # 미언급 → 둘 다
    return [0,1]