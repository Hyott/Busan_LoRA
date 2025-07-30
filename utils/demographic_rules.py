# utils/demographic_rules.py
import re
from mapping.lexicon_region import GENDER_TERMS, ALL_AGES

def parse_gender(text: str) -> list[int]:
    # 다중 용어가 공존하면 합집합
    result = set()
    for k, v in GENDER_TERMS.items():
        if k in text:
            result.update(v)
    # 명시가 없으면 '전체' 정책(필요 시 바꿀 수 있음)
    return sorted(result) if result else [0,1]

def parse_ages(text: str) -> list[int]:
    # 공백 제거
    t = text.replace(" ", "")
    
    # "전연령", "모든연령", "전체연령" 처리
    if "전연령" in t or "모든연령" in t or "전체연령" in t:
        return ALL_AGES['전 연령'].copy()
    
    # "30~40대" 또는 "30-40대" 형태 처리
    m_range = re.search(r'(\d{2})\s*[~|-]\s*(\d{2})\s*대', t)
    if m_range:
        key = f"{int(m_range.group(1))}~{int(m_range.group(2))}대"
        return ALL_AGES.get(key, ALL_AGES['전 연령']).copy()
    
    # "30,40대" 또는 "30와40대" 형태 처리
    m_multi = re.search(r'(\d{2})\s*(?:,|와|및|/)\s*(\d{2})\s*대', t)
    if m_multi:
        start_age = int(m_multi.group(1))
        end_age = int(m_multi.group(2))
        # 연속적인 나이 범위인지 확인
        if start_age in ALL_AGES['전 연령'] and end_age in ALL_AGES['전 연령']:
            key = f"{start_age}~{end_age}대" if start_age < end_age else f"{end_age}~{start_age}대"
            return ALL_AGES.get(key, ALL_AGES['전 연령']).copy()
    
    # "30대"와 같은 단일 나이 처리
    m_single = re.search(r'(\d{2})\s*대', t)
    if m_single:
        key = f"{int(m_single.group(1))}대"
        return ALL_AGES.get(key, ALL_AGES['전 연령']).copy()
    
    # "30,40,50대"와 같은 다중 나이 처리
    m_list = re.findall(r'(\d{2})\s*대', t)
    if m_list:
        ages = sorted([int(x) for x in m_list])
        # 연속적인지 확인하여 키 생성
        if len(ages) > 1 and all(ages[i] + 10 == ages[i+1] for i in range(len(ages)-1)):
            key = f"{ages[0]}~{ages[-1]}대"
            return ALL_AGES.get(key, ALL_AGES['전 연령']).copy()
        # 비연속적인 경우, 개별 나이 합집합
        result = []
        for age in ages:
            key = f"{age}대"
            result.extend(ALL_AGES.get(key, []))
        return sorted(list(set(result))) if result else ALL_AGES['전 연령'].copy()
    
    # 유효하지 않은 입력은 기본값으로 '전 연령' 반환
    return ALL_AGES['전 연령'].copy()