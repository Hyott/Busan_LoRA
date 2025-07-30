# quick_ptrn_test.py
from mapping.ptrn_normalizer import normalize_ptrn_to_codes, remap_legacy_codes

tests = [
    "2025년 7월 부산 사상구 직장 인구 통계 알려줘",
    "광주광역시 방문자 30대 남녀",
    "생활인구 기준으로 김해시 40대",
    "춘천시 20,30대 데이터",   # 미언급 → [0,1,2]
    "일 때문에 강남구 30대 여성",   # 직장 ADV
    "관광하러 제주도 20대",        # 방문 ADV
    "살아서 거주해서 수원시",       # 거주 ADV
]
for t in tests:
    print(t, "=>", normalize_ptrn_to_codes(t))

print("legacy [5] ->", remap_legacy_codes([5]))
print("legacy [3,5] ->", remap_legacy_codes([3,5]))