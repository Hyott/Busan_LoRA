# utils/date_rules.py
import re
from datetime import datetime
from dateutil.relativedelta import relativedelta

TZ = "Asia/Seoul"  # 운영 환경에서 실제 TZ 바인딩

def parse_base_ym(text: str, now: datetime) -> int | None:
    m = re.search(r'(\d{4})\s*년\s*(\d{1,2})\s*월', text)
    if m:
        y, mm = int(m.group(1)), int(m.group(2))
        return y*100 + mm
    # 상대표현
    t = text.replace(" ", "")
    if "이번달" in t:
        return now.year*100 + now.month
    if "지난달" in t:
        prev = now - relativedelta(months=1)
        return prev.year*100 + prev.month
    # TODO: "어제", "이번 주"→ 월로 내림/올림 규칙 확정 시 추가
    return None