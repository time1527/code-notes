import datetime
from pytz import timezone

tool_template = """
{name}：{description}，参数：{parameters}，必须是json格式。
"""

selected = [
    "Africa/Cairo",  # 开罗（埃及）
    "Africa/Johannesburg",  # 约翰内斯堡（南非）
    "America/New_York",  # 纽约（美国）
    "America/Los_Angeles",  # 洛杉矶（美国）
    "America/Sao_Paulo",  # 圣保罗（巴西）
    "Europe/London",  # 伦敦（英国）
    "Europe/Paris",  # 巴黎（法国）
    "Asia/Tokyo",  # 东京（日本）
    "Asia/Shanghai",  # 上海（中国）
    "Australia/Sydney",  # 悉尼（澳大利亚）
]


class GetTime:
    name = "GetTime"
    description = "获取给定时区的当前时间"
    parameters = [
        {
            "name": "tzone",
            "description": f"时区，例如：Asia/shanghai，必须是{selected}中的一个。",
            "type": "string",
            "required": True,
        },
    ]

    def call(self, args):
        tzone = args.get("tzone", "")
        if not tzone:
            return "时区不能为空。"
        if tzone not in selected:
            return f"时区错误，必须是{selected}中的一个。"
        tz = timezone(tzone)
        current_time = datetime.datetime.now(tz)
        return current_time.strftime("%Y-%m-%d %H:%M:%S")


class AnswerBook:
    name = "AnswerBook"
    description = "答案之书，可以直接回答一切问题"
    parameters = []

    def call(self, args):
        return "万事如意，心想事成"
