"""Conservative filtering for exported chat messages, not ordinary prose."""
import re

_PLACEHOLDERS = {
    "<media omitted>", "image omitted", "video omitted", "audio omitted",
    "document omitted", "sticker omitted", "null", "(file attached)",
    "this message was deleted", "you deleted this message",
}
_NOTICE = re.compile(
    r"^(?:messages and calls are end-to-end encrypted\b.*|"
    r"(?:you |[^:]{1,80} )?(?:created (?:this )?group|changed the subject|"
    r"changed this group|joined using this group|changed (?:their |to a new )phone number)\b.*|"
    r"you were added\b.*|you joined\b.*|missed (?:voice|video) call)$", re.I)


def is_system_body(body):
    text = str(body).strip().lstrip("\u200e\u200f").strip()
    return text.lower() in _PLACEHOLDERS or bool(_NOTICE.fullmatch(text))
