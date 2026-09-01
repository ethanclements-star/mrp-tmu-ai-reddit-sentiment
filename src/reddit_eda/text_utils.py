import re

# http(s) links, www. links, and reddit preview URLs
URL_PATTERN = re.compile(
    r"https?://[^\s\)\]>\"']+|www\.[^\s\)\]>\"']+",
    flags=re.IGNORECASE,
)
# Markdown links: [label](url) -> label
MARKDOWN_LINK_PATTERN = re.compile(r"\[([^\]]*)\]\([^\)]+\)")
WHITESPACE_PATTERN = re.compile(r"\s+")


def strip_urls(text: str) -> str:
    """Remove URLs and markdown link targets from comment text."""
    if not text:
        return ""
    cleaned = MARKDOWN_LINK_PATTERN.sub(r"\1", text)
    cleaned = URL_PATTERN.sub(" ", cleaned)
    return WHITESPACE_PATTERN.sub(" ", cleaned).strip()
