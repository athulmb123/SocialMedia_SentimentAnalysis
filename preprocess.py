"""Text cleaning for social media posts."""
import re

URL_RE = re.compile(r"https?://\S+|www\.\S+")
MENTION_RE = re.compile(r"@\w+")
HASHTAG_RE = re.compile(r"#(\w+)")
REPEAT_RE = re.compile(r"(.)\1{2,}")          # soooo -> soo
HTML_RE = re.compile(r"&\w+;")
SPACE_RE = re.compile(r"\s+")

EMOTICONS = {
    r":-?\)+|:-?D|;-?\)|<3|:p": " happyface ",
    r":-?\(+|:'\(|:-?/": " sadface ",
}


def clean_text(text: str) -> str:
    """Lowercase, strip URLs/mentions/HTML, keep hashtag words, map emoticons,
    squeeze elongated words."""
    text = str(text).lower()
    text = URL_RE.sub(" ", text)
    text = HTML_RE.sub(" ", text)
    text = MENTION_RE.sub(" ", text)
    text = HASHTAG_RE.sub(r" \1 ", text)
    for pattern, repl in EMOTICONS.items():
        text = re.sub(pattern, repl, text)
    text = REPEAT_RE.sub(r"\1\1", text)
    text = re.sub(r"[^a-z0-9'\s!?]", " ", text)
    return SPACE_RE.sub(" ", text).strip()


if __name__ == "__main__":
    print(clean_text("@bob I LOOOVE this phone!!! http://t.co/x #BestDay :)"))
