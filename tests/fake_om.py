"""`memshort om` with only the model replaced, for the adapter tests: JSON on stdin, `om.run`'s JSON on stdout.

Each observer call writes one line, `saw <the conversation's first words>`, so a test can see which run of
conversation produced which observation and where it landed around the preserved blocks.
"""

import json
import sys

from memshort import om


def complete(role: str, prompt: str) -> tuple[str, dict]:
    conversation = prompt.rsplit("<conversation>\n", 1)[1].split("\n</conversation>")[0]
    words = " ".join(conversation.split()[:4])
    return (
        f"<observations>\n* 🟡 saw {words}\n</observations>\n<current-task>\ntask\n</current-task>",
        {},
    )


om.complete = complete
print(json.dumps(om.run(json.load(sys.stdin)), ensure_ascii=False))
