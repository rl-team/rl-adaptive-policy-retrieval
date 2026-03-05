"""Split parsed CMS policy text into paragraph-level chunks."""

import re


MIN_CHUNK_CHARS = 40
MAX_CHUNK_WORDS = 500


def chunk_text(text, min_chars=MIN_CHUNK_CHARS, max_words=MAX_CHUNK_WORDS):
    """Split text into paragraph-level chunks.

    Splits on sentence-ending boundaries to produce chunks of roughly
    paragraph size (up to max_words each).
    """
    # Split on patterns that look like paragraph boundaries.
    # Note: the second branch splits on ". A" (period + uppercase), which may
    # incorrectly split on abbreviations like "Dr. Smith". CMS policy text is
    # formal and rarely uses such abbreviations, so this is acceptable here.
    raw_parts = re.split(r"(?:\.\s{2,})|(?:\.\s*(?=[A-Z][a-z]))", text)

    chunks = []
    current = ""
    for part in raw_parts:
        part = part.strip()
        if not part:
            continue
        # Re-add the period that was consumed by the split
        candidate = (current + ". " + part).strip() if current else part
        if len(candidate.split()) > max_words and current:
            if len(current) >= min_chars:
                chunks.append(current.strip().rstrip(".") + ".")
            current = part
        else:
            current = candidate
            
        # Hard-split fallback if a single part is still too long.
        # Splits at exactly max_words word boundaries, which may cut mid-sentence.
        while len(current.split()) > max_words:
            words = current.split()
            chunk_str = " ".join(words[:max_words]).strip()
            chunks.append(chunk_str.rstrip(".") + ".")
            current = " ".join(words[max_words:])

    if current and len(current) >= min_chars:
        chunks.append(current.strip().rstrip(".") + ".")

    return chunks


def build_corpus(parsed_data):
    """Convert parsed CMS data into a flat list of deduplicated chunk dicts.

    A chunk is identified by its (policy_id, text) pair. When the same chunk
    text comes from the same source document but is relevant to multiple
    procedures, a single corpus entry is created and all procedure codes are
    collected in the ``procedure_codes`` list rather than duplicating the chunk.

    Args:
        parsed_data: dict mapping procedure_code -> list of
                     {source, section_type, text}

    Returns:
        List of {chunk_id, policy_id, section_type, text, procedure_codes}
    """
    corpus: list[dict] = []
    chunk_counter = 0
    # (policy_id, text) -> index into corpus, for deduplication
    seen: dict[tuple[str, str], int] = {}

    for proc_code, sections in parsed_data.items():
        for section in sections:
            parts = chunk_text(section["text"])
            for part in parts:
                key = (section["source"], part)
                if key in seen:
                    existing = corpus[seen[key]]
                    if proc_code not in existing["procedure_codes"]:
                        existing["procedure_codes"].append(proc_code)
                else:
                    seen[key] = len(corpus)
                    corpus.append({
                        "chunk_id": f"chunk_{chunk_counter:03d}",
                        "policy_id": section["source"],
                        "section_type": section["section_type"],
                        "text": part,
                        "procedure_codes": [proc_code],
                    })
                    chunk_counter += 1

    return corpus
