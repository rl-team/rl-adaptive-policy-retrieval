"""Parse CMS Medicare Coverage Database CSVs into structured JSON."""

import csv
import re
import sys
import os
import json

csv.field_size_limit(sys.maxsize)

RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "mcd_raw")

SECTION_MAP = {
    "indication": "coverage_criteria",
    "diagnoses_support": "coverage_criteria",
    "diagnoses_dont_support": "exclusions",
    "coding_guidelines": "billing",
    "doc_reqs": "billing",
    "itm_srvc_desc": "coverage_criteria",
    "indctn_lmtn": "coverage_criteria",
    "description": "billing",
}

# Sources to parse for each target procedure
# Hard-code to two procedures for the MVP
LCD_SOURCES = {
    "72148": {
        "lcd_ids": ["34220"],
        "ncd_sections": ["220.2"],
    },
    "29881": {
        "lcd_ids": ["36575"],
        "ncd_sections": ["150.9"],
        "article_ids": ["52369"],
    },
}


def strip_html(text):
    if not text:
        return ""
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"&[a-z]+;", " ", text)
    text = re.sub(r"&#\d+;", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _read_lcd(lcd_ids, text_fields=None):
    if text_fields is None:
        text_fields = ["indication", "diagnoses_support", "diagnoses_dont_support",
                       "coding_guidelines", "doc_reqs"]
    results = []
    path = os.path.join(RAW_DIR, "current_lcd", "current_lcd_csv", "lcd.csv")
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["lcd_id"] in lcd_ids:
                for field in text_fields:
                    raw = row.get(field, "")
                    clean = strip_html(raw)
                    if len(clean) > 30:
                        results.append({
                            "source": f"LCD_{row['lcd_id']}",
                            "section_type": SECTION_MAP.get(field, "coverage_criteria"),
                            "text": clean,
                        })
    return results


def _read_ncd(ncd_sections, text_fields=None):
    if text_fields is None:
        text_fields = ["itm_srvc_desc", "indctn_lmtn"]
    results = []
    path = os.path.join(RAW_DIR, "ncd", "ncd_csv", "ncd_trkg.csv")
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        for row in reader:
            sect = row.get("NCD_mnl_sect", "").strip()
            if sect in ncd_sections:
                for field in text_fields:
                    raw = row.get(field, "")
                    clean = strip_html(raw)
                    if len(clean) > 30:
                        results.append({
                            "source": f"NCD_{sect}",
                            "section_type": SECTION_MAP.get(field, "coverage_criteria"),
                            "text": clean,
                        })
    return results


def _read_articles(article_ids):
    results = []
    path = os.path.join(RAW_DIR, "current_article", "current_article_csv", "article.csv")
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("article_id", "") in article_ids:
                clean = strip_html(row.get("description", ""))
                if len(clean) > 30:
                    results.append({
                        "source": f"Article_{row['article_id']}",
                        "section_type": SECTION_MAP.get("description", "billing"),
                        "text": clean,
                    })
    return results


def _split_exclusions(sections):
    """Post-process sections to split out exclusion content from coverage text."""
    result = []
    # Patterns that indicate start of exclusion / noncovered content
    excl_patterns = [
        r"(?:Nationally\s+)?Non-?[Cc]overed\s+Indications",
        r"Contraindications",
        r"(?:are|is)\s+not\s+(?:covered|reasonable)",
    ]
    combined = "|".join(excl_patterns)

    for s in sections:
        text = s["text"]
        match = re.search(combined, text)
        if match and match.start() > 100:
            before = text[:match.start()].strip()
            after = text[match.start():].strip()
            if before:
                result.append({**s, "text": before})
            if after:
                result.append({**s, "text": after, "section_type": "exclusions"})
        else:
            result.append(s)
    return result


def parse_cms_data():
    """Parse all relevant CMS sources for target procedures.

    Returns dict mapping procedure_code -> list of {source, section_type, text}.
    """
    parsed = {}
    for proc_code, sources in LCD_SOURCES.items():
        sections = []
        sections.extend(_read_lcd(sources.get("lcd_ids", [])))
        sections.extend(_read_ncd(sources.get("ncd_sections", [])))
        if "article_ids" in sources:
            sections.extend(_read_articles(sources["article_ids"]))
        sections = _split_exclusions(sections)
        parsed[proc_code] = sections
    return parsed


if __name__ == "__main__":
    data = parse_cms_data()
    for proc, sections in data.items():
        print(f"{proc}: {len(sections)} sections, "
              f"{sum(len(s['text']) for s in sections)} total chars")
    out_path = os.path.join(os.path.dirname(__file__), "..", "data", "cms_parsed.json")
    with open(out_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Wrote {out_path}")
