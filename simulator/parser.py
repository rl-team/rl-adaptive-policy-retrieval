"""Parse CMS Medicare Coverage Database CSVs into structured JSON."""

import csv
import json
import os
import re
import sys

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


def _get_lcd_sources():
    with open(os.path.join(RAW_DIR, "..", "templates.json"), "r") as f:
        templates = json.load(f)
    return {proc: t.get("sources", {}) for proc, t in templates.items()}


def get_sources_for_code(hcpc_code):
    article_ids = set()
    lcd_ids = set()

    # 1. Find articles by HCPC code
    art_hcpc_path = os.path.join(
        RAW_DIR, "current_article", "current_article_csv", "article_x_hcpc_code.csv")
    if os.path.exists(art_hcpc_path):
        with open(art_hcpc_path, "r", encoding="utf-8", errors="replace") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get("hcpc_code_id") == hcpc_code:
                    article_ids.add(row.get("article_id"))

    # 2. Find LCDs by HCPC code
    lcd_hcpc_path = os.path.join(
        RAW_DIR, "current_lcd", "current_lcd_csv", "lcd_x_hcpc_code.csv")
    if os.path.exists(lcd_hcpc_path):
        with open(lcd_hcpc_path, "r", encoding="utf-8", errors="replace") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get("hcpc_code_id") == hcpc_code:
                    lcd_ids.add(row.get("lcd_id"))

    # 3. Find related LCDs from articles
    art_rel_path = os.path.join(
        RAW_DIR, "current_article", "current_article_csv", "article_related_documents.csv")
    if os.path.exists(art_rel_path):
        with open(art_rel_path, "r", encoding="utf-8", errors="replace") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get("article_id") in article_ids:
                    r_lcd = row.get("r_lcd_id")
                    if r_lcd:
                        lcd_ids.add(r_lcd)

    # 4. Find related articles from LCDs
    lcd_rel_path = os.path.join(
        RAW_DIR, "current_lcd", "current_lcd_csv", "lcd_related_documents.csv")
    if os.path.exists(lcd_rel_path):
        with open(lcd_rel_path, "r", encoding="utf-8", errors="replace") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get("lcd_id") in lcd_ids:
                    r_art = row.get("r_article_id")
                    if r_art:
                        article_ids.add(r_art)
                # Reverse relation
                if row.get("r_article_id") in article_ids:
                    r_lcd = row.get("lcd_id")
                    if r_lcd:
                        lcd_ids.add(r_lcd)

    # 5. Look for related NCDs
    ncd_ids = set()
    ncd_sect_map = {}
    ncd_trkg_path = os.path.join(RAW_DIR, "ncd", "ncd_csv", "ncd_trkg.csv")
    if os.path.exists(ncd_trkg_path):
        with open(ncd_trkg_path, "r", encoding="utf-8", errors="replace") as f:
            reader = csv.DictReader(f)
            for row in reader:
                ncd_id = row.get("NCD_id")
                sect = row.get("NCD_mnl_sect")
                if ncd_id and sect:
                    ncd_sect_map[ncd_id] = sect

    lcd_rel_ncd = os.path.join(
        RAW_DIR, "current_lcd", "current_lcd_csv", "lcd_related_ncd_documents.csv")
    if os.path.exists(lcd_rel_ncd):
        with open(lcd_rel_ncd, "r", encoding="utf-8", errors="replace") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get("lcd_id") in lcd_ids:
                    ncd_id = row.get("r_ncd_id")
                    if ncd_id and ncd_id != "0":
                        ncd_ids.add(ncd_id)

    art_rel_ncd = os.path.join(
        RAW_DIR, "current_article", "current_article_csv", "article_related_ncd_documents.csv")
    if os.path.exists(art_rel_ncd):
        with open(art_rel_ncd, "r", encoding="utf-8", errors="replace") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get("article_id") in article_ids:
                    ncd_id = row.get("r_ncd_id")
                    if ncd_id and ncd_id != "0":
                        ncd_ids.add(ncd_id)

    ncd_sections = set()
    for nid in ncd_ids:
        sect = ncd_sect_map.get(nid)
        if sect:
            ncd_sections.add(sect)

    return {
        "lcd_ids": sorted(list(lcd_ids)),
        "article_ids": sorted(list(article_ids)),
        "ncd_sections": sorted(list(ncd_sections))
    }


LCD_SOURCES = _get_lcd_sources()


def strip_html(text):
    if not text:
        return ""
    # Convert block tags to periods to aid in sentence chunking
    text = re.sub(r"<(br|p|li|tr|div|h[1-6])[^>]*>", ". ", text, flags=re.IGNORECASE)
    text = re.sub(r"</(p|li|tr|div|h[1-6])>", ". ", text, flags=re.IGNORECASE)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"&[a-z]+;", " ", text)
    text = re.sub(r"&#\d+;", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    # Collapse consecutive periods introduced by stripping
    text = re.sub(r"\.\s*\.", ".", text)
    text = re.sub(r"^\.", "", text).strip()
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
    path = os.path.join(RAW_DIR, "current_article",
                        "current_article_csv", "article.csv")
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


def _split_exclusions(sections, min_prefix_length=100):
    """Post-process sections to split out exclusion content from coverage text.
    
    Args:
        sections: List of section dicts.
        min_prefix_length: Minimum characters before the exclusion pattern to split.
            This threshold (default 100) prevents splitting on Table of Contents 
            or very short introductory text.
    """
    result = []
    # Patterns that indicate start of exclusion / noncovered content
    excl_patterns = [
        r"(?:Nationally\s+)?Non-?[Cc]overed\s+Indications",
        r"Contraindications",
    ]
    combined = "|".join(excl_patterns)

    for s in sections:
        text = s["text"]
        match = re.search(combined, text)
        if match and match.start() > min_prefix_length:
            before = text[:match.start()].strip()
            after = text[match.start():].strip()
            if before:
                result.append({**s, "text": before})
            if after:
                result.append(
                    {**s, "text": after, "section_type": "exclusions"})
        else:
            result.append(s)
    return result


def parse_cms_data():
    """Parse all relevant CMS sources for target procedures.

    Returns dict mapping procedure_code -> list of {source, section_type, text, procedure_code}.
    """
    parsed = {}
    for proc_code, sources in LCD_SOURCES.items():
        if not sources or (not sources.get("lcd_ids") and not sources.get("article_ids") and not sources.get("ncd_sections")):
            print(
                f"[parse] Automatically determining sources for {proc_code}...")
            sources = get_sources_for_code(proc_code)
            print(f"[parse] Found for {proc_code}: {sources}")

        sections = []
        sections.extend(_read_lcd(sources.get("lcd_ids", [])))
        sections.extend(_read_ncd(sources.get("ncd_sections", [])))
        if "article_ids" in sources:
            sections.extend(_read_articles(sources["article_ids"]))
        sections = _split_exclusions(sections)
        for s in sections:
            s["procedure_code"] = proc_code
        parsed[proc_code] = sections

    return parsed


if __name__ == "__main__":
    data = parse_cms_data()
    for proc, sections in data.items():
        print(f"{proc}: {len(sections)} sections, "
              f"{sum(len(s['text']) for s in sections)} total chars")
    out_path = os.path.join(os.path.dirname(__file__),
                            "..", "data", "cms_parsed.json")
    with open(out_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Wrote {out_path}")
