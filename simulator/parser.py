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
def _get_lcd_sources():
    with open(os.path.join(RAW_DIR, "..", "templates.json"), "r") as f:
        templates = json.load(f)
    return {proc: t.get("sources", {}) for proc, t in templates.items()}

LCD_SOURCES = _get_lcd_sources()


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

    Returns dict mapping procedure_code -> list of {source, section_type, text, procedure_code}.
    """
    parsed = {}
    for proc_code, sources in LCD_SOURCES.items():
        sections = []
        sections.extend(_read_lcd(sources.get("lcd_ids", [])))
        sections.extend(_read_ncd(sources.get("ncd_sections", [])))
        if "article_ids" in sources:
            sections.extend(_read_articles(sources["article_ids"]))
        sections = _split_exclusions(sections)
        for s in sections:
            s["procedure_code"] = proc_code
        parsed[proc_code] = sections
        
    # Add some generic/distractor chunks
    general_sections = []
    general_sections.append({
        "source": "admin-100-01",
        "section_type": "coverage_criteria",
        "text": "Medicare Part B covers medically necessary services and preventive services. Medically necessary services are defined as services or supplies needed to diagnose or treat a medical condition and that meet accepted standards of medical practice.",
        "procedure_code": "general"
    })
    general_sections.append({
        "source": "admin-100-01",
        "section_type": "coverage_criteria",
        "text": "Prior authorization is required for certain Medicare Part B services as specified by CMS. The prior authorization process ensures that services meet Medicare coverage requirements before they are provided. Providers must submit clinical documentation supporting medical necessity.",
        "procedure_code": "general"
    })
    general_sections.append({
        "source": "admin-100-02",
        "section_type": "coverage_criteria",
        "text": "Evidence-based clinical guidelines should be used to support prior authorization decisions. The American Medical Association and specialty societies publish guidelines that inform coverage determinations. Payers should reference the most current published guidelines.",
        "procedure_code": "general"
    })
    general_sections.append({
        "source": "admin-100-02",
        "section_type": "billing",
        "text": "Claims for prior authorized services must include the prior authorization number on the CMS-1500 or UB-04 claim form. Failure to include the authorization number may result in claim denial. Electronic claims should include the authorization in the appropriate loop and segment.",
        "procedure_code": "general"
    })
    general_sections.append({
        "source": "admin-100-03",
        "section_type": "coverage_criteria",
        "text": "The appeals process for denied prior authorization requests is available to all Medicare beneficiaries. The first level of appeal is a redetermination by the Medicare Administrative Contractor. Subsequent appeal levels include reconsideration by a Qualified Independent Contractor and hearing by an Administrative Law Judge.",
        "procedure_code": "general"
    })
    parsed["general"] = general_sections
    
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
