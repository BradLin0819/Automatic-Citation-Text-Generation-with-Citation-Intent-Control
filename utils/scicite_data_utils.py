import os
import re
import math
import json
import spacy
import scispacy


nlp = spacy.load('en_core_sci_lg')


def load_data(data_path):
    with open(data_path, "r") as f:
        for line in f:
            yield json.loads(line)


def get_citation_marker_from_span(text, citation_marker_span):
    target_reference_sign = None

    cite_start, cite_end = (int(idx) if not math.isnan(idx) else idx
                            for idx in citation_marker_span)

    if not math.isnan(cite_start) and not math.isnan(cite_end):
        target_reference_sign = text[cite_start:cite_end]

        # Some given cite_start and cite_end should be right shift by 1
        # Otherwise, we will get wrong reference signs e.g., (Martin, 201
        if target_reference_sign.startswith("(") and not target_reference_sign.endswith(")"):
            target_reference_sign = text[cite_start+1:cite_end+1]
    return target_reference_sign


def process_citation_with_cite_pattern(text, cite_pattern):
    reference_sign_matches = re.finditer(cite_pattern, text)

    for reference_sign_match in reference_sign_matches:
        original_reference_sign = reference_sign_match.group(0)

        if "#REF" in original_reference_sign:
            text = text.replace(original_reference_sign, "#REF")
        else:
            text = text.replace(
                original_reference_sign, "#OTHERREF")
    return text


def process_apa_citation(text):
    cite_pattern = "(([^\(\)]+\d{1,4})|(#REF))"
    apa_incitation_pattern = f"\({cite_pattern}(; {cite_pattern})*\)"
    text = process_citation_with_cite_pattern(
        text, apa_incitation_pattern)
    return text


def process_ieee_citation(text):
    cite_pattern = "((\d{1,}(-\d{1,})*)|(#REF))"
    ieee_incitation_pattern = f"\[{cite_pattern}(,\s?{cite_pattern})*\]"
    text = process_citation_with_cite_pattern(
        text, ieee_incitation_pattern)
    return text


def process_reference_marker(text, citation_marker=None):
    processed_text = text
    if citation_marker is not None:
        processed_text = processed_text.replace(citation_marker, "#REF")

    # Handle APA citation
    processed_text = process_apa_citation(
        processed_text)

    # Handle IEEE citation
    processed_text = process_ieee_citation(
        processed_text)

    return processed_text

# preprocess citing or cited input content


def process_source_input(text):
    text = text.replace(
        '\n', ' ').replace("\r", " ").strip().lower()

    return text


def process_seciton_name(section_name):
    section_name = str(section_name).lower()
    section_name = re.sub(
        "(\d+\.?)+\d?[\s\t]+(.+)", r"\2", section_name)
    return section_name


def get_citing_paper_input(data_pair, citing_input_mode='abstract'):
    if citing_input_mode == "title":
        return_input = data_pair["citingTitle"]
    elif citing_input_mode == "abstract":
        return_input = data_pair["citingAbstract"]
    else:
        raise TypeError("Unknown input mode")
    return return_input


def is_valid_citation_text(text):
    # sentence starts with "...", it's an incomplete sentence
    invalid_charaters = (b'\xe2\x80\xa6'.decode(), ",")
    return not text.startswith(invalid_charaters)
