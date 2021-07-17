import os
import json
import logging
import argparse
from scicite_data_utils import *

logger = logging.getLogger(__name__)


class SciciteDataPreprocessor:
    def __init__(self,
                 file_path):
        self._citances = load_data(file_path)

    def _clean_citances(self):
        citation_string_set = set()
        for citance in self._citances:
            # ignore replicated citation texts
            if (citance["string"] != "") and (citance["string"] not in citation_string_set):
                citation_string_set.add(citance["string"])
                yield citance

    def get_src_tgt_pairs(self):
        src_tgt_pairs = []

        for cleaned_citance in self._clean_citances():
            citation_marker = None

            # augmented data from semantic scholar
            if "citation_marker" in cleaned_citance:
                citation_marker = cleaned_citance["citation_marker"]
                if citation_marker == "":
                    citation_marker = None
            # original scicite data
            elif "citeStart" in cleaned_citance and "citeEnd" in cleaned_citance:
                citation_marker_span = (
                    cleaned_citance["citeStart"], cleaned_citance["citeEnd"])
                citation_marker = get_citation_marker_from_span(cleaned_citance["string"],
                                                                citation_marker_span)
            # filter out data without abstract
            if cleaned_citance["citedAbstract"].lower() not in ("", "abstract"):
                src_tgt_pairs.append({
                    "unique_id": cleaned_citance["unique_id"],
                    "citingPaperId": cleaned_citance["citingPaperId"],
                    "citedPaperId": cleaned_citance["citedPaperId"],
                    "citingTitle": cleaned_citance["citingTitle"],
                    "citingAbstract": cleaned_citance["citingAbstract"],
                    "citedTitle": cleaned_citance["citedTitle"],
                    "citedAbstract": cleaned_citance["citedAbstract"],
                    "citation_text": cleaned_citance["string"],
                    "sectionName": cleaned_citance["sectionName"],
                    "citation_marker": citation_marker,
                    "intent": cleaned_citance["label"]
                })

        return src_tgt_pairs

    def _preprocess(self, src_tgt_pairs, intent=None,
                    prepend_token=False, citing_input_mode='abstract'):
        preprocessd_src_tgt_pairs = []

        for src_tgt_pair in src_tgt_pairs:
            # Check if valid citation text
            if is_valid_citation_text(src_tgt_pair["citation_text"]):
                if intent is not None and intent != src_tgt_pair["intent"]:
                    continue

                citing_src = get_citing_paper_input(
                    src_tgt_pair, citing_input_mode)
                citing_src_preprocessed = process_source_input(citing_src)
                cited_src_preprocessed = process_source_input(
                    src_tgt_pair["citedAbstract"])
                src = citing_src_preprocessed + " " + cited_src_preprocessed
                control_codes = ""

                # prepend citation intent token
                if prepend_token:
                    control_codes += f'@{src_tgt_pair["intent"]} '
                    src = f"{control_codes} {src}"

                tgt = process_reference_marker(
                    src_tgt_pair["citation_text"], src_tgt_pair["citation_marker"]).replace('\n', ' ').strip()

                if src != "" and tgt != "":
                    preprocessd_src_tgt_pairs.append({
                        "unique_id": src_tgt_pair["unique_id"],
                        "citingPaperId": src_tgt_pair["citingPaperId"],
                        "citedPaperId": src_tgt_pair["citedPaperId"],
                        "citingTitle": src_tgt_pair["citingTitle"],
                        "citingAbstract": src_tgt_pair["citingAbstract"],
                        "citedTitle": src_tgt_pair["citedTitle"],
                        "citedAbstract": src_tgt_pair["citedAbstract"],
                        "citingAbstract_preprocessed": process_source_input(src_tgt_pair["citingAbstract"]),
                        "citedAbstract_preprocessed": cited_src_preprocessed,
                        "intent": src_tgt_pair["intent"],
                        "model_src": src,
                        "citation_text": tgt
                    })

        return preprocessd_src_tgt_pairs

    def get_preprocessed_src_tgt_pairs(self, intent=None, prepend_token=False,
                                       citing_input_mode='abstract'):
        return self._preprocess(self.get_src_tgt_pairs(), intent=intent, prepend_token=prepend_token,
                                citing_input_mode=citing_input_mode)

    def _export_to_transformers_input_file(self, dataset, dataset_type,
                                           out_dir='scicite_data'):
        data_path = os.path.join(out_dir, dataset_type)

        with open(f"{data_path}.source", "w") as f_src, open(f"{data_path}.target", "w") as f_tgt:
            for data in dataset:
                f_src.write(data["model_src"])
                f_src.write('\n')

                f_tgt.write(data["citation_text"])
                f_tgt.write('\n')

    def _export_to_jsonl_file(self, dataset, dataset_type,
                              out_dir='scicite_data'):
        data_path = os.path.join(out_dir, dataset_type)

        with open(f"{data_path}.jsonl", "w") as f_src, open(f"{data_path}.jsonl", "w") as f_tgt:
            for data in dataset:
                f_src.write(json.dumps(data))
                f_src.write('\n')

                f_tgt.write(json.dumps(data))
                f_tgt.write('\n')

    def export_to_transformers_input_file(self, dataset_type, out_dir='scicite_data',
                                          intent=None, prepend_token=False, citing_input_mode='abstract'):
        dataset = self.get_preprocessed_src_tgt_pairs(
            intent=intent, prepend_token=prepend_token, citing_input_mode=citing_input_mode)

        if not os.path.isdir(out_dir):
            os.mkdir(out_dir)

        self._export_to_transformers_input_file(
            dataset, dataset_type, out_dir=out_dir)

    def export_to_jsonl_file(self, dataset_type, out_dir='scicite_data',
                             intent=None,  citing_input_mode='abstract'):
        dataset = self.get_preprocessed_src_tgt_pairs(
            intent=intent, prepend_token=prepend_token, citing_input_mode=citing_input_mode)

        if not os.path.isdir(out_dir):
            os.mkdir(out_dir)

        self._export_to_jsonl_file(
            dataset, dataset_type, out_dir=out_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str,
                        help="Path of parsed dataset to be exported to transformers input format")
    parser.add_argument("--out_dir", type=str,
                        help="Path of output diretory")
    parser.add_argument("--dataset_type", type=str, required=False,
                        help="Dataset type (train/val/test)")
    parser.add_argument("--intent", type=str, required=False,
                        help="Citation dataset with <intent>")
    parser.add_argument("--citing_input_mode", type=str, default="abstract",
                        help="The input content of citing paper ['abstract' | 'title']")
    parser.add_argument("--prepend_token", action="store_true",
                        help="Prepend intent token")
    parser.add_argument("--outfile_type", type=str, required=True,
                        help="export file type (huggingface transformers format (hf) or jsonl)")
    args = parser.parse_args()

    scicite_preprocessor = SciciteDataPreprocessor(args.input_file)

    if args.dataset_type is None:
        raise ValueError("Please input dataset type!")

    if args.outfile_type == "jsonl":
        scicite_preprocessor.export_to_jsonl_file(
            args.dataset_type, out_dir=args.out_dir, intent=args.intent,
            citing_input_mode=args.citing_input_mode)
    elif args.outfile_type == "hf":
        scicite_preprocessor.export_to_transformers_input_file(
            args.dataset_type, out_dir=args.out_dir, intent=args.intent,
            prepend_token=args.prepend_token, citing_input_mode=args.citing_input_mode)
