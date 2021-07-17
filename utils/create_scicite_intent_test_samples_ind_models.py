import sys
import json


num_of_background = 869
num_of_method = 558
num_of_result = 219

ori_dataset = ["background"] * num_of_background + ["method"] * num_of_method + ["result"] * num_of_result

scicite_model_dataset = []
generated_citation_texts_path = sys.argv[1]
output_file_path = sys.argv[2]

with open(generated_citation_texts_path, "r") as fin:
    genearted_citation_texts = [line.strip() for line in fin]

assert len(genearted_citation_texts) == len(ori_dataset)

with open(output_file_path, "w") as fout:
    for idx, ori_data in enumerate(ori_dataset):
        scicite_model_data = {
            "source": "",
            "citeEnd": None,
            "sectionName": "",
            "citeStart": None,
            "string": genearted_citation_texts[idx],
            "label": ori_data,
            "citingPaperId": "",
            "citedPaperId": "",
            "isKeyCitation": None,
            "id": "",
            "unique_id": "",
            "excerpt_index": None
        }
        fout.write(json.dumps(scicite_model_data))
        fout.write("\n")
