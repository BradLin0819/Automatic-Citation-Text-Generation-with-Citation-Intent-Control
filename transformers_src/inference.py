import torch
import argparse
from logging import getLogger

from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer
)
from utils import (
    use_task_specific_params,
    parse_numeric_n_bool_cl_kwargs,
    add_special_tokens_
)


logger = getLogger(__name__)

DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def process_source_input(text):
    text = text.replace(
        '\n', ' ').replace("\r", " ").strip().lower()

    return text


def generate_citation_text(
    src: str,
    model_name: str,
    model_state_dict_path: str = None,
    device: str = DEFAULT_DEVICE,
    fp16=False,
    prefix=None,
    **generate_kwargs,
) -> str:
    model_name = str(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)
    if fp16:
        model = model.half()

    if model_state_dict_path is not None:
        model.load_state_dict(torch.load(model_state_dict_path))

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    logger.info(f"Inferred tokenizer type: {tokenizer.__class__}")
    # add_special_tokens_(model, tokenizer)
    use_task_specific_params(model, 'summarization')
    if prefix is None:
        prefix = prefix or getattr(model.config, "prefix", "") or ""

    src = prefix + src
    tokenized_src = tokenizer(src, return_tensors="pt",
                              truncation=True).to(device)

    citation_text_preds = model.generate(
        input_ids=tokenized_src.input_ids,
        attention_mask=tokenized_src.attention_mask,
        **generate_kwargs,
    )

    citation_text = tokenizer.batch_decode(
        citation_text_preds, skip_special_tokens=True, clean_up_tokenization_spaces=True)

    return citation_text


def run_generate(verbose=True):
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str,
                        help="like facebook/bart-large-cnn,t5-base, etc.")
    parser.add_argument("--pretrained_model_path", type=str,
                        default=None, help="path of pretrained model")
    parser.add_argument("--citing_context", type=str,
                        help="Context of citing paper (e.g. abstract of citing paper)")
    parser.add_argument("--cited_context", type=str,
                        help="Context of cited paper(e.g. abstract of cited paper)")
    parser.add_argument("--intent", type=str,
                        default="background", help="Specified citation intent")
    parser.add_argument("--device", type=str, required=False,
                        default=DEFAULT_DEVICE, help="cuda, cuda:1, cpu etc.")
    parser.add_argument(
        "--prefix", type=str, required=False, default=None, help="will be added to the begininng of src examples"
    )
    parser.add_argument("--fp16", action="store_true")

    # Unspecified args like --num_beams=2 --decoder_start_token_id=4 are passed to model.generate
    args, rest = parser.parse_known_args()
    parsed_args = parse_numeric_n_bool_cl_kwargs(rest)
    if parsed_args and verbose:
        print(f"parsed the following generate kwargs: {parsed_args}")

    citing_src = args.citing_context
    cited_src = args.cited_context
    model_src = citing_src + " " + cited_src

    model_src = " " + \
        model_src.rstrip() if "t5" in args.model_name else model_src.rstrip()

    model_src = f'@{args.intent} {process_source_input(model_src)}'

    citation_text = generate_citation_text(
        model_src,
        args.model_name,
        args.pretrained_model_path,
        device=args.device,
        fp16=args.fp16,
        prefix=args.prefix,
        **parsed_args,
    )

    return citation_text


if __name__ == '__main__':
    print(run_generate())
