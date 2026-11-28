"""Token counts of the 5-shot Belebele requests under the Llama-3.2-1B tokenizer (revision 5a8abab4a5d6).

    pip install lm_eval==0.4.13 datasets tokenizers huggingface_hub
    python token_counts.py belebele_hin_Deva belebele_kan_Knda belebele_tel_Telu belebele_tam_Taml belebele_ben_Beng

Builds the requests with lm_eval (num_fewshot 5, seed 42, no chat template), as run_wide_gguf.py does, and counts
context + continuation tokens plus one BOS. Output of the run of 25 Nov: token_counts.txt.
"""
import json
import statistics as st
import sys

from huggingface_hub import hf_hub_download
from lm_eval.tasks import TaskManager, get_task_dict
from tokenizers import Tokenizer

REV = "5a8abab4a5d6f164389b1079fb721cfab8d7126c"
tok = Tokenizer.from_file(hf_hub_download("unsloth/Llama-3.2-1B-Instruct", "tokenizer.json", revision=REV))
tm = TaskManager()

for name in sys.argv[1:]:
    task = get_task_dict([name], tm)[name]
    task.set_fewshot_seed(42)
    task._config.num_fewshot = 5
    task.build_all_requests(limit=None, rank=0, world_size=1, apply_chat_template=False, fewshot_as_multiturn=False)
    lens = sorted(len(tok.encode(ctx + cont, add_special_tokens=False).ids) + 1   # +1 for BOS
                  for ctx, cont in (inst.args for inst in task.instances))
    n = len(lens)
    r = dict(requests=n, mean=round(st.mean(lens)), p50=lens[n // 2], p95=lens[int(n * .95)], max=lens[-1],
             over4096=sum(x > 4096 for x in lens), over8192=sum(x > 8192 for x in lens),
             over12288=sum(x > 12288 for x in lens), over16384=sum(x > 16384 for x in lens))
    print(name, json.dumps(r), flush=True)
