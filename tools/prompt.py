"""The Needle 3 prompt on the wire ("Porting Needle 3", docs/upstream/SOURCES.md, and
needle.model.finetune.render_example, which training used)."""
import json

IM_START, IM_END = "<|im_start|>", "<|im_end|>"


def render_tools(tools):
    keep = ("name", "description", "parameters")
    clean = [{k: t[k] for k in keep if k in t} for t in tools]
    return json.dumps(clean, separators=(",", ":"), ensure_ascii=False)


def render_prefix(tools, system=""):
    pre = IM_START + "system\n" + system + IM_END + "\n" if system else ""
    return pre + IM_START + "user\n<tools>" + render_tools(tools) + "</tools>"


def render_turn(query):
    return "\n" + query + IM_END + "\n" + IM_START + "assistant\n"


def render_prompt(tools, query, system=""):
    return render_prefix(tools, system) + render_turn(query)
