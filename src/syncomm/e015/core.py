"""E015 core: model loading, concept directions, steering hook, batched greedy gen.

Conventions (frozen in the preregistration note):
  * Base models, plain-text prompts, no chat template.
  * Injection: h <- h + alpha * d_unit at ONE decoder layer, all positions,
    where alpha = beta * sigma_m(layer), sigma = median residual L2 norm.
  * Greedy decoding only. bf16 on GPU, fp32 on CPU.
"""
from __future__ import annotations
try:
    from syncomm.models import resolve_model
except ImportError:  # run outside the package: references must already be local paths
    def resolve_model(ref):
        return str(ref)

import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODELS = {
    "qwen3-0.6b": "/path/to/models/Qwen3-0.6B",
    "qwen3-1.7b": "/path/to/models/Qwen3-1.7B",
    "qwen3-8b-base": "/path/to/models/Qwen3-8B-Base",
    "qwen3-14b-base": "/path/to/models/Qwen3-14B-Base",
    "qwen3-8b": "/path/to/models/Qwen3-8B",
    "qwen3-14b": "/path/to/models/Qwen3-14B",
    "llama-3.1-8b": "/path/to/models/Llama-3.1-8B",
    # ---- E026 compatibility atlas: 17 hook-safe dense models, 8 families
    # (hybrid-attention models screened out: Ministral, OLMo-3, gpt-oss) ----
    "qwen3-8b-base": "/path/to/models/Qwen3-8B-Base",
    "qwen3-14b-base": "/path/to/models/Qwen3-14B-Base",
    "qwen3-8b": "/path/to/models/Qwen3-8B",
    "qwen3-4b": "/path/to/models/Qwen3-4B",
    # The POSTTRAIN0826 controlled ladder (lock BLANK-B3), added 2026-08-28:
    # one backbone (qwen3, 2560x36) in four known post-training states, so the
    # question "how MUCH post-training moves the address space" has a
    # dose-response rather than the two lineages plus a control the primary has.
    "qwen3-4b-instruct": "/path/to/models/Qwen3-4B-Instruct-2507",
    "qwen3-4b-rl-code": "/path/to/models/Qwen3-4B-Non-Thinking-RL-Code-Step300",
    "qwen3-4b-rl-math": "/path/to/models/Qwen3-4B-Non-Thinking-RL-Math-Step500",
    "qwen3-14b": "/path/to/models/Qwen3-14B",
    "qwen3-32b": "/path/to/models/Qwen3-32B",
    # width-vs-parameters panel (2026-08-05): MoE buys experts, not residual width
    "qwen3-30b-a3b": "/path/to/models/Qwen3-30B-A3B-Instruct-2507",
    # 2026 field-standard roster additions (survey of 22 verified top-tier papers,
    # 2026-08-10): Gemma-3-27B (8/22 papers incl. CMU AgentArk + the Anthropic-line
    # introspection work) and Llama-3.1-70B (Meta/CMU/NVIDIA). OLMo-3-7B already
    # present. glm-4-9b / phi-4 appear in 0/22 and are demoted to appendix breadth.
    "gemma-3-27b": "/path/to/models/gemma-3-27b-it",
    "llama-3.1-70b": "/path/to/models/Llama-3.1-70B-Instruct",
    "olmo-3-7b": "/path/to/models/OLMo-3-7B-Instruct",
    "qwen2.5-72b-instruct": "/path/to/models/Qwen2.5-72B-Instruct",
    "r1-llama-70b": "/path/to/models/DeepSeek-R1-Distill-Llama-70B",
    "yi-1.5-34b-chat": "/path/to/models/Yi-1.5-34B-Chat",
    "qwen2.5-7b": "/path/to/models/Qwen2.5-7B",
    "qwen2.5-7b-instruct": "/path/to/models/Qwen2.5-7B-Instruct",
    "qwen2.5-14b-instruct": "/path/to/models/Qwen2.5-14B-Instruct",
    # Added 2026-09-05: the PRETRAINED Qwen2.5-14B, the base DeepSeek lists for
    # R1-Distill-Qwen-14B (model card). POSTTRAIN0826 lineage B used the Instruct
    # sibling as its reference checkpoint because this base was not on disk;
    # downloaded 2026-09-05 (8 safetensors shards, headers verified).
    # 2026-09-05T0715Z: the shared-storage copy was deleted the same hour because it
    # exhausted the storage quota (all writes under anon/ failed);
    # the weights are staged NODE-LOCALLY (and mirrored on the dev box) at the path
    # below. Volatile by construction (node /tmp is wiped on re-provision): the
    # study lock records the HF revision + per-shard sizes, and re-staging is
    # `hf download Qwen/Qwen2.5-14B --local-dir /tmp/models/Qwen2.5-14B`
    # on the dev box + rsync to the node.
    "qwen2.5-14b": "/tmp/models/Qwen2.5-14B",
    "llama-3.1-8b": "/path/to/models/Llama-3.1-8B",
    "llama-3.1-8b-instruct": "/path/to/models/Llama-3.1-8B-Instruct",
    "yi-1.5-9b-chat": "/path/to/models/Yi-1.5-9B-Chat",
    "mistral-nemo": "/path/to/models/Mistral-Nemo-Instruct-2407",
    "phi-4": "/path/to/models/phi-4",
    "glm-4-9b": "/path/to/models/GLM-4-9B-0414",
    "granite-4.1-8b": "/path/to/models/granite-4.1-8b",
    "r1-llama-8b": "/path/to/models/DeepSeek-R1-Distill-Llama-8B",
    "r1-qwen-14b": "/path/to/models/DeepSeek-R1-Distill-Qwen-14B",
    # ---- 2026-generation families (added 2026-07-30 after an architecture screen
    # that verified what this method actually needs: model.model.layers reachable,
    # unembedding accessible, zero-injection an exact no-op, steering effective.
    # The earlier exclusion of these was MY BUG: the screen read
    # num_hidden_layers at the config top level, but these are unified
    # multimodal configs with the text dims nested under text_config, so they
    # looked dimension-less. Hybrid linear attention costs speed (no
    # flash-linear-attention kernel on the node), never hook compatibility,
    # because nothing here touches KV. Note sigma spans 11.8 (qwen3.5-4b) to
    # 2058 (gpt-oss-20b) — 174x — which is why doses are normalised per model. ----
    "qwen3.5-9b-base": "/path/to/models/Qwen3.5-9B-Base",
    "qwen3.5-9b": "/path/to/models/Qwen3.5-9B",
    "qwen3.5-4b": "/path/to/models/Qwen3.5-4B",
    "qwen3.5-2b": "/path/to/models/Qwen3.5-2B",
    "gpt-oss-20b": "/path/to/models/gpt-oss-20b",
    "qwen3.6-35b-a3b": "/path/to/models/Qwen3.6-35B-A3B",
    "qwen3.5-27b": "/path/to/models/Qwen3.5-27B",
    "seed-oss-36b": "/path/to/models/Seed-OSS-36B-Instruct",
    "qwen3.5-122b": "/path/to/models/Qwen3.5-122B-A10B",
    "qwen3.5-397b": "/path/to/models/Qwen3.5-397B-A17B",
    "qwen3.6-27b": "/path/to/models/Qwen3.6-27B",
}


def load(model_key: str, device: str = "cuda", dtype: str | None = None,
         device_map: str | None = None):
    """device_map='auto' shards across visible GPUs (needed for >183GB models);
    hooks and autograd work unchanged across devices."""
    path = MODELS[model_key]
    tok = AutoTokenizer.from_pretrained(resolve_model(path))
    if dtype is None:
        dtype = "bfloat16" if device != "cpu" else "float32"
    if device_map:
        model = AutoModelForCausalLM.from_pretrained(
            resolve_model(path), dtype=getattr(torch, dtype), device_map=device_map)
    else:
        model = AutoModelForCausalLM.from_pretrained(resolve_model(path), dtype=getattr(torch, dtype))
        model.to(device)
    model.eval()
    model.requires_grad_(False)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    # Batched generate on a causal LM requires left padding.
    tok.padding_side = "left"
    return model, tok


def decoder_layers(model):
    m = getattr(model, "model", model)
    if hasattr(m, "layers"):
        return m.layers
    # Multimodal wrappers (Gemma3ForConditionalGeneration -> Gemma3Model) nest
    # the text decoder one level further down as language_model; text-only
    # forwards, hooks and hidden_states are unaffected by the vision tower.
    lm = getattr(m, "language_model", None)
    if lm is not None and hasattr(lm, "layers"):
        return lm.layers
    raise AttributeError(
        f"no decoder layers on {type(model).__name__}: tried model.model.layers "
        f"and model.model.language_model.layers")


def tap_layer(model, frac: float) -> int:
    n = len(decoder_layers(model))
    li = int(round(frac * n))
    return max(1, min(n - 1, li))


def single_token_id(tok, word: str) -> int | None:
    """Token id of ' word' if it is a single token, else None."""
    ids = tok(" " + word, add_special_tokens=False)["input_ids"]
    return ids[0] if len(ids) == 1 else None


def concept_token_ids(tok, text: str) -> list[int]:
    """All token ids of ' text' — the multi-token generalisation of
    single_token_id. Used by the multi-token/phrase dictionary."""
    return tok(" " + text.strip(), add_special_tokens=False)["input_ids"]


def direction(model, tok, word: str, kind: str,
              multi: bool = False) -> torch.Tensor | None:
    """Unit-norm model-native direction for a concept.

    kind='unembed': row of lm_head weight (W_U). kind='input_embed': row of the
    input embedding.
    multi=False (default, unchanged): single-token concepts only, None otherwise
    — this preserves every result measured so far.
    multi=True: multi-token concepts and phrases are represented by the MEAN of
    their constituent rows, renormalised. This is a documented APPROXIMATION to
    the phrase/template lens (proposal §4.2 requires approximations to be
    labelled as such): it ignores composition order and treats the phrase as a
    bag of its wordpieces. It is the cheapest construction that generalises the
    dictionary beyond single tokens; a template-lens version is the principled
    alternative and remains unbuilt.
    """
    if multi:
        ids = concept_token_ids(tok, word)
        if not ids:
            return None
        W = (model.get_output_embeddings() if kind == "unembed"
             else model.get_input_embeddings()).weight
        w = W[torch.tensor(ids, device=W.device)].detach().float().mean(0)
        return w / (w.norm() + 1e-8)
    tid = single_token_id(tok, word)
    if tid is None:
        return None
    if kind == "unembed":
        w = model.get_output_embeddings().weight[tid]
    elif kind == "input_embed":
        w = model.get_input_embeddings().weight[tid]
    else:
        raise ValueError(kind)
    w = w.detach().float()
    return w / (w.norm() + 1e-8)


@torch.no_grad()
def residual_sigma(model, tok, li: int, device) -> float:
    """Median residual-stream L2 norm at layer li's OUTPUT over neutral text."""
    texts = [
        "The quick brown fox jumps over the lazy dog.",
        "In recent years, many people have started to",
        "Q: What time is it?\nA: It is",
        "1, 2, 3, 4, 5, 6, 7, 8",
    ]
    norms = []
    for t in texts:
        ids = tok(t, return_tensors="pt").input_ids.to(device)
        hs = model(ids, output_hidden_states=True).hidden_states
        # hidden_states[li+1] is the output of decoder layer li (index 0 = embeddings)
        h = hs[li + 1][0].float()
        norms.append(h.norm(dim=-1))
    return float(torch.cat(norms).median())


class Steer:
    """Forward hook adding a fixed vector to decoder layer li's output (all
    positions, every forward chunk incl. cached decoding steps)."""

    def __init__(self, model, li: int, vec: torch.Tensor | None):
        self.vec = vec
        self.handle = None
        if vec is not None:
            self.handle = decoder_layers(model)[li].register_forward_hook(self._hk)

    def _hk(self, mod, args, out):
        h = out[0] if isinstance(out, tuple) else out
        h = h + self.vec.to(h.dtype).to(h.device)
        return (h,) + tuple(out[1:]) if isinstance(out, tuple) else h

    def remove(self):
        if self.handle is not None:
            self.handle.remove()
            self.handle = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.remove()


@torch.no_grad()
def generate_batch(model, tok, prompts: list[str], vec: torch.Tensor | None,
                   li: int, max_new_tokens: int = 32) -> list[str]:
    """Greedy continuations under optional steering. Returns new text only."""
    enc = tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False)
    dev = next(model.parameters()).device
    ids, attn = enc.input_ids.to(dev), enc.attention_mask.to(dev)
    with Steer(model, li, vec):
        out = model.generate(ids, attention_mask=attn,
                             max_new_tokens=max_new_tokens, do_sample=False,
                             pad_token_id=tok.pad_token_id)
    return tok.batch_decode(out[:, ids.shape[1]:], skip_special_tokens=True)


@torch.no_grad()
def choice_logit_gap(model, tok, stems: list[str], tid_a: int, tid_b: int,
                     vec: torch.Tensor | None, li: int) -> list[float]:
    """logit(a) - logit(b) at the next-token position for each stem."""
    enc = tok(stems, return_tensors="pt", padding=True, add_special_tokens=False)
    dev = next(model.parameters()).device
    with Steer(model, li, vec):
        logits = model(enc.input_ids.to(dev),
                       attention_mask=enc.attention_mask.to(dev)).logits
    last = logits[:, -1].float()  # left padding -> last position is the real end
    return (last[:, tid_a] - last[:, tid_b]).tolist()


def hidden_size(model) -> int:
    """Residual width, tolerating unified multimodal configs.

    This is the SAME defect the 2026-07-30 note above records for
    num_hidden_layers, still live for hidden_size until 2026-08-28: Gemma3Config
    (and the other unified configs) nest the text dims under `text_config`, so
    `model.config.hidden_size` raises AttributeError rather than returning the
    width. It killed the gemma-3-27b steerability profile inside
    zero_injection_check — i.e. inside the integrity check every adapter must
    pass — and left gemma a declared hole in the PROBELAW0828 roster.

    Raises rather than guessing: a wrong width would silently build a
    mis-shaped zero vector, and a zero-injection check that cannot be trusted
    is worse than one that fails loudly.
    """
    for cfg in (model.config, getattr(model.config, "text_config", None)):
        if cfg is None:
            continue
        d = getattr(cfg, "hidden_size", None)
        if isinstance(d, int) and d > 0:
            return d
    raise ValueError(
        f"cannot read hidden_size from {type(model.config).__name__}; "
        f"checked config and config.text_config")


@torch.no_grad()
def zero_injection_check(model, tok, li: int, device) -> float:
    """Max |logit diff| between hook-with-zero-vector and no hook. Must be 0."""
    d = hidden_size(model)
    zero = torch.zeros(d, device=device)
    ids = tok("The capital of France is", return_tensors="pt").input_ids.to(device)
    base = model(ids).logits
    with Steer(model, li, zero):
        hooked = model(ids).logits
    return float((base - hooked).abs().max())


@torch.no_grad()
def meandiff_direction(model, tok, word: str, li: int, device) -> torch.Tensor:
    """Activation mean-difference direction (positive control): states from
    concept-focused text minus neutral text, last-8-token mean, at layer li."""
    pos_texts = [f"Let me tell you about the {word}. The {word} is",
                 f"Facts about the {word}: the {word}",
                 f"My favorite thing is the {word}. I love the {word}"]
    neg_texts = ["Let me tell you about it. It is",
                 "Facts about this topic: it",
                 "My favorite thing is this. I love it"]

    def states(texts):
        hs = []
        for t in texts:
            ids = tok(t, return_tensors="pt").input_ids.to(device)
            h = model(ids, output_hidden_states=True).hidden_states[li + 1][0]
            hs.append(h[-8:].float().mean(0))
        return torch.stack(hs).mean(0)

    v = states(pos_texts) - states(neg_texts)
    return v / (v.norm() + 1e-8)


def append_jsonl(path, row: dict):
    with open(path, "a") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def done_keys(path) -> set:
    keys = set()
    try:
        with open(path) as f:
            for line in f:
                try:
                    keys.add(json.loads(line)["key"])
                except Exception:
                    continue
    except FileNotFoundError:
        pass
    return keys
