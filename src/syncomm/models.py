"""Resolve the model references stored in run manifests to loadable checkpoints.

Manifests record checkpoints as ``/path/to/models/<Name>`` (the directory names used when the experiments ran).
``resolve_model`` maps such a reference, a bare ``<Name>``, or a Hugging Face ID to a local directory:

* if ``SYNC_MODEL_ROOT`` is set, ``$SYNC_MODEL_ROOT/<Name>`` is used (offline machines, local mirrors);
* otherwise ``<Name>`` is looked up in ``HF_IDS`` and downloaded with ``huggingface_hub.snapshot_download``
  (respecting ``HF_HOME``/``HF_TOKEN``; Llama and Gemma checkpoints are gated and need an accepted license).

Any other string (an existing local directory or an unknown Hugging Face ID) is returned unchanged.
"""
import os
from pathlib import Path

PLACEHOLDER = '/path/to/models/'

# Directory name -> Hugging Face model ID (each ID verified to exist on the Hub).
HF_IDS = {
    'Qwen3-0.6B': 'Qwen/Qwen3-0.6B',
    'Qwen3-1.7B': 'Qwen/Qwen3-1.7B',
    'Qwen3-4B': 'Qwen/Qwen3-4B',
    'Qwen3-4B-Instruct-2507': 'Qwen/Qwen3-4B-Instruct-2507',
    'Qwen3-8B': 'Qwen/Qwen3-8B',
    'Qwen3-8B-Base': 'Qwen/Qwen3-8B-Base',
    'Qwen3-14B': 'Qwen/Qwen3-14B',
    'Qwen3-14B-Base': 'Qwen/Qwen3-14B-Base',
    'Qwen3-32B': 'Qwen/Qwen3-32B',
    'Qwen3-30B-A3B-Instruct-2507': 'Qwen/Qwen3-30B-A3B-Instruct-2507',
    'Qwen3.5-2B': 'Qwen/Qwen3.5-2B',
    'Qwen3.5-4B': 'Qwen/Qwen3.5-4B',
    'Qwen3.5-9B': 'Qwen/Qwen3.5-9B',
    'Qwen3.5-9B-Base': 'Qwen/Qwen3.5-9B-Base',
    'Qwen3.5-27B': 'Qwen/Qwen3.5-27B',
    'Qwen3.5-122B-A10B': 'Qwen/Qwen3.5-122B-A10B',
    'Qwen3.5-397B-A17B': 'Qwen/Qwen3.5-397B-A17B',
    'Qwen3.6-27B': 'Qwen/Qwen3.6-27B',
    'Qwen3.6-35B-A3B': 'Qwen/Qwen3.6-35B-A3B',
    'Qwen2.5-7B': 'Qwen/Qwen2.5-7B',
    'Qwen2.5-7B-Instruct': 'Qwen/Qwen2.5-7B-Instruct',
    'Qwen2.5-14B': 'Qwen/Qwen2.5-14B',
    'Qwen2.5-14B-Instruct': 'Qwen/Qwen2.5-14B-Instruct',
    'Qwen2.5-72B-Instruct': 'Qwen/Qwen2.5-72B-Instruct',
    'Llama-3.1-8B': 'meta-llama/Llama-3.1-8B',
    'Llama-3.1-8B-Instruct': 'meta-llama/Llama-3.1-8B-Instruct',
    'Llama-3.1-70B-Instruct': 'meta-llama/Llama-3.1-70B-Instruct',
    'gemma-3-27b-it': 'google/gemma-3-27b-it',
    'gemma-4-26B-A4B-it': 'google/gemma-4-26B-A4B-it',
    'GLM-4-9B-0414': 'zai-org/GLM-4-9B-0414',
    'phi-4': 'microsoft/phi-4',
    'granite-4.1-8b': 'ibm-granite/granite-4.1-8b',
    'OLMo-3-7B-Instruct': 'allenai/Olmo-3-7B-Instruct',
    'Mistral-Nemo-Instruct-2407': 'mistralai/Mistral-Nemo-Instruct-2407',
    'Mistral-Small-3.2-24B-Instruct-2506': 'mistralai/Mistral-Small-3.2-24B-Instruct-2506',
    'gpt-oss-20b': 'openai/gpt-oss-20b',
    'Yi-1.5-9B-Chat': '01-ai/Yi-1.5-9B-Chat',
    'Yi-1.5-34B-Chat': '01-ai/Yi-1.5-34B-Chat',
    'DeepSeek-R1-Distill-Qwen-14B': 'deepseek-ai/DeepSeek-R1-Distill-Qwen-14B',
    'DeepSeek-R1-Distill-Llama-8B': 'deepseek-ai/DeepSeek-R1-Distill-Llama-8B',
    'DeepSeek-R1-Distill-Llama-70B': 'deepseek-ai/DeepSeek-R1-Distill-Llama-70B',
}


def model_name(ref):
    ref = str(ref)
    if ref.startswith(PLACEHOLDER):
        return ref[len(PLACEHOLDER):].strip('/')
    return ref


def resolve_model(ref):
    ref = str(ref)
    if os.path.isdir(ref):
        return ref
    name = model_name(ref)
    root = os.environ.get('SYNC_MODEL_ROOT')
    if root and (Path(root) / name).is_dir():
        return str(Path(root) / name)
    hf_id = HF_IDS.get(name, name if '/' in name and not ref.startswith(PLACEHOLDER) else None)
    if hf_id is None:
        raise FileNotFoundError(
            f'No Hugging Face ID is known for {name!r}; set SYNC_MODEL_ROOT to a directory containing {name}/.')
    from huggingface_hub import snapshot_download
    return snapshot_download(hf_id)
