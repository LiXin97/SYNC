"""Batch-one counterfactual runner. Frozen before confirmation is allowed.

Reuses the previous writer/solver via an immutable source snapshot. New task,
chat conventions, answer parser and calibration are explicitly separate from
the historical experiments. No existing runner or result is modified.
"""
try:
    from syncomm.models import resolve_model
except ImportError:  # run outside the package: references must already be local paths
    def resolve_model(ref):
        return str(ref)
from pathlib import Path
import argparse, contextlib, copy, gc, hashlib, json, os, sys, time
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

ROOT = next(p for p in Path(__file__).resolve().parents if (p / 'README.md').exists())
OLD = ROOT / 'artifacts/reviewbinding0917/snapshot/src'
sys.path.insert(0, str(OLD))
from syncomm import run_demo as rd
from syncomm.e015.core import decoder_layers, tap_layer
from syncomm.packet import (Packet, Lane, serialize, deserialize, inventory_digest,
                          dequantize)
from syncomm.run_j0c import mean_h, LITERAL
from syncomm.writer import SlotWrite, _solve_a0
from syncomm import solver


def sha(data):
    return hashlib.sha256(data).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(value, sort_keys=True, indent=2) + '\n'
    if path.exists():
        assert json.loads(path.read_text()) == value, f'resume drift: {path}'
    else:
        tmp = path.with_suffix(path.suffix + '.tmp')
        tmp.write_text(text)
        os.replace(tmp, path)


def load_model(path, dtype=torch.float32):
    tok = AutoTokenizer.from_pretrained(resolve_model(path))
    config = json.loads((Path(path) / 'config.json').read_text())
    cls = AutoModelForCausalLM
    if config['model_type'] == 'qwen3_5':
        from transformers import AutoModelForImageTextToText
        cls = AutoModelForImageTextToText
    model = cls.from_pretrained(resolve_model(path), dtype=dtype, device_map='cuda:0',
                               attn_implementation='eager').eval()
    model.requires_grad_(False)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = 'left'
    return model, tok


def dictionary(path):
    blob = torch.load(path, map_location='cpu', weights_only=False)
    return blob['dirs'].float().cuda(), blob['words']


def extract(m, args, bank):
    model, tok = load_model(m['models'][args.sender], dtype=torch.bfloat16)
    D, words = dictionary(m['dictionaries'][args.sender])
    mid = tap_layer(model, .5)
    result = {}
    cache = {}
    dest = Path(m['root']) / 'packets' / f'{args.split}_{args.sender}.json'
    for s in bank:
        for w in s['worlds']:
            lanes = []
            for i, evidence in enumerate(w['evidence']):
                if evidence not in cache:
                    with torch.inference_mode():
                        h = mean_h(model, tok, evidence + '\n', mid, 'cuda:0')
                        a = _solve_a0(h.float().unsqueeze(0), D, .05, .001)[0]
                    cache[evidence] = rd.quantize_topk(a.cpu(), topk=4, qbits=8)
                ids, weights = cache[evidence]
                lanes.append(Lane(i, ids, weights))
            # Empty selections remain in the denominator and yield no write.
            packet = Packet(inventory_digest(words), 8, tuple(lanes))
            raw = serialize(packet)
            assert serialize(deserialize(raw)) == raw
            result[f"{s['scene']}|{w['world']}"] = dict(hex=raw.hex(), sha256=sha(raw),
                bytes=len(raw), evidence_sha256=sha(json.dumps(w['evidence']).encode()),
                empty_lanes=sum(not ln.concept_ids for ln in packet.lanes))
        print('EXTRACT', args.sender, s['scene'], flush=True)
    write_json(dest, dict(sender=args.sender, split=args.split, packets=result,
        dictionary_sha256=sha(Path(m['dictionaries'][args.sender]).read_bytes()),
        words=words, unique_evidence_inputs=len(cache)))


class ObservedWrite(SlotWrite):
    """Observe the existing operation without changing its floating-point path."""
    def __init__(self, *args, report, span, **kwargs):
        self.report, self.span = report, span
        super().__init__(*args, **kwargs)

    def _hk(self, mod, args, out):
        h = out[0] if isinstance(out, tuple) else out
        if h.shape[1] == 1:
            return super()._hk(mod, args, out)
        positions = self.positions[0]
        before = h[0, positions].detach().float().clone()
        result = super()._hk(mod, args, out)
        after = result[0] if isinstance(result, tuple) else result
        delta = after[0, positions].detach().float() - before
        for j, pos in enumerate(positions):
            self.report[f'{self.li}|{self.span}|{pos}'] = dict(
                actual=float(delta[j].norm()), before=float(before[j].norm()),
                computed=(self.applied_norms[j] if len(self.applied_norms) > j else 0.))
        return result


class Receiver:
    def __init__(self, m, key, task):
        self.m, self.key, self.task = m, key, task
        self.model, self.tok = load_model(m['models'][key])
        self.D, self.words = dictionary(m['dictionaries'][key])
        self.band = sorted(set(tap_layer(self.model, t) for t in (.4, .5, .6)))
        self.chat = key not in ('qwen3-14b-base',)
        self.cache = {}
        self.mag = self.calibrate_magnitude()

    def calibrate_magnitude(self):
        # Fixed construction/calibration entities, unrelated to the new outcomes.
        mags = []
        mid = tap_layer(self.model, .5)
        for word in self.m['magnitude_words']:
            with torch.inference_mode():
                h = mean_h(self.model, self.tok, LITERAL.format(e=word), mid, 'cuda:0')
            a = solver.extract(h.cpu(), self.D.cpu(), lam1=.05, lam2=.001).a
            if float(a.max()) > 1e-6:
                mags.append(float(a.max()))
        assert mags
        return float(torch.tensor(mags).median())

    def prompt(self, scene, qi, extra):
        text, carriers = self.task.receiver_body(scene, qi, extra)
        if self.chat:
            text = self.tok.apply_chat_template([{'role':'user', 'content':text}],
                tokenize=False, add_generation_prompt=True, enable_thinking=False)
        enc = self.tok(text, return_offsets_mapping=True, add_special_tokens=not self.chat)
        positions = []
        for carrier in carriers:
            start = text.rfind(carrier)
            assert start >= 0
            positions.append([i for i, (a, b) in enumerate(enc['offset_mapping'])
                              if a < start + len(carrier) and b > start])
        assert all(positions)
        return text, positions

    @contextlib.contextmanager
    def hooks(self, positions, packet, gamma, writer, zero, report):
        hooks = []
        band = self.band if writer == 'band' else self.band[:1]
        # gamma=0 is the true no-write candidate, not target-coefficient zeroing.
        if zero:
            for li in band:
                for i, pos in enumerate(positions):
                    hooks.append(ObservedWrite(self.model, li, [pos], 'add',
                        vec=torch.zeros(self.D.shape[1], device='cuda:0'),
                        report=report, span=i))
        elif packet is not None and gamma > 0:
            for li in band:
                for ln in packet.lanes:
                    if not ln.concept_ids:
                        continue
                    q = dequantize(ln.q, packet.qbits).cuda()
                    weights = q * (self.mag * gamma / q.sum()) if q.sum() > 0 else q
                    hooks.append(ObservedWrite(self.model, li, [positions[ln.slot_id]],
                        'replace', D=self.D, support=list(ln.concept_ids), a_hat=weights,
                        lam1=.05, lam2=.001, report=report, span=ln.slot_id))
        try:
            yield hooks
        finally:
            for h in hooks:
                h.remove()

    def evaluate(self, scene, qi, packet=None, gamma=0, writer='band', extra=None,
                 zero=False, replay=False):
        prompt, positions = self.prompt(scene, qi, extra)
        payload = serialize(packet).hex() if packet is not None else ''
        key = (prompt, payload, gamma, writer, zero)
        if key in self.cache:
            return copy.deepcopy(self.cache[key])
        enc = self.tok(prompt, return_tensors='pt', add_special_tokens=not self.chat).to('cuda:0')
        ids = enc['input_ids']
        report = {}
        with torch.inference_mode(), self.hooks(positions, packet, gamma, writer, zero, report) as hooks:
            out = self.model(**enc, use_cache=True)
        logits = out.logits[0, -1].float()
        cache = out.past_key_values
        hook_count = sum(h.n_writes for h in hooks)
        logprobs = torch.log_softmax(logits, dim=-1)
        alias_ids = [self.tok(' ' + a, add_special_tokens=False)['input_ids'] for a in scene['aliases']]
        pieces = []
        with torch.inference_mode():
            for tokens in alias_ids:
                lp = [float(logprobs[tokens[0]])]
                # Use one-token cached continuations for all model families.
                branch = copy.deepcopy(cache)
                for prev, target in zip(tokens[:-1], tokens[1:]):
                    step = self.model(torch.tensor([[prev]], device='cuda:0'),
                        past_key_values=branch, use_cache=True)
                    branch = step.past_key_values
                    lp.append(float(torch.log_softmax(step.logits[0, -1].float(), -1)[target]))
                pieces.append(lp)
        generated = []
        branch = cache
        current = logits
        eos = self.model.generation_config.eos_token_id
        eos = {eos} if isinstance(eos, int) else set(eos or [self.tok.eos_token_id])
        stop = 'budget'
        with torch.inference_mode():
            for _ in range(256):
                token = int(current.argmax())
                generated.append(token)
                text = self.tok.decode(generated, skip_special_tokens=True)
                if token in eos:
                    stop = 'eos'
                    break
                if '\n' in text.lstrip():
                    stop = 'line'
                    break
                step = self.model(torch.tensor([[token]], device='cuda:0'),
                    past_key_values=branch, use_cache=True)
                branch, current = step.past_key_values, step.logits[0, -1].float()
        # A single deterministic continuation gives both prefix budgets; there
        # is no outcome-based choice of whether to run the longer sensitivity.
        primary_text = self.tok.decode(generated[:64], skip_special_tokens=True)
        long_text = self.tok.decode(generated, skip_special_tokens=True)
        check = None
        if replay:
            # Second prefill must be exact and zero hooks must not alter logits.
            r2 = {}
            with torch.inference_mode(), self.hooks(positions, packet, gamma, writer, zero, r2):
                logits2 = self.model(**enc, use_cache=True).logits[0, -1].float()
            assert torch.equal(logits, logits2), 'prefill replay drift'
            # Complete-prompt scoring has different tensor shapes; record its
            # actual discrepancy, never silently pool two numerical regimes.
            full = []
            for tokens in alias_ids:
                prefix = torch.cat([ids, torch.tensor([tokens], device='cuda:0')], 1)
                with torch.inference_mode(), self.hooks(positions, packet, gamma, writer, zero, {}):
                    x = self.model(prefix, use_cache=False).logits[0, ids.shape[1]-1:ids.shape[1]-1+len(tokens)].float()
                lp = torch.log_softmax(x, -1)
                full.append(sum(float(lp[i, t]) for i, t in enumerate(tokens)))
            cached_gap = sum(pieces[0]) - sum(pieces[1])
            check = dict(prefix_exact=True, cached_gap=cached_gap, full_gap=full[0]-full[1],
                difference=abs(cached_gap - (full[0]-full[1])))
            assert check['difference'] <= .5, 'full-sequence cross-check failed'
        result = dict(prompt_sha256=sha(prompt.encode()), payload_sha256=sha(bytes.fromhex(payload)),
            payload_bytes=len(bytes.fromhex(payload)), prompt_tokens=ids.shape[1],
            gap=sum(pieces[0])-sum(pieces[1]), first_gap=float(logits[alias_ids[0][0]]-logits[alias_ids[1][0]]),
            alias_token_ids=alias_ids, alias_logprob_pieces=pieces,
            text64=primary_text, text256=long_text, tokens=generated,
            truncated64=len(generated)>64 or (len(generated)==64 and stop=='budget'),
            stop=stop, sites=report, hook_writes=hook_count, replay=check,
            logits_sha256=sha(logits.cpu().numpy().tobytes()),
            text_bytes=len(extra.encode('utf8')) if extra else 0)
        self.cache[key] = copy.deepcopy(result)
        return result


def packet_text(packet, words, aliases, weights=False):
    parts = []
    for ln in packet.lanes:
        items = [words[i] + (f' [{q}/255]' if weights else '')
                 for i, q in zip(ln.concept_ids, ln.q)]
        parts.append(aliases[ln.slot_id] + ': ' + ', '.join(items))
    prefix = 'Selected words and exact relative weights (integers divided by 255): ' if weights else 'Selected words: '
    return prefix + '; '.join(parts) + '.'


def evaluate(m, args, bank, task):
    pair = m['pairs'][args.pair]
    packets = json.loads((Path(m['root'])/'packets'/f"{args.split}_{pair['sender']}.json").read_text())
    selected = bank[args.shard::args.shards]
    receiver = Receiver(m, pair['receiver'], task)
    assert receiver.words == packets['words']
    out = Path(m['root'])/'raw'/f"{args.split}_p{args.pair}_s{args.shard}_{args.mode}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    done = {}
    if out.exists():
        for line in out.open():
            r = json.loads(line); assert r['key'] not in done; done[r['key']] = r
    if args.mode == 'smoke':
        selected = selected[:2]
        configs = [('none',0,'band'),('zero',0,'band'),('oracle',0,'band'),
                   ('selected',0,'band'),('exact_text',0,'band'),('write',4,'band'),('write',4,'single')]
    elif args.split == 'dev':
        configs = [('none',0,'band'),('zero',0,'band'),('oracle',0,'band'),
                   ('selected',0,'band'),('exact_text',0,'band')]
        configs += [('write',g,w) for w in ['band','single'] for g in [1,2,4,8,16]]
    else:
        raise RuntimeError('Confirmation runner requires the separately frozen selection and control plan')
    for si, scene in enumerate(selected):
        for world in scene['worlds']:
            rec = packets['packets'][f"{scene['scene']}|{world['world']}"]
            packet = deserialize(bytes.fromhex(rec['hex']))
            for qi in (0, 1):
                gold = world['answers'][qi]
                for arm,gamma,writer in configs:
                    key = '|'.join(map(str,[scene['scene'],world['world'],qi,arm,gamma,writer]))
                    if key in done:
                        continue
                    extra = (' '.join(world['evidence']) if arm=='oracle' else
                        packet_text(packet, receiver.words, scene['aliases'],arm=='exact_text')
                        if arm in ['selected','exact_text'] else None)
                    result = receiver.evaluate(scene, qi, packet if arm=='write' else None,
                        gamma, writer, extra, zero=arm=='zero', replay=(args.mode=='smoke' and si==0))
                    if arm=='zero':
                        refkey = '|'.join(map(str,[scene['scene'],world['world'],qi,'none',0,'band']))
                        ref=done[refkey]
                        assert result['logits_sha256']==ref['logits_sha256'] and result['tokens']==ref['tokens']
                    row = dict(key=key, pair=args.pair, scene=scene['scene'], family=scene['family'],
                        split=args.split, domain=scene['domain'], template=scene['template'],
                        world=world['world'], question=qi, aliases=scene['aliases'], gold=gold,
                        arm=arm, gamma=gamma, writer=writer, magnitude=receiver.mag,
                        sender_packet_sha256=rec['sha256'], sender_payload_bytes=rec['bytes'],
                        empty_lanes=rec['empty_lanes'], **result)
                    row['score64']=task.score_answer(row['text64'],scene['aliases'],gold)
                    row['score256']=task.score_answer(row['text256'],scene['aliases'],gold)
                    with out.open('a') as f:
                        f.write(json.dumps(row)+'\n')
                    done[key]=row
        print('EVAL', args.pair,args.shard,args.mode,si+1,'/',len(selected),scene['scene'],flush=True)
    expected=len(selected)*4*len(configs)
    assert len(done)==expected,(len(done),expected)
    write_json(out.with_suffix('.done.json'),dict(status='PASS',rows=len(done),scenes=len(selected),
        pair=args.pair,shard=args.shard,mode=args.mode,split=args.split,mag=receiver.mag,
        sender_packets_sha256=sha((Path(m['root'])/'packets'/f"{args.split}_{pair['sender']}.json").read_bytes())))


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--manifest',type=Path,required=True)
    ap.add_argument('--operation',choices=['extract','evaluate'],required=True)
    ap.add_argument('--sender');ap.add_argument('--pair',type=int,default=0)
    ap.add_argument('--split',choices=['dev','eval'],default='dev')
    ap.add_argument('--mode',choices=['smoke','full'],default='smoke')
    ap.add_argument('--shard',type=int,default=0);ap.add_argument('--shards',type=int,default=1)
    args=ap.parse_args();m=json.loads(args.manifest.read_text())
    for path,digest in m['pins'].items():
        assert sha(Path(path).read_bytes())==digest,path
    sys.path.insert(0,m['snapshot'])
    import content_counterfactual as task
    bank=[s for s in json.loads(Path(m['bank']).read_text()) if s['split']==args.split]
    if args.split == 'eval':
        raise RuntimeError('Development lock: confirmation awaits a separately frozen selection')
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = args.operation != 'extract'
    if args.operation=='extract':extract(m,args,bank)
    else:evaluate(m,args,bank,task)


if __name__=='__main__':main()
