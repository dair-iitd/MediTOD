import json
import numpy as np

import re
import argparse
from copy import deepcopy
from utils import ChatGPTPairwiseComparator
from tqdm import tqdm


def read_cli():
    parser = argparse.ArgumentParser(description='Evaluation POL')
    parser.add_argument(
        "-data",
        "--data",
        help="Path to dialog.json",
        required=True,
        type=str,
    )
    parser.add_argument(
        "-pred",
        "--pred",
        help="Path to predictions",
        required=True,
        type=str,
    )
    parser.add_argument(
        "-model",
        "--model",
        help="Model",
        required=True,
        choices=['PPTOD', 'Flan-T5', 'BioGPT', 'LLM'],
        type=str,
    )
    parser.add_argument(
        "-mode",
        "--mode",
        help="mode",
        required=False,
        default='all',
        choices=['canon_only', 'all'],
        type=str,
    )
    args = vars(parser.parse_args())

    return args


def load_json(fname):
    with open(fname, 'r') as fp:
        obj = json.load(fp)
    return obj


def parse_pptod_string(aspan):
    text = aspan.replace('<sos_a>', '')
    text = text.replace('<eos_a>', '')
    text = text.strip()

    pattern = r"@[\-a-z_]+@"
    entry_sidxs = []
    for ee in re.finditer(pattern, text):
        span = ee.span()
        assert span[1] - span[0] > 0
        entry_sidxs.append(span[0])
    entry_sidxs.append(len(text))

    ret_data = []
    for ii in range(len(entry_sidxs) - 1):
        ttext = deepcopy(text[entry_sidxs[ii]:entry_sidxs[ii + 1]])
        _, action, ttext = ttext.split('@', 2)
        action = action.strip()

        ttext = ttext.strip()
        pattern = r"\[[a-z_]+\]"
        sidxs = []
        for ee in re.finditer(pattern, ttext):
            span = ee.span()
            assert span[1] - span[0] > 0
            sidxs.append(span[0])
        sidxs.append(len(ttext))

        action_data = {
            'action': action
        }
        for jj in range(len(sidxs) - 1):
            vspan = ttext[sidxs[jj]:sidxs[jj + 1]]
            key, tmp = vspan.split(']', 1)
            key = key.strip('[').strip()
            tmp = tmp.strip()

            pattern = r"\(.*?\)"
            act_data = []
            for ee in re.finditer(pattern, tmp):
                ss, ee = ee.span()
                vts = tmp[ss:ee].strip('(').strip(')').strip()
                checks = dict()
                kvs = dict()
                for vt in vts.split('#'):
                    kk, vv = vt.split('=')
                    kk = kk.strip()
                    vv = vv.strip()

                    if 'values' in kk:
                        vv = vv.strip().split('|')
                        vv = [z.strip() for z in vv]

                    # kk and vv are here.
                    if 'checks-type' == kk:
                        checks[vv] = {
                            'type': vv
                        }
                    elif '-values' in kk:
                        ctype = kk.split('-', 1)[0].strip()
                        checks[ctype]['values'] = vv
                    else:
                        kvs[kk] = vv
                if len(checks) > 0:
                    kvs['checks'] = [checks[kk] for kk in checks]
                act_data.append(kvs)
            action_data[key] = act_data
        ret_data.append(action_data)

    return ret_data


def get_key_value_pairs_old(actions, ignore_intent=False):
    canon_slots = [
        'symptom',
        'medical_history',
        'family_history',
        'habit',
        'exposure',
        'medical_test',
        'medication',
        'disease'
    ]

    canon_kvs = set()
    noncanon_kvs = set()
    for entry in actions:
        if ignore_intent:
            intent = 'dummy'
        else:
            intent = entry['action']
        akeys = list(entry.keys())
        akeys.remove('action')
        loc_kvs = set()

        for slot in akeys:
            data = entry[slot]
            assert type(data) == list
            for vv in data:
                head = vv.get('value', 'dummy')
                if 'value' in vv:
                    loc_kvs.add((intent, slot, head, 'value', vv['value']))
                for xx in vv.get('checks', []):
                    ctype = xx.get('type', 'dummy')
                    if 'type' in xx:
                        loc_kvs.add((intent, slot, head, 'checks', ctype))
                    for yy in xx.get('values', []):
                        loc_kvs.add((intent, slot, head, f'{ctype}-value', yy))

                # # Ideally, we should not need the following loop as action can only have value and checks.
                # # However, parsing errors may cause some problems without this.
                # for gg in vv:
                #     if gg in ['value', 'checks']:
                #         continue
                #     if type(vv[gg]) == list:
                #         for zz in vv[gg]:
                #             loc_kvs.add((intent, slot, head, gg, zz))
                #     else:
                #         loc_kvs.add((intent, slot, head, gg, vv[gg]))

        if len(loc_kvs) == 0:
            loc_kvs.add((intent,))

        for ee in loc_kvs:
            if len(ee) == 1:
                # Null is canon
                canon_kvs.add(ee)

            elif ee[1] in canon_slots and ee[3] == 'value':
                canon_kvs.add(ee)

            elif ee[1] == 'medication' and ee[3] == 'respone_to':
                canon_kvs.add(ee)

            elif ee[1] in canon_slots and ee[3] == 'checks':
                canon_kvs.add(ee)
                
            else:
                noncanon_kvs.add(ee)

    return canon_kvs, noncanon_kvs


def process_single_entry(entry):
    slots_with_canon_heads = [
        'symptom',
        'habit',
        'medical_history',
        'family_history',
        'medication',
        'medical_test',
        'exposure',
        'disease'
    ]
    slots_without_canon_heads = [
        'occupation',
        'travel',
        'basic_information',
        'residence'
    ]

    intent = entry['action']
    slots = deepcopy(entry)
    del slots['action']

    if len(slots) == 0:
        return {(intent,)}, set()

    canon_kvs = set()
    noncanon_kvs = set()
    for slot, data in slots.items():
        stype = 'medical'
        if slot == 'symptom':
            HEAD_KEY = 'value'
            canon_attrs = [
                'location', 'progression', 'severity', 'lesion_size',
                'rash_swollen', 'itching', 'lesions_peel_off'
            ]

        elif any(x in slot for x in [
            'medical_history', 'family_history', 'habit',
            'medical_test', 'exposure'
        ]):
            HEAD_KEY = 'value'
            canon_attrs = []

        elif 'medication' in slot:
            HEAD_KEY = 'value'
            canon_attrs = ['respone_to', 'impact']

        elif 'disease' in slot:
            HEAD_KEY = 'value'
            canon_attrs = []

        else:
            stype = 'non_medical'

        if stype == 'medical':
            assert type(data) == list
            for ee in data:
                if HEAD_KEY in ee:
                    src = ee[HEAD_KEY]
                    canon_kvs.add((intent, slot, src))
                else:
                    src = 'dummy'

                for ck in ee.get('checks', []):
                    ctype = ck.get('type', 'dummy')
                    is_canon = ctype in canon_attrs

                    if 'type' in ck:
                        # Type is always canon
                        canon_kvs.add((intent, slot, src, ctype))

                    for vv in ck.get('values', []):
                        if is_canon:
                            canon_kvs.add((intent, slot, src, ctype, vv))
                        else:
                            noncanon_kvs.add((intent, slot, src, ctype, vv))
            continue

        if slot in ['occupation', 'residence', 'travel']:
            for ck in data:
                if 'value' in ck:
                    noncanon_kvs.add((intent, slot, ck['value']))

                for ck in ck.get('checks', []):
                    ctype = ck.get('type', 'dummy')
                    is_canon = ctype in ['status']

                    if 'type' in ck:
                        # Type is always canon
                        canon_kvs.add((intent, slot, ctype))

                    for vv in ck.get('values', []):
                        if is_canon:
                            canon_kvs.add((intent, slot, ctype, vv))
                        else:
                            noncanon_kvs.add((intent, slot, ctype, vv))

        elif slot in ['basic_information']:
            for ck in data:
                if 'value' in ck:
                    noncanon_kvs.add((intent, slot, ck['value']))

        else:
            print(entry)
            print('UNKNOWN SLOT', slot)

    return canon_kvs, noncanon_kvs


def get_key_value_pairs(actions, ignore_intent=False):
    canon_kvs = set()
    noncanon_kvs = set()
    # assert type(actions) == list, f"{type(actions)}"
    if type(actions) == dict:
        actions = [actions]
    for entry in actions:
        try:
            can, ncan = process_single_entry(entry)
        except:
            print(entry)
            can, ncan = set(), set()
        canon_kvs = canon_kvs.union(can)
        noncanon_kvs = noncanon_kvs.union(ncan)

    return canon_kvs, noncanon_kvs


def compute_precision_recall_f1(gold_acts, pred_acts, ignore_intent=False, mode='all'):
    CTP, CFP, CFN = 0.0, 0.0, 0.0
    NTP, NFP, NFN = 0.0, 0.0, 0.0
    joint_acc = 0

    total_cnt, noncanon_cnt = 0, 0
    for gacts, pacts in zip(gold_acts, pred_acts):
        if len(gacts) == 0:
            continue
        if len(gacts) == 1 and gacts[0]['action'] == 'inquire' and len(gacts[0]) == 1:
            continue

        gold_canon_kvs, gold_noncanon_kvs = get_key_value_pairs(gacts, ignore_intent)
        pred_canon_kvs, pred_noncanon_kvs = get_key_value_pairs(pacts, ignore_intent)

        total_cnt += 1
        if len(gold_noncanon_kvs) > 0:
            noncanon_cnt += 1

        gg, pp = gold_canon_kvs, pred_canon_kvs
        TP = len(gg.intersection(pp))
        FP = len(pp - gg)
        FN = len(gg - pp)
        CTP += TP
        CFP += FP
        CFN += FN

        if mode == 'canon_only':
            joint_acc += int(gold_canon_kvs == pred_canon_kvs)
            continue

        gg, pp = gold_noncanon_kvs, pred_noncanon_kvs
        TP = len(gg.intersection(pp))
        FP = len(pp - gg)
        FN = len(gg - pp)
        NTP += TP
        NFP += FP
        NFN += FN

        gold_kvs = gold_canon_kvs.union(gold_noncanon_kvs)
        pred_kvs = pred_canon_kvs.union(pred_noncanon_kvs)
        joint_acc += int(gold_kvs == pred_kvs)

    ret = dict()
    num_turns = len(gold_acts)
    joint_acc /= num_turns

    ret['samples'] = num_turns
    ret['joint_accuracy'] = joint_acc

    TP, FP, FN = CTP, CFP, CFN
    prec = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    rec = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    f1 = (2.0 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0
    ret['canon_precision'] = prec
    ret['canon_recall'] = rec
    ret['canon_f1'] = f1
    ret['canon_total'] = TP + FN

    if mode == 'canon_only':
        for kk in ret:
            ret[kk] = round(ret[kk], 4)
        return ret

    TP, FP, FN = NTP, NFP, NFN
    prec = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    rec = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    f1 = (2.0 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0
    ret['noncanon_precision'] = prec
    ret['noncanon_recall'] = rec
    ret['noncanon_f1'] = f1
    ret['noncanon_total'] = TP + FN

    TP, FP, FN = CTP + NTP, CFP + NFP, CFN + NFN
    prec = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    rec = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    f1 = (2.0 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0
    ret['precision'] = prec
    ret['recall'] = rec
    ret['f1'] = f1
    ret['total'] = TP + FN

    for kk in ret:
        ret[kk] = round(ret[kk], 4)

    print('TOTAL', total_cnt)
    print('NONCANON', noncanon_cnt)
    print(noncanon_cnt / total_cnt)
    return ret


def compute_precision_k(gold_actions, pred_actions, K, use_jaccard=False, mode='all'):
    acc = 0
    total = 0

    for gactions, pactions in zip(gold_actions, pred_actions):
        gold_canon_kvs = []
        gold_noncanon_kvs = []
        gold_kvs = []
        for actions in gactions:
            can, ncan = get_key_value_pairs(actions)
            gold_canon_kvs.append(can)
            gold_noncanon_kvs.append(ncan)
            if mode == 'all':
                gold_kvs.append(can.union(ncan))
            elif mode == 'canon_only':
                gold_kvs.append(can)

        for ii in range(len(pactions)):
            can, ncan = get_key_value_pairs(pactions[ii])
            if mode == 'all':
                kvs = can.union(ncan)
            elif mode == 'canon_only':
                kvs = can

            vals = []
            for jj in range(ii, min(ii + K, len(pactions)), 1):
                if use_jaccard:                    
                    numer = len(kvs.intersection(gold_kvs[jj])) * 1.0
                    denom = len(kvs.union(gold_kvs[jj]))
                    vals.append(numer / denom if denom > 0 else 0)

                else:
                    vals.append(int(kvs == gold_kvs[jj]))

            acc += max(vals)
            total += 1

    ret = {
        'accuracy': round(acc / total, 4),
        'total': total,
        'K': K
    }
    return ret



def compute_soft_metrics_chatgpt(gold_acts, pred_acts, ignore_intent=False):
    # 1. Create groups first based on (intent, slot, value, attr)
    # 2. Compute group wise precision. For each predicted group, find the corresponding group in gold.
    # 3. If gold group exists, perform matching similar to BertScore.
    ## If gold group exists, remove the hard matches first. Then perform soft matching by pairing.

    def get_groups(kvs):
        groups = dict()

        for entry in kvs:
            if len(entry) == 1 and entry[0] not in groups:
                print('---->', entry)
                groups[entry[:1]] = {entry[0]}
                continue

            kk = entry[:-1]
            if kk not in groups:
                groups[kk] = set()

            # print(entry)
            groups[kk].add(entry[-1])

        groups  = {k: sorted(v, key=lambda x: (-len(x), x)) for k, v in groups.items()}

        return groups


    scorer = ChatGPTPairwiseComparator()
    logs = []
    total_comparisons = 0
    for gold_act, pred_act in tqdm(zip(gold_acts, pred_acts)):
        if len(gold_act) == 0:
            continue
        if len(gold_act) == 1 and gold_act[0]['action'] == 'inquire' and len(gold_act[0]) == 1:
            continue

        gold_canon_kvs, gold_noncanon_kvs = get_key_value_pairs(gold_act, ignore_intent)
        pred_canon_kvs, pred_noncanon_kvs = get_key_value_pairs(pred_act, ignore_intent)

        log = {
            'canon': {
                'TP': len(gold_canon_kvs.intersection(pred_canon_kvs)),
                'FP': len(pred_canon_kvs - gold_canon_kvs),
                'FN': len(gold_canon_kvs - pred_canon_kvs),
                'gold_kvs': list(gold_canon_kvs),
                'pred_kvs': list(pred_canon_kvs),
            },
            'noncanon': {
                'TP': 0, 'FP': 0, 'FN': 0,
                'gold_kvs': list(gold_noncanon_kvs),
                'pred_kvs': list(pred_noncanon_kvs),
                'chatgpt_logs': []
            }
        }


        ggroups = get_groups(gold_noncanon_kvs)
        pgroups = get_groups(pred_noncanon_kvs)

        for gid in ggroups:
            if gid not in pgroups:
                log['noncanon']['FN'] += len(ggroups[gid])
                continue

            scores = scorer.get_scores(ggroups[gid], pgroups[gid])
            # scores = np.zeros((len(ggroups[gid]), len(pgroups[gid])))
            total_comparisons += len(ggroups[gid]) * len(pgroups[gid])
            tscores = deepcopy(scores)
            log['noncanon']['chatgpt_logs'].append((gid, ggroups[gid], pgroups[gid], [[int(x) for x in y] for y in scores]))

            matched_preds = []
            for ii in range(len(ggroups[gid])):
                matched = False
                for jj in range(len(pgroups[gid])):
                    if tscores[ii, jj] > 0 and jj not in matched_preds:
                        matched = True
                        break

                if matched:
                    # Only one match per gold.
                    log['noncanon']['TP'] += 1
                    matched_preds.append(jj)
                else:
                    log['noncanon']['FN'] += 1
            log['noncanon']['FP'] += len(pgroups[gid]) - len(matched_preds)

        for gid in pgroups:
            if gid not in ggroups:
                log['noncanon']['FP'] += len(pgroups[gid])

        logs.append(log)

    OTP, OFP, OFN = 0, 0, 0
    for tag in ['canon', 'noncanon']:
        TP, FP, FN = 0, 0, 0
        for log in logs:        
            TP += log[tag]['TP']
            FP += log[tag]['FP']
            FN += log[tag]['FN']

        prec = TP / (TP + FP) if (TP + FP) > 0 else 0
        rec = TP / (TP + FN) if (TP + FN) > 0 else 0
        f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0

        print(f'{tag}_precision', round(prec, 4))
        print(f'{tag}_recall', round(rec, 4))
        print(f'{tag}_f1', round(f1, 4))
        print(f'{tag}_total', TP + FN)

        OTP += TP
        OFP += FP
        OFN += FN

    prec = OTP / (OTP + OFP) if (OTP + OFP) > 0 else 0
    rec = OTP / (OTP + OFN) if (OTP + OFP) > 0 else 0
    f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0
    print(f'precision', round(prec, 4))
    print(f'recall', round(rec, 4))
    print(f'f1', round(f1, 4))
    print(f'Total Comparisons', total_comparisons)

    return logs


if __name__ == '__main__':
    args = read_cli()

    data = load_json(args['data'])

    gold_pols = []
    pred_pols = []
    ordered_dids_uids = []

    if args['model'] == 'PPTOD':
        pred_data = load_json(args['pred'])

        dids = list(pred_data)
        for ii, did in enumerate(dids):
            loc_gold = []
            loc_pred = []

            tdid = did[4:]
            dlg = data[tdid]
            uids = sorted(pred_data[did], key=lambda x: int(x))
            pids = [
                jj for jj in range(len(dlg['utterances']))
                if dlg['utterances'][jj]['speaker'] == 'doctor' and jj != 0
            ]

            for jj, uid in enumerate(uids):
                ordered_dids_uids.append((did, pids[jj]))
                gold = dlg['utterances'][pids[jj]]['actions']
                loc_gold.append(gold)

                pred = pred_data[did][uid]['aspn_gen']
                pred = parse_pptod_string(pred)
                loc_pred.append(pred)

            gold_pols.append(loc_gold)
            pred_pols.append(loc_pred)

    elif args['model'] in ['Flan-T5', 'BioGPT']:
        pred_data = load_json(args['pred'])

        did2samples = dict()
        for entry in pred_data:
            did = entry['dial_id'][4:]
            if did not in did2samples:
                did2samples[did] = dict()
            did2samples[did][entry['turn_num']] = entry

        dids = list(did2samples)
        for ii, did in enumerate(dids):
            loc_gold = []
            loc_pred = []

            dlg = data[did]
            uids = sorted(did2samples[did], key=lambda x: int(x))
            pids = [
                jj for jj in range(len(dlg['utterances']))
                if dlg['utterances'][jj]['speaker'] == 'doctor' and jj != 0
            ]

            for jj, uid in enumerate(uids):
                ordered_dids_uids.append((did, pids[jj]))
                gold = dlg['utterances'][pids[jj]]['actions']
                loc_gold.append(gold)

                pred = did2samples[did][uid]['prediction']
                pred = parse_pptod_string(pred)
                loc_pred.append(pred)

            gold_pols.append(loc_gold)
            pred_pols.append(loc_pred)

    elif args['model'] in ['LLM']:
        pred_data = load_json(args['pred'])[:100]

        did2samples = dict()
        for entry in pred_data:
            did = entry['did']
            if did not in did2samples:
                did2samples[did] = dict()
            did2samples[did][entry['uid']] = entry

        dids = list(did2samples)
        for ii, did in enumerate(dids):
            loc_gold = []
            loc_pred = []

            dlg = data[did]
            uids = sorted(did2samples[did], key=lambda x: int(x))
            pids = [
                jj for jj in range(len(dlg['utterances']))
                if dlg['utterances'][jj]['speaker'] == 'doctor' and jj != 0
            ]
            for jj, uid in enumerate(uids):
                ordered_dids_uids.append((did, pids[jj]))
                gold = dlg['utterances'][pids[jj]]['actions']
                loc_gold.append(gold)

                assert did2samples[did][uid]['nlg'] == dlg['utterances'][pids[jj]]['text']

                pred = did2samples[did][uid]['prediction']
                loc_pred.append(pred)

            gold_pols.append(loc_gold)
            pred_pols.append(loc_pred)

    allg = [x for y in gold_pols for x in y]
    allp = [x for y in pred_pols for x in y]

    ret = compute_precision_recall_f1(allg, allp, mode=args['mode'])
    for k in [1, 2, 4, 6, 8, 10, 1000000]:
        ret2 = compute_precision_k(gold_pols, pred_pols, K=k, mode=args['mode'])
        ret[f'precision@{k}'] = ret2['accuracy']
    print(json.dumps(ret, indent=2))
    compute_soft_metrics_chatgpt(allg, allp, ignore_intent=False)
