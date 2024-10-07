import json
import numpy as np

import re
import argparse
from copy import deepcopy
from tqdm import tqdm

from utils import PairwiseScorer, ChatGPTPairwiseComparator


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
        default='all'
        choices=['canon_only', 'all'],
        type=str,
    )
    args = vars(parser.parse_args())

    return args


def load_json(fname):
    with open(fname, 'r') as fp:
        obj = json.load(fp)
    return obj


def parse_pptod_string(bspan):
    list_fields = {
        'aggravating_factor',
        'alleviating_factor',
        'location',
        'negative_characteristics',
        'not_aggravating_factor',
        'not_alleviating_aggravating_factor',
        'not_alleviating_factor',
        'positive_characteristics',
        'unknown_ana_factor',
        'unknown_characteristics',
        'unknown_factor'
    }

    text = bspan.replace('<sos_b>', '')
    text = text.replace('<eos_b>', '')
    text = text.strip()

    # pattern = r"\[[a-z_]+\]\s*\[[a-z_]+\]"
    pattern = r"@[\-a-z_]+@"
    entry_sidxs = []
    for ee in re.finditer(pattern, text):
        span = ee.span()
        assert span[1] - span[0] > 0
        entry_sidxs.append(span[0])
    entry_sidxs.append(len(text))

    intent_data = []
    for ii in range(len(entry_sidxs) - 1):
        ttext = deepcopy(text[entry_sidxs[ii]:entry_sidxs[ii + 1]])
        _, intent, ttext = ttext.split('@', 2)
        intent = intent.strip()

        ttext = ttext.strip()
        pattern = r"\[[a-z_]+\]"
        sidxs = []
        for ee in re.finditer(pattern, ttext):
            span = ee.span()
            assert span[1] - span[0] > 0
            sidxs.append(span[0])
        sidxs.append(len(ttext))

        slot_data = dict()
        for jj in range(len(sidxs) - 1):
            vspan = ttext[sidxs[jj]:sidxs[jj + 1]]
            key, tmp = vspan.split(']', 1)
            key = key.strip('[').strip()
            tmp = tmp.strip()

            pattern = r"\(.*?\)"
            kvs_arr = []
            for ee in re.finditer(pattern, tmp):
                ss, ee = ee.span()
                vts = tmp[ss:ee].strip('(').strip(')').strip()
                kvs = dict()
                for vt in vts.split('#'):
                    kk, vv = vt.split('=')
                    kk = kk.strip()
                    vv = vv.strip()
                    if kk in list_fields:
                        vv = vv.strip().split('|')
                        vv = [z.strip() for z in vv]
                    kvs[kk.strip()] = vv
                kvs_arr.append(kvs)

            slot_data[key] = kvs_arr

        intent_data.append({
            'intent': intent,
            'slots': slot_data
        })

    return intent_data


def get_key_value_pairs_old(nlu, ignore_intent=False):
    # 1. Standalone intents
    # 2. Canonicalised values
    # 3. Non-canonicalized values

    # Head field here is value
    canon_slots = [
        'negative_exposure',
        'negative_family_history',
        'negative_habit',
        'negative_medical_history',
        'negative_medication',
        'negative_symptom',

        'positive_exposure',
        'positive_family_history',
        'positive_habit',
        'positive_medical_history',
        'positive_medication',
        'positive_symptom',

        'unknown_exposure',
        'unknown_family_history',
        'unknown_habit',
        'unknown_medical_history',
        'unknown_symptom',

        'avail_medical_test',
        'unavail_medical_test',
    ]

    canon_kvs = []
    noncanon_kvs = []
    for entry in nlu:
        intent = entry['intent']
        slots = entry.get('slots', dict())

        if len(slots) == 0:
            canon_kvs.append((intent,))
            continue

        for slot, data in slots.items():
            if slot in ['occupation', 'residence']:
                HEAD_FIELD = 'value'
            elif slot in ['travel']:
                HEAD_FIELD = 'destination'
            elif slot in ['basic_information']:
                HEAD_FIELD = 'name'
            elif slot in canon_slots:
                HEAD_FIELD = 'value'
            else:
                print('Error....', slot)

            data = deepcopy(data)
            if type(data) == dict:
                data = [data]

            for ee in data:
                assert type(ee) == dict
                prm = ee.get(HEAD_FIELD, 'dummy')
                for kk, vv in ee.items():
                    if type(vv) == list:
                        vlist = vv
                    else:
                        assert type(vv) == str
                        vlist = [vv]

                    if (
                        (kk == HEAD_FIELD and slot in canon_slots) or
                        (slot in ['positive_medication', 'negative_medication', 'unknown_medication'] and kk == 'respone_to')
                    ):
                        canon_kvs.extend([(intent, slot, prm, kk, tvv) for tvv in vlist])

                    else:
                        # Everything else is non-canon
                        noncanon_kvs.extend([(intent, slot, prm, kk, tvv) for tvv in vlist])

    return set(canon_kvs), set(noncanon_kvs)


def process_single_entry(entry):
    slots_with_canon_heads = [
        'symptom',
        'habit',
        'medical_history',
        'family_history',
        'medication',
        'medical_test',
        'exposure',
        # 'disease'
    ]
    slots_without_canon_heads = [
        'occupation',
        'travel',
        'basic_information',
        'residence'
    ]

    intent = entry['intent']
    slots = entry.get('slots', dict())

    if len(slots) == 0:
        return {(intent,)}, set()

    canon_kvs = set()
    noncanon_kvs = set()
    for slot, data in slots.items():
        stype = 'medical'
        if '_symptom' in slot:
            HEAD_KEY = 'value'
            canon_attrs = [
                'location', 'progression', 'severity', 'lesion_size',
                'rash_swollen', 'itching', 'lesions_peel_off'
            ]

        elif any(x in slot for x in [
            '_medical_history', '_family_history', '_habit',
            '_medical_test', '_exposure'
        ]):
            HEAD_KEY = 'value'
            canon_attrs = []

        elif '_medication' in slot:
            HEAD_KEY = 'value'
            canon_attrs = ['respone_to', 'impact']

        else:
            stype = 'non_medical'

        if stype == 'medical':
            for ee in data:
                if HEAD_KEY in ee:
                    src = ee[HEAD_KEY]
                    canon_kvs.add((intent, slot, src))
                else:
                    src = 'dummy'

                for key in ee:
                    if key == HEAD_KEY:
                        continue

                    vlist = ee[key] if type(ee[key]) == list else [ee[key]]
                    if key in canon_attrs:
                        # canon
                        for vv in vlist:
                            canon_kvs.add((intent, slot, src, key, vv))
                    else:
                        # non canon
                        for vv in vlist:
                            noncanon_kvs.add((intent, slot, src, key, vv))
            continue

        if slot in ['occupation', 'residence', 'travel']:
            for ee in data:
                for key in ee:
                    vlist = ee[key] if type(ee[key]) == list else [ee[key]]
                    if key in ['status']:
                        # canon
                        for vv in vlist:
                            canon_kvs.add((intent, slot, key, vv))
                    else:
                        # non canon
                        for vv in vlist:
                            noncanon_kvs.add((intent, slot, key, vv))

        elif slot in ['basic_information']:
            for ee in data:
                for key in ee:
                    vlist = ee[key] if type(ee[key]) == list else [ee[key]]
                    for vv in vlist:
                        canon_kvs.add((intent, slot, key, vv))

        else:
            print('UNKNOWN SLOT', slot)

    return canon_kvs, noncanon_kvs


def get_key_value_pairs(nlu, ignore_intent=False):
    canon_kvs = set()
    noncanon_kvs = set()
    for entry in nlu:
        can, ncan = process_single_entry(entry)
        canon_kvs = canon_kvs.union(can)
        noncanon_kvs = noncanon_kvs.union(ncan)

    return canon_kvs, noncanon_kvs


def compute_precision_recall_f1(gold_nlus, pred_nlus, ignore_intent=False, mode='all'):
    CTP, CFP, CFN = 0.0, 0.0, 0.0
    NTP, NFP, NFN = 0.0, 0.0, 0.0
    joint_acc = 0
    format_errs = 0
    for gacts, pacts in zip(gold_nlus, pred_nlus):
        if len(gacts) == 0:
            continue
        if len(gacts) == 1 and gacts[0]['intent'] == 'inform' and len(gacts[0].get('slots', dict())) == 0:
            continue

        gold_canon_kvs, gold_noncanon_kvs = get_key_value_pairs(gacts, ignore_intent)
        try:
            pred_canon_kvs, pred_noncanon_kvs = get_key_value_pairs(pacts, ignore_intent)
        except:
            format_errs += 1
            pred_canon_kvs, pred_noncanon_kvs = set(), set()

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
    num_turns = len(gold_nlus)
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

    TP, FP, FN = CTP + NTP, CFP + NFP, CFN + NFN
    prec = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    rec = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    f1 = (2.0 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0
    ret['precision'] = prec
    ret['recall'] = rec
    ret['f1'] = f1
    ret['total'] = TP + FN

    ret['format_errors'] = format_errs / len(gold_nlus)

    for kk in ret:
        ret[kk] = round(ret[kk], 4)

    return ret


def compute_soft_metrics(gold_nlus, pred_nlus, ignore_intent=False):
    # 1. Create groups first based on (intent, slot, value, attr)
    # 2. Compute group wise precision. For each predicted group, find the corresponding group in gold.
    # 3. If gold group exists, perform matching similar to BertScore.
    ## If gold group exists, remove the hard matches first. Then perform soft matching by pairing.

    def get_groups(kvs):
        groups = dict()

        for entry in kvs:
            if len(entry) == 1 and entry[0] not in groups:
                groups[entry[:1]] = {entry[0]}
                continue

            kk = entry[:-1]
            if kk not in groups:
                groups[kk] = set()

            # print(entry)
            groups[kk].add(entry[-1])

        groups  = {k: sorted(v) for k, v in groups.items()}

        return groups

    precision, recall, f1 = 0.0, 0.0, 0.0
    all_cnt = 0
    scorer = PairwiseScorer()
    total_pairwise_cnt = 0
    format_errs = 0

    for gold_nlu, pred_nlu in tqdm(zip(gold_nlus, pred_nlus)):
        gold_canon_kvs, gold_noncanon_kvs = get_key_value_pairs(gold_nlu, ignore_intent)
        try:
            pred_canon_kvs, pred_noncanon_kvs = get_key_value_pairs(pred_nlu, ignore_intent)
        except:
            format_errs += 1
            pred_canon_kvs, pred_noncanon_kvs = set(), set()

        # pred_canon_kvs, pred_noncanon_kvs = get_key_value_pairs(pred_nlu, ignore_intent)

        gold_kvs = gold_canon_kvs.union(gold_noncanon_kvs)
        pred_kvs = pred_canon_kvs.union(pred_noncanon_kvs)

        ggroups = get_groups(gold_kvs)
        pgroups = get_groups(pred_kvs)

        # recall
        rtotal, ptotal = 0.0, 0.0
        rcnt, pcnt = 0.0, 0.0
        for gid in ggroups:
            rcnt += len(ggroups[gid])

            if gid not in pgroups:
                continue

            scores = scorer.get_scores(ggroups[gid], pgroups[gid])
            rtotal += np.sum(np.max(scores, axis=1))
            ptotal += np.sum(np.max(scores, axis=0))
            total_pairwise_cnt += len(ggroups[gid]) * len(pgroups[gid])

        for gid in pgroups:
            pcnt += len(pgroups[gid])

        all_cnt += 1

        prec = ptotal / pcnt if pcnt > 0 else 0
        rec = rtotal / rcnt if rcnt > 0 else 0
        ff1 = (2.0 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0

        precision += prec
        recall += rec
        f1 += ff1

    print(precision / all_cnt)
    print(recall / all_cnt)
    print(f1 / all_cnt)
    print(total_pairwise_cnt)


def compute_soft_metrics_chatgpt(gold_nlus, pred_nlus, ignore_intent=False):
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
    format_errs = 0
    for gold_nlu, pred_nlu in tqdm(zip(gold_nlus, pred_nlus)):
        if len(gold_nlu) == 0:
            continue
        if len(gold_nlu) == 1 and gold_nlu[0]['intent'] == 'inform' and len(gold_nlu[0].get('slots', dict())) == 0:
            continue

        gold_canon_kvs, gold_noncanon_kvs = get_key_value_pairs(gold_nlu, ignore_intent)
        # pred_canon_kvs, pred_noncanon_kvs = get_key_value_pairs(pred_nlu, ignore_intent)
        try:
            pred_canon_kvs, pred_noncanon_kvs = get_key_value_pairs(pred_nlu, ignore_intent)
        except:
            format_errs += 1
            pred_canon_kvs, pred_noncanon_kvs = set(), set()


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
    print(f'Format errors', format_errs / len(gold_nlus))

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
                if dlg['utterances'][jj]['speaker'] == 'patient' # and jj != 0
            ]

            for jj, uid in enumerate(uids):
                ordered_dids_uids.append((did, pids[jj]))
                gold = dlg['utterances'][pids[jj]]['nlu']
                loc_gold.append(gold)

                pred = pred_data[did][uid]['bspn_nlu_gen']
                try:
                    pred = parse_pptod_string(pred)
                except:
                    pred = []
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
                if dlg['utterances'][jj]['speaker'] == 'patient' # and jj != 0
            ]

            for jj, uid in enumerate(uids):
                ordered_dids_uids.append((did, pids[jj]))
                gold = dlg['utterances'][pids[jj]]['nlu']
                loc_gold.append(gold)

                pred = did2samples[did][uid]['prediction']
                try:
                    pred = parse_pptod_string(pred)
                except:
                    pred = []
                loc_pred.append(pred)

            gold_pols.append(loc_gold)
            pred_pols.append(loc_pred)

    elif args['model'] in ['LLM']:
        pred_data = load_json(args['pred'])
        curr_did = pred_data[0]['did']
        loc_gold = []
        loc_pred = []
        for entry in pred_data:
            did = entry['did']
            if did != curr_did:
                gold_pols.append(loc_gold)
                pred_pols.append(loc_pred)

                loc_gold = []
                loc_pred = []

            loc_gold.append(entry['nlu'])
            loc_pred.append(entry['prediction'])

        gold_pols.append(loc_gold)
        pred_pols.append(loc_pred)

    print(loc_gold[:5])
    print(loc_pred[:5])

    allg = [x for y in gold_pols for x in y]
    allp = [x for y in pred_pols for x in y]
    ret = compute_precision_recall_f1(allg, allp, mode=args['mode'])
    print(json.dumps(ret, indent=2))
    compute_soft_metrics_chatgpt(allg, allp, ignore_intent=False)
