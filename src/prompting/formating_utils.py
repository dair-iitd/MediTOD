from copy import deepcopy


def format_nlu(nlu):
    ft_nlu = []

    for entry in nlu:
        ft_entry = dict()
        ft_entry['intent'] = entry['intent']

        if 'slots' not in entry:
            ft_nlu.append(ft_entry)
            continue

        ft_entry['slots'] = dict()

        for slot in entry['slots']:
            status = None
            repl = None
            if 'positive_' in slot:
                status = 'positive'
                repl = 'positive_'
            elif 'negative_' in slot:
                status = 'negative'
                repl = 'negative_'
            elif 'unknown_' in slot:
                status = 'unknown'
                repl = 'unknown_'
            elif 'unavail_' in slot:
                status = 'unavail'
                repl = 'unavail_'
            elif 'avail_' in slot:
                status = "avail"
                repl = "avail_"

            if status is None:
                ft_entry['slots'][slot] = deepcopy(entry['slots'][slot])
                continue

            new_arr = []
            for ee in entry['slots'][slot]:
                tee = deepcopy(ee)
                tee['status'] = status
                new_arr.append(tee)

            ft_entry['slots'][slot.replace(repl, '')] = new_arr
        ft_nlu.append(ft_entry)

    return ft_nlu


def deformat_nlu(ft_nlu):
    nlu = []

    for ft_entry in ft_nlu:
        entry = dict()
        entry['intent'] = ft_entry['intent']

        if 'slots' not in ft_entry:
            nlu.append(entry)
            continue

        entry['slots'] = dict()

        for slot in ft_entry['slots']:
            if slot in [
                'symptom', 'medical_history', 'family_history',
                'habit', 'exposure', 'medication', 'medical_test'
            ]:
                for ee in ft_entry['slots'][slot]:
                    st = ee.get('status')
                    if st in ['positive', 'negative', 'unknown'] and slot != 'medical_test':
                        dslot = f"{st}_{slot}"

                    elif st in ['avail', 'unavail'] and slot == 'medical_test':
                        dslot = f"{st}_{slot}"

                    else:
                        dslot = f"unknown_{slot}"

                    if dslot not in entry['slots']:
                        entry['slots'][dslot] = []

                    tee = deepcopy(ee)
                    if 'status' in tee:
                        del tee['status']
                    entry['slots'][dslot].append(tee)

            else:
                entry['slots'][slot] = deepcopy(ft_entry['slots'][slot])

        nlu.append(entry)

    return nlu

