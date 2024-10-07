class CollectData {
    constructor(utterances, opmode) {
        this.annotations = {};
        this.annotation_rec_idx = {};
        for (var ii = 0; ii < utterances.length; ii++) {
            this.annotations[String(ii)] = new Array();
            this.annotation_rec_idx[String(ii)] = 0;
        }

        this.opmode = "main";
        if (typeof opmode != "undefined" ) {
            this.opmode = opmode;
        }

        this.tracking_slot_values = {};
        for (var ii = 0; ii < tracking_slots.length; ii++) {
            this.tracking_slot_values[tracking_slots[ii]] = {};
        }

        if (this.opmode != "tutorial") {
            return;
        }
        this.gold_annotations = {};
        for (var ii = 0; ii < utterances.length; ii++) {
            this.gold_annotations[String(ii)] = utterances[ii]['annotation'];
        }
    }

    update_tracking_information() {
        for (var ii = 0; ii < tracking_slots.length; ii++) {
            delete this.tracking_slot_values[tracking_slots[ii]];
            this.tracking_slot_values[tracking_slots[ii]] = {};
        }

        for (let uid in this.annotations) {
            let results = this.annotations[uid];
            for (let jj = 0; jj < results.length; jj++) {
                let res = results[jj];
                if (('slot' in res) && tracking_slots.includes(res['slot']) && typeof res['value'] != "undefined") {
                    var toks = res['value'].split(',');
                    var status = "not defined";
                    if ("status" in res) {
                        if (good_values.includes(res["status"])) {
                            status = "green";
                        } else if (bad_values.includes(res["status"])) {
                            status = "red";
                        } else if (not_sure_values.includes(res["status"])) {
                            status = "grey"
                        }
                    }
                    for (var tt = 0; tt < toks.length; tt++) {
                        if ((toks[tt] in this.tracking_slot_values[res['slot']])) {
                            if (status != "not defined") {
                                this.tracking_slot_values[res['slot']][toks[tt]] = status;
                            }
                        } else {
                            this.tracking_slot_values[res['slot']][toks[tt]] = status;
                        }
                    }        
                }
            }
        }
    }

    add_form_results(uidx, form) {
        var res = {};
        for (var jj = 0; jj < form.elements.length; jj++) {
            var elem = form.elements[jj];
    
            if (elem.type === 'checkbox') {
                if (elem.checked || (elem.hasAttribute("always_store") && elem.getAttribute("always_store") == "true")) {
                    res[elem.name] = elem.checked;
                }
            } else {
                if (elem.name.includes("_fuzy")) {
                    continue;
                }
                if (elem.value != '') {
                    res[elem.name] = elem.value;
                }
            }
        }
        res["identifier"] = this.annotation_rec_idx[uidx];
        this.annotation_rec_idx[uidx]++;
        this.annotations[uidx].push(res);

        this.update_tracking_information();

        return res;
    }

    fill_form_results(res, form) {
        var intent_element = form.elements[0];
        intent_element.value = res["intent"];
        var change_event = new Event("change", {bubbles: true});
        intent_element.dispatchEvent(change_event);

        if (form.elements.length > 1) {
            var slot_element = form.elements[1];
            slot_element.value = res["slot"];
            slot_element.dispatchEvent(change_event);
        }

        for (var jj = 0; jj < form.elements.length; jj++) {
            var elem = form.elements[jj];
    
            if (elem.type === 'checkbox') {
                if (res[elem.name]) {
                    elem.checked = res[elem.name];
                }
            } else {
                if (elem.name.includes("_fuzy")) {
                    continue;
                }
                if (res[elem.name]) {
                    elem.value = res[elem.name];
                }
            }
        }
    }

    remove_form_results(uidx, identifier) {
        var ident = parseInt(identifier);
        var iloc = -1;
        for (var ii = 0; ii < this.annotations[uidx].length; ii++) {
            if (this.annotations[uidx][ii]['identifier'] == ident) {
                iloc = ii;
                break;
            }
        }
        if (iloc >= 0) {
            this.annotations[uidx].splice(iloc, 1);
        }

        this.update_tracking_information();

        return this.annotations[uidx].length;
    }

    get_form_results(uidx, identifier) {
        var ident = parseInt(identifier);
        var iloc = -1;
        for (var ii = 0; ii < this.annotations[uidx].length; ii++) {
            if (this.annotations[uidx][ii]['identifier'] == ident) {
                iloc = ii;
                break;
            }
        }
        //console.log(iloc);
        var value_dict = this.annotations[uidx][iloc];
        //console.log(value_dict["intent"]);
        return value_dict;
    }

    get_all_results() {
        var all_uidxs = [];
        for (var key in this.annotation_rec_idx) {
            all_uidxs.push(key);
        }
        all_uidxs.sort(function(a, b){return parseInt(a) - parseInt(b)});

        var results = [];
        for (var ii = 0; ii < all_uidxs.length; ii++) {
            for (var jj = 0; jj < this.annotations[all_uidxs[ii]].length; jj++) {
                var tres = JSON.parse(JSON.stringify(this.annotations[all_uidxs[ii]][jj]));
                delete tres['identifier'];
                tres['uid'] = all_uidxs[ii];
                results.push(tres);
            }
        }

        return results;
    }

    set_all_results(results) {
        var all_res = new Array();
        for (var ii = 0; ii < results.length; ii++) {
            var res = Object.assign({}, results[ii]);
            var uidx = res['uid'];
            delete res['uid'];

            res["identifier"] = this.annotation_rec_idx[uidx];
            this.annotation_rec_idx[uidx]++;
            this.annotations[uidx].push(res);
            all_res.push(res);
        }

        this.update_tracking_information();

        return all_res;
    }

    check_results_utterance(uttr_idx) {
        if (this.opmode != "tutorial") {
            return true, ""
        }

        var annotations = this.annotations[uttr_idx];
        var gold_annotations = this.gold_annotations[uttr_idx];

        if (annotations.length == 0) {
            return [false, "<p>Please annotate the utterance before checking.</p>"];
        }

        function match_dicts(gold, pred) {
            var flag = 1;
            var ignore_keys = ["uid", "explaination", "identifier"];

            for (var key of Object.keys(gold)) {
                let tflag = false;
                for (var kk of ignore_keys) {
                    if (kk === key) {
                        tflag = true;
                        break;
                    }
                }
                if (tflag) {
                    continue;
                }
                if (!pred.hasOwnProperty(key)) {
                    flag = 0;
                    break;
                }
                if (gold[key] != pred[key]) {
                    flag = 0;
                    break
                }
            }

            return flag;
        }

        var matching = new Array();
        for (var ii = 0; ii < annotations.length; ii++) {
            matching.push(new Array());
            for (var jj = 0; jj < gold_annotations.length; jj++) {
                matching[ii].push(match_dicts(gold_annotations[jj], annotations[ii]));
            }
        }

        var matched_gold = 0;
        for (var jj = 0; jj < gold_annotations.length; jj++) {
            var cnt = 0;
            for (var ii = 0; ii < annotations.length; ii++) {
                cnt += matching[ii][jj];
            }
            if (cnt > 0) {
                // Note: we allow multiple matches
                matched_gold++;
            }
        }

        var matched_pred = 0;
        for (var ii = 0; ii < annotations.length; ii++) {
            var cnt = 0;
            for (var jj = 0; jj < gold_annotations.length; jj++) {
                cnt += matching[ii][jj];
            }
            if (cnt > 0) {
                // Note: we allow multiple matches
                matched_pred++;
            }
        }

        var html_msg = '<table>';
        for (var ii = 0; ii < gold_annotations.length; ii++) {
            // html_msg += "<p>" + String(ii) + " " + gold_annotations[ii]["explaination"] + "</p>";
            html_msg += "<p>" + gold_annotations[ii]["explanation"] + "</p>";
        }

        if ((matched_pred == annotations.length) && (matched_gold == gold_annotations.length)) {
            return [true, gold_annotations];
        }
        return [false, gold_annotations];
    }
}

var global_collected_data;
