function add_new_form_div(index) {
    var base_id = "";
    base_id = base_id.concat("uttr_", index.toString());

    var div = document.createElement("div");
    div.id = base_id.concat("_div");
    div.classList.add('row');
    div.classList.add('d-none');
    div.classList.add('p-3');

    div.classList.add("text-white");
    div.classList.add("bg-dark");

    var ann_box = document.getElementById("ann_box");
    ann_box.appendChild(div);
}


function add_label_form(parent_div, uttr_idx, speaker, uttr) {
    for (var ii = 0; ii < parent_div.children.length; ii++) {
        if (parent_div.children[ii].nodeName == 'FORM') {
            alert("Submit current label first before adding new one.")
            return;
        }
    }
    var parent_id = parent_div.id.slice(0, -4);
    var new_form = document.createElement("form");
    new_form.id = parent_id.concat("_form_");
    new_form.setAttribute("uttr_idx", uttr_idx);
    new_form.setAttribute("speaker", speaker);
    new_form.setAttribute("autocomplete", "off");

    new_form.onkeydown = function(event) {
        var keyCode = event.which;
        if (keyCode === 13) { 
            event.preventDefault();
            return false;
        }
    };

    new_form.onkeyup = function(event) {
        var keyCode = event.which;
        if (keyCode === 13) { 
            event.preventDefault();
            return false;
        }
    };

    var uttr_text = document.createElement("p");
    uttr_text.innerHTML = "<b><i>" + uttr.trim() + "</i></b>";
    new_form.appendChild(uttr_text);
    new_form.appendChild(document.createElement("hr"));

    var input_div = get_input_skeleton(6);

    var info_icon = document.createElement("i");
    info_icon.id = new_form.id.concat("intent_info");
    info_icon.className = 'fa fa-question-circle';
    info_icon.setAttribute("data-toggle", "tooltip");
    info_icon.setAttribute("title", intent_config['help_message']);

    var desc = document.createElement("span");
    desc.innerHTML = intent_config['question']

    input_div.children[0].appendChild(desc);
    input_div.children[0].appendChild(info_icon);
    input_div.children[0].for = new_form.id.concat("intent_select");
    input_div.children[0].id = new_form.id.concat("intent_label");

    var sel_elem = document.createElement("select");
    sel_elem.className = "form-select form-select-sm";
    sel_elem.id = new_form.id.concat("intent_select");
    sel_elem.name = "intent";
    sel_elem.onchange = intent_select_event;

    var opt = document.createElement("option");
    opt.value = "None";
    opt.text = "-- select an intent --";
    sel_elem.add(opt);

    for (var ii = 0; ii < intent_config['intent_list'].length; ii++) {
        let intent_info = intent_config['intent_list'][ii];
        if (!intent_info['applicable_speakers'].includes(speaker)) {
            continue;
        }
        var opt = document.createElement("option");
        opt.value = intent_info['value'];
        if (intent_info.hasOwnProperty("text")) {
            opt.text = intent_info['text'];
        } else {
            opt.text = intent_info['value'];
        }

        if (intent_info.hasOwnProperty("help_message")) {
            opt.setAttribute("data-toggle", "tooltip");
            opt.setAttribute("title", intent_info['help_message']);
        }
        sel_elem.add(opt);
    }
    input_div.children[1].appendChild(sel_elem);
    new_form.appendChild(input_div);

    var slot_div = document.createElement("div");
    slot_div.id = new_form.id.concat("slot_div");
    new_form.appendChild(slot_div);

    var custom_div = document.createElement("div");
    custom_div.id = new_form.id.concat("custom_div");
    new_form.appendChild(custom_div);

    var footer_div = document.createElement("div");
    footer_div.className = "row";

    var sbt_btn = document.createElement("button");
    sbt_btn.id = new_form.id.concat("sbt_btn");
    sbt_btn.type = "button";
    sbt_btn.className = "btn btn-danger btn-sm btn-block col-11";
    sbt_btn.textContent = "Submit";
    sbt_btn.onclick = submit_label_form;
    sbt_btn.onkeydown = function(event) {
        if (event.which == 13) {
            submit_label_form(event);
        }
    }
    footer_div.appendChild(sbt_btn);

    var x_btn = document.createElement("button");
    x_btn.type = "button";
    x_btn.className = "btn-close btn-close-white border border-5";
    x_btn.onclick = remove_label_form;
    var t_div = document.createElement("div");
    t_div.className = "col-1";
    t_div.appendChild(x_btn);
    footer_div.appendChild(t_div);

    var tmp = document.createElement("hr");
    tmp.setAttribute("style", "height:3px; width:100%; border-width:0; color:red; background-color:red");
    new_form.appendChild(tmp);

    new_form.appendChild(footer_div);
    parent_div.insertBefore(new_form, parent_div.firstChild);

    // parent_div.insertBefore(document.createElement("hr"), parent_div.firstChild);
    // var uttr_text = document.createElement("p");
    // uttr_text.innerHTML = "<b><i>" + uttr.trim() + "</i></b>";
    // parent_div.insertBefore(uttr_text, parent_div.firstChild);

    var elem = document.getElementById(new_form.id.concat("intent_select"));
    waitForElement(new_form.id.concat("intent_select"), function () { elem.focus(); });
}


function intent_select_event(event) {
    var intent = event.target;
    var intent_type = intent.options[intent.selectedIndex].value;
    var form = intent.form;

    var slot_div = document.getElementById(form.id.concat("slot_div"));
    while (slot_div.firstChild) {
        slot_div.removeChild(slot_div.firstChild);
    }
    var custom_div = document.getElementById(form.id.concat("custom_div"));
    while (custom_div.firstChild) {
        custom_div.removeChild(custom_div.firstChild);
    }

    if (intent_type == "other") {
        var input_div = get_input_skeleton(6);
        input_div.children[0].textContent = "Enter custom intent type";
        input_div.children[0].for = form.id.concat("intent_other");
        input_div.children[0].id = form.id.concat("intent_other_label");

        var elem = document.createElement("input");
        elem.type = "text";
        elem.className = "form-control form-control-sm"
        elem.id = form.id.concat("intent_other");
        elem.name = "custom_intent_value";
        input_div.children[1].appendChild(elem);
        slot_div.appendChild(input_div);
    }

    var input_div = get_input_skeleton(6);

    var info_icon = document.createElement("i");
    info_icon.className = 'fa fa-question-circle';
    info_icon.setAttribute("data-toggle", "tooltip");
    info_icon.setAttribute("title", slot_config['help_message']);

    var desc = document.createElement("span");
    desc.innerHTML = slot_config['question'];

    input_div.children[0].appendChild(desc);
    input_div.children[0].appendChild(info_icon);
    input_div.children[0].for = form.id.concat("slot_select");
    input_div.children[0].id = form.id.concat("slot_label");

    var sel_elem = document.createElement("select");
    sel_elem.className = "form-select form-select-sm";
    sel_elem.id = form.id.concat("slot_select");
    sel_elem.name = "slot";
    sel_elem.onchange = slot_select_event;

    var flag = false;

    for (var ii = 0; ii < slot_config['slot_list'].length; ii++) {
        let cfg = slot_config['slot_list'][ii];
        if (cfg['parent_intents'].includes(intent_type)) {
            if (!flag) {
                var opt = document.createElement("option");
                opt.value = 'None';
                opt.text = '-- select a slot type --';
                sel_elem.add(opt);
                flag = true;
            }

            var opt = document.createElement("option");
            opt.value = cfg['value'];
            if (cfg.hasOwnProperty("text")) {
                opt.text = cfg['text'];
            } else {
                opt.text = cfg['value'];
            }

            if (cfg.hasOwnProperty("help_message")) {
                opt.setAttribute("data-toggle", "tooltip");
                opt.setAttribute("title", cfg['help_message']);
            }
            sel_elem.add(opt);
        }
    }
    if (flag) {
        input_div.children[1].appendChild(sel_elem);
        slot_div.appendChild(input_div);
    }
}


function slot_select_event(event) {
    var slot = event.target;
    var slot_type = slot.options[slot.selectedIndex].value;
    var form = slot.form;
    var spk = form.getAttribute("speaker");
    var intent = document.getElementById(slot.id.replace("slot", "intent")).value;

    var custom_div = document.getElementById(form.id.concat("custom_div"));
    while (custom_div.firstChild) {
        custom_div.removeChild(custom_div.firstChild);
    }

    if (slot_type === 'None') {
        return;
    }

    var slot_cfg;
    for (var ii = 0; ii < slot_config['slot_list'].length; ii++) {
        if (slot_type === slot_config['slot_list'][ii]['value']) {
            slot_cfg = slot_config['slot_list'][ii]['config'];
        }
    }

    for (var ii = 0; ii < slot_cfg.length; ii++) {
        var cfg = slot_cfg[ii];
        var chkbox_on = cfg.hasOwnProperty('chkbox_on') ? cfg['chkbox_on'] : [];

        if (cfg.type === 'datalist') {
            if (chkbox_on.includes(intent)) {
                var tcfg = {
                    "name": cfg.name,
                    "type": "checkbox",
                    "label": cfg.label,
                }
                add_checkbox_to_form(tcfg, custom_div, form.id, slot_type, spk);
                continue;
            }
            add_datalist_to_form(cfg, custom_div, form.id, intent, slot_type, spk);
        } else if (cfg.type === "text") {
            if (chkbox_on.includes(intent)) {
                var tcfg = {
                    "name": cfg.name,
                    "type": "checkbox",
                    "label": cfg.label,
                }
                add_checkbox_to_form(tcfg, custom_div, form.id, slot_type, spk);
                continue;
            }
            add_text_to_form(cfg, custom_div, form.id, intent, slot_type, spk);
        } else if (cfg.type === "checkbox") {
            add_checkbox_to_form(cfg, custom_div, form.id);
        } else if (cfg.type === "select") {
            if (chkbox_on.includes(intent)) {
                var tcfg = {
                    "name": cfg.name,
                    "type": "checkbox",
                    "label": cfg.label,
                }
                add_checkbox_to_form(tcfg, custom_div, form.id);
                continue;
            }
            add_select_to_form(cfg, custom_div, form.id, intent, slot_type, spk);
        }
    }
}
