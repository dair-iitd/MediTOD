var button_active = false;
var active_div = '';


function submit_label_form(event) {
    if (in_edit_mode) {
        in_edit_mode = false;
        var close_btn = document.getElementById("status_xbtn_" + edit_status_code);
        close_btn.click();
    }

    var sbt_btn = event.target;
    var form = sbt_btn.form;
    var uttr_idx = form.getAttribute("uttr_idx");

    for (var ii = 0; ii < form.elements.length; ii++) {
        if (form.elements[ii].name === 'intent' && form.elements[ii].value === 'None') {
            alert("Intent can not be empty.");
            return;
        }

        if (form.elements[ii].name === 'slot' && form.elements[ii].value === 'None') {
            alert("Slot can not be empty.");
            return;
        }
    }

    var res = global_collected_data.add_form_results(uttr_idx, form);
    update_status_table(uttr_idx, res);

    var parent_div = form.parentElement;
    for (var ii = 0; ii < parent_div.children.length; ii++) {
        if (parent_div.children[ii].nodeName === "FORM") {
            parent_div.removeChild(parent_div.children[ii]);
            break;
        }
    }

    var all_divs = document.getElementById("ann_box").children;
    var all_btns = document.getElementById("dlg_box").children;
    for (var ii = 0; ii < all_divs.length; ii++) {
        if (!all_divs[ii].classList.contains("d-none")) {
            all_divs[ii].classList.add("d-none");
        }

        var tbtn = all_btns[ii].children[1].children[0];
        tbtn.disabled = false;
    }

    var tokens = form.id.split("_", 3);
    var btn_id = tokens[0] + '_' + tokens[1];
    var btn = document.getElementById(btn_id);
    if (btn.getAttribute("is_clicked") === "false") {
        var span = document.createElement("i");
        span.className = "fa fa-check";
        btn.appendChild(span);
        global_progress_bar.increment_bar();
    }
    btn.setAttribute("is_clicked", "true");
    btn.focus();
    button_active = false;
    active_div = '';
}


function remove_label_form(event) {
    in_edit_mode = false;

    var parent_div = event.target.parentElement.parentElement.parentElement.parentElement;
    for (var ii = 0; ii < parent_div.children.length; ii++) {
        if (parent_div.children[ii].nodeName === "FORM") {
            parent_div.removeChild(parent_div.children[ii]);
            break;
        }
    }

    var all_divs = document.getElementById("ann_box").children;
    var all_btns = document.getElementById("dlg_box").children;
    for (var ii = 0; ii < all_divs.length; ii++) {
        if (!all_divs[ii].classList.contains("d-none")) {
            all_divs[ii].classList.add("d-none");
        }

        var tbtn = all_btns[ii].children[1].children[0];
        tbtn.disabled = false;
    }

    var tokens = parent_div.id.split("_", 3);
    var btn_id = tokens[0] + '_' + tokens[1];
    var btn = document.getElementById(btn_id);
    btn.focus();
    button_active = false;
    active_div = '';
}


function add_utterance_form_div(index) {
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


function utterance_button_event(event) {
    var btn;
    if (event.target.nodeName === "I") {
        btn = event.target.parentElement;
    } else {
        btn = event.target;
    }

    var div_id = btn.id.concat("_div");
    var div_elem = document.getElementById(div_id);

    add_label_form(div_elem, btn.getAttribute("uttr_idx"), btn.getAttribute("speaker"), btn.textContent);

    var all_divs = document.getElementById("ann_box").children;
    var all_btns = document.getElementById("dlg_box").children;
    for (var ii = 0; ii < all_divs.length; ii++) {
        if (!all_divs[ii].classList.contains("d-none")) {
            all_divs[ii].classList.add("d-none");
        }

        var tbtn = all_btns[ii].children[1].children[0];
        if (!tbtn.disabled) {
            tbtn.disabled = true;
        }
    }
    div_elem.classList.remove('d-none');
    btn.disabled = false;
    button_active = true;
    active_div = div_id;
}


function add_uttrance(speaker, index, utterance) {
    var outer_div = document.createElement("div");
    outer_div.className = "row";

    var div = document.createElement("div");
    div.className = "col-1 text-center";
    div.textContent = String(index) + "."
    outer_div.appendChild(div);

    var element = document.createElement("button");
    element.type = "button";

    let elem_id = "uttr_".concat(index.toString());
    let class_name = "btn btn-block"
    let text = ""
    let icon;
    if (speaker == "doctor") {
        class_name = class_name.concat(" btn-info");
        icon = '<i class="fa fa-user-md"></i> &nbsp';
        text = text.concat('Doctor: ', utterance, ' ');
    } else if (speaker == "patient") {
        class_name = class_name.concat(" btn-warning");
        icon = '<i class="fa fa-wheelchair"></i> &nbsp';
        text = text.concat('Patient: ', utterance, ' ');
    } else if (speaker == "control") {
        class_name = class_name.concat(" btn-secondary");
        icon = '<i class="fa fa-gears"></i> &nbsp';
        text = text.concat(utterance, ' ');
    }
    element.id = elem_id;
    element.className = class_name;
    element.innerHTML = icon + text;
    element.onclick = utterance_button_event;

    element.onkeydown = utterance_keydown_event;
    element.onfocus = utterance_onfocus_event;
    element.onblur = utterance_onblur_event;

    element.setAttribute("is_clicked", "false");
    element.setAttribute("uttr_idx", index);
    element.setAttribute("speaker", speaker);

    var div = document.createElement("div");
    div.className = "col d-grid";
    div.appendChild(element);
    outer_div.appendChild(div);

    var dlg_box = document.getElementById("dlg_box");
    dlg_box.appendChild(outer_div);
}


function add_new_utterance(entry) {
    add_uttrance(entry.speaker, entry.uttr_id, entry.text);
    add_new_form_div(entry.uttr_id);
}


function utterance_keydown_event(event) {
    var key = event.which;
    // 27 is escape, 38 up, 40 down, 13 enter
    if (key != 38 && key != 40) {
        return;
    }

    event.preventDefault();
    var uttr_id = event.target.id;

    var tokens = uttr_id.split("_", 2);
    var uidx = parseInt(tokens[1]);

    if (key == 38) {
        if (uidx != 0) {
            uidx--;
        }
        next_uttr_id = tokens[0] + '_' + parseInt(uidx).toString();
    } else if (key == 40) {
        if (uidx != global_progress_bar.total - 1) {
            uidx++;
        }
        next_uttr_id = tokens[0] + '_' + parseInt(uidx).toString();
    }

    var next_uttr = document.getElementById(next_uttr_id);
    next_uttr.focus();
}


function utterance_onfocus_event(event) {
    var row = event.target.parentElement.parentElement;
    // var row = event.target.parentElement;
    row.classList.add("border");
    row.classList.add("border-primary");
    row.classList.add("border-3");
}


function utterance_onblur_event(event) {
    if (button_active) {
        return;
    }
    var row = event.target.parentElement.parentElement;
    // var row = event.target.parentElement;
    row.classList.remove("border");
    row.classList.remove("border-primary");
    row.classList.remove("border-3");
}
