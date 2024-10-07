function get_input_skeleton(labsize, insize, insize2) {
    var outer_div = document.createElement("div");
    outer_div.className = "row";

    var label = document.createElement("label");
    label.classList.add("form-label");
    label.classList.add("col-" + String(labsize));

    var div = document.createElement("div");
    if (typeof insize === "undefined") {
        div.className = "col-" + String(12 - labsize);
    } else {
        div.className = "col-" + String(insize);
    }

    outer_div.appendChild(label);
    outer_div.appendChild(div);

    if (typeof insize2 != "undefined") {
        var div = document.createElement("div");
        div.className = "col-" + String(insize2);
        outer_div.appendChild(div);
    }

    return outer_div;
}


function get_grided_inputs(inid, num_elems, elems_per_row) {
    const num_rows = Math.floor(num_elems / elems_per_row);
    const num_elems_last_row = num_elems % elems_per_row;
    var num_cols_per_elem = String(Math.floor(12 / (elems_per_row * 2)));
    var rows_arr = new Array();

    for (var ii = 0; ii < num_rows; ii += 1) {
        var row_div = document.createElement("div");
        row_div.className = "row";
        row_div.id = inid.concat('_row_', String(ii));

        for (var jj = 0; jj < elems_per_row; jj++) {
            var label = document.createElement("label");
            label.classList.add("form-label");
            label.classList.add("col-" + num_cols_per_elem);
        
            var div = document.createElement("div");
            div.className = "col-" + String(num_cols_per_elem);
            row_div.appendChild(label);
            row_div.appendChild(div);
        }
        rows_arr.push(row_div);
    }

    if (num_elems_last_row == 0) {
        return rows_arr;    
    }

    // var num_cols_per_elem = String(Math.floor(12 / (num_elems_last_row * 2)));
    var row_div = document.createElement("div");
    row_div.className = "row";
    row_div.id = inid.concat('_row_', String(rows_arr.length));

    for (var jj = 0; jj < num_elems_last_row; jj++) {
        var label = document.createElement("label");
        label.classList.add("form-label");
        label.classList.add("col-" + num_cols_per_elem);
    
        var div = document.createElement("div");
        div.className = "col-" + String(num_cols_per_elem);
        row_div.appendChild(label);
        row_div.appendChild(div);
    }
    rows_arr.push(row_div);

    return rows_arr;
}


function slot_checkbox_onchage_event(event) {
    // TODO: Nasty way of referring to form.
    var chkbox = event.target;
    for (var kk = 0; kk < chkbox.enables.length; kk++) {
        var elem = document.getElementById(chkbox.enables[kk]);
        var contains = elem.parentElement.parentElement.classList.contains("d-none");

        // elem.disabled = !chkbox.checked;
        if (chkbox.checked) {
            if (contains) {
                elem.parentElement.parentElement.classList.remove("d-none");
            }
        } else {
            if (!contains) {
                elem.parentElement.parentElement.classList.add("d-none");
            }
        }
    }
} 


function add_checkbox_to_form(cfg, parent_div, form_id, slot_type, spk) {
    if (cfg.type != 'checkbox') {
        return;
    }

    var div = get_input_skeleton(11);

    var elem = document.createElement("input");
    elem.type = cfg.type;    
    elem.className = "form-check-input"
    elem.id = form_id.concat("slot_value_", cfg.name);
    elem.name = cfg.name;
    elem.value = "true";
    elem.onchange = slot_checkbox_onchage_event;
    elem.enables = new Array();
    if (cfg.hasOwnProperty("enables")) {
        for (var kk = 0; kk < cfg.enables.length; kk++) {
            elem.enables.push(form_id.concat("slot_value_", cfg.enables[kk]));
        }
    }
    var always_store = cfg.hasOwnProperty('always_store') ? cfg['always_store'] : false;
    if (always_store) {
        elem.setAttribute("always_store", "true");
    }

    div.children[1].classList.add("form-check");
    div.children[1].appendChild(elem);

    var label = div.children[0];
    label.classList.add("form-check-label");
    label.textContent = cfg.label;
    label.for = elem.id;
    label.id = elem.id.concat("_label");

    parent_div.appendChild(div);
}


function add_datalist_to_form(cfg, parent_div, form_id, intent, slot_type, spk) {
    if (cfg.type != 'datalist') {
        return;
    }

    var aux_flag = "none";
    if (cfg.hasOwnProperty("label")) {
        if (cfg.hasOwnProperty("add_checkbox") && cfg["add_checkbox"].includes(intent)) {
            var div = get_input_skeleton(6, 5, 1);
            aux_flag = "checkbox";
        } else if (cfg.hasOwnProperty("add_select") && cfg["add_select"].includes(intent)) {
            var div = get_input_skeleton(6, 4, 2);
            aux_flag = "select";
        } else {
            var div = get_input_skeleton(6);
        }
    } else {
        var div = get_input_skeleton(0);
    }

    var elem = document.createElement("input");
    elem.className = "custom_dropdown_input form-control form-control-sm";
    elem.id = form_id.concat("slot_value_", cfg.name);
    elem.name = cfg.name;
    elem.placeholder = "Type to search";
    elem_id = elem.id;
    elem.onblur = custom_dropdown_input_onblur;
    elem.onfocus = custom_dropdown_input_onfocus;
    elem.onkeyup = custom_dropdown_keyup;
    elem.onkeydown = custom_dropdown_keydown;

    div.children[1].appendChild(elem);
    div.children[1].classList.add("custom_dropdown");
    // To be used for search
    var search_tag = slot_type + "_" + cfg.name;
    div.children[1].setAttribute("search_tag", search_tag);
    search_controller.add_index(search_tag, cfg.values);

    var content_div = document.createElement("div");
    content_div.className = "custom_dropdown-content d-none";
    content_div.setAttribute("tabindex", "-1");
    for (var val of cfg.values) {
        var a_elem = document.createElement("a");
        a_elem.className = "custom_dropdown_entry";
        a_elem.setAttribute("value", val.code);
        if (val.hasOwnProperty("description")) {
            a_elem.innerHTML = "<b>" + val.code + "</b> " + val.description;
        } else {
            a_elem.innerHTML = val.code;
        }

        a_elem.onclick = custom_dropdown_entry_click;
        content_div.appendChild(a_elem);
    }
    div.children[1].appendChild(content_div);

    if (aux_flag === "checkbox") {
        var elem = document.createElement("input");
        elem.type = "checkbox";
        elem.className = "form-check-input"
        elem.id = form_id.concat("slot_value_", cfg.name, "_check");
        elem.name = cfg.name + " check";
        elem.value = "true";
        div.children[2].classList.add("form-check");
        div.children[2].appendChild(elem);
    } else if (aux_flag === "select") {
        var elem = document.createElement("select");
        elem.type = "select";
        elem.className = "form-select form-select-sm";
        elem.id = form_id.concat("slot_value_", cfg.name, "_status");
        elem.name = cfg.name + " status";

        var opt = document.createElement("option");
        opt.value = '';
        opt.text = '-- select --';
        elem.add(opt);

        for (var ii = 0; ii < cfg.aux_values.length; ii++) {
            var opt = document.createElement("option");
            opt.value = cfg.aux_values[ii]['code'];
            opt.text = cfg.aux_values[ii]['description'];
            elem.add(opt);
        }
        div.children[2].appendChild(elem);
    }

    if (cfg.hasOwnProperty("label")) {
        var label = div.children[0];
        label.textContent = cfg.label;
        label.for = elem.id;
        label.id = elem.id.concat("_label");    
    } else {
        div.removeChild(div.firstChild);
    }

    if (cfg.hasOwnProperty("default")) {
        elem.placeholder = cfg.default;
    }

    if (cfg.hasOwnProperty("disabled")) {
        div.classList.add("d-none");
    }

    parent_div.appendChild(div);
}


function add_text_to_form(cfg, parent_div, form_id, intent, slot_type, spk) {
    if (cfg.type != 'text') {
        return;
    }

    var checkbox_flag = false;
    if (cfg.hasOwnProperty("label")) {
        if (cfg.hasOwnProperty("add_checkbox") && cfg["add_checkbox"].includes(intent)) {
            var div = get_input_skeleton(6, 5, 1);
            checkbox_flag = true;
        } else {
            var div = get_input_skeleton(6);
        }
    } else {
        var div = get_input_skeleton(0);
    }

    elem = document.createElement("input");
    elem.type = cfg.type;    
    elem.className = "form-control form-control-sm"
    elem.id = form_id.concat("slot_value_", cfg.name);
    elem.name = cfg.name;
    div.children[1].appendChild(elem);

    if (checkbox_flag) {
        var elem = document.createElement("input");
        elem.type = "checkbox";
        elem.className = "form-check-input"
        elem.id = form_id.concat("slot_value_", cfg.name, "_check");
        elem.name = cfg.name + " check";
        elem.value = "true";
        div.children[2].classList.add("form-check");
        div.children[2].appendChild(elem);
    }

    if (cfg.hasOwnProperty("label")) {
        var label = div.children[0];
        label.textContent = cfg.label;
        label.for = elem.id;
        label.id = elem.id.concat("_label");    
    } else {
        div.removeChild(div.firstChild);
    }

    if (cfg.hasOwnProperty("default")) {
        elem.placeholder = cfg.default;
    }

    if (cfg.hasOwnProperty("disabled")) {
        div.classList.add("d-none");
    }

    parent_div.appendChild(div);
}


function add_select_to_form(cfg, parent_div, form_id, intent, slot_type, spk) {
    if (cfg.type != 'select') {
        return;
    }

    var aux_flag = "none";
    if (cfg.hasOwnProperty("label")) {
        if (cfg.hasOwnProperty("add_checkbox") && cfg["add_checkbox"].includes(intent)) {
            var div = get_input_skeleton(6, 5, 1);
            aux_flag = "checkbox";
        } else if (cfg.hasOwnProperty("add_text") && cfg["add_text"].includes(intent)) {
            var div = get_input_skeleton(6, 4, 2);
            aux_flag = "text";
        } else {
            var div = get_input_skeleton(6);
        }
    } else {
        var div = get_input_skeleton(0);
    }

    elem = document.createElement("select");
    elem.type = cfg.type;    
    elem.className = "form-select form-select-sm";
    elem.id = form_id.concat("slot_value_", cfg.name);
    elem.name = cfg.name;

    var opt = document.createElement("option");
    opt.value = '';
    opt.text = '-- select --';
    elem.add(opt);

    var has_default = cfg.hasOwnProperty("default");
    var defval;
    if (has_default && typeof cfg.default === "string") {
        defval = cfg.default;
    } else if (has_default && typeof cfg.default === "object" && spk in cfg.default) {
        defval = cfg.default[spk];
    } else {
        has_default = false;
    }

    if (has_default && defval === "disabled") {
        // console.log(defval, spk);
        return;
    }

    for (var ii = 0; ii < cfg.values.length; ii++) {
        var opt = document.createElement("option");
        opt.value = cfg.values[ii];
        opt.text = cfg.values[ii];
        // if (cfg.hasOwnProperty("default") && cfg["default"] === cfg.values[ii]) {
        if (has_default && defval === cfg.values[ii]) {
            opt.selected = "selected";
        }
        elem.add(opt);
    }
    // selected="selected"
    div.children[1].appendChild(elem);

    if (aux_flag === "checkbox") {
        var elem = document.createElement("input");
        elem.type = "checkbox";
        elem.className = "form-check-input"
        elem.id = form_id.concat("slot_value_", cfg.name, "_check");
        elem.name = cfg.name + " check";
        elem.value = "true";
        div.children[2].classList.add("form-check");
        div.children[2].appendChild(elem);
    } else if (aux_flag === "text") {
        var elem = document.createElement("input");
        elem.type = "text";
        elem.className = "form-control form-control-sm"
        if (cfg.hasOwnProperty("aux_value")) {
            let ttag = cfg["aux_value"]["tag"];
            elem.id = form_id.concat("slot_value_", cfg.name, "_" + ttag);
            elem.name = cfg.name + " " + ttag;
            elem.placeholder = cfg["aux_value"]["placeholder"];
        } else {
            elem.id = form_id.concat("slot_value_", cfg.name, "_det");
            elem.name = cfg.name + " text";
        }
        div.children[2].appendChild(elem);
    }

    if (cfg.hasOwnProperty("label")) {
        var label = div.children[0];
        label.textContent = cfg.label;
        label.for = elem.id;
        label.id = elem.id.concat("_label");    
    } else {
        div.removeChild(div.firstChild);
    }

    if (cfg.hasOwnProperty("disabled")) {
        div.classList.add("d-none");
    }

    parent_div.appendChild(div);
}


function waitForElement(id, callback){
    var poops = setInterval(function(){
        if(document.getElementById(id)){
            clearInterval(poops);
            callback();
        }
    }, 100);
}
