var dialog_loded = false;
var ann_loded = false;

console.log("Initializing in", opmode, "mode.");


document.getElementById('import').onclick = function() {
    if (dialog_loded) {
        alert("Dialog already imported.");
        return;
    }
    var files = document.getElementById('selectFiles').files;
    if (files.length <= 0) {
        return false;
    }

    var fr = new FileReader();

    fr.onload = function(e) {
        var result = JSON.parse(e.target.result);
        result.utterances.forEach(add_new_utterance);
        global_collected_data = new CollectData(result.utterances, opmode);
        global_progress_bar = new ProgressBar(result.utterances.length);

        var uttr = document.getElementById('uttr_0');
        uttr.focus();
    }
    fr.readAsText(files.item(0));

    var ann_box = document.getElementById("ann_box");
    ann_box.classList.add("border");
    dialog_loded = true;
    setup_tracking_table();
}


function gather_all_responses() {
    var results = global_collected_data.get_all_results()
    var data = JSON.stringify(results);
    // console.log(data);

    var element = document.createElement('a');
    element.setAttribute('href', 'data:text/plain;charset=utf-8,' + encodeURIComponent(data));
    element.setAttribute('download', "data.json");

    element.style.display = 'none';
    document.body.appendChild(element);

    element.click();

    document.body.removeChild(element);
}
document.getElementById('save').onclick = gather_all_responses;

document.getElementById('import_ann').onclick = function() {
    if (!dialog_loded) {
        alert("Import dialog before loading the annotations.");
        return;
    }

    if (ann_loded) {
        alert("Annotations already loaded.")
        return;
    }

    var files = document.getElementById('selectFilesAnn').files;
    if (files.length <= 0) {
        return false;
    }

    var fr = new FileReader();

    fr.onload = function(e) {
        var result = JSON.parse(e.target.result);
        var entries = global_collected_data.set_all_results(result);

        for (var ii = 0; ii < entries.length; ii++) {
            var uttr_idx = result[ii]['uid'];
            update_status_table(uttr_idx, entries[ii]);

            var btn = document.getElementById('uttr_' + uttr_idx);
            if (btn.getAttribute("is_clicked") === "false") {
                var span = document.createElement("i");
                span.className = "fa fa-check";
                btn.appendChild(span);
                global_progress_bar.increment_bar();
            }
            btn.setAttribute("is_clicked", "true");
        }
    }
    ann_loded = true;
    fr.readAsText(files.item(0));
}

window.onbeforeunload = function ()
{
    return "";
};


window.onkeyup = function(event) {
    // This means that the form is being shown.
    if (event.which == 27 && active_div.length > 0) {
        var fdiv = document.getElementById(active_div);
        var xbtn = fdiv.getElementsByClassName("btn-close");
        xbtn[0].click();
    }
}



function delete_entry(btn) {
    var parent_row = btn.target.parentElement.parentElement;
    var uttr_idx = parent_row.getAttribute("uttr_idx");
    var identifier = parent_row.getAttribute("identifier");
    var status_table = document.getElementById("status_table");

    var iloc = 1;

    while (iloc < status_table.rows.length) {
        let tuttr_idx = status_table.rows[iloc].getAttribute("uttr_idx");
        let tidentifier = status_table.rows[iloc].getAttribute("identifier");

        if (uttr_idx == tuttr_idx && identifier == tidentifier) {
            break;
        }
        iloc++;
    }
    status_table.deleteRow(iloc); 
    var len = global_collected_data.remove_form_results(uttr_idx, identifier);

    if (len == 0) {
        var btn = document.getElementById("uttr_" + uttr_idx);
        var ii = 0
        while (ii < btn.children.length) {
            if (btn.children[ii].nodeName == "I" && btn.children[ii].classList.contains("fa-check")) {
                btn.removeChild(btn.children[ii]);
                break;
            }
            ii++;
        }
        btn.setAttribute("is_clicked", "false");
        global_progress_bar.decrement_bar();
    }
    update_tracking_table();
}


function update_status_table(uttr_idx, tres) {
    var res = Object.assign({}, tres)
    // var status_table = document.getElementById("status_table");
    var status_table = document.getElementById("status_table").getElementsByTagName('tbody')[0];
    // var iloc = 1;
    var iloc = 0;
    var uidx = parseInt(uttr_idx);

    while (iloc < status_table.rows.length) {
        let tidx = parseInt(status_table.rows[iloc].getAttribute("uttr_idx"));
        if (tidx > uidx) {
            break;
        }
        iloc++;
    }

    var row = status_table.insertRow(iloc);
    row.id = "status_row_" + String(uttr_idx) + "_" + String(iloc);
    row.setAttribute("uttr_idx", uttr_idx);
    row.setAttribute("identifier", res['identifier']);
    var uidx_cell = row.insertCell(0);
    uidx_cell.className = "text-center";
    var aelem = document.createElement("a");
    aelem.text = uttr_idx;
    aelem.className = "btn btn-outline-secondary";
    aelem.setAttribute("uttr_idx", uttr_idx);
    aelem.onclick = goto_utterance;
    uidx_cell.appendChild(aelem);
    delete res['identifier'];

    var stat_cell = row.insertCell(1);
    var pre = document.createElement("pre");
    pre.innerHTML = JSON.stringify(res, undefined, 2);
    stat_cell.appendChild(pre);

    var cancel_cell = row.insertCell(2);
    cancel_cell.className = "text-center";
    var x_btn = document.createElement("button");
    x_btn.type = "button";
    x_btn.id = "status_xbtn_" + String(uttr_idx) + "_" + String(iloc);
    x_btn.className = "btn-close border border-5";
    cancel_cell.appendChild(x_btn);
    x_btn.onclick = delete_entry;

    update_tracking_table();
}


function goto_utterance(event) {
    in_edit_mode = true; //added for edit mode
    var btn = event.target;
    var uttr_idx = btn.getAttribute("uttr_idx");

    var uttr_btn = document.getElementById("uttr_" + uttr_idx);
    uttr_btn.click();

    var status_row = btn.parentElement.parentElement;
    var status_row_name = status_row.id;
    var status_code = status_row_name.substring(11);
    edit_status_code = status_code; //added for edit mode

    var uttr_idx = status_row.getAttribute("uttr_idx");
    var identifier = status_row.getAttribute("identifier");
    var value_dict = global_collected_data.get_form_results(uttr_idx, identifier);

    //console.log(value_dict["intent"]);

    var form_sbt_btn = document.getElementById("uttr_" + uttr_idx + "_form_sbt_btn");
    var form = form_sbt_btn.form;
    global_collected_data.fill_form_results(value_dict, form);

    //var close_btn = document.getElementById("status_xbtn_" + status_code);
    //close_btn.click();
}


function setup_tracking_table() {
    var tracking_table = document.getElementById("tracking_table").getElementsByTagName('tbody')[0];

    for (var ii = 0; ii < tracking_slots.length; ii++) {
        var row = tracking_table.insertRow(ii);
        row.id = "tracking_row_" + tracking_slots[ii];
        row.setAttribute("tracking_slot", tracking_slots[ii]);

        var uidx_cell = row.insertCell(0);
        uidx_cell.className = "text-center";
        uidx_cell.innerHTML = tracking_slots[ii];

        var uidx_cell = row.insertCell(1);
        uidx_cell.className = "text-center";
        uidx_cell.innerHTML = '';
    }
}


function update_tracking_table() {
    var tracking_info = global_collected_data.tracking_slot_values;
    var tracking_table = document.getElementById("tracking_table").getElementsByTagName('tbody')[0];

    for (var ii = 0; ii < tracking_slots.length; ii++) {
        if (!(tracking_slots[ii] in tracking_info)) {
            continue;
        }
        let row = tracking_table.rows[ii];
        let row_slot = row.getAttribute("tracking_slot");

        if (row_slot != tracking_slots[ii]) {
            console.log("Error: Tracking misaligned.");
        }

        // let value = ''
        while (row.cells[1].firstChild) {
            row.cells[1].removeChild(row.cells[1].firstChild);
        }
    
        for (let entity in tracking_info[row_slot]) {
            let status = tracking_info[row_slot][entity];

            var aelem = document.createElement("a");
            aelem.text = entity;
            if (status === "green") {
                aelem.className = "btn btn-sm btn-success border-white";
            } else if (status === "red") {
                aelem.className = "btn btn-sm btn-danger border-white";
            } else if (status === "not defined") {
                aelem.className = "btn btn-sm btn-secondary border-white";
            } else {
                aelem.className = "btn btn-sm btn-primary border-white";
            }
            // aelem.className = "btn btn-outline-secondary";
            aelem.setAttribute("data-toggle", "tooltip");
            aelem.setAttribute("data-placement", "top");
            aelem.setAttribute("title", "Click to copy");

            aelem.setAttribute("slot_value", entity);
            aelem.onmousedown = copy_to_clipboard;
            aelem.onmouseout = reset_tooltip;
            row.cells[1].appendChild(aelem);
        }
    }
}


function copy_to_clipboard(event) {
    event.preventDefault();
    var act_elem = document.activeElement;
    if (act_elem.nodeName != 'INPUT') {
        return;
    }

    var btn = event.target;
    var slot_value = btn.getAttribute("slot_value");
    if (act_elem.value === '') {
        act_elem.value = slot_value;
    } else {
        act_elem.value = act_elem.value + ',' + slot_value;
    }
}


function reset_tooltip(event) {
    var btn = event.target;
    btn.title = "Click to copy";
}
