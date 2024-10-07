var drop_down_sel_idx = -1;


function update_input_value(current_value, new_value) {
    var tokens = current_value.split(',')

    if (tokens.includes(new_value)) {
        return current_value;
    } 
    var ret = '';

    for (var ii = 0; ii < tokens.length - 1; ii++) {
        tok = tokens[ii];
        if (tok.length > 0) {
            ret += tok + ',';
        }
    }
    ret += new_value;

    return ret ;
}


function get_custom_dropdown_element(outer_div, clstype) {
    var content_div = null;
    for (var ii = 0; ii < outer_div.children.length; ii++) {
        if (outer_div.children[ii].classList.contains(clstype)) {
            content_div = outer_div.children[ii];
            break;
        }
    }

    return content_div;
}


function custom_dropdown_input_onfocus(event) {
    drop_down_sel_idx = -1;
    var content_div = get_custom_dropdown_element(
        event.target.parentElement, "custom_dropdown-content"
    );
    content_div.classList.remove("d-none");
}


function custom_dropdown_input_onblur(event) {
    // We wait for 500ms for any option click event. Not ideal but works.
    setTimeout(function () {
        var content_div = get_custom_dropdown_element(
            event.target.parentElement, "custom_dropdown-content"
        );
        content_div.classList.add("d-none");
    }, 400);
}


function custom_dropdown_entry_click(event) {
    // Very fragile code. This will not work if <a> contains HTML other than <b>
    var outer_div = null;
    var value = null;
    // console.log(event.target.nodeName);
    if (event.target.nodeName === 'A') {
        outer_div = event.target.parentElement.parentElement;
        value = event.target.getAttribute('value');
    } else if (event.target.nodeName === 'B') {
        outer_div = event.target.parentElement.parentElement.parentElement;
        value = event.target.parentElement.getAttribute('value');
    }

    var input_div = get_custom_dropdown_element(outer_div, "custom_dropdown_input");
    input_div.value = update_input_value(input_div.value, value);
}


function custom_dropdown_keydown(event) {
    var key = event.which;

    if (key != 38 && key != 40 && key != 13) {
        return;
    }

    var input = event.target;
    var content_div = get_custom_dropdown_element(
        input.parentElement, "custom_dropdown-content"
    );
    var a = content_div.getElementsByTagName("a");

    var vis_idx = [];
    for (i = 0; i < a.length; i++) {
        txtValue = a[i].textContent || a[i].innerText;
        if (a[i].style.display != "none") {
            vis_idx.push(i);
        }
    }

    if (key == 38) {
        drop_down_sel_idx--;
        if (drop_down_sel_idx < 0) {
            drop_down_sel_idx = 0;
        }
    } else if (key == 40) {
        drop_down_sel_idx++;
        if (drop_down_sel_idx >= a.length) {
            drop_down_sel_idx = a.length - 1;
        }
    } else {
        if (drop_down_sel_idx == -1) {
            return;
        }
        if (a[vis_idx[drop_down_sel_idx]].classList.contains('custom_dropdown_entry-highlight')) {
            a[vis_idx[drop_down_sel_idx]].classList.remove('custom_dropdown_entry-highlight');
        }
        value = a[vis_idx[drop_down_sel_idx]].getAttribute('value');
        input.value = update_input_value(input.value, value);
        // input.blur();
        input.focus();
        return;
    }

    for (i = 0; i < vis_idx.length; i++) {
        if (i == drop_down_sel_idx) {
            a[vis_idx[i]].classList.add('custom_dropdown_entry-highlight');
        } else {
            if (a[vis_idx[i]].classList.contains('custom_dropdown_entry-highlight')) {
                a[vis_idx[i]].classList.remove('custom_dropdown_entry-highlight');
            }
        }
    }

    var topPos = a[vis_idx[drop_down_sel_idx]].offsetTop;
    var div_ht = content_div.offsetHeight;
    if (topPos < content_div.scrollTop) {
        content_div.scrollTop = topPos;
    } else {
        var diff = topPos - content_div.scrollTop;
        if (diff > div_ht) {
            content_div.scrollTop = topPos;
        }
    }
}


class FlexSearchController {
    constructor() {
        this.indices = {};
        // Allowing spelling errors upto this many characters
        this.tolerance = 1;
    }

    add_index(tag, values) {
        if (this.indices.hasOwnProperty(tag)) {
            return;
        }

        var index = lunr(function () {
            this.ref('idx')
            this.field('text')

            var cid = 0;
            var str;
            for (var val of values) {
                var str = val.code;
                if (val.hasOwnProperty("description")) {
                    str = str + " " + val.description;
                }

                if (val.hasOwnProperty("keywords")) {
                    for (var kw of val.keywords) {
                        str = str + " " + kw;
                    }
                }
                
                this.add({
                    'idx': cid, 'text': str
                });
                cid++;
            }    
        });
        this.indices[tag] = index;
    }

    search(tag, query) {
        if (!this.indices.hasOwnProperty(tag)) {
            console.log("Can not search. Index is missing.", tag);
            return;
        }
        var index = this.indices[tag];
        // var results = index.search('*' + query + '*' + '~' + this.tolerance);
        var results = index.search('*' + query + '*');

        var ret = [];
        for (var ee of results) {
            ret.push(parseInt(ee.ref));
        }

        return ret;
    }
}

var search_controller = new FlexSearchController();

function custom_dropdown_keyup(event) {
    if (event.which == 38 || event.which == 40 || event.which == 13) {
        return;
    }

    var input = event.target;
    var query = input.value;

    var tokens = query.split(',');
    query = tokens[tokens.length - 1];

    var match_idx = null;
    drop_down_sel_idx = -1;

    var content_div = get_custom_dropdown_element(
        input.parentElement, "custom_dropdown-content"
    );
    var a = content_div.getElementsByTagName("a");

    if (query.length > search_controller.tolerance) {
        var search_tag = input.parentElement.getAttribute("search_tag");
        match_idx = search_controller.search(search_tag, query);
    } else {
        match_idx = [];
        filter = query.toUpperCase();
        for (i = 0; i < a.length; i++) {
            txtValue = a[i].textContent || a[i].innerText;            
            if (txtValue.toUpperCase().indexOf(filter) > -1) {
                match_idx.push(i)
            }
        }
    }

    for (i = 0; i < a.length; i++) {
        if (match_idx.includes(i)) {
            a[i].style.display = "";
        } else {
            a[i].style.display = "none";
        }
    }
}
