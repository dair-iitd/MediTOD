class ProgressBar {
    constructor(uttr_cnt) {
        this.completed = 0;
        this.total = uttr_cnt;
        this.p_bar = document.getElementById('progress_bar');
        this.p_bar.style.width = '0%';
        this.p_bar.setAttribute('aria-valuenow', '0');
        this.p_bar.setAttribute('aria-valuemin', '0');
        this.p_bar.setAttribute('aria-valuemax', uttr_cnt.toString());
        this.p_bar.innerHTML = '0 / ' + uttr_cnt.toString();
    }

    increment_bar() {
        this.completed++;

        var perc = 100.0 * this.completed / this.total;
        this.p_bar.style.width = perc.toString() + '%';
        this.p_bar.setAttribute('aria-valuenow', this.completed.toString());
        this.p_bar.innerHTML = this.completed.toString() + ' / ' + this.total.toString();
    }

    decrement_bar() {
        this.completed--;

        var perc = 100.0 * this.completed / this.total;
        this.p_bar.style.width = perc.toString() + '%';
        this.p_bar.setAttribute('aria-valuenow', this.completed.toString());
        this.p_bar.innerHTML = this.completed.toString() + ' / ' + this.total.toString();
    }
}

var global_progress_bar;
