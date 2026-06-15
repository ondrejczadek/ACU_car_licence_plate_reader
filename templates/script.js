async function updateHistory() {
    try {
        const res = await fetch('/api/history');
        const data = await res.json();
        const el = document.getElementById('history');
        if (data.length === 0) {
            el.innerHTML = '<tr><td colspan="5" class="prazdno">Zadna data</td></tr>';
            return;
        }
        el.innerHTML = data.map((d, i) => {
            const t = new Date(d.timestamp);
            const datum = t.toLocaleDateString('cs-CZ') + '  ' + t.toLocaleTimeString('cs-CZ');
            return '<tr>' +
                '<td class="col-num">' + (i + 1) + '</td>' +
                '<td class="col-spz">' + d.text + '</td>' +
                '<td class="col-datum">' + datum + '</td>' +
                '<td class="col-pocet">' + d.count + 'x</td>' +
                '<td class="col-akce"><button class="smazat" onclick="smazat(\'' + d.text + '\')">&times;</button></td>' +
                '</tr>';
        }).join('');
    } catch (e) { }
}

async function smazat(text) {
    await fetch('/api/history/' + encodeURIComponent(text), { method: 'DELETE' });
    updateHistory();
}

document.getElementById('videoFeed').onerror = function () {
    setTimeout(() => { this.src = '/video_feed?' + Date.now(); }, 2000);
};

setInterval(updateHistory, 2000);
updateHistory();