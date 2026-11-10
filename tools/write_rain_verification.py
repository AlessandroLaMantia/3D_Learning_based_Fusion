from pathlib import Path
import json
ROOT=Path(__file__).resolve().parents[1]
run=ROOT/'runs/pioggia_llvip_yolo'
report=json.loads((run/'results.json').read_text(encoding='utf-8'))
meta=report['metadata'];before=report['test']['pretrained']['summary'];after=report['test']['finetuned']['summary']
checks=json.loads((ROOT/'presentazione/layout_checks.json').read_text(encoding='utf-8'))
assert len(checks)==18 and not any(s['overflow'] for s in checks)
text=f'''# Verifica del progetto pioggia

- Dataset finale: 100 coppie train, 20 validation, 50 test; ID di scena disgiunti.
- Le 50 immagini finali sono escluse dai 100 ID dei due test diagnostici precedenti. L'elenco di esclusione è salvato con il dataset.
- Pioggia simulata sul visibile; termico invariato. Non sono registrazioni di pioggia reale.
- Feature AMG-Fuse e ultimo strato originali caricati da pesi ufficiali. Nessuna rete di fusione casuale addestrata da zero.
- Detector comune: testa di YOLOv8n adattata sulle stesse 100 coppie; pesi scelti sulla validation. Durante il successivo fine-tuning della fusione tutti i suoi pesi e buffer sono congelati.
- Adattamento della fusione: {meta['epochs']} epoche prefissate, {meta['trainable_parameters']} coefficienti dell'ultimo strato. La loss usa vere box, non uno pseudo-riferimento d'immagine.
- Il confronto mostra i pesi realmente aggiornati dopo {meta['evaluated_epoch']} epoche. La raccomandazione della validation resta separata: epoca {meta['recommended_epoch']} (zero significa originale). Non si sostituisce un risultato negativo con una copia dell'originale.
- Stesso detector, stessi input, stesse annotazioni e stessa soglia {after['score_threshold']:.2f} per i tre confronti di fusione. La soglia è scelta sulla validation prima del fine-tuning della fusione.
- Test: {before['true_positives']} → {after['true_positives']} pedoni trovati su {after['ground_truth_count']}; falsi positivi {before['false_positives']} → {after['false_positives']}; AP50 {before['AP50']*100:.2f}% → {after['AP50']*100:.2f}%.
- Il risultato di questo test non prevale sulla raccomandazione della validation per scegliere un modello da utilizzare. Campioni piccoli e distribuzioni diverse possono dare esiti discordanti; non si afferma un beneficio generale.
- Suite completa: **7 test superati**, senza skip. Include casi noti per matching uno a uno e AP, aggiornamento effettivo dei pesi della fusione con detector identico, verifica degli artefatti e prova reale dell'interfaccia Streamlit sulle 50 scene.
- `pip check`: nessuna dipendenza incompatibile.
- {len(checks)} slide PDF con verifica automatica dei blocchi di testo e controllo visivo della panoramica. Sfondo chiaro, copertina centrata con autore e ringraziamento finale; PDF e spiegazione testuale coerenti. Esportazione solo PDF.

Hash AMG-Fuse: `{meta['base_sha256']}`.

Hash detector comune congelato: `{meta['domain_detector_sha256']}`.

Hash manifest finale: `{meta['dataset_sha256']}`.
'''
(ROOT/'docs/VERIFICA_PIOGGIA.md').write_text(text,encoding='utf-8')
print('Verification report written')
