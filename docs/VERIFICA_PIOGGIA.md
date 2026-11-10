# Verifica del progetto pioggia

- Dataset finale: 100 coppie train, 20 validation, 50 test; ID di scena disgiunti.
- Le 50 immagini finali sono escluse dai 100 ID dei due test diagnostici precedenti. L'elenco di esclusione è salvato con il dataset.
- Pioggia simulata sul visibile; termico invariato. Non sono registrazioni di pioggia reale.
- Feature AMG-Fuse e ultimo strato originali caricati da pesi ufficiali. Nessuna rete di fusione casuale addestrata da zero.
- Detector comune: testa di YOLOv8n adattata sulle stesse 100 coppie; pesi scelti sulla validation. Durante il successivo fine-tuning della fusione tutti i suoi pesi e buffer sono congelati.
- Adattamento della fusione: 3 epoche prefissate, 2592 coefficienti dell'ultimo strato. La loss usa vere box, non uno pseudo-riferimento d'immagine.
- Il confronto mostra i pesi realmente aggiornati dopo 3 epoche. La raccomandazione della validation resta separata: epoca 0 (zero significa originale). Non si sostituisce un risultato negativo con una copia dell'originale.
- Stesso detector, stessi input, stesse annotazioni e stessa soglia 0.30 per i tre confronti di fusione. La soglia è scelta sulla validation prima del fine-tuning della fusione.
- Test: 73 → 84 pedoni trovati su 128; falsi positivi 11 → 17; AP50 66.79% → 71.28%.
- Il risultato di questo test non prevale sulla raccomandazione della validation per scegliere un modello da utilizzare. Campioni piccoli e distribuzioni diverse possono dare esiti discordanti; non si afferma un beneficio generale.
- Suite completa: **7 test superati**, senza skip. Include casi noti per matching uno a uno e AP, aggiornamento effettivo dei pesi della fusione con detector identico, verifica degli artefatti e prova reale dell'interfaccia Streamlit sulle 50 scene.
- `pip check`: nessuna dipendenza incompatibile.
- 18 slide PDF con verifica automatica dei blocchi di testo e controllo visivo della panoramica. Sfondo chiaro, copertina centrata con autore e ringraziamento finale; PDF e spiegazione testuale coerenti. Esportazione solo PDF.

Hash AMG-Fuse: `1598a25c6b3e08f5819b26b618afb0bf74fc94a26f6576a60fc78c7fcdd85a7b`.

Hash detector comune congelato: `0033aaef4b6bf270784be2db07d4fe25c2b6868835ed073f620a288f5901892b`.

Hash manifest finale: `fc47f826047ce5052f924104324e9e7505f00ab09ebc397ce4b0110cef0ebbda`.
