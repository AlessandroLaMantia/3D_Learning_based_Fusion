# WeatherFusion — pedoni nella pioggia

**Alessandro La Mantia · Learning-based Data Fusion**

Obiettivo: misurare se adattare una fusione visibile–termico aiuta a riconoscere più pedoni sotto pioggia simulata. Il confronto usa vere annotazioni, tenendo identico e congelato il detector.

## Avvio

Per scaricare il progetto completo da GitHub, installa Git LFS e poi esegui:

```powershell
git lfs install
git clone https://github.com/AlessandroLaMantia/3D_Learning_based_Fusion.git
cd 3D_Learning_based_Fusion
git lfs pull
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-meteo-lock.txt
```

I pesi e le cache `.pt`/`.pth` sono inclusi tramite Git LFS; l'ambiente `.venv` va creato localmente. L'interfaccia Streamlit è in inglese. Le presentazioni sono disponibili in italiano e in inglese nella cartella `presentazione`.

Apri `AVVIA_DEMO.cmd`, oppure:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app_pioggia.py --server.port 64001
```

Demo: http://127.0.0.1:64001. La prima scheda mostra risultati aggregati; la seconda permette di esplorare tutte le 50 scene con box vere, previsioni e mappe di fusione; la terza spiega l'apprendimento. I risultati e le feature sono già salvati: la demo non ripete il training.

## Dati e fine-tuning realmente eseguiti

Sono state scaricate **170 coppie reali LLVIP** con annotazioni XML dei pedoni: **100 train, 20 validation, 50 test**. Il conteggio delle box si trova nel manifest e nel report. Il dataset completo contiene 15.488 coppie; abbiamo letto l'indice dell'archivio remoto e scaricato solo i file necessari, verificandone CRC e SHA-256. Manifest e provenienza: `data/llvip_subset/`.

Gli ID di scena scelti per train, validation e test sono disgiunti, come verificato dal codice. Non è certificata l'assenza di LLVIP dal pretraining degli autori di AMG-Fuse. La separazione garantita riguarda il nostro fine-tuning.

**Il meteo è simulato:** strisce di pioggia, leggera sfocatura e attenuazione del visibile; il termico resta invariato. Intensità 0,4–0,9 nel train e 0,7 in validation e test. LLVIP è soprattutto un dataset di scene poco illuminate, non un dataset di pioggia reale. Non generalizziamo questi risultati a riprese piovose reali.

Abbiamo eseguito **3 epoche**, aggiornando **2,592 pesi** dello strato finale originale di AMG-Fuse. Il confronto valuta il modello realmente adattato dopo le **3 epoche prefissate**, anche se peggiora. La validation raccomanda l'**epoca 0** tramite AP50: zero significa conservare l'originale. La raccomandazione resta distinta dall'effetto sperimentale del fine-tuning. Il test non sceglie il modello da mostrare.

## Come funziona la rete

Il punto di partenza è il checkpoint ufficiale **AMG-Fuse Haze**, già specializzato nella fusione per foschia, trasferito qui alla pioggia simulata. Non viene addestrata una rete casuale. Tutte le feature congelate e i coefficienti iniziali dell'ultimo strato provengono dal checkpoint originale.

Prima del confronto viene preparato un detector comune: la sola testa di YOLOv8n COCO è adattata per sei epoche sulle stesse 100 coppie, alternando visibile, termico e media fissa. La validation sceglie anche i suoi pesi. Poi l'intero detector viene congelato per tutti e tre i confronti e per il fine-tuning della fusione. Non vengono aggiunte immagini di training. Il test finale esclude i 100 ID usati nelle prove diagnostiche iniziali; l'elenco necessario a riprodurre questa separazione è in `data/llvip_subset/initial_trial_test_ids.json`.

Controllo sul test finale, mantenendo la fusione originale: senza la preparazione comune del detector si trovano **41 pedoni**, con AP50 **40.08%**; dopo la sua preparazione si trovano **73 pedoni**, con AP50 **66.79%**. Questo controllo cambia il detector e non va attribuito al fine-tuning della fusione. Il confronto principale sottostante mantiene invece il detector fisso.

Per lavorare su questa CPU usiamo una **variante guidata da AMG-Fuse**, distinta dalla sua uscita nativa a piena risoluzione:

1. La parte congelata di AMG-Fuse estrae feature a 40 × 32 pixel, salvate in cache FP16 e rilette in FP32.
2. Lo strato finale originale, una convoluzione 96 → 3, ricostruisce un risultato ridotto. Solo i suoi 2.592 coefficienti vengono adattati.
3. Dalle distanze quadratiche di quel risultato rispetto ai due ingressi ridotti ricaviamo α = sigmoid(4 × (distanza_termico − distanza_visibile)). Ingrandiamo α con interpolazione bilineare.
4. Combiniamo gli ingressi originali a 320 × 256 pixel: F = αV + (1−α)T. Questo conserva i dettagli delle misure. α è un coefficiente di combinazione, non una probabilità di affidabilità dei sensori.
5. Un **YOLOv8n COCO preaddestrato** cerca i pedoni. La loss di classificazione e posizione, calcolata sulle vere box, attraversa il detector congelato e aggiorna solo lo strato della fusione. Si aggiunge 0,1 volte l'errore quadratico tra α e la sua versione iniziale come regolarizzazione.

La media fissa impone α = 0,5. Le due varianti neurali usano identica architettura e identiche feature congelate: cambiano soltanto i coefficienti dello strato finale. Il detector mantiene sempre identici pesi e statistiche, verificati da un test con aggiornamento reale del gradiente. **“Senza NN” riguarda la fusione:** il detector è neurale in tutti e tre i casi.

## Risultati sulle 50 immagini di test

| Fusione | Pedoni trovati | Mancati | Falsi positivi | Recall | Precision | AP50 |
|---|---:|---:|---:|---:|---:|---:|
| Media fissa | 67 | 61 | 16 | 52.34% | 80.72% | 57.01% |
| Prima del fine-tuning | 73 | 55 | 11 | 57.03% | 86.90% | 66.79% |
| Dopo il fine-tuning | 84 | 44 | 17 | 65.62% | 83.17% | 71.28% |


Differenza dopo − prima: **+11 pedoni correttamente trovati** e **+8.59 punti percentuali di recall**. Va letta insieme ai falsi positivi e ad AP50; un maggior numero di proposte non è automaticamente un miglioramento.

Matching uno a uno, IoU ≥ 0,50. Conteggi, precision e recall alla soglia condivisa 0.30, scelta sulla validation massimizzando F1 con la fusione originale, prima del fine-tuning della fusione. YOLO riceve le immagini ingrandite a 640 × 512; le box sono riportate nelle coordinate 320 × 256. AP50 integra l'inviluppo della curva precision–recall sulle proposte con score ≥ 0,05 e massimo 100 proposte per immagine. È AP50 di questo esperimento a una classe, non COCO mAP 0,50:0,95. La confidence di una box non è l'accuratezza del sistema.

`runs/pioggia_llvip_yolo/results.json` conserva tutte le box vere e previste, split, metadati e metriche. `training.json` contiene la cronologia della validation. `evaluated_head.pt` conserva lo strato dopo le tre epoche realmente confrontato. `best.pt` e `best_head.pt` conservano invece la raccomandazione della validation, che può restare l'epoca zero. Non si sostituisce il risultato negativo di un modello adattato con una copia dell'originale.

## Presentazione e spiegazione

- `presentazione/WeatherFusion.pdf`: 15 slide aggiornate, sfondo chiaro, autore in apertura e ringraziamento finale.
- `presentazione/english/WeatherFusion.pdf`: stessa presentazione in inglese, con note in `presentazione/english/Speaker_notes.txt`.
- `Spiegazione_slides.txt`: obiettivo, punto di partenza e spiegazione di tutte le slide attuali.


## Ripetere l'esperimento su questo PC

Ambiente configurato: Python 3.11, PyTorch CPU e Torchvision compatibile. Le risorse sono già presenti.

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-meteo.txt
.\.venv\Scripts\python.exe tools/download_llvip_subset.py
.\.venv\Scripts\python.exe prepare_rain_detector.py
.\.venv\Scripts\python.exe train_rain_detection.py --epochs 3
.\.venv\Scripts\python.exe tools/build_rain_presentation.py
```

Il download viene ripreso usando i file già verificati; le feature congelate vengono riutilizzate. Il training registra il progresso in `runs/pioggia_llvip_yolo/progress.log`. Per un nuovo ambiente occorrono anche i pesi AMG-Fuse e YOLOv8n; le loro fonti e gli hash sono nel report e nei file di provenienza. Non cambiare manifest o pesi conservando una cache precedente: il codice ne verifica la corrispondenza.

## Fonti

- [LLVIP, dataset e annotazioni](https://github.com/bupt-ai-cz/LLVIP): Jia et al., ICCV Workshops 2021. Uso non commerciale per ricerca e didattica; licenza in `data/llvip_subset/LICENSE.md`.
- [Mirror LLVIP utilizzato](https://huggingface.co/datasets/McCheng/LLVIP), revisione e50976a; seed 17 per train/validation e 2028 per il test finale, escludendo gli ID dei test diagnostici.
- [AMG-Fuse, codice e checkpoint ufficiali](https://github.com/ixilai/AMG-Fuse); codice originale con licenza nella cartella vendor.
- [YOLOv8n e pesi ufficiali](https://docs.ultralytics.com/models/yolov8/).
