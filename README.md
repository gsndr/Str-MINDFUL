# STREAM MINDFUL

## Struttura del repository

    
        .
        ├── datasets       	           # contiene i vari file contenenti i dati necessari alla computazione (file .txt, .csv, .npy)
        │    ├ numeric         		# contiene i dataset sottoposti al preprocessing dei dati 
        │    ├ original        		# contiene i dataset originali contenenti i dati che modellano i flussi di rete
        │    └ utils           		# contiene tutti i file contenenti dati di supporto alla computazione 
        ├── models         		   # contiene i modelli di rappresentazione e classificazione dei dati
        ├── plot           		   # contiene tutti i grafici esplicanti i risultati ottenuti
        ├── results    		   # contiene i file .csv e .txt in cui risiedono i risultati sperimentali
        ├── tesi    		   # contiene pdf della tesi di laurea e presentazione per la seduta
        ├── AutoencoderHypersearch.py  # consultare versione base di mindful
        ├── CNN1D.py		   # consultare versione base di mindful
        ├── CNN Hypersearch.py	   # consultare versione base di mindful
        ├── CountWindow.py		   # finestra scorrevole basata sul numero di elementi al suo interno
        ├── DatasetConfig.py	   # consultare versione base di mindful
        ├── global_config.py	   # consultare versione base di mindful
        ├── main.py			   # file di lancio del programma
        ├── MINDFUL.conf		   # file di configurazione dell'esecuzione
        ├── PageHinkleyTest.py	   # implementazione del PageHinkley test 
        ├── Plot.py			   # consultare versione base di mindful
        ├── Preprocessing.py	   # consultare versione base di mindful
        ├── RunStream.py		   # implementazione dell'algoritmo di gestione dello stream di dati
        ├── StreamLearner.py	   # classe contenente tutte le funzioni necessarie all'aggiornamento della conoscenza del sistema e dei modelli
        ├── TimeStream.py		   # classe che istanzia un oggetto che simula uno stream a partire da una sorgente di dati (DataFrame)
        ├── TimeWindow.py		   # finestra scorrevole basata sull'intervallo temporale tra i valori timestamp dell'esempio più recente e più datato contenuti
        ├── requirments.txt		   # requisiti per funzionamento del codice
        └── README.md
All'interno di result vi è un sistema di sottocartelle per organizzare i risultati delle varie sperimentazioni

Tutti i file sorgente la cui descrizione è *"consultare versione base di mindful"*  provengono dalla versione base di MINDFUL e non sono stati quasi per nulla variati, salvo piccole modifiche per adattarli al programma stream (esempio: in DatasetConfig è stata aggiunta una funzione di preprocessing del dataset destinato a diventare stream)

### Valutazione

Per ogni sperimentazione, sulla base dei dati raccolti, sono stati stipulati i seguenti file:

    CICIDS2017_result_stream.csv	# valori delle metriche di valutazione e della matrice di confusione
    Informations on errors.png      # grafico riportante le informazioni sui drift e sugli errori di classificazione lungo tutto lo stream
    Results Stream Algorithm.txt	# file di log riportante le informazioni dell'esecuzione dell'algoritmo



## Installazione

    pip install -r requirements.txt

**Python  3.6.12**

Packeges:

* [Tensorflow 2.3.1](https://www.tensorflow.org/) 
* [Keras 2.4.3](https://github.com/keras-team/keras)
* [Matplotlib 3.3.2](https://matplotlib.org/)
* [Pandas 1.1.3](https://pandas.pydata.org/)
* [Numpy 1.18.5](https://www.numpy.org/)
* [Scikit-learn 0.23.2](https://scikit-learn.org/stable/)
* [Scikit-multiflow 0.5.3](https://scikit-multiflow.github.io/)
* [Hyperas 0.4.1](https://github.com/maxpumperla/hyperas)
* [Hyperopt 0.2.5](https://github.com/hyperopt/hyperopt)
* [Pillow 7.2.0](https://pillow.readthedocs.io/en/stable/)



## Utilizzo

Per configurare l'esecuzione dell'algoritmo è necessario impostare i parametri dell'area *settings* di MINDFUL.conf. I dettagli sono qui spiegati: 

    [setting]
    EXECUTION_TYPE              # 0=>esecuzione batch 1=>esecuzione streaming
    N_CLASSES                   # numero classi che il classificatore distingue
    PREPROCESSING1              # 0=>non eseguire preprocessing 1=>esegui preprocessing
    LOAD_AUTOENCODER_ADV        # 0=>addestra autoencoder attachi da zero  1=> carica autoencoder
    LOAD_AUTOENCODER_NORMAL     # 0=>addestra autoencoder normali da zero  1=> carica autoencoder
    LOAD_CNN                    # 0=>addestra CNN1D da zero  1=> carica CNN1D
    VALIDATION_SPLIT            # percentuale split train/validation
    WINDOW_TYPE                 # 0=>count window  1=>time window
    DECREASE_BATCH_DATASET      # 0=>utilizza dataset batch completo durante l'app stream  1=>riduci la dimensione del dataset batch 

Oltre alla configurazione dei valori per l'esecuzione, è necessario anche inserire le informazioni relative al dataset che si intende utilizzare:

    [nome_dataset]
    pathModels          # path in cui contenere i modelli
    pathPlot            # path in cui inserire i grafici
    pathDataset         # path in cui sono contenuti i dataset originali
    path                # nome del file del dataset batch
    pathTest            # nome del file del dataset usato come test nella versione base
    pathStream          # nome del file del dataset utilizzato per lo streaming
    testPath            # coincide col nome del dataset
    pathDatasetNumeric  # path in cui contenere i file dei dataset numerici preprocessati
    pathUtils           # path della cartella in cui inserire i file di supporto alla computazione streaming
    countWindowNormal   # dimensione della countWindow degli esempi normali
    countWindowAttack   # dimensione della countWindow degli esempi di attacco
    smallerSize         # dimensione del dataset batch dopo il campionamento
    timeDeltaNormal     # lasso di tempo consentito nella timeWindow degli esempi normali
    timeDeltaAttack     # lasso di tempo consentito nella timeWindow degli esempi di attaco

Una volta impostati tutti i valori necessari all'interno del file di configurazione, per la corretta esecuzione dell'algoritmo stream sono necessarie due fasi: un addestramento preliminare dei modelli di rappresentazione e classificazione dei dati e l'esecuzione della fase streaming.

### Addestramento preliminare

Impostare il valore *EXECUTION_TYPE=0*, il valore *path* con il nome del dataset che si intende utilizzare come batch e lanciare main.py utilizzando come argomento il nome del dataset di input desiderato.

    python3 main.py nome_dataset

In questo modo verrà eseguita la versione base di MINDFUL, la quale si occuperà dell'addestramento preliminare dei modelli, attraverso la scelta dei migliori iperparametri. Se si desiderano ulteriori informazioni su questa fase è possibile consultare [questo repository](https://github.com/gsndr/MINDFUL)

### Esecuzione app streaming

Una volta ottenuti i modelli addestrati con conoscenza batch è possibile impostare *EXECUTION_TYPE=1*, il valore di *pathStream* con il nome del dataset che si vuole rendere in forma stream e lanciare nuovamente main.py con il comando precedente.

NOTA BENE: La versione corrente di STREAM MINDFUL è stata implementata sulla base dell'utilizzo del dataset CICIDS2017. Se si desidera utilizzare un altro dataset è necessario aggiungere nella funzione *Datasetconfig.preprocessing2()* l'implementazione del preprocessing del dataset desiderato, in modo che esso possa creare un dataset numerico standardizzato unico, ad hoc per il sistema streaming

#### Preprocessing

Durante la fase di preprocessing i dati contenuti nel dataset vengono standardizzati e ripuliti da eventuali valori nulli o infiniti. In seguito i dati vengono ordinati sulla base dei valori contenuti nella colonna Timestemp in modo che, quando il set di dati verrà tramutato in sorgente streaming, i dati forniti seguiranno un andamento temporale come succede nella realtà. Successivamente, i valori temporali vengono separati dal dataset e salvati in un file .npy in *dataset/nome_dataset/utils/*. Infine i dati risultanti vengono passati alla classe *TimeStream* per gestiti tramite un flusso.

#### Algoritmo streaming

Il funzionamento dell'algoritmo streaming è esplicato dal seguente pseudo codice:

    PER OGNI ESEMPIO DELLO STREAM:
        esempio, etichetta, orario <- stream.leggere esempio()
        normPred <- autoencoderN.predizione(esempio)
        attPred <- autoencoderA.predizione(esempio)
        immagine <- [esempio, normPred, attPred]
        etichettaPred <- CNN1D.predizione(immagine)          
        SE (etichetta == Normale):
                finestraN.aggiungi(esempio , etichettaPred, orario)
                PHN <- PHT.Page Hinkley Test Normal( mse(esempio, normPred) )
                SE (PHN == True):
                         tipo <- 'normal'
                         datasetTrain <- aggiorna dataset(datasetTrain, finestraN, tipo)
                         autoencoderN <- addestra autoencoder(datasetTrain, autoencoderN, tipo)
                         CNN1D <- addestra CNN1D(datasetTrain, CNN1D, autoencoderN, autoencoderA)
                         Errori Normali <- calcola Errori(datasetTrain, autoencoderN, tipo)
                         Errori Classificazione <- calcola Errori(datasetTrain, CNN1D, autoencoderN, autoencoderA)
                         PHT.Inizializza PHT Normali(Errori Normali, Errori Classificazione)
        ALTRIMENTI:
                finestraA.aggiungi(esempio , etichettaPred, orario)
                PHA <- PHT.Page Hinkley Test Attack( mse(esempio, attPred) )
                SE (PHN == True):
                         tipo <- 'attack'
                         datasetTrain <- aggiorna dataset(datasetTrain, finestraN, tipo)
                         autoencoderA <- addestra autoencoder(datasetTrain, autoencoderA, tipo)
                         CNN1D <- addestra CNN1D(datasetTrain, CNN1D, autoencoderN, autoencoderA)
                         Errori Attacchi <- calcola Errori(datasetTrain, autoencoderA, tipo)
                         Errori Classificazione <- calcola Errori(datasetTrain, CNN1D, autoencoderN, autoencoderA)
                         PHT.Inizializza PHT Attacchi(Errori Attacchi, Errori Classificazione)
        SE (PHN == False and PHA == False)
                PHC <- Page Hinkley Test Classifier( abs(etichetta - etichettaPred) )
                SE (PHC == True):
                         tipo <- 'classifier'
                         datasetTrain <- aggiorna dataset(datasetTrian, finestraN, finestraA)
                         CNN1D <- addestra CNN1D(datasetTrain, CNN1D, autoencoderN, autoencoderA)
                         Errori Classificazione <- calcola Errori(datasetTrain, CNN1D, autoencoderN, autoencoderA)
                         PHT.Inizializza PHT Classifizione(Errori Classificazione)

Per ogni esempio proveniente dallo stream, il sistema esegue due fasi: una fase di analisi dell'esempio e una di valutazione delle performance del sistema stesso.

Durante la fase di analisi, lo stream fornisce al sistema un esempio in formato [x, y, t],dove t è il valore temporale nel quale si verifica l’esempio. Vengono così costruite le ricostruzioni di x per mezzo degli autpencoder, le quali verranno integrate con l’esempio originale per costruire l’esempio multicanale, secondo la stessa metodologia utilizzata dalla versione base di MINDFUL. Successivamente l’esempio multicanale viene classificato per mezzo della CNN1D per ottenere l’etichetta predetta, concludendo la fase di analisi dell’esempio. Le etichette originali e quelle predette vengono salvate su un file di log man mano che vengono lette/generate.

Durante la fase di valutazione delle peformance del sistema, vengono effettuati i test di Page Hinkley per determinare eventuali concept drift nel monitoraggio degli errori di ricostruzione degli autoencoder o di classificazione della CNN1D. Nel momento in cui dovesse insorgere un concept drift causato da uno degli autoencoder, quest'ultimo e la CNN1D verranno readdestrati con il dataset batch aggiornato con gli esempi contenuti nella relativa finestra scorrevole fino a quel momento; nel caso in cui, invece il drift dovesse essere causato direttamente dal classificatore, solo la CNN1D verrà readdestrata con il dataset aggiornato mediante entrambe le finestre scorrevoli. Durante l'esecuzione del programma, tutti i valori degli errori effettuati dal sistema vengono salvati su dei file di log, al fine di generare i relativi grafici alla fine della computazione.

#### Valutazione 

La valutazione finale delle prestazioni dell'algoritmo viene effettuata mediante matrice di confusione costruita attraverso i dati recuperati dai due file di log contenenti le etichette reali degli esempi e le etichette predette dalla CNN1D. Dalla matrice saranno poi calcolate le metriche *OverAll Accuracy*, *Average Accuracy*, *Precision*, *Recall*, *F1-score*, *False Allarm Rate*. Sia la matrice di confusione che le metriche verranno salvate sia sul file testuale di output del programma che su un documento .csv.

#### File necessari all'esecuzione dell'app streaming

Per rendere possibile l'esecuzione dell'app streaming sono necessari i seguenti file:

* Il dataset originale dei dati streaming nella directory *datasets/nome_dataset/original/*
* Il dataset batch numerico con cui sono stati addestrati preliminarmente i modelli, preprocessato dalla versione base di MINDFUL, nella directory *datasets/nome_dataset/numeric/*
* I tre modelli preaddestrati dalla versione base di MINDFUL. I file sono *autoencoderAttacks.h5*, *autoencoderNormal.h5* e *MINDFUL.h5*, tutti nella directory *models/nome_dataset/*
* I tre file .npy con gli errori di ricostruzione e di classificazione effettuati dai modelli durante l'esecuzione della versione base di MINDFUL, i quali verranno utilizzati per l'inizializzazione delle istante della classe *PageHinkleytest*. I tre file devono trovarsi all'interno di *datasets/nome_dataset/utils/*.

Infine bisogna ricordare che, se nel file di configurazione si imposta *PREPROCESSING1=0*, sono necessari anche i seguenti file:

* Il dataset numerico dei dati streaming, preprocessato da una esecuzione precedentedell'app streaming, nella directory *datasets/nome_dataset/numeric/*
* Il file .npy con i valori timestamp ordinati relativi ai dati del dataset streaming, nella directory *datasets/nome_dataset/utils/*.

### Classi e funzioni implementate

#### RunStream
Questa è la classe di controllo del sistema che contiene l'esecuzione dell'intero algoritmo, la quale è inserita nel metodo **Run()**. Il costruttore della classe necessita dei parametri *dsConfig* e *config*, ossia rispettivamente le configurazioni del dataset di input e del processo sopra indicate, ricavate da *MINDFUL.conf*

#### TimeStream
Questa classe è la responsabile della creazione e della gestione dell'oggetto che simula lo stream di dati. Nel costruttore, il quale riceve come parametri *dsConf* e *conf*, la classe legge il nome del dataset da tramutare in stream (dal campo *pathStream* del file di configurazione), carica il relativo file csv, effettua la pre-elaborazione dei dati e costruisce lo stream. I metodi **getNextExamples()** e **hasMoreExamples()** permettono, rispettivamente, di ottenere un nuovo esempio dallo stream e di controllare se lo stream contiene almeno un esempio.

#### PageHinkleyTest
Le istanze di questa classe hanno il compito di eseguire il Page Hinkley Test. Il costruttore riceve come parametri *dsConf*, ossia la configurazione del dataset di input, e *classLabel*, una variabile String che può assumere i valori in {Normal, Attack, Classifier}, la quale specifica su quale modello viene eseguito il test; inoltre instanzia i riferimenti a due file di log, i quali vengono usati iterativamente per raccogliere le informazioni relative agli errori del modello monitorato e a eventuali concept drift.
Il metodo **test(example, prediction)** esegue il PHTest calcolando l'errore commesso dal modello monitorato, il quale si ottiene calcolando l'mse tra *example* e *prediction* se *classLabel* indica uno dei due autoencoder o il valore assoluto della differenza altrimenti, e confrontandolo con gli errori fino ad allora registrati. *example* è un esempio di flusso originale o una label originale ottenuto dallo stream, mentre *prediction* è lo stesso esempio ricostruito da un autoencoder o la label predetta dalla CNN1D. Il metodo restituisce *True* se il PH test ha esito positivo, *False* altrimenti.
Il metodo **updatePHT(errors)** viene richiamato dopo ogni riaddestramento di un modello del sistema e serve a reinizializzare il PH test con i dati contenuti nel parametro *errors*, ossia un array contenente i nuovi errori effettuati dal modello monitorato riaddestrato sulla conoscenza aggiornata.

#### CountWindow
Le istanze di questa classe sono le finestre scorrevoli di cui fa uso l'algoritmo per mantenere dinamicamente gli esempi provenienti dallo stream. Il costruttore riceve come parametri *dsConf*, da cui ricava la dimensione massima da impostare, e *classLabel*, parametro che specifica se l'istanza immagazzinerà esempi normali o di attacco. Il metodo **insertExample(example, prediction, time)** inserisce un nuovo esempio, la sua etichetta predetta dalla CNN1D e il suo valore TimeStamp nella finestra. Prima dell'inserimento, il metodo effettua un controllo: se la finestra ha raggiunto la massima dimensione, l'esempio in coda viene scartato per far posto al nuovo esempio in testa. Quando risulta necessario aggiornare la conoscenza del sistema, il metodo **getWindow(columns)** costruisce e restituisce un dataframe Pandas, le cui colonne sono quelle specificate nel parametro *columns* e le righe sono tutti gli esempi che erano in quel momento contenuti nella finestra.

#### TimeWindow
Le istanze di questa classe sono finestre scorrevoli che seguono la stessa funzionalita delle CountWindow. La differenza sta nella strategia di mantenimento degli esempi, la quale, in questo caso, utilizza l'intervallo di tempo tra il primo e l'ultimo esempio contenuti per delimitare la quantità di elementi nella finestra. Se tale intervallo eccede un certo limite, specificato dal file di configurazione, vengono rimossi gli esempi necessari per rietrarvici.

#### Stream Learner
I metodi di questa classe sono i responsabili delle procedure necessarie al riaddestramento dei modelli del sistema. Il costruttore, riceve il parametro *dsConf*, ossia le informazioni relative alla configurazione del dataset di input. Durante la costruzione dell'istanza, vengono anche caricati i valori batch size dei modelli dai csv dei risultati della versione base di MINDFUL; tali valori verranno impostati come parametri batch size nel momento in cui i modelli verranno readdestrati.
I metodi **loadDataset()** e **saveDataset(dataset)** servono rispettivamente al caricamento e al salvataggio su disco del dataset con tenente la conoscenza del sistema, nel momento in cui quest'ultima debba essere aggiornata.
Il metodo **updateTrainingSet(dataset, window, classLabel)** si occupa dell'aggiornamento della conoscenza del sistema, ricevuta dal parametro *dataset*. Gli esempi contenuti in *dataset* vengono scissi nei sottoinsiemi di esempi normali ed esempi d'attacco. Da uno dei due sottoinsiemi, scelto sulla base del valore di *classLabel*, vengono successivamente eliminati casualmente un numero di esempi uguale al numero di esempi contenuti in quel momento nella finestra scorrevole *window*. Eseguita l'eliminazione, vengono nuovamente concatenati i sottoinsiemi degli esempi normali e d'attacco a cui si aggiungono gli esempi ottenuti dalla finestra scorrevole.
Il metodo **updateAutoencoder(autoencoder, dataX, dataY, classLabel)** riceve in input uno degli autoencoder, tramite il parametro *autoencoder*, e lo riaddestra utilizzando i dati di training *dataX* e le etichette *dataY* e eseguendo una divisione fra training set e validation set del 20%. Il parametro *classLabel* specifica se l'autoencoder in input è l'autoencoder degli attacchi o degli esempi normali, informazione necessaria per sapere quale valore batch size selezionare. Il metodo restituisce l'autoencoder riaddestrato.
Il metodo **updateCNN1D(CNN1D, imageX, dataY)** riceve in input la CNN1D, tramite il parametro *CNN1D*, e la riaddestra utilizzando l'immagine multicanale *imageX*, le etichette *dataY* e eseguendo una divisione fra training set e validation set del 20%. Il metodo restituisce il classificatore riaddestrato.
Il metodo **getAutoencoderErrors(autoencoder, dataX, classLabel)** calcola gli errori di ricostruzione dell'autoencoder, passato in input tramite *autoencoder*, sugli esempi contenuti nel parametro array *dataX*. L'autoencoder ricostruisce gli esempi in input, per poi calcolare gli mse su gli esempi originali e quelli ricostruiti. Gli mse calcolati vengono restituiti in output.
Il metodo **getCNN1DErrors(self, CNN1D, dataX, dataY)** calcola gli errori di classificazione effettuati dal classificatore passato in input, tramite *CNN1D*. Il classificatore predice gli esempi contenuti nell'array *dataX*, le cui etichette vengono confrontate con quelle contenute nell'array *dataY*, per calcolare il valore assoluto delle differenze. Gli errori calcolati vengono restituiti in output.
Nello stesso file della classe, ma esternamente ad essa, sono implementate le funzioni **preprocessAutoencoderDataset(dataset, classLabel)** e **preprocessClassifierDataset(dataset, autoencoderN, autoencoderA, n_classes)**, due funzioni che si occupano di eseguire la pre-elaborazione dei dati contenuti in *dataset* nella stessa maniera in cui venivano trattati nella versione base di MINDFUL.
