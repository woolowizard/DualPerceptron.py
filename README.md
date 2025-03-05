# Documentazione del Progetto

## Struttura del Progetto

1. **DualPerceptron**
   - Il codice relativo all'implementazione del Perceptron Duale è contenuto nel file `DualPerceptron.py`.

2. **Funzioni Utility**
   - I codici per lo split dei dataset, il calcolo delle statistiche e la generazione dei plot sono contenuti nel file `projectUtils.py`.

3. **Esecuzione del Codice**
   - Nella cartella `run_code` sono presenti quattro sottocartelle, ciascuna denominata `df_{nome_df}` (una per ogni dataset).
   - All'interno di ogni sottocartella è presente un file `run_{nome_df}.py` contenente il codice per:
     - Importazione del dataset
     - Pulizia e preparazione dei dati
     - Esecuzione dei tre modelli

4. **Pesi dei Modelli**
   - I pesi dei modelli utilizzati nella relazione sono salvati nella cartella `/run_code/df_{name_df}/model_params`.
   - Per testarli è disponibile la funzione `new_prediction()`.

### Dettagli sulla Funzione `new_prediction()`

La funzione `new_prediction()` consente di caricare un modello DualPerceptron con parametri già stimati e di effettuare previsioni.

#### Sintassi
```python
new_prediction(params, X_test, y_test)
```

#### Argomenti
- **params (dict)**: Dizionario contenente i parametri del modello. Esempio:
  ```python
  {
      'alpha': array([...]),
      'b': float,
      'X_train': array([...]),
      'y_train': array([...])
  }
  ```
- **X_test (array)**: Dati di test su cui effettuare la previsione.
- **y_test (array)**: Etichette di test per il calcolo dell'accuratezza.

#### Valore Restituito
- **tuple**:
  - Array delle predizioni
  - Statistiche del modello

#### Esempio di Utilizzo
```python
# Caricamento dei parametri del modello
params = load_params('/Users/andreacommodi/Downloads/model_params_linear (2).pkl')
# Predizioni e calcolo accuratezza
obj = new_prediction(params, X_test, y_test)
# Print delle stats e predizioni
print(obj['stats'])
print(obj['pred'])
```

5. **Relazione del Progetto**
   - La relazione completa è disponibile nel file `RelazioneCommodiAndrea.pdf`.
