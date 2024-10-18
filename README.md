- **Analisi esplorativa.ipynb**: cleaning dei dati estratti dallo scraper e analisi esplorativa. Analisi sentiment (in fase successiva)
- **Sentiment labelling.ipynb**: uso di BioBERT per estrarre il sentiment dai vari messaggi 
- **fine_tune_success.py**: processo di fine-tuning di BioBERT
- **negative_to_positive_detection.ipynb**: identificazione di positive sentiment shift (da negativo a positivo)
- **network_modelling.ipynb**: network modelling e estrazione metriche
- **positive_to_negative_detection.ipynb**: identificazione di negative sentiment shift (da positivo a negativo), seconda domanda di ricerca
- **preprocessing_functions.py**: funzioni per il text preprocessing
- **scraper.ipynb**: scraper per estrarre i dati dalle community
- **topic_modelling_first_research_question.ipynb**: notebook contenente topic modelling e analisi prima domanda di ricerca
- **tweets_clean.csv**: dataset SemEval usato per il fine-tuning di BioBERT

<br><br>
**Dati nel drive** (troppo grandi per GitHub, dimensioni da 50 MB a 330 MB)


- **df_a_type_interaction_su_u.csv**: dataframe Asthma UK contenente interazioni con solo superusers o interazioni ibride
- **df_a_type_interaction_su_u.csv**: dataframe Asthma UK contenente solo interazioni user-user
- **df_asthma_negative_to_positive.csv**: dataframe Asthma UK contenente interazioni con negative sentiment shift (da positivo a negativo). Il nome del file è un refuso, dobrebbe essere positive_to_negative

- **df_lungs_type_interaction.csv**: dataframe BLF contenente interazioni con solo superusers o interazioni ibride
- **df_lungs_type_interaction_user_user_total.csv**: dataframe BLF contenente solo interazioni user-user
- **df_asthma_negative_to_positive.csv**: dataframe BLF contenente interazioni con negative sentiment shift (da positivo a negativo). Il nome del file è un refuso, dobrebbe essere positive_to_negative
