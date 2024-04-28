Dies ist der praktische Anteil meiner komplexen Leistung.
Es handelt sich um ein Klassifizierungsprojekt für Hirntumore, bei welchen
Convolutional Neural Networks auf verschiedensten Wegen Bilder auf Tumore untersuchen sollen.

Hierfür wird hauptsächlich die Bibliothek Tensorflow genutzt, aber auch Matplotlib zur Datenvisualisierung und
Numpy für den generellen Umgang mit den Daten sowie einige Weitere (siehe requirements.txt).

----------

Datensatz 1:
240x240 schwarz-weiß MRT-Scans des Hirnes, sortiert nach Tumor (1) und kein Tumor(0).
→ Hier erreichte das beste Modell (dataset_1/models/model_28) eine Genauigkeit von etwa 97,8 %, das Ergebnis wurde
über einen Bruteforce-Ansatz erzielt (siehe dataset_1/models_data/logs.txt für mehr Informationen) und über eigene
Anteilserrechnung der korrekten Vorhersagen im Testdatensatz eingestuft.
https://doi.org/10.34740/KAGGLE/DSV/1370629 (Stand: 27.04.2024)

Datensatz 2:
512x512 schwarz-weiß MRT-Scans des Hirnes, sortiert nach kein Tumor (no tumor bzw. 2) und
3 Unterkategorien der Tumore (labels 0, 1, 3).
→ Das beste Modell (dataset_2/models/model_11) erreichte eine Genauigkeit von etwa 97,1 %, das Ergebnis wurde
über Trial-and-Error bzw. Interpretation vorheriger Testergebnisse (dataset_2/models/model_1-8) erreicht und über die
sparse-categorial-accuracy eingestuft. (Anmerkung: Da die Modelle dieses Datensatzes zum Teil parallel trainiert wurden,
sind die logs teils durcheinander, eine Zuordnung ist über die Graphen möglich)
https://doi.org/10.34740/KAGGLE/DSV/2645886 (Stand: 27.04.2024)

----------

Alte Modelle:
Nur das Modell 11 des letzten Testvorganges sowie der Modell-Creator für den ersten Datensatz sind unmittelbar
aufzufinden, die vorherigen Modell-Dateien finden sich im Unterordner legacy_models.
Vor der Nutzung dieser sollten sie zurück in den Überordner geführt werden, da sonst Konflikte mit den Dateinamen
innerhalb der Pythonfiles entstehen können.

Anmerkung zu GitHub:
Aufgrund von hoher Datengröße finden sich weder die trainierten Modelle noch die Datensätze auf Github. Hier ist lediglich
der Code und die Code-History aufzufinden. Ein Link zum Download des kompletten Projektes folgt.
