import json

import numpy as np
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.model_selection import train_test_split

from prueba_optimal_threshold import Procedimiento, MoE

N_CLUSTERS=[4, 5, 6, 7, 8, 9, 11, 13, 14, 16]
CLUSTERING_MODEL=['K', 'B']
TEXT_MODEL=['A', 'B']
ROUTED=[True, False]
with open('/Users/magaliboulanger/Documents/Dataset/twitter_dataset_formatted.jsonl', 'rt', encoding='utf-8') as f:
    dev = [json.loads(l) for l in f]
salida_texto = ""
for tm in TEXT_MODEL:
    p = Procedimiento(tm)
    for cl in CLUSTERING_MODEL:
        for nc in N_CLUSTERS:
            for r in ROUTED:
                y_pred, report = p.proceder(0.01, nc, cl, r)
                false_pos_rate, true_pos_rate, proba = roc_curve(y,y_pred)
                optimal_proba_cutoff = sorted(list(zip(np.abs(true_pos_rate - false_pos_rate), proba)), key=lambda i: i[0], reverse=True)[0][1]
                y_pred2, report2 = p.proceder(optimal_proba_cutoff,nc,cl,r)
                salida_texto+= str(optimal_proba_cutoff)+'-'+str(cl)+'-'+str(tm)+'-'+str(nc)+'clusters-Routed'+str(r)+"=  accuracy: ",report2['accuracy'] + '\n'
                print(str(optimal_proba_cutoff)+'-'+str(cl)+'-'+str(tm)+'-'+str(nc)+'clusters-Routed'+str(r)+"=  accuracy: ",report2['accuracy'] )#suma / 10
                f = open("twitter_dataset_outputs/" + str(optimal_proba_cutoff)+'-'+str(cl)+'-'+str(tm)+'-'+str(nc)+'clusters-Routed'+str(r)+".txt", "w")
                f.write(str(report2))
                f.close()
print("Todos los resultados: ")
print(salida_texto)
