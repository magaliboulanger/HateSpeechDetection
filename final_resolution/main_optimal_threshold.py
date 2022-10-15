import json

import numpy as np
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.model_selection import train_test_split

from prueba_optimal_threshold import Procedimiento, MoE

N_CLUSTERS=[5]
CLUSTERING_MODEL=['K']
TEXT_MODEL=['A']
ROUTED=[True]
with open('/Users/magaliboulanger/Documents/Dataset/dev_sr_final.jsonl', 'rt', encoding='utf-8') as f:
    dev = [json.loads(l) for l in f]
salidas=[]
for tm in TEXT_MODEL:
    p = Procedimiento(tm)
    train_img, dev_img, test_img = p.extract_features_images()
    train_text, dev_text, test_text = p.generate_embeddings_text()
    x = np.concatenate((train_img, train_text), axis=1)
    y = np.asarray([t['label'] for t in dev])
    for cl in CLUSTERING_MODEL:
        for nc in N_CLUSTERS:
            for r in ROUTED:
                y_pred, report = p.proceder(0.5, nc, cl, r)
                false_pos_rate, true_pos_rate, proba = roc_curve(y,y_pred)
                optimal_proba_cutoff = sorted(list(zip(np.abs(true_pos_rate - false_pos_rate), proba)), key=lambda i: i[0], reverse=True)[0][1]
                suma = 0
                for i in range(0, 10):
                    print(i)
                    y_pred2, report2 = p.proceder(optimal_proba_cutoff,nc,cl,r)
                    suma += report2['accuracy']
                    salidas.append({"num_iteracion": i, "resultado": report2})
                print(str(optimal_proba_cutoff)+'-'+str(cl)+'-'+str(tm)+'-'+str(nc)+'clusters-Routed'+str(r)+"=  accuracy: ", suma / 10)
                salidas.append({"acc_avg": suma / 10})
                # change name according to the run
                f = open("outputs_ROC_sr/"+str(optimal_proba_cutoff)+'-'+str(cl)+'-'+str(tm)+'-'+str(nc)+'clusters-Routed'+str(r)+".txt", "w")
                f.write(str(salidas))
                f.close()
