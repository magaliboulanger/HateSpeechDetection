# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.
# Press the green button in the gutter to run the script.
from Modelo_Pruebas_Octubre import Procedimiento

if __name__ == '__main__':
    salidas=[]
    suma=0
    for i in range(0,10):
        print(i)
        p = Procedimiento()
        report = p.proceder()
        suma+=report['accuracy']
        salidas.append({"num_iteracion":i, "resultado": report})
    print("accuracy: ", suma/10)
    salidas.append({"acc_avg": suma/10})

    #change name according to the run
    f = open("outputs/0.005-KMeans-allminiLML6V2-5clusters-notRouted.txt", "w")
    f.write(str(salidas))
    f.close()


