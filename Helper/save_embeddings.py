from helper import Helper
from text_processing.text_processing import TextProcessing

tp = TextProcessing()
helper = Helper(tp)
texts = helper.get_texts_of_json('train')

w, h = 2, 8500
output = [[0 for x in range(w)] for y in range(h)]
i=0
for text in texts:
    output[i][0]=text
    output[i][1]=tp.encode_sentences(text)
    i+=1

f = open ('results/output_embeddings.txt','w')
f.write(output)
f.close()

g = open ('results/output_texts.txt','w')
g.write(texts)
g.close()