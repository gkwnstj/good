import numpy as np
import pandas as pd

predicted_labels = np.load("pred_seq2seq.npy")
##print(data)



index_list = []
for i in range(0,len(predicted_labels)-1):
    index_list.append(f"S{i+1:05d}")
    
prediction = pd.DataFrame(columns=['id', 'pred'])

prediction["id"] = index_list
prediction["pred"] = predicted_labels[1:]

prediction = prediction.reset_index(drop=True)

prediction.to_csv('./result/20221119_하준서_sent_class.pred11.csv', index = False)

