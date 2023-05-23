import numpy as np
import pandas as pd

predicted_labels = np.load("C:/Users/USER/Desktop/Lecture/NLP/Lab4/demo3/2023-nlp-lab-4-gkwnstj-main/result/pred_seq2seq.npy")
##print(data)


##predicted_labels = predicted_labels.detach().cpu().numpy()
index_list = []
for i in range(0,len(predicted_labels)):
    index_list.append(f"S{i+1:05d}")
    
prediction = pd.DataFrame(columns=['id', 'pred'])

prediction["id"] = index_list
prediction["pred"] = predicted_labels

prediction = prediction.reset_index(drop=True)

prediction.to_csv('20221119_하준서_sent_class.pred.csv', index = False)

#index_list
prediction
