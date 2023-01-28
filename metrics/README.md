# The Metrics for ERNN
Our primary evaluation metric for classification model performance is the overall $Accuracy$ and $F1$, which are defined in below equations. $\eqref{eq:f1}$. $TP$ and $FP$ refer to True Positive and False Positive, respectively. $TN$ and $FN$ indicate True Negative and False Negative, respectively. $Accuracy$ is the ratio of correct predictions over the total samples.

$$
Accuracy = \frac{{\sum\nolimits_{i = 1}^k {\left({T{P_i} + T{N_i}}\right)} }}{{\sum\nolimits_{i = 1}^k {\left( {T{P_i} + T{N_i} + F{P_i} + F{N_i}} \right)} }}
$$

$Precision$ is the ratio of $TP$ to all predicted positive samples: 

$$
Precision = \frac{{\sum\nolimits_{i = 1}^k {T{P_i}} }}{{\sum\nolimits_{i = 1}^k {\left( {T{P_i} + F{P_i}} \right)} }} 
$$

and $Recall$ is the ratio of $TP$ to all positive samples: 

$$
Recall = \frac{{\sum\nolimits_{i = 1}^k {T{P_i}} }}{{\sum\nolimits_{i = 1}^k {\left( {T{P_i} + F{N_i}} \right)} }} 
$$

where $k$ denotes the number of categories (e.g., $k$ = 13 in Section 5 and $k$ = 2 in Section 6. The F1-score can be calculated as shown in below equation: 

$$
F1 = \frac{{2 \cdot Precision \cdot Recall}}{{Precision + Recall}} 
$$
