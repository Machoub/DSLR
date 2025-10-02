🧠 DSLR - Logistic Regression Classifier
📌 Project Description

This project implements a generalized logistic regression classifier from scratch in Python to predict a Hogwarts house based on student features.

While logistic regression is typically used for binary classification, we apply a one-vs-all strategy to extend it to a multi-class setting. The implementation is modular and reusable, allowing the classifier to be applied to other datasets beyond this project.

In addition, the project includes a data analysis and visualization section to better understand the dataset and extract meaningful insights before training the model.

📊 Methodology
🔁 Logistic Regression

Like linear regression, logistic regression uses gradient descent to minimize a loss function. However, it applies a sigmoid activation to map inputs into a probability between 0 and 1, which reflects the likelihood of belonging to a certain class.

🧮 Loss Function - Cross Entropy
J(θ)=−1m∑i=1m[y(i)log⁡(hθ(x(i)))+(1−y(i))log⁡(1−hθ(x(i)))]
J(θ)=−
m
1
	​

i=1
∑
m
	​

[y
(i)
log(h
θ
	​

(x
(i)
))+(1−y
(i)
)log(1−h
θ
	​

(x
(i)
))]

with:

hθ(x)=σ(θTx)andσ(z)=11+e−z
h
θ
	​

(x)=σ(θ
T
x)andσ(z)=
1+e
−z
1
	​

📈 Gradient Descent

The partial derivative with respect to each parameter 
θj
θ
j
	​

 is given by:

∂J∂θj=1m∑i=1m(hθ(x(i))−y(i))xj(i)
∂θ
j
	​

∂J
	​

=
m
1
	​

i=1
∑
m
	​

(h
θ
	​

(x
(i)
)−y
(i)
)x
j
(i)
	​


This allows for updating the weights during training using gradient descent.

🗂️ Program Structure
📄 describe.py

Usage:
