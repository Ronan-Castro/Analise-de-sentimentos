#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re


# In[3]:


import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('rslp')
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import TweetTokenizer
from sklearn.model_selection import train_test_split


# In[4]:


get_ipython().run_cell_magic('capture', '', '!pip install contractions\nimport contractions')


# In[5]:


plt.style.use('seaborn-pastel')


# + Visão geral do DataFrame

# In[6]:


df = pd.read_csv('Tweets.csv', usecols=['airline_sentiment', 'text'])
df.head(5)


# + Verificando a distribuição dos dados.

# In[7]:


df.airline_sentiment.value_counts()


# In[8]:


df.airline_sentiment.value_counts().plot(kind='bar')


# + Os dados estão desbalanceados. Requerendo um processo de oversampling e/ou undersampling.
# + Primeiramente, aplicaremos um processo de transformação dos dados.
#   - A classe com valor `negative` receberá o valor -1;
#   - A classe `neutral` receberá o valor 0; e
#   - A classe `positive` receberá o valor 1.

# In[9]:


condlist = [
  df.airline_sentiment == 'negative',
  df.airline_sentiment == 'neutral',
  df.airline_sentiment == 'positive'
]

choicelist = [-1, 0, 1]

df.airline_sentiment = np.select(condlist, choicelist, default=2)


# In[10]:


neg, neu, pos = df.airline_sentiment.value_counts()
print('neg:', neg)
print('neu:', neu)
print('pos:', pos)


# In[11]:


# Separação do DataFrame em classes distintas
df_negative = df[df['airline_sentiment'] == -1]
df_neutral  = df[df['airline_sentiment'] ==  0]
df_positive = df[df['airline_sentiment'] ==  1]

# Undersampling nas classes negatives e positivas
df_negative_under = df_negative.sample(pos)
df_neutral_under  = df_neutral.sample(pos)

# Junção dos 3 DataFrames em um único
df_under = pd.concat([df_negative_under, df_neutral_under, df_positive],
                     axis=0)


# In[12]:


df_under.airline_sentiment.value_counts().plot(kind='bar')


# In[13]:


df_under.reset_index(drop=True, inplace=True)


# In[14]:


# Remoção das entradas duplicadas
df_under.drop_duplicates(['text'], inplace=True)
df = df_under
df.shape


# # Pré-Processamento dos Dados:

# In[15]:


# Definição das funções de limpeza de strings

def format_string(text):
  text = text.lower()
  # Remoção de links
  formated_text = re.sub(r'http\S+', '', text)
  # Remoção das mentions (eg. @fulanodetal)
  formated_text = re.sub(r'@[\w]* ', '', formated_text)
  # Remoção das contrações (eg. I'm --> I am)
  formated_text = contractions.fix(formated_text)
  # Simplificação do texto
  formated_text = re.sub(r'[^a-zA-Z0-9@# ]', '', formated_text)
  return (''.join(formated_text)).strip()

def remove_stopwords(text, tokenizer):
  words = tokenizer.tokenize(text)
  newwords = [word for word in words if word not in stopwords.words('english')]
  return (' '.join(newwords)).strip()

def lemmatize_string(text, lemmatizer):
  final_text = ''
  for word in text.split(' '):
    final_text += lemmatizer.lemmatize(word) + ' '
  return final_text

def stemming_string(text, stemmer):
  final_text = ''
  for word in text.split(' '):
    final_text += stemmer.stem(word) + ' '
  return final_text


# In[16]:


# Instanciação das classes exigidas
tweet_tokenizer = TweetTokenizer()

string_lemmatizer = WordNetLemmatizer()

string_stemmer = SnowballStemmer('english')


# In[17]:


classes, tweets = df['airline_sentiment'], df['text']


# + Aplicando formatação nos comentários

# In[18]:


tweets = tweets.map(format_string)


# + Removendo as stopword dos comentários

# In[19]:


tweets = tweets.map(lambda txt: remove_stopwords(txt, tweet_tokenizer))


# + Stemming

# In[20]:


tweets = tweets.map(lambda txt: stemming_string(txt, string_stemmer))


# + Lemmatization

# In[21]:


tweets = tweets.map(lambda txt: lemmatize_string(txt, string_lemmatizer))


# In[22]:


tweets.drop_duplicates(inplace=True)


# # Aplicação dos algoritmos

# In[23]:


from sklearn.feature_extraction.text import CountVectorizer
# from sklearn.linear_model import PassiveAggressiveClassifier
# from sklearn.linear_model import SGDClassifier
# from sklearn.naive_bayes import MultinomialNB
# from sklearn.linear_model import Perceptron

from sklearn.svm import LinearSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier


# In[24]:


processed_df = pd.concat([tweets, classes], axis=1)
processed_df.dropna(how='any', inplace=True)


# In[25]:


# Visão geral do DataFrame
processed_df


# In[26]:


vectorizer = CountVectorizer(analyzer="word", tokenizer=tweet_tokenizer.tokenize)


# + Dados vetorizados

# In[27]:


freq_tweets = vectorizer.fit_transform(processed_df['text'])
freq_tweets.shape


# + Separação em conjunto de treinamento e teste

# In[30]:


X_train, X_test, y_train, y_test =   train_test_split(freq_tweets,
                   processed_df['airline_sentiment'],
                   test_size=0.33, 
                   random_state=42)
y_train


# + Testando múltiplos algoritmos

# In[28]:


#BA = Basic Algorithms
classificadores_BA = {
    'Linear SVC': LinearSVC(random_state=0, tol=1e-5),
    'Logistic Regression':LogisticRegression(random_state=0, max_iter=1000),
    'Multi-Layered Perceptron':MLPClassifier(random_state=1, max_iter=300),
    'KNN':KNeighborsClassifier(n_neighbors=3)
}

cls_stats_BA = {}

for model in classificadores_BA:
    cls_stats_BA[model] = 0.0

for model, cls in classificadores_BA.items():
        cls.fit(X_train, y_train)
        cls_stats_BA[model] = cls.score(X_test, y_test)
        print(f'{model} done')

# Resultado do modelos -- acurácia bruta
print(f'{"model":<32} | {"accuracy":<15}\n{"-"*30}')
for model in cls_stats_BA:
  accuracy = cls_stats_BA[model]
  print(f'{model:<32} | {accuracy:<1.5f}')
    
cls_stats_BA_save = cls_stats_BA


# In[29]:


fig, ax = plt.subplots(figsize=(10, 5))
y_values = list(cls_stats_BA.values())
# ymin = min(y_values) * .997
# ymax = max(y_values) * 1.002
ax.bar(list(cls_stats_BA.keys()), y_values, width=0.35)
# ax.set_ylim(ymin, ymax)
ax.set_title('Acurácia dos modelos')
plt.show()


# + Teste do modelo

# In[30]:


from sklearn.model_selection import GridSearchCV


# In[31]:


solvers = ['newton-cg', 'lbfgs', 'liblinear']
penalty = ['l2']
c_values = np.logspace(-4, 4, 20)
max_iter = [1000, 1500, 2000]
tol = [0.0001, 0.001, 0.01]

grid = dict(solver=solvers,
            penalty=penalty,
            C=c_values,
            max_iter=max_iter,
            tol=tol)

LR = LogisticRegression(random_state=0, n_jobs=-1)

clf = GridSearchCV(LR, grid, scoring='accuracy', n_jobs=-1)


# In[32]:


clf.fit(X_train, y_train)


# In[33]:


clf.best_estimator_


# In[34]:


print(f'LR acurácia: {clf.best_score_}')


# + Treinamento do modelo por cross-validation

# In[35]:


from sklearn.model_selection import cross_validate


# In[36]:


classificadores_BA = {
    'Linear SVC': LinearSVC(random_state=0, tol=1e-5),
    'Logistic Regression':LogisticRegression(random_state=0, max_iter=1000),
    'Multi-Layered Perceptron':MLPClassifier(random_state=1, max_iter=300),
    'KNN':KNeighborsClassifier(n_neighbors=3)
}

cv_result_BA = {}

for model in classificadores_BA:
  cv_result_BA[model] = []

for model in classificadores_BA:
  cv_result_BA[model] = cross_validate(classificadores_BA[model],
                                    X=freq_tweets,
                                    y=processed_df['airline_sentiment'],
                                    cv=10,
                                    scoring='accuracy',
                                    return_estimator=True)


# + Valores de acurácia dos modelos via Cross Validation

# In[37]:


accuracy_values_BA = []
fit_time_values_BA = []
score_time_values_BA = []

for cls in cv_result_BA:
  accuracy_values_BA.append(cv_result_BA[cls]["test_score"])
  score_time_values_BA.append(cv_result_BA[cls]["score_time"])
  fit_time_values_BA.append(cv_result_BA[cls]["fit_time"])


# In[38]:


fig, ax = plt.subplots(figsize=(8,4))
ax.boxplot(accuracy_values_BA)
ax.set_xticklabels(cv_result_BA.keys())
ax.set_title('Acurácia dos modelos via CV')
plt.show()

meanAcurracy = pd.DataFrame            (list(zip(accuracy_values_BA[0],accuracy_values_BA[1],accuracy_values_BA[2],accuracy_values_BA[3])),            columns = ['Linear SVC','Logistic Regression','Multi-Layered Perceptron','KNN'])
print(f'acurácia média nos testes:\n\n{meanAcurracy.mean()}')


# In[39]:


fig, ax = plt.subplots(figsize=(10,4))
ax.boxplot(fit_time_values_BA)
ax.set_xticklabels(cv_result_BA.keys())
ax.set_title('Tempo para treinar os modelos via CV')
plt.show()
meanTimeTrain = pd.DataFrame            (list(zip(fit_time_values_BA[0],fit_time_values_BA[1],fit_time_values_BA[2],fit_time_values_BA[3])),            columns = ['Linear SVC','Logistic Regression','Multi-Layered Perceptron','KNN'])
print(f'tempo médio de treino:\n\n{meanTimeTrain.mean()}')


# In[40]:


fig, ax = plt.subplots(figsize=(8,4))
ax.boxplot(score_time_values_BA)
ax.set_xticklabels(cv_result_BA.keys())
ax.set_title('Tempo para testar os modelos via CV')
plt.show()


meanTimeTest = pd.DataFrame            (list(zip(score_time_values_BA[0],score_time_values_BA[1],score_time_values_BA[2],score_time_values_BA[3])),            columns = ['Linear SVC','Logistic Regression','Multi-Layered Perceptron','KNN'])
print(f'tempo médio de teste:\n\n{meanTimeTest.mean()}')


# In[41]:


# Resultado do modelos -- acurácia no cross validation
actual = 0
print(f'{"model":<32} | {"accuracy":<15}\n{"-"*43}')
for cls in cv_result_BA:
  accuracy = cv_result_BA[cls]["test_score"].mean()
  print(f'{cls:<32} | {accuracy:<1.5f}')
  if (actual < accuracy) :
    cv_result_BA_best_estimator = cv_result_BA[cls]['estimator'][0]


# + Matriz de confusão para modelo treinado em Holdout

# In[42]:


from sklearn.metrics import plot_confusion_matrix


# In[43]:


#print(cv_result_BA)
for cls in cv_result_BA:
    print(cv_result_BA[cls]['estimator'][0])


# In[48]:


fig, axs = plt.subplots(2, 2, figsize=(12, 12))

for i, clf in enumerate(cv_result_BA):
  x = i//2
  y = i%2
  axs[x, y].set_title(clf)
  cv_model = cv_result_BA[clf]['estimator'][0]
  plot_confusion_matrix(cv_model, X_test, y_test, normalize='true', ax=axs[x, y])

plt.show() 


# + Aplicando alguns algoritmos avançados

# In[53]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier


# In[54]:


#AA = Advanced Algorithm
classificadores_AA = {
    'Random Forest': RandomForestClassifier(max_depth=2, random_state=0),
    'AdaBoost': AdaBoostClassifier(n_estimators=100, random_state=0),
    'XGBoost': GradientBoostingClassifier(n_estimators=100, learning_rate=1.0,max_depth=1, random_state=0),
}

cls_stats_AA = {}

for model in classificadores_AA:
    cls_stats_AA[model] = 0.0

for model, cls in classificadores_AA.items():
        cls.fit(X_train, y_train)
        cls_stats_AA[model] = cls.score(X_test, y_test)
        #print(f'{model} done')

# Resultado do modelos - acurácia bruta
print(f'{"model":<32} | {"accuracy":<15}\n{"-"*30}')
for model in cls_stats_AA:
  accuracy = cls_stats_AA[model]
  print(f'{model:<32} | {accuracy:<1.5f}')
        


# In[55]:


clsvalues = []
for model in cls_stats_BA_save:
    aux = cls_stats_BA_save[model]
    clsvalues.append(aux)
for model in cls_stats_AA:
    aux = cls_stats_AA[model]
    clsvalues.append(aux)
aux1 = list(cls_stats_BA_save.keys())
aux2 = list(cls_stats_AA.keys())
df = pd.Series((v for v in aux1))
df2 = pd.Series((v for v in aux2), index=[4, 5, 6])
df = df.append(df2)


# In[56]:


fig, ax = plt.subplots(figsize=(15, 5))
y_values = clsvalues
ax.bar(df, y_values, width=0.55)
ax.set_title('Acurácia dos modelos')
plt.show()
i = 0
for model in df:
    print(model,f'{clsvalues[i]:.4f}')
    i+=1


# In[57]:


classificadores_AA = {
    'Random Forest': RandomForestClassifier(max_depth=2, random_state=0),
    'AdaBoost':AdaBoostClassifier(n_estimators=100, random_state=0),
    'XGBoost': GradientBoostingClassifier(n_estimators=100, learning_rate=1.0,max_depth=1, random_state=0),
}

cv_result_AA = {}

for model in classificadores_AA:
  cv_result_AA[model] = []

for model in classificadores_AA:
  cv_result_AA[model] = cross_validate(classificadores_AA[model],
                                    X=freq_tweets,
                                    y=processed_df['airline_sentiment'],
                                    cv=10,
                                    scoring='accuracy',
                                    return_estimator=True)


# In[58]:


accuracy_values_AA = []
fit_time_values_AA = []
score_time_values_AA = []

for cls in cv_result_AA:
  accuracy_values_AA.append(cv_result_AA[cls]["test_score"])
  score_time_values_AA.append(cv_result_AA[cls]["score_time"])
  fit_time_values_AA.append(cv_result_AA[cls]["fit_time"])


# In[59]:


fig, ax = plt.subplots(figsize=(10,4))
ax.boxplot(accuracy_values_AA)
ax.set_xticklabels(cv_result_AA.keys())
ax.set_title('Acurácia dos modelos via CV')
plt.show()

meanAccuracy = pd.DataFrame            (list(zip(accuracy_values_AA[0],accuracy_values_AA[1],accuracy_values_AA[2])),            columns = ['Random Forest','AdaBoost','XGBoost'])
print(f'acurácia média nos testes:\n\n{meanAccuracy.mean()}')


# In[60]:


fig, ax = plt.subplots(figsize=(10,4))
ax.boxplot(fit_time_values_AA)
ax.set_xticklabels(cv_result_AA.keys())
ax.set_title('Tempo para treinar os modelos via CV')
plt.show()
meanTimeTrain = pd.DataFrame            (list(zip(fit_time_values_AA[0],fit_time_values_AA[1],fit_time_values_AA[2])),            columns = ['Random Forest','AdaBoost','XGBoost'])
print(f'tempo médio de treino:\n\n{meanTimeTrain.mean()}')


# In[61]:


fig, ax = plt.subplots(figsize=(8,4))
ax.boxplot(score_time_values_AA)
ax.set_xticklabels(cv_result_AA.keys())
ax.set_title('Tempo para testar os modelos via CV')
plt.show()

meanTimeTest = pd.DataFrame            (list(zip(score_time_values_AA[0],score_time_values_AA[1],score_time_values_AA[2])),            columns = ['Random Forest','AdaBoost','XGBoost'])
print(f'tempo médio de teste:\n\n{meanTimeTest.mean()}')


# In[62]:


# Resultado do modelos -- acurácia no cross validation
actual = 0
print(f'{"model":<32} | {"accuracy":<15}\n{"-"*43}')
for cls in cv_result_AA:
  accuracy = cv_result_AA[cls]["test_score"].mean()
  print(f'{cls:<32} | {accuracy:<1.5f}')
  if (actual < accuracy) :
    cv_result_AA_best_estimator = cv_result_AA[cls]['estimator'][0]


# In[ ]:


fig, axs = plt.subplots(1, 3, figsize=(18, 5))

for i, clf in enumerate(cv_result_AA):
  axs[i].set_title(clf)
  cv_model = cv_result_AA[clf]['estimator'][0]
  plot_confusion_matrix(cv_model, X_test, y_test, normalize='true', ax=axs[i])

plt.show() 


# 

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




