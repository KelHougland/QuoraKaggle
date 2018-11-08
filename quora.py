import numpy as np
import pandas as pd
import string
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from lightgbm import LGBMClassifier
from sklearn.naive_bayes import MultinomialNB

# import os
# print(os.listdir("../input"))
# train = pd.read_csv('../input/train.csv')
# test = pd.read_csv('../input/test.csv')


train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

train['preLength'] = train['question_text'].apply(len)
test['preLength'] = test['question_text'].apply(len)

def cleanString(astring):
    newString = [char for char in astring if char not in string.punctuation]
    newString = ''.join(newString)
    finString = []
    for word in newString.split():
        if word.lower() not in ['a','the','of','i','did','we', 'you', 'my', 'your', 'they','their','is','as','was']:
            finString.append(word)  
    return finString

def countPunc(astring):
    numPunc = 0
    for char in astring:
        if char in string.punctuation:
            numPunc +=1
    return numPunc

X = train.drop('target', axis=1)
y = train['target']

messages = pd.concat([train['question_text'],test['question_text']])

#X['cleaned'] = X['question_text'].apply(cleanString)
#test['cleaned'] = test['question_text'].apply(cleanString)
#X['numPunc'] = X['question_text'].apply(countPunc)
#test['numPunc'] = test['question_text'].apply(countPunc)

#X = X.drop('question_text', axis=1)
#test = test.drop('question_text', axis=1)

bow_transformer = CountVectorizer(analyzer=cleanString).fit(messages)
print(len(bow_transformer.vocabulary_))

messages_bow = bow_transformer.transform(messages)
train_bow = bow_transformer.transform(train['question_text'])
test_bow = bow_transformer.transform(test['question_text'])

tfidf_transformer = TfidfTransformer().fit(messages_bow)

tfidf_train = tfidf_transformer.transform(train_bow)
tfidf_test = tfidf_transformer.transform(test_bow)

quoModNB = MultinomialNB(fit_prior=True).fit(tfidf_train,train['target'])
quoModRF = RandomForestClassifier(n_estimators=20,class_weight='balanced').fit(tfidf_train,train['target'])
quoModKN = KNeighborsClassifier(n_neighbors=50).fit(tfidf_train,train['target'])
quoModBM = LGBMClassifier(class_weight='balanced').fit(tfidf_train,train['target'])

NBpreds = quoModNB.predict_proba(tfidf_test)
RFpreds = quoModRF.predict_proba(tfidf_test)
KNpreds = quoModKN.predict_proba(tfidf_test)
BMpreds = quoModBM.predict_proba(tfidf_test)

allPreds = [NBpreds[:,1], RFpreds[:,1], KNpreds[:,1], BMpreds[:,1]]
subMatrix = np.array(allPreds).transpose()
weights = [0.25,0.25,0.25,0.25]
predProbas = np.matmul(allPreds,weights)

preds = predProbas < 0.7


submission = pd.DataFrame({'qid': test['qid'],'prediction': preds},columns = ['qid','prediction'])
submission.to_csv('submission.csv', index=False)