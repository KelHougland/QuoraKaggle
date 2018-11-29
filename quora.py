import numpy as np
import pandas as pd
import string
from tqdm import tqdm
import os
from gensim.models import KeyedVectors
import operator
import re

print(os.listdir("../input"))



tqdm.pandas()

news_path = 'input/GoogleNews-vectors-negative300.bin'


# train = pd.read_csv('../input/train.csv')
# test = pd.read_csv('../input/test.csv')
train = pd.read_csv('input/train.csv')
test = pd.read_csv('input/test.csv')

embeddings_index = KeyedVectors.load_word2vec_format(news_path, binary=True)


def build_vocab(sentences, verbose = True):
    vocab = {}
    for sentence in tqdm(sentences, disable = (not verbose)):
        for word in sentence:
            try:
                vocab[word] +=1
            except KeyError:
                vocab[word] = 1
    return vocab
    

def check_coverage(vocab,embeddings_index):
    a = {}
    oov = {}
    k = 0
    i = 0
    for word in tqdm(vocab):
        try:
            a[word] = embeddings_index[word]
            k += vocab[word]
        except:
            oov[word] = vocab[word]
            i += vocab[word]
            pass

    print('Found embeddings for {:.2%} of vocab'.format(len(a)/len(vocab)))
    print('Found embeddings for {:.2%} of all text'.format(k/(k+1)))
    sorted_x = sorted(oov.items(), key=operator.itemgetter(1))[::-1]

    return sorted_x

def clean_text(x):

    x = str(x)
    for punct in "/-'":
        x = x.replace(punct, ' ')
    for punct in '&':
        x = x.replace(punct, f' {punct} ')
    for punct in '?!.,"#$%\'()*+-/:;<=>@[\\]^_`{|}~' + '“”’':
        x = x.replace(punct, '')
    return x


def replace_typical_misspell(text):
    def replace(match):
        return mispellings[match.group(0)]

    return mispellings_re.sub(replace, text)

def _get_mispell(mispell_dict):
    misplell_re = re.compile('(%s)' % '|'.join(mispell_dict.keys()))
    return mispell_dict, misplell_re

def clean_numbers(x):
    x = re.sub('[0-9]{5,}', '#####', x)
    x = re.sub('[0-9]{4,}', '####', x)
    x = re.sub('[0-9]{3,}', '###', x)
    x = re.sub('[0-9]{2,}', '##', x)
    return x

sentences = train["question_text"].progress_apply(lambda x: x.split()).values

vocab = build_vocab(sentences)
print({k: vocab[k] for k in list(vocab)[:5]})


mispell_dict = {'colour':'color',
                'centre':'center',
                'didnt':'did not',
                'doesnt':'does not',
                'isnt':'is not',
                'shouldnt':'should not',
                'favourite':'favorite',
                'travelling':'traveling',
                'counselling':'counseling',
                'theatre':'theater',
                'cancelled':'canceled',
                'labour':'labor',
                'organisation':'organization',
                'wwii':'world war 2',
                'citicise':'criticize',
                'instagram': 'social medium',
                'whatsapp': 'social medium',
                'snapchat': 'social medium',
                'Snapchat': 'social medium',
                'bitcoin' : 'internet currency',
                'cryptocurrency' : 'internet currency',
                'cryptocurrencies' : 'internet currency',
                'bitcoins' : 'internet currency'

                }

mispellings, mispellings_re = _get_mispell(mispell_dict)



train["question_text"] = train["question_text"].progress_apply(lambda x: replace_typical_misspell(x))
sentences = train["question_text"].progress_apply(lambda x: x.split())
to_remove = ['a','to','of','and']
sentences = [[word for word in sentence if not word in to_remove] for sentence in tqdm(sentences)]
vocab = build_vocab(sentences)


oov = check_coverage(vocab, embeddings_index)
oov[:10]



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