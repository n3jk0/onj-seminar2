import numpy as np
from nltk import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import cross_val_predict
from sklearn.svm import LinearSVC
from sklearn import metrics
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import StratifiedKFold
from sklearn import svm
from scipy import interp
from scipy.sparse import csr_matrix, hstack
import matplotlib.pyplot as plt

import re


def k_mers(text, k=2):
    n = len(text)
    for i in range(n - k + 1):
        yield tuple(text[i:i + k])


class CategoryOfRedditComments:

    def __init__(self, fileName):

        self.pos_tagged = {}
        self.tokens = {}
        self.text = {}
        self.y = list()
        with open(fileName, 'rt') as data:
            i = 0
            for line in data:
                line = line.split("\t")
                comment = re.sub("http\S+", "", line[2])
                # nba == 0 ; politics == 1
                self.y.append(0 if line[1] == "nba" else 1)
                self.text[i] = comment.rstrip()
                i += 1

        self.y = np.array(self.y)
        self.tokenize()

    def tokenize(self):
        for comment_id, text in self.text.items():
            self.tokens[comment_id] = word_tokenize(text)

    def featurize(self):
        vect = TfidfVectorizer(strip_accents="unicode", analyzer="word", stop_words="english")
        X = csr_matrix(vect.fit_transform(self.text.values()))

        word_3_mers = self.build_kmers_featurize(k=3)
        word_2_mers = self.build_kmers_featurize(k=2)

        X = hstack([X, word_3_mers])
        X = hstack([X, word_2_mers])

        return X

    def build_kmers_featurize(self, k=2):

        dic = {}
        i = 0

        for comment_id, comment in self.tokens.items():
            for k_mer in list(k_mers(comment, k=k)):
                if k_mer not in dic:
                    dic[k_mer] = i
                    i += 1

        res = csr_matrix((len(self.tokens), len(dic)))

        for comment_id, comment in self.tokens.items():
            for k_mer in list(k_mers(comment, k=k)):
                res[comment_id, dic[k_mer]] = 1  # row is id of a comment, column is value in k-mer_dict for this k-mer

        return res


FILENAME_REDDIT_COMMENTS = "./data/RC_2018-01-01_filtered_final"
crc = CategoryOfRedditComments(FILENAME_REDDIT_COMMENTS)
X = crc.featurize()

randpred = np.random.randint(2, size=len(crc.y))
predicted = cross_val_predict(LinearSVC(), X, crc.y, cv=3)

rand_acc = metrics.accuracy_score(crc.y, randpred)
rand_prec = metrics.precision_recall_fscore_support(crc.y, randpred, average='binary', pos_label=1)

print("Random classifier:")
print('Accuracy', rand_acc, '| Precision', rand_prec[0], '| Recall', rand_prec[1], '| F-score', rand_prec[2])

acc = metrics.accuracy_score(crc.y, predicted)
prec = metrics.precision_recall_fscore_support(crc.y, predicted, average='binary', pos_label=1)

print('My prediction')
print('Accuracy', acc, '| Precision', prec[0], '| Recall', prec[1], '| F-score', prec[2])

# ROC
cv = StratifiedKFold(n_splits=3, shuffle=True)
classifier = svm.SVC(kernel='linear', probability=True)
all_tprs = []
mean_fpr = np.linspace(0, 1, 50)
i = 0
for train, test in cv.split(X, crc.y):
    probas = classifier.fit(X.tocsr()[train], crc.y[train]).predict_proba(X.tocsr()[test])
    fpr, tpr, _ = roc_curve(crc.y[test], probas[:, 1])
    roc_auc = auc(fpr, tpr)
    all_tprs.append(interp(mean_fpr, fpr, tpr))
    all_tprs[-1][0] = 0.0
    plt.plot(fpr, tpr, lw=1, alpha=0.4, label='ROC fold %d (AUC = %.2f)' % (i, roc_auc))
    i += 1

plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='b', label='Random', alpha=.8)

mean_tpr = np.mean(all_tprs, axis=0)
mean_tpr[-1] = 1.0
plt.plot(mean_fpr, mean_tpr, color='r', label='Mean ROC (AUC = %.2f)' % auc(mean_fpr, mean_tpr), lw=2, alpha=.8)

plt.title('Subreddit detection ROC')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.legend(loc="lower right")
plt.show()
