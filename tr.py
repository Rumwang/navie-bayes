import re
import string
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

file_index = open('label/index', encoding='utf-8')
stop_words = set(stopwords.words('english'))    # get stop word set
columns = ['label', 'word_count', 'text']
del_table = string.punctuation
str_table = str.maketrans('', '', '1234567890')

dict1 = {}
count = 0
file = open('output.txt', 'w')
spam_word = []

ps = PorterStemmer()
df = pd.DataFrame(columns=columns)



for line in file_index.readlines():
    tmp = line.split(' ')
    tmp_label = 1 if tmp[0] == 'spam' else 0 # mail label, spam -> 1, ham -> 0
    tmp_path = tmp[1]   # mail's path
    tmp_path = tmp_path.replace('\n', '')   # remove '\n' in path
    tmp_path = tmp_path.replace('../', '')  # remove '../' in path
    if tmp_label == 1:
        try:
            count += 1
            print(count)

            mail_file = open(tmp_path, encoding='utf-8')
            mail_text = mail_file.read()
            #mail_text = re.sub('[%s]' % re.escape(del_table), ' ', mail_text)  # remove number and punctuation
            #mail_text = mail_text.lower()   # lower case
            word_tokens = re.split(r'\W+', mail_text)
            word_tokens2 = []
            for tok in word_tokens:
                tok = tok.translate(str_table)
                if 20 > len(tok) > 2:
                    word_tokens2.append(tok.lower())
            # remove stop words
            #word_tokens = word_tokenize(mail_text)
            filtered_sentence = [w for w in word_tokens2 if w not in stop_words]
            # stem
            stem_sentence = []
            for w in filtered_sentence:
                stem_sentence.append(ps.stem(w))
                if w not in dict1:
                    dict1[w] = 1
                else:
                    dict1[w] += 1
        except UnicodeDecodeError:
            pass

c = 0
result = sorted(dict1.items(), key=lambda k:k[1], reverse=True)
for tmp in result:
    if c < 12000:
        file.write(tmp[0] + ':' + str(tmp[1]) + '\n')
        c += 1
file.close()
print(len(dict1))
