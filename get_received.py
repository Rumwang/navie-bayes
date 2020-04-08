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
file = open('mail_type.txt', 'w')
spam_word = []

ps = PorterStemmer()
df = pd.DataFrame(columns=columns)



for line in file_index.readlines():
    tmp = line.split(' ')
    tmp_label = 1 if tmp[0] == 'spam' else 0 # mail label, spam -> 1, ham -> 0
    tmp_path = tmp[1]   # mail's path
    tmp_path = tmp_path.replace('\n', '')   # remove '\n' in path
    tmp_path = tmp_path.replace('../', '')  # remove '../' in path
    if tmp_label == 0:
        try:
            count += 1
            print(count)
            mail_file = open(tmp_path, encoding='utf-8')
            mail_type = mail_file.readline()
            mail_type = mail_type.split(' ')
            if len(mail_type) > 2:
                mail_type = mail_type[2]
                tmp = mail_type.split('.')
                res = ''
                for i in range(1, len(tmp)):
                    if i != len(tmp)-1:
                        res += (tmp[i]) + '.'
                    else:
                        res += tmp[i]
                if res not in dict1:
                    dict1[res] = 1
                else:
                    dict1[res] += 1

        except UnicodeDecodeError:
            pass

c = 0
result = sorted(dict1.items(), key=lambda k:k[1],reverse=True)
for tmp in result:
    # if c < 12000:
    file.write(tmp[0] + ':' + str(tmp[1]) + '\n')
        # c += 1
file.close()
print(len(dict1))
