import re
import string
import pandas as pd
import numpy as np
import random
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

file_index = open('label/index', encoding='utf-8')
file_output = open('output.txt', encoding='utf-8')
file_mail_type = open('mail_type.txt', encoding='utf-8')

stop_words = set(stopwords.words('english'))    # get stop word set
#columns = ['label', 'word_count', 'mail_type', 'text']
columns = ['label', 'word_count', 'text']
del_table = string.punctuation
str_table = str.maketrans('', '', '1234567890')

dict_word_index = {}  # a dictionary which get word index
word_list = []  # word vector (12000)
count_word_index = 0
count = 0
total_data = 0
mail_type_list = []
dict_mail_type_index = {}
mail_type_count = 0


ps = PorterStemmer()
df = pd.DataFrame(columns=columns)

#   get dictionary from outside
for line in file_output.readlines():
    tmp_list = line.split(':')
    word_list.append(tmp_list[0])
    dict_word_index[tmp_list[0]] = count_word_index
    count_word_index += 1

#   get mail type from outside
for line in file_mail_type.readlines():
    tmp_list = line.split(':')
    mail_type_list.append(tmp_list[0])
    dict_mail_type_index[tmp_list[0]] = mail_type_count
    mail_type_count += 1


for line in file_index.readlines():
    tmp = line.split(' ')
    tmp_label = 1 if tmp[0] == 'spam' else 0  # mail label, spam -> 1, ham -> 0
    tmp_path = tmp[1]   # mail's path
    tmp_path = tmp_path.replace('\n', '')   # remove '\n' in path
    tmp_path = tmp_path.replace('../', '')  # remove '../' in path
    try:
        mail_file = open(tmp_path, encoding='utf-8')
    #   handle mail type
    #     mail_type = mail_file.readline()
    #     mail_type = mail_type.split(' ')
    #     tmp_mail_type_list = [0]*(mail_type_count+1)
    #     fin_mail_type = ''
    #     if len(mail_type) > 2:
    #         mail_type = mail_type[2]
    #         tmp = mail_type.split('.')
    #         for i in range(1, len(tmp)):
    #             if i != len(tmp) - 1:
    #                 fin_mail_type += (tmp[i]) + '.'
    #             else:
    #                 fin_mail_type += tmp[i]
    #     if fin_mail_type in mail_type_list:
    #         tmp_mail_type_list[dict_mail_type_index[fin_mail_type]] = 1
    #     else:
    #         tmp_mail_type_list[mail_type_count] = 1  # other type
    #   handle words
        mail_text = mail_file.read()
        total_data += 1
        print(total_data)

        word_tokens = re.split(r'\W+', mail_text)   # remove punctuation in mail
        #   lower case and filter
        word_tokens2 = []
        for tok in word_tokens:
            tok = tok.translate(str_table)
            if 20 > len(tok) > 2:
                word_tokens2.append(tok.lower())
        # remove stop words
        filtered_sentence = [w for w in word_tokens2 if w not in stop_words]
        # stem
        stem_sentence = []
        for w in filtered_sentence:
            stem_sentence.append(ps.stem(w))
        # convert to bag of words
        tmp_word_list = [0] * 12000
        for w in stem_sentence:
            if w in word_list:
                get_index = dict_word_index[w]
                tmp_word_list[get_index] += 1
        # count word
        tmp_word_count = len(stem_sentence)
        # add a line to the DataFrame
        # tmp_line = {'label': tmp_label, 'word_count': tmp_word_count, 'mail_type': tmp_mail_type_list,
        #             'text': tmp_word_list}
        tmp_line = {'label': tmp_label, 'word_count': tmp_word_count, 'text': tmp_word_list}
        df = df.append(tmp_line, ignore_index=True)
    except UnicodeDecodeError:
        pass


#   divide into 5 groups
group1 = list(range(int(total_data/5)))
group2 = list(range(int(total_data/5), int(total_data/5*2)))
group3 = list(range(int(total_data/5*2), int(total_data/5*3)))
group4 = list(range(int(total_data/5*3), int(total_data/5*4)))
group5 = list(range(int(total_data/5*4), total_data))
train_group = group2 + group3 + group4 + group1
test_group = group5

train_group = random.sample(train_group, int(len(train_group)*0.5))

train_data = len(train_group)


#   train data
p1_word_num = np.ones(len(word_list))
p0_word_num = np.ones(len(word_list))
p1_denom = 12000.0
p0_denom = 12000.0



spam_count = 0
ham_count = 0
#p1_mail_type = np.zeros(mail_type_count+1)
#p0_mail_type = np.zeros(mail_type_count+1)

for i in train_group:
    print(i)
    vec = df.loc[i, 'text']
    #m_type = df.loc[i, 'mail_type']
    if df.loc[i, 'label'] == 1:
        p1_word_num += np.array(vec)
        #p1_mail_type += np.array(m_type)
        p1_denom += df.loc[i, 'word_count']
        spam_count += 1
    else:
        p0_word_num += np.array(vec)
        #p0_mail_type += np.array(m_type)
        p0_denom += df.loc[i, 'word_count']
        ham_count += 1

p1_vec = np.log(p1_word_num / p1_denom)
p0_vec = np.log(p0_word_num / p0_denom)
p_spam = float(spam_count) / float(train_data)
#p1_mail_type = p1_mail_type / float(spam_count)
#p0_mail_type = p0_mail_type / float(ham_count)


# test
correct = 0
test_data = len(test_group)
print('start test.')
print('train_data:' + str(train_data))
print('test_data:' + str(test_data))
for i in test_group:
    vec = df.loc[i, 'text']
    #m_type = df.loc[i, 'mail_type']
    correct_label = df.loc[i, 'label']
    predict_label = 0
    p1 = sum(np.array(vec) * p1_vec) + np.log(p_spam)
    p0 = sum(np.array(vec) * p0_vec) + np.log(1-p_spam)
    if p1 > p0:
        predict_label = 1
    if predict_label == correct_label:
        correct += 1

print("accuracy:" + str(correct/float(test_data)))
