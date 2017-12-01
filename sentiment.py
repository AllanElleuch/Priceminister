# from nltk.corpus import sentiwordnet as swn
import nltk
# nltk.download('wordnet')
# python -m nltk.downloader sentiwordnet
# breakdown = swn.senti_synset('breakdown.n.03')
# print(breakdown)
# breakdown.pos_score()
# breakdown.neg_score()
# breakdown.obj_score()
# list(swn.senti_synsets('slow'))
# happy = swn.senti_synsets('happy', 'a')
# print( swn.all_senti_synsets())
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
sentence = "Iphone6 camera is awesome for low light "
token = nltk.word_tokenize(sentence)
print(token)
tagged = nltk.pos_tag(token)
print(tagged)
