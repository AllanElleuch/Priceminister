from nltk.tokenize import RegexpTokenizer
from textstat.textstat import textstat

class FeaturesCalculator():
    tokenizer = RegexpTokenizer(r'\w+')
    # @staticmethod
    @classmethod
    def features_difficultword_content(cls, row):
        data = row['review_content']
        token = cls.tokenizer.tokenize(data)
        if(token == None or len(token)==0):
            return 0
        # val = textstat.smog_index(row['review_content'])  # 0.0970957994444
        # val = textstat.dale_chall_readability_score(row['review_content'])  # 0.0655506963852
        val = textstat.difficult_words(row['review_content'])  # 0.119689698366
        # val = textstat.linsear_write_formula(row['review_content'])  # 0.0393165149095
        # val = textstat.gunning_fog(row['review_content'])  # 0.064893772836

        # val = textstat.flesch_reading_ease(row['review_content'])  # -0.000962802863895
        # val = textstat.automated_readability_index(row['review_content']) #0.0206780263383
        if(val != None):
            return val
        else:
            return 0
    @classmethod
    def features_smog_content(cls, row):
        data = row['review_content']

        token = cls.tokenizer.tokenize(data)
        if(token == None or len(token)==0):
            return 0
        val = textstat.smog_index(row['review_content'])  # 0.0970957994444
        # val = textstat.dale_chall_readability_score(row['review_content'])  # 0.0655506963852
        # val = textstat.difficult_words(row['review_content'])  # 0.119689698366
        # val = textstat.linsear_write_formula(row['review_content'])  # 0.0393165149095
        # val = textstat.gunning_fog(row['review_content'])  # 0.064893772836

        # val = textstat.flesch_reading_ease(row['review_content'])  # -0.000962802863895
        # val = textstat.automated_readability_index(row['review_content']) #0.0206780263383
        if(val != None):
            return val
        else:
            return 0
    @classmethod
    def features_linsear_title(cls, row):
        data = row['review_title']
        token = cls.tokenizer.tokenize(data)
        if(token == None or len(token)==0):
            return 0
        # val = textstat.smog_index(row['review_title']) # #-0.00216594769872
        # val = textstat.dale_chall_readability_score(row['review_title'])  # #  0.025131347883
        val = textstat.difficult_words(row['review_title']) #   # 0.0363778452286
        # val = textstat.linsear_write_formula(row['review_title'])  # #-0.00557553587525
        # val = textstat.gunning_fog(row['review_title'])  # 0.0202643684371

        # val = textstat.flesch_reading_ease(row['review_title'])  # -0.0207657385707
        # val = textstat.automated_readability_index(row['review_title']) ##0.0142244017091
        if(val != None):
            return val
        else:
            return 0
    # @staticmethod
    @classmethod
    def features_tokenized_content(cls,row):
        data = row['review_content']
        token = cls.tokenizer.tokenize(data)
        if(token == None or len(token)==0):
            return 0
        else:
            return 1
    @classmethod
    def features_tokenized_title(cls, row):
        data = row['review_title']
        token = cls.tokenizer.tokenize(data)
        if(token == None or len(token)==0):
            return -1
        else:
            return 1
