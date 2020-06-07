from __future__ import print_function, division, absolute_import, unicode_literals
import json
import multiprocessing
import os
import re
import sys
from builtins import range, zip, str, object
from past.utils import old_div
from collections import OrderedDict
from pattern.text.en import Sentence, parse, modality
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer as Vader_Sentiment
from decorator import contextmanager


class Lexicons(object):
    """Lexicon is a class with static members for managing the existing lists of words.
    Use Lexicon.list(key) in order to access the list with name key.
    """
    pth = os.path.join(os.path.dirname(__file__), 'lexicon.json')
    if os.path.isfile(pth):
        with open(pth, 'r') as filp:
            wordlists = json.loads(filp.read())
    else:
        print(pth, "... file does not exist.")
        wordlists = {}
    # print(list(wordlists.keys()))

    @classmethod
    def list(cls, name):
        """list(name) get the word list associated with key name"""
        return cls.wordlists[name]



def split_into_sentences(text):
    caps = "([A-Z])"
    prefixes = "(Mr|St|Mrs|Ms|Dr)[.]"
    suffixes = "(Inc|Ltd|Jr|Sr|Co)"
    starters = "(Mr|Mrs|Ms|Dr|He\s|She\s|It\s|They\s|Their\s|Our\s|We\s|But\s|However\s|That\s|This\s|Wherever)"
    acronyms = "([A-Z][.][A-Z][.](?:[A-Z][.])?)"
    websites = "[.](com|net|org|io|gov)"
    digits = "([0-9])"

    text = " " + text + "  "
    text = re.sub(prefixes, "\\1<prd>", text)
    text = re.sub(websites, "<prd>\\1", text)
    if "Ph.D" in text:
        text = text.replace("Ph.D.", "Ph<prd>D<prd>")
    if "e.g." in text:
        text = text.replace("e.g.", "e<prd>g<prd>")
    if "i.e." in text:
        text = text.replace("i.e.", "i<prd>e<prd>")
    text = re.sub("\s" + caps + "[.] ", " \\1<prd> ", text)
    text = re.sub(acronyms + " " + starters, "\\1<stop> \\2", text)
    text = re.sub(caps + "[.]" + caps + "[.]" + caps + "[.]", "\\1<prd>\\2<prd>\\3<prd>", text)
    text = re.sub(caps + "[.]" + caps + "[.]", "\\1<prd>\\2<prd>", text)
    text = re.sub(" " + suffixes + "[.] " + starters, " \\1<stop> \\2", text)
    text = re.sub(" " + suffixes + "[.]", " \\1<prd>", text)
    text = re.sub(" " + caps + "[.]", " \\1<prd>", text)
    text = re.sub(digits + "[.]" + digits, "\\1<prd>\\2", text)
    if "”" in text:
        text = text.replace(".”", "”.")
    if "\"" in text:
        text = text.replace(".\"", "\".")
    if "!" in text:
        text = text.replace("!\"", "\"!")
    if "?" in text:
        text = text.replace("?\"", "\"?")
    text = text.replace("\n", " <stop>")
    text = text.replace(".", ".<stop>")
    text = text.replace("?", "?<stop>")
    text = text.replace("!", "!<stop>")
    text = text.replace("<prd>", ".")
    sentences = text.split("<stop>")
    sentences = sentences[:-1]
    sentences = [s.strip() for s in sentences]
    sentences = [s for s in sentences if len(s) >= 2]
    return sentences


def find_ngrams(input_list, n):
    return list(zip(*[input_list[i:] for i in range(n)]))


def syllable_count(text):
    exclude = '!"#$%&\'()*+,-./:;<=>?@[\]^_`{|}~'
    count = 0
    vowels = 'aeiouy'
    text = text.lower()
    text = "".join(x for x in text if x not in exclude)

    if text is None:
        return 0
    elif len(text) == 0:
        return 0
    else:
        if text[0] in vowels:
            count += 1
        for index in range(1, len(text)):
            if text[index] in vowels and text[index - 1] not in vowels:
                count += 1
        if text.endswith('e'):
            count -= 1
        if text.endswith('le'):
            count += 1
        if count == 0:
            count += 1
        count = count - (0.1 * count)
        return count


def lexicon_count(text, removepunct=True):
    exclude = '!"#$%&\'()*+,-./:;<=>?@[\]^_`{|}~'
    if removepunct:
        text = ''.join(ch for ch in text if ch not in exclude)
    count = len(text.split())
    return count


def sentence_count(text):
    ignore_count = 0
    sentences = split_into_sentences(text)
    for sentence in sentences:
        if lexicon_count(sentence) <= 2:
            ignore_count = ignore_count + 1
    sentence_cnt = len(sentences) - ignore_count
    if sentence_cnt < 1:
        sentence_cnt = 1
    return sentence_cnt


def avg_sentence_length(text):
    lc = lexicon_count(text)
    sc = sentence_count(text)
    a_s_l = float(old_div(lc, sc))
    return round(a_s_l, 1)


def avg_syllables_per_word(text):
    syllable = syllable_count(text)
    words = lexicon_count(text)
    try:
        a_s_p_w = old_div(float(syllable), float(words))
        return round(a_s_p_w, 1)
    except ZeroDivisionError:
        # print "Error(ASyPW): Number of words are zero, cannot divide"
        return 1


def flesch_kincaid_grade(text):
    a_s_l = avg_sentence_length(text)
    a_s_w = avg_syllables_per_word(text)
    f_k_r_a = float(0.39 * a_s_l) + float(11.8 * a_s_w) - 15.59
    return round(f_k_r_a, 1)


def count_feature_freq(feature_list, tokens_list, txt_lwr):
    cnt = 0
    # count unigrams
    for w in tokens_list:
        if w in feature_list:
            cnt += 1
        # count wildcard features
        for feature in feature_list:
            if str(feature).endswith('*') and str(w).startswith(feature[:-1]):
                cnt += 1
    # count n_gram phrase features
    for feature in feature_list:
        if " " in feature and feature in txt_lwr:
            cnt += str(txt_lwr).count(feature)
    return cnt


def check_quotes(text):
    quote_info = dict(has_quotes=False,
                      quoted_list=None,
                      mean_quote_length=0,
                      nonquoted_list=split_into_sentences(text),
                      mean_nonquote_length=avg_sentence_length(text))
    quote = re.compile(r'"([^"]*)"')
    quotes = quote.findall(text)
    if len(quotes) > 0:
        quote_info["has_quotes"] = True
        quote_info["quoted_list"] = quotes
        total_qte_length = 0
        nonquote = text
        for qte in quotes:
            total_qte_length += avg_sentence_length(qte)
            nonquote = nonquote.replace(qte, "")
            nonquote = nonquote.replace('"', '')
            re.sub(r'[\s]+', ' ', nonquote)
        quote_info["mean_quote_length"] = round(old_div(float(total_qte_length), float(len(quotes))), 4)
        nonquotes = split_into_sentences(nonquote)
        if len(nonquotes) > 0:
            quote_info["nonquoted_list"] = nonquotes
            total_nqte_length = 0
            for nqte in nonquotes:
                total_nqte_length += avg_sentence_length(nqte)
            quote_info["mean_nonquote_length"] = round(old_div(float(total_nqte_length), float(len(nonquotes))), 4)
        else:
            quote_info["nonquoted_list"] = None
            quote_info["mean_nonquote_length"] = 0

    return quote_info


def check_neg_persp(input_words, vader_neg, vader_compound, include_nt=True):
    """
    Determine the degree of negative perspective of text
    Returns an float for score (higher is more negative)
    """
    neg_persp_score = 0.0
    neg_words = ["aint", "arent", "cannot", "cant", "couldnt", "darent", "didnt", "doesnt",
                  "ain't", "aren't", "can't", "couldn't", "daren't", "didn't", "doesn't",
                  "dont", "hadnt", "hasnt", "havent", "isnt", "mightnt", "mustnt", "neither",
                  "don't", "hadn't", "hasn't", "haven't", "isn't", "mightn't", "mustn't",
                  "neednt", "needn't", "never", "none", "nope", "nor", "not", "nothing", "nowhere",
                  "oughtnt", "shant", "shouldnt", "uhuh", "wasnt", "werent",
                  "oughtn't", "shan't", "shouldn't", "uh-uh", "wasn't", "weren't",
                  "without", "wont", "wouldnt", "won't", "wouldn't", "rarely", "seldom", "despite"]
    for word in neg_words:
        if word in input_words:
            neg_persp_score += 1
    if include_nt:
        for word in input_words:
            if "n't" in word and word not in neg_words:
                neg_persp_score += 1
    if vader_neg > 0.0:
        neg_persp_score += vader_neg
    if vader_compound < 0.0:
        neg_persp_score += abs(vader_compound)
    return neg_persp_score



ref_lexicons = Lexicons()

presup = ref_lexicons.list('presupposition')

doubt = ref_lexicons.list('doubt_markers')

partisan = ref_lexicons.list('partisan')

value_laden = ref_lexicons.list('value_laden')
vader_sentiment_analysis = Vader_Sentiment()

figurative = ref_lexicons.list('figurative')

attribution = ref_lexicons.list('attribution')

self_refer = ref_lexicons.list('self_reference')


def extract_bias_features(text, do_get_caster=False):
    features = OrderedDict()
    if sys.version_info < (3, 0):
        # ignore conversion errors between utf-8 and ascii
        text = text.decode('ascii', 'ignore')
    text_nohyph = text.replace("-", " ")  # preserve hyphenated words as separate tokens
    txt_lwr = str(text_nohyph).lower()
    words = ''.join(ch for ch in txt_lwr if ch not in '!"#$%&()*+,-./:;<=>?@[\]^_`{|}~').split()
    unigrams = sorted(list(set(words)))
    bigram_tokens = find_ngrams(words, 2)
    bigrams = [" ".join([w1, w2]) for w1, w2 in sorted(set(bigram_tokens))]
    trigram_tokens = find_ngrams(words, 3)
    trigrams = [" ".join([w1, w2, w3]) for w1, w2, w3 in sorted(set(trigram_tokens))]

    ## SENTENCE LEVEL MEASURES
    # word count
    features['word_cnt'] = len(words)

    # unique word count
    features['unique_word_cnt'] = len(unigrams)

    # Flesch-Kincaid Grade Level (reading difficulty) using textstat
    features['fk_gl'] = flesch_kincaid_grade(text)

    # compound sentiment score using VADER sentiment analysis package
    vader_sentiment = vader_sentiment_analysis.polarity_scores(text)
    vader_negative_proportion = vader_sentiment['neg']
    vader_compound_sentiment = vader_sentiment['compound']
    features['vader_sentiment'] = vader_compound_sentiment
    features['vader_senti_abs'] = abs(vader_compound_sentiment)

    # negative-perspective
    features['neg_persp'] = check_neg_persp(words, vader_negative_proportion, vader_compound_sentiment)

    # modality (certainty) score and mood using  http://www.clips.ua.ac.be/pages/pattern-en#modality
    sentence = parse(text, lemmata=True)
    sentence_obj = Sentence(sentence)
    features['certainty'] = round(modality(sentence_obj), 4)

    # quoted material
    quote_dict = check_quotes(text)
    features["has_quotes"] = quote_dict["has_quotes"]
    features["quote_length"] = quote_dict["mean_quote_length"]
    features["nonquote_length"] = quote_dict["mean_nonquote_length"]

    ## LEXICON LEVEL MEASURES
    # presupposition markers
    count = count_feature_freq(presup, words, txt_lwr)
    features['presup_cnt'] = count
    features['presup_rto'] = round(old_div(float(count), float(len(words))), 4)

    # doubt markers
    count = count_feature_freq(doubt, words, txt_lwr)
    features['doubt_cnt'] = count
    features['doubt_rto'] = round(old_div(float(count), float(len(words))), 4)

    # partisan words and phrases
    count = count_feature_freq(partisan, words, txt_lwr)
    features['partisan_cnt'] = count
    features['partisan_rto'] = round(old_div(float(count), float(len(words))), 4)

    # subjective value laden word count
    count = count_feature_freq(value_laden, words, txt_lwr)
    features['value_cnt'] = count
    features['value_rto'] = round(old_div(float(count), float(len(words))), 4)

    # figurative language markers
    count = count_feature_freq(figurative, words, txt_lwr)
    features['figurative_cnt'] = count
    features['figurative_rto'] = round(old_div(float(count), float(len(words))), 4)

    # attribution markers
    count = count_feature_freq(attribution, words, txt_lwr)
    features['attribution_cnt'] = count
    features['attribution_rto'] = round(old_div(float(count), float(len(words))), 4)

    # self reference pronouns
    count = count_feature_freq(self_refer, words, txt_lwr)
    features['self_refer_cnt'] = count
    features['self_refer_rto'] = round(old_div(float(count), float(len(words))), 4)

    # Contextual Aspect Summary and Topical-Entity Recognition (CASTER)
    if do_get_caster:
        """ May incur a performance cost in time to process """
        caster_dict = get_caster(text)
        features['caster_dict'] = caster_dict

    return features

# order-preserved list of multiple linear regression model coefficients
modelbeta = [0.844952,
             -0.015031,
             0.055452,
             0.064741,
             -0.018446,
             -0.008512,
             0.048985,
             0.047783,
             0.028755,
             0.117819,
             0.269963,
             -0.041790,
             0.129693]

# order-preserved list of multiple linear regression model features
modelkeys = ['word_cnt',
             'vader_senti_abs',
             'neg_persp',
             'certainty',
             'quote_length',
             'presup_cnt',
             'doubt_cnt',
             'partisan_cnt',
             'value_cnt',
             'figurative_cnt',
             'attribution_cnt',
             'self_refer_cnt']

# unordered associative array (reference dictionary) containing the 
#   multiple linear regression model features and coefficients
mlrmdict = {# 'intercept'    : 0.844952,
            'word_cnt'       : -0.01503,
            'vader_senti_abs': 0.055452,
            'neg_persp'      : 0.064741,
            'certainty'      : -0.01845,
            'quote_length'   : -0.00851,
            'presup_cnt'     : 0.048985,
            'doubt_cnt'      : 0.047783,
            'partisan_cnt'   : 0.028755,
            'value_cnt'      : 0.117819,
            'figurative_cnt' : 0.269963,
            'attribution_cnt': -0.04179,
            'self_refer_cnt' : 0.129693}


def measure_feature_impact(sentence):
    """ Calculate the (normalized) impact of each feature for a given sentence using  
        the top half of the logistic function sigmoid. 
        Returns a Python dictionary of the impact score for each feature."""
    impact_dict = {}
    e = 2.7182818284590452353602874713527  # e constant (Euler's number)
    ebf = extract_bias_features(sentence)
    for k in mlrmdict.keys():
        impact_dict[k] = (2 * (1 / (1 + e**(-abs(ebf[k])))) - 1) * abs(mlrmdict[k]) * 100
    return impact_dict

            
def featurevector(features):
    """Extract the features into a vector in the right order, prepends a 1 for constant term."""
    l = [1]
    l.extend(features[k] for k in modelkeys)
    return l


def normalized_features(features):
    """Normalize the features by dividing by the coefficient."""
    beta = modelbeta
    fvec = featurevector(features)
    norm = lambda i: old_div(fvec[i], modelbeta[i])
    return [norm(i) for i in range(len(modelbeta))]


def compute_bias(sentence_text):
    """run the trained regression coefficients against the feature dict"""
    features = extract_bias_features(sentence_text)
    coord = featurevector(features)
    bs_score = sum(modelbeta[i] * coord[i] for i in range(len(modelkeys)))
    return bs_score



@contextmanager
def poolcontext(*args, **kwargs):
    """poolcontext makes it easier to run a function with a process Pool.
    Example:
            with poolcontext(processes=n_jobs) as pool:
                bs_scores = pool.map(compute_bias, sentences)
                avg_bias = sum(bs_scores)
    """
    pool = multiprocessing.Pool(*args, **kwargs)
    yield pool
    pool.terminate()


def roundmean(avg_bias, sentences, k=4):
    """Compute the average and round to k places"""
    avg_bias = round(old_div(float(avg_bias), float(len(sentences))), k)
    return avg_bias


def compute_avg_statement_bias_mp(statements_list_or_str, n_jobs=1):
    """compute_statement_bias_mp a version of compute_statement_bias
    with the multiprocessing pool manager."""
    sentences = list()
    if not isinstance(statements_list_or_str, list):
        if isinstance(statements_list_or_str, str):
            sentences.extend(split_into_sentences(statements_list_or_str))
        else:
            logmessage = "-- Expecting type(list) or type(str); type({}) given".format(type(statements_list_or_str))
            print(logmessage)
    # max_len = max(map(len, sentences))

    if len(sentences) == 0:
        return 0

    with poolcontext(processes=n_jobs) as pool:
        bs_scores = pool.map(compute_bias, sentences)
        total_bias = sum(bs_scores)

    if len(sentences) > 0:
        avg_bias = roundmean(total_bias, sentences)
    else:
        avg_bias = 0

    return avg_bias


def compute_avg_statement_bias(statements_list_or_str):
    """compute the bias of a statement from the test.
    returns the average bias over the entire text broken down by sentence.
    """
    sentences = list()
    if not isinstance(statements_list_or_str, list):
        if isinstance(statements_list_or_str, str):
            sentences.extend(split_into_sentences(statements_list_or_str))
        else:
            logmessage = "-- Expecting type(list) or type(str); type({}) given".format(type(statements_list_or_str))
            print(logmessage)

    # max_len = max(map(len, sentences))

    if len(sentences) == 0:
        return 0

    bs_scores = []
    for sent in sentences:
        bs_scores.append(compute_bias(sent))

    total_bias = sum(bs_scores)

    if len(sentences) > 0:
        avg_bias = roundmean(total_bias, sentences)
    else:
        avg_bias = 0

    return avg_bias
