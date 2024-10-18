import re
import spacy
from bs4 import BeautifulSoup
import emoji
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer

nlp = spacy.load("en_core_web_sm")
nlp.disable_pipe("parser")
nlp.add_pipe("sentencizer")
stemmer = PorterStemmer()

def expand_contractions(text, contractions_dict):
    for contraction, expansion in contractions_dict.items():
        text = text.replace(contraction, expansion)
    return text

def custom_remove_entities(text):
    doc = nlp(text)
    filtered_tokens = [token.text for token in doc if not token.ent_type_]
    text_without_spacy_entities = ' '.join(filtered_tokens)
    text_cleaned = re.sub(r'\b[A-Z]\.[A-Za-z]+\b', '', text_without_spacy_entities)
    return text_cleaned

def remove_html_tags(text):
    soup = BeautifulSoup(text, "html.parser")
    clean_text = soup.get_text(separator=" ")
    clean_text = re.sub(r'\s+', ' ', clean_text).strip()
    return clean_text

def remove_urls(text):
    clean_text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    clean_text = re.sub(r'\s+', ' ', clean_text).strip()
    return clean_text

def remove_emojis(text):
    clean_text = emoji.replace_emoji(text, replace='')
    clean_text = re.sub(r'\s+', ' ', clean_text).strip()
    return clean_text

def remove_punctuation(text):
    clean_text = text.translate(str.maketrans('', '', string.punctuation))
    clean_text = re.sub(r'\s+', ' ', clean_text).strip()
    return clean_text

def remove_extra_whitespace(text):
    clean_text = re.sub(r'\s+', ' ', text).strip()
    return clean_text

def remove_numbers(text):
    clean_text = re.sub(r'\d+', '', text)
    clean_text = re.sub(r'\s+', ' ', clean_text).strip()
    return clean_text

def handle_negations(text, negations_dict):
    for negation, replacement in negations_dict.items():
        text = re.sub(r'\b' + re.escape(negation) + r'\b', replacement, text, flags=re.IGNORECASE)
    return re.sub(r'\s+', ' ', text).strip()

def remove_stop_words(text):
    stop_words = set(stopwords.words('english'))
    return ' '.join(word for word in text.split() if word not in stop_words)

def remove_filler_words(text, filler_words):
    return ' '.join(word for word in text.split() if word not in filler_words)

def stem_text(tokens):
    return ' '.join(stemmer.stem(word) for word in tokens)

def pos_tagging(tokens):
    return nltk.pos_tag(tokens)

def extract_adjectives_adverbs(pos_tagged_review):
    return ' '.join(word for word, pos in pos_tagged_review if pos.startswith('JJ') or pos.startswith('RB'))

def clean_text_pipeline(user_input):
    contractions_dict = {
    "ain't": "is not",
    "aren't": "are not",
    "can't": "cannot",
    "couldn't": "could not",
    "didn't": "did not",
    "doesn't": "does not",
    "don't": "do not",
    "hadn't": "had not",
    "hasn't": "has not",
    "haven't": "have not",
    "he'd": "he would",
    "he'll": "he will",
    "he's": "he is",
    "I'd": "I would",
    "I'll": "I will",
    "I'm": "I am",
    "I've": "I have",
    "isn't": "is not",
    "it's": "it is",
    "let's": "let us",
    "mightn't": "might not",
    "might've": "might have",
    "mustn't": "must not",
    "must've": "must have",
    "shan't": "shall not",
    "she'd": "she would",
    "she'll": "she will",
    "she's": "she is",
    "shouldn't": "should not",
    "should've": "should have",
    "that'll": "that will",
    "that's": "that is",
    "there'd": "there would",
    "there's": "there is",
    "they'd": "they would",
    "they'll": "they will",
    "they're": "they are",
    "they've": "they have",
    "we'd": "we would",
    "we'll": "we will",
    "we're": "we are",
    "we've": "we have",
    "weren't": "were not",
    "what'll": "what will",
    "what're": "what are",
    "what's": "what is",
    "what've": "what have",
    "when's": "when is",
    "where's": "where is",
    "who'd": "who would",
    "who'll": "who will",
    "who's": "who is",
    "why's": "why is",
    "won't": "will not",
    "wouldn't": "would not",
    "you'd": "you would",
    "you'll": "you will",
    "you're": "you are",
    "you've": "you have",
    "y'all": "you all",
    "could've": "could have",
    "would've": "would have",
    "should've": "should have",
    "it'd": "it would",
    "didn't": "did not",
    "doesn't": "does not",
    "haven't": "have not",
    "hasn't": "has not",
    "hadn't": "had not",
    "there'd": "there would",
    "how's": "how is",
    "where'd": "where did",
    "how'd": "how did",
    "why'd": "why did",
    "let's": "let us",
    "needn't": "need not",
    "oughtn't": "ought not",
    "daren't": "dare not",
}

    negations_dict = {  
        "not good": "bad",
        "not great": "bad",
        "not bad": "good",
        "not happy": "unhappy",
        "not satisfied": "unsatisfied",
        "not recommend": "recommend against",
        "never": "ever",
        "no": "any",
        "n't": "",
        "not like": "dislike",
        "not true": "false",
        "not worth": "worthless",
        "not interested": "disinterested",
        "not enough": "insufficient",
        "not clear": "unclear",
        "not easy": "difficult",
        "not helpful": "unhelpful",
        "not possible": "impossible",
        "not worth it": "worthless",
        "not sure": "unsure",
        "not appealing": "unappealing",
        "not necessary": "unnecessary",
        "not exciting": "uninspiring",
        "not convenient": "inconvenient",
        "not reliable": "unreliable",
        "not good enough": "inadequate",
        "not friendly": "unfriendly",
        "not the best": "inferior",
        "not comfortable": "uncomfortable",
        "not great at": "poor at",
        "not enjoyable": "unenjoyable",
        "not worth the time": "a waste of time",
        "not impressed": "underwhelmed",
        "not what I expected": "disappointing",
        "not suitable": "unsuitable",
        "not confident": "doubtful",
        "not ideal": "suboptimal",
        "not quite": "somewhat",
        "not always": "rarely",
        "not really": "barely",
        "not entirely": "partially",
        "not as good as": "inferior to",
        "not better than": "worse than",
        "not much of": "little to no",
        "not my favorite": "least favorite",
        "not loving": "disliking",
        "not thankful": "ungrateful",
        "not suitable for": "unsuitable for",
        "not worth mentioning": "insignificant",
        "not to my taste": "unappealing",
        "not engaging": "unengaging",
        "not thrilling": "unthrilling",
        "not memorable": "forgettable",
        "not entertaining": "uninspiring",
        "not fun": "unenjoyable",
        "not well made": "poorly made",
        "not well acted": "badly acted",
        "not realistic": "unrealistic",
        "not well written": "poorly written",
    }

    filler_words = [
        "um", "uh", "like", "you know", "actually", "basically", "sort of", "kind of",
        "I mean", "you see", "well", "so", "right", "okay", "anyways", "look", "intially"
        "listen", "believe me", "virtually", "practically", "probably", "maybe", "perhaps",
        "frankly", "nearly", "almost", "essentially", "utterly", "completely","really"
        "just", "only", "mainly", "stuff", "thing", "things", "somewhat", "seemingly"
        "in a sense", "at the end of the day", "to be honest", "you know what I mean",
        "to tell you the truth", "in other words", "as a matter of fact", "in fact",
        "for the most part", "on the whole", "generally", "usually", "frequently",
        "often", "occasionally", "at least", "kind of", "sort of", "more or less",
        "up to a point", "in many ways", "in some way", "as it were", "in effect",
        "in reality", "to some extent", "to a certain extent", "likely", "really",
        "presumably", "ostensibly", "evidently", "seemingly", "apparently",
        "probably", "likely", "somehow", "someway", "let’s say", "let's",
        "roughly", "you know", "at all", "basically", "let me be clear",
        "if you will", "as such", "let's be honest", "in truth", "sort of like",
        "in general", "like I said", "as far as I know", "suffice it to say",
        "for what it's worth", "I guess", "I suppose", "certainly", "absolutely",
        "totally", "merely", "specifically", "explicitly", "implicitly","exactly"
        "in particular", "especially", "not really", "not entirely", "particularly"
        "not totally", "to be fair", "to clarify", "to put it simply",
        "to put it mildly", "let me put it this way", "the thing is",
        "it’s like", "I mean to say", "just saying", "and all",
        "if you know what I mean", "as far as I'm concerned", "for instance",
        "for example", "such as", "you might say", "that is",
        "like I said before", "and stuff", "or something", "or whatever",
        "basically", "kind of like", "just about", "in some respects",
        "to make a long story short", "when it comes to", "in this case",
        "in light of", "and so on", "and whatnot",
        'actually', 'ah', 'alright', 'anyway', 'apparently', 'basically', 'er', 'frankly',
        'honestly', 'just', 'kinda', 'like', 'literally', 'look', 'obviously', 'okay',
        'really', 'right', 'so', 'truthfully', 'uh', 'um', 'well', 'yeah',"Aaah", "Umm",
        "Uhh", "Awww", "Eh", "Erm", "Hmmm", "Oh", "Mmm", "Like", "You know", "Sort of",
        "Kind of", "I mean", "Actually", "Basically", "Literally", "Right", "Okay", "Well",
        "So", "Uh-huh", "Yeah", "Hmm", "Ahh", "Hmmmm", "Ehhh", "Y'know", "Let’s see", "Alright",
        "Right?", "Honestly", "Seriously", "Just", "Anyway", "Look", "Listen", "Believe me", "To be honest",
        "At the end of the day","Aah", "Aaah", "Aaaaah", "Umm", "Ummm", "Ummmm", "Uhh", "Uhhh", "Uhhhh", "Awww",
        "Awwww", "Awwwww", "Hmm", "Hm", "Hmmmm", "Oh", "Ooh", "Ohhh", "Oooooh", "Er", "Err", "Errr", "Yeah", "Yeaah",
        "Yeeeah", "Yeeeaaaah", "Like", "Liiike", "Liiiiike", "Okay", "Okaaay", "Ok", "Oooookay", "Right", "Riiight", "Riiiight", "So", "Sooo", "Soooo", "Sooooo", "No", "Nooo", "Noooo", "Nooooo", "Well", "Weell", "Weeellll"
    ]
    text = expand_contractions(user_input, contractions_dict)
    text = custom_remove_entities(text)
    text = remove_html_tags(text)
    text = remove_urls(text)
    text = remove_emojis(text)
    text = remove_punctuation(text)
    text = remove_extra_whitespace(text)
    text = remove_numbers(text)
    text = handle_negations(text, negations_dict)
    text = remove_stop_words(text)
    text = remove_filler_words(text, filler_words)
    text = text.lower()

    tokens = word_tokenize(text)
    cleaned_stemmed_text = stem_text(tokens)

    pos_tagged_tokens = pos_tagging(tokens)
    adjectives_adverbs = extract_adjectives_adverbs(pos_tagged_tokens)

    return cleaned_stemmed_text, adjectives_adverbs  

def tfidf_vectorization(corpus, max_features=500, min_df=0.05, ngram_range=(1, 1)):
    vectorizer = TfidfVectorizer(max_features=max_features, min_df=min_df, ngram_range=ngram_range)
    tfidf_matrix = vectorizer.fit_transform(corpus)
    return tfidf_matrix, vectorizer.get_feature_names_out()

user_input = "Ooh I loved the movie, was soooo amazing 😊!"
cleaned_review, adjectives_adverbs = clean_text_pipeline(user_input)

corpus = [cleaned_review]

tfidf_matrix, feature_names = tfidf_vectorization(corpus)


