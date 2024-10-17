import re
import nltk
from bs4 import BeautifulSoup
import string
import emoji
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
import spacy

nlp = spacy.load("en_core_web_sm")
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

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

number_words = { 
    'zero': '', 'one': '', 'two': '', 'three': '', 'four': '', 'five': '',
    'six': '', 'seven': '', 'eight': '', 'nine': '', 'ten': '',
    'eleven': '', 'twelve': '', 'thirteen': '', 'fourteen': '', 'fifteen': '',
    'sixteen': '', 'seventeen': '', 'eighteen': '', 'nineteen': '',
    'twenty': '', 'thirty': '', 'forty': '', 'fifty': '', 'sixty': '',
    'seventy': '', 'eighty': '', 'ninety': '',
    'twenty-one': '', 'twenty-two': '', 'twenty-three': '', 'twenty-four': '', 'twenty-five': '',
    'twenty-six': '', 'twenty-seven': '', 'twenty-eight': '', 'twenty-nine': '',
    'thirty-one': '', 'thirty-two': '', 'thirty-three': '', 'thirty-four': '', 'thirty-five': '',
    'thirty-six': '', 'thirty-seven': '', 'thirty-eight': '', 'thirty-nine': '',
    'forty-one': '', 'forty-two': '', 'forty-three': '', 'forty-four': '', 'forty-five': '',
    'forty-six': '', 'forty-seven': '', 'forty-eight': '', 'forty-nine': '',
    'fifty-one': '', 'fifty-two': '', 'fifty-three': '', 'fifty-four': '', 'fifty-five': '',
    'fifty-six': '', 'fifty-seven': '', 'fifty-eight': '', 'fifty-nine': '',
    'sixty-one': '', 'sixty-two': '', 'sixty-three': '', 'sixty-four': '', 'sixty-five': '',
    'sixty-six': '', 'sixty-seven': '', 'sixty-eight': '', 'sixty-nine': '',
    'seventy-one': '', 'seventy-two': '', 'seventy-three': '', 'seventy-four': '', 'seventy-five': '',
    'seventy-six': '', 'seventy-seven': '', 'seventy-eight': '', 'seventy-nine': '',
    'eighty-one': '', 'eighty-two': '', 'eighty-three': '', 'eighty-four': '', 'eighty-five': '',
    'eighty-six': '', 'eighty-seven': '', 'eighty-eight': '', 'eighty-nine': '',
    'ninety-one': '', 'ninety-two': '', 'ninety-three': '', 'ninety-four': '', 'ninety-five': '',
    'ninety-six': '', 'ninety-seven': '', 'ninety-eight': '', 'ninety-nine': '',
    'hundred': '', 'thousand': '', 'million': '', 'billion': ''
}

negations = {  
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

def preprocess_text(text):
    text = text.lower()
    text = BeautifulSoup(text, "html.parser").get_text()  
    text = re.sub(r'@\w+', '', text)  
    text = re.sub(r'#\w+', '', text)  
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)  
    text = emoji.replace_emoji(text, "") 
    text = re.sub(r'\d+', '', text)  
    text = text.translate(str.maketrans("", "", string.punctuation)) 
    
    for contraction, expansion in contractions_dict.items():
        text = text.replace(contraction, expansion)
    
    for negation, replacement in negations.items():
        text = text.replace(negation, replacement)
    
    words = text.split()
    words = [word for word in words if word not in stop_words]
    
    stemmer = nltk.PorterStemmer()
    words = [stemmer.stem(word) for word in words]
    
    return ' '.join(words)