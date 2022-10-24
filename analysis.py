import spacy
#print(spacy.__version__)
from sklearn import svm
import ru_core_news_lg

nlp = ru_core_news_lg.load()
#print(nlp)

class Standarts:
    GREETINGS = "GREETINGS"

    OFFER_TO_BUY = "OFFER_TO_BUY"
    OFFER_TO_BUY_PACKAGE = "OFFER_TO_BUY_PACKAGE"

    PRICE_SAY = "PRICE_SAY"
    PAY_TYPE_QUESTION = "PAY_TYPE_QUESTION"
    THANKING_FOR_BUYING = "THANKING_FOR_BUYING"
    END_OF_SHOPPING = "END_OF_SHOPPING"
    SHORT_CHANGE = "SHORT_CHANGE"
    GOODBYE = "GOODBYE"

nlp = spacy.load("ru_core_news_lg")
train_x = ["доброе утро", "добрый день", "добрый вечер", "здраствуйте", "привет",
           "не забудьте приобрести товар месяца", "по привлекательной цене",
           "cумма покупки",
           "cпасибо за покупку", "cпасибо за покупку приходите ещё",
           "это вся ваша покупка", "это всё что вы хотели бы приобрести",
           "не хотите ли приобрести пакет", "пакет большой или маленький", "пакет нужен", "большой", "маленький",
           "оплата наличными или картой", "наличные? карта?", "оплатить покупку",
           "всего доброго", "до свидания", "приходите ещё", "ждём вас ещё", "всего хорошего",
           "ваша сдача", "сдача", "пожалуйста, Ваша сдача 100 рублей"
           ]

docs = [nlp(text) for text in train_x]
train_x_word_vectors = [x.vector for x in docs]

train_y = [
    Standarts.GREETINGS, Standarts.GREETINGS, Standarts.GREETINGS, Standarts.GREETINGS, Standarts.GREETINGS,
    Standarts.OFFER_TO_BUY, Standarts.OFFER_TO_BUY,
    Standarts.PRICE_SAY,
    Standarts.THANKING_FOR_BUYING, Standarts.THANKING_FOR_BUYING,
    Standarts.END_OF_SHOPPING, Standarts.END_OF_SHOPPING,
    Standarts.OFFER_TO_BUY_PACKAGE, Standarts.OFFER_TO_BUY_PACKAGE, Standarts.OFFER_TO_BUY_PACKAGE,
    Standarts.OFFER_TO_BUY_PACKAGE, Standarts.OFFER_TO_BUY_PACKAGE,
    Standarts.PAY_TYPE_QUESTION, Standarts.PAY_TYPE_QUESTION, Standarts.PAY_TYPE_QUESTION,
    Standarts.GOODBYE, Standarts.GOODBYE, Standarts.GOODBYE, Standarts.GOODBYE, Standarts.GOODBYE,
    Standarts.SHORT_CHANGE, Standarts.SHORT_CHANGE, Standarts.SHORT_CHANGE
]

clf_svm_wm = svm.SVC(kernel="linear")
clf_svm_wm.fit(train_x_word_vectors, train_y)

def Voice_tag(text):
    text_esp = [text]
    test_docs = [nlp(text) for text in text_esp]
    test_x_word_vectors = [x.vector for x in test_docs]
    a = clf_svm_wm.predict(test_x_word_vectors)
    i = 0
    for el in a:
        print(el, text_esp[i])
        i += 1
