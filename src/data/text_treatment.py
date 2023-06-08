import pandas as pd
import unicodedata
import re
import nltk
import spacy
from spacy.lang.pt.examples import sentences 
from utils import utils
import logging

#ToDo: Executar python -m spacy download pt_core_news_lg

logger = logging.getLogger(__name__)
logger = utils.get_logger(logger=logger)

class text_preparer:
    """Class dedicated to the data cleaning and preparation
    """    
    def __init__(self, input_path: str, output_path: str) -> None:
        """Constructor Method of the class text_prepares

        Parameters
        ----------
        input_path : str
            Path where the raw file should be readed
        output_path : str
            Path where the processed file should be written
        """        
        self.input_path = input_path
        self.output_path = output_path
        self.data = pd.read_csv(input_path)
    
    @classmethod
    def __clear_chars__(self, text: str) -> str:
        """Clear the special chars from the text

        Parameters
        ----------
        text : str
            Text to be treated

        Returns
        -------
        str
            Text with no accent or special caracter
        """        
        logging.info("\t Changing caracters with accents")
        # Remoção pontuação
        text_without_accents = ''.join(c for c in unicodedata.normalize('NFD', text)
                        if unicodedata.category(c) != 'Mn')
        
        # Limpeza de caracteres especiais
        logging.info("\t Cleanning special caracters")
        rgx_lists = re.findall('[a-zA-Z0-9\s]', text_without_accents)
        
        # Junção de todos os termos coletados
        output = ''.join(rgx_lists) 

        return output

    @classmethod
    def __remove_stop_words__(self, text: str) -> str:
        """Removes stop words of the text

        Parameters
        ----------
        text : str
            Text to be cleaned

        Returns
        -------
        str
            Text with no stop words
        """      
        logging.info("\rRemovving Stop Words")
        stop_words = set(nltk.corpus.stopwords.words('portuguese'))
        word_tokens = nltk.word_tokenize(text)
        caracteres_sentenca_filtrada = [word for word in word_tokens if not word in stop_words]
        sentenca_filtrada = ' '.join(caracteres_sentenca_filtrada)

        return sentenca_filtrada

    @classmethod
    def __lemmatization__(self, text: str) -> str:
        """Applies the process of lemmatization, which is responsible for transforming the word in its primitive form

        Parameters
        ----------
        text : str
            Text to have the lemmatization applied

        Returns
        -------
        str
            Text with lemmatization applied
        """        
        # Amigos > Amigo
        # Amigo  > Amigo
        # Amiga  > Amigo

        logging.info("\tAplying lemmatization")
        load_model = spacy.load('pt_core_news_sm', disable = ['parser','ner'])
        doc = load_model(text)
        output = " ".join([token.lemma_ for token in doc])
        return output

    def prepare_text(self) -> None:
        """Main method responsible for apply all the methods to treat the text data, and stores it in the text_preparer.data            
        """        
        logging.info("Starting the data cleaning...")
        
        # Copiando para evitar problemas com 
        df_int = self.data.copy()

        # Remoção de nan
        df_int = df_int[df_int['input'].notna()]

        # Declaração do tipo como string
        df_int['x_input'] = df_int['input'].astype(str)

        # Tranformação dos caracteres para letras minúsculas;
        df_int['x_input'] = df_int['x_input'].str.lower()

        # Remoção de palavras irrelevantes (ex: preposições, conjunções, artigos e etc);
        df_int['x_input'] = df_int['x_input'].apply(self.__remove_stop_words__)

        # Aplicação de Lematização:
        df_int['x_input'] = df_int['x_input'].apply(self.__lemmatization__)

        # Limpeza de caracteres especiais e remoção pontuação com RegEx
        df_int['x_input'] = df_int['x_input'].apply(self.__clear_chars__)

        logging.info("Data preparation Done!")

        self.data = df_int.copy()
  
    def export_text(self) -> None:
        """Exports the text in the specified dir as output_path
        """        
        logging.info(f"Exporting treated data in {self.output_path}")
        data = self.data.copy()
        data.to_csv(self.output_path, sep=";", index=False)

