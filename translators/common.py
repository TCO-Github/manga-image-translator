import os
import readline
from typing import List, Tuple
from abc import ABC, abstractmethod

from utils import ModelWrapper

class LanguageUnsupportedException(Exception):
    def __init__(self, language_code: str, translator: str = None, supported_languages: List[str] = None):
        error = 'Language not supported for %s: "%s"' % (translator if translator else 'chosen translator', language_code)
        if supported_languages:
            error += '. Supported languages: "%s"' % ','.join(supported_languages)
        super().__init__(error)

class MTPEAdapter():
    async def dispatch(self, queries: List[str], translations: List[str]) -> List[str]:
        new_translations = []
        print('\n -- Running Machine Translation Post Editing (MTPE)')
        for i, (query, translation) in enumerate(zip(queries, translations)):
            print(f'[{i + 1}/{len(queries)}] {query}:')
            readline.set_startup_hook(lambda: readline.insert_text(translation.replace('\n', '\\n')))
            new_translation = ''
            try:
                new_translation = input(' -> ').replace('\\n', '\n')
            finally:
                readline.set_startup_hook()
            new_translations.append(new_translation)
        return new_translations

class CommonTranslator(ABC):
    _LANGUAGE_CODE_MAP = {}

    def __init__(self):
        super().__init__()
        self.mtpe_adapter = MTPEAdapter()

    def supports_languages(self, from_lang: str, to_lang: str, fatal: bool = False) -> bool:
        supported_src_languages = ['auto'] + list(self._LANGUAGE_CODE_MAP)
        supported_tgt_languages = list(self._LANGUAGE_CODE_MAP)

        if from_lang not in supported_src_languages:
            if fatal:
                raise LanguageUnsupportedException(from_lang, self.__class__.__name__, supported_src_languages)
            return False
        if to_lang not in supported_tgt_languages:
            if fatal:
                raise LanguageUnsupportedException(to_lang, self.__class__.__name__, supported_tgt_languages)
            return False
        return True

    def parse_language_codes(self, from_lang: str, to_lang: str, fatal: bool = False) -> Tuple[str, str]:
        if not self.supports_languages(from_lang, to_lang, fatal):
            return None, None

        _from_lang = self._LANGUAGE_CODE_MAP.get(from_lang) if from_lang != 'auto' else 'auto'
        _to_lang = self._LANGUAGE_CODE_MAP.get(to_lang)
        return _from_lang, _to_lang

    async def translate(self, from_lang: str, to_lang: str, queries: List[str], mtpe: bool = False) -> List[str]:
        '''
        Translates list of queries of one language into another.
        '''
        if from_lang == to_lang:
            result = []
        else:
            result = await self._translate(*self.parse_language_codes(from_lang, to_lang, fatal=True), queries)

        translated_sentences = []
        if len(result) < len(queries):
            translated_sentences.extend(result)
            translated_sentences.extend([''] * (len(queries) - len(result)))
        elif len(result) > len(queries):
            translated_sentences.extend(result[:len(queries)])
        else:
            translated_sentences.extend(result)
        if mtpe:
            translated_sentences = self.mtpe_adapter.dispatch(queries, translated_sentences)
        return translated_sentences

    @abstractmethod
    async def _translate(self, from_lang: str, to_lang: str, queries: List[str]) -> List[str]:
        pass

class OfflineTranslator(CommonTranslator, ModelWrapper):
    _MODEL_DIR = os.path.join(ModelWrapper._MODEL_DIR, 'translators')

    async def _translate(self, *args, **kwargs):
        return await self.forward(*args, **kwargs)

    @abstractmethod
    async def _forward(self, from_lang: str, to_lang: str, queries: List[str]) -> List[str]:
        pass

    async def load(self, from_lang: str, to_lang: str, device: str):
        return await super().load(device, *self.parse_language_codes(from_lang, to_lang))

    @abstractmethod
    async def _load(self, from_lang: str, to_lang: str, device: str):
        pass

    async def reload(self, from_lang: str, to_lang: str, device: str):
        return await super().reload(device, from_lang, to_lang)
