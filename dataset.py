from pathlib import Path
from tqdm import tqdm
import pandas as pd
from typing import Callable, List, Dict, Tuple
import logging
logging.basicConfig()


class SlavNERDataset(object):

    def __init__(self,
                 data_dir: Path,
                 word_tokenizer: Callable,
                 sent_tokenizer: Callable,
                 annotation_prefix: str = 'annotated',
                 raw_prefix: str = 'raw',
                 output_file_suffix: str = '.out'):
        self.data_dir = data_dir
        self.word_tokenizer = word_tokenizer
        self.sent_tokenizer = sent_tokenizer
        self.annotation_prefix = annotation_prefix
        self.raw_prefix = raw_prefix
        self.output_file_suffix = output_file_suffix

    def to_df(self) -> pd.DataFrame:
        annotated_path = self.data_dir / Path(self.annotation_prefix)
        sentences = []
        for p_topic in annotated_path.glob('*'):
            topic = p_topic.name
            for p_topic_lang in p_topic.glob('*'):
                lang = p_topic_lang.name
                files = list(self.list_file_pairs(p_topic_lang))
                for a_path, r_path in tqdm(files, total=len(files),
                                           desc=f"{topic}: {lang}"):
                    for sentence in self.process_pair(a_path,
                                                      r_path,
                                                      self.sent_tokenizer,
                                                      self.word_tokenizer,
                                                      lang=lang):
                        sentence['topic'] = topic
                        sentence['lang'] = lang

                        sentences.append(sentence)

        return pd.DataFrame(sentences)

    def load_annotated(
        self,
        f_annotated: Path,
        word_tokenizer: Callable,
        lang: str
    ) -> List:

        text = f_annotated.read_text()
        lines = text.strip().split('\n')
        file_id = lines[0]

        annotations = {}

        for i, line in enumerate(lines[1:]):
            split = line.split('\t')

            if len(split) != 4:
                logging.warning('File %s: Line %d "%s" is split '
                                'into %d columns',
                                file_id, i + 1, line, len(split))
                continue

            token, lemma, tag, entity = split

            token = tuple(word_tokenizer(token.strip(), lang=lang))
            match = {
                "token": token,
                "lemma": lemma,
                "tag": tag,
                "entity": entity
            }

            existing = annotations.get(token)
            # If a match already exists, there is a chance something is messed
            # up
            if existing:
                logging.debug('File %s: Line %d "%s" contains seen token.',
                              file_id, i + 1, line)

                # If the new match is not the same as it was before, there is a
                # chance at least some part of the labeling is wrong.
                if existing != match:
                    logging.debug('File %s: Line %d, Ours: %r Existing: %r',
                                  file_id, i + 1, match, existing)

                continue

            annotations[tuple(match['token'])] = match
        return file_id, annotations

    def load_raw(self, f_raw: Path) -> List:
        text = f_raw.read_text()
        lines = text.strip().split('\n')

        txt_id, language, creation_date, url, title = lines[:5]

        # add title and skip empty lines
        text = ' '.join([line for line in lines[4:] if line.strip()])

        return txt_id, language, creation_date, url, title, text

    def match_annotations_in_sentence(
        self,
        sentence_tokens: str,
        annotations: Dict,
        other_marker: str = 'O'
    ) -> Dict:

        pos, last_pos = 0, 0
        sentence_len = len(sentence_tokens)

        # Go through the whole sentence
        while pos < sentence_len:
            for tokens in annotations:
                n_tokens = len(tokens)
                # Check if the annotated tokens match the current prefix in the
                # sentence
                if tokens == tuple(sentence_tokens[pos:pos+n_tokens]):

                    yield {
                        "token": sentence_tokens[last_pos:pos],
                        "lemma": other_marker,
                        "tag": other_marker,
                        "entity": other_marker
                    }
                    yield annotations[tokens]
                    pos += n_tokens

                    last_pos = pos

            pos += 1

        yield {
            "token": sentence_tokens[last_pos:pos],
            "lemma": other_marker,
            "tag": other_marker,
            "entity": other_marker
        }

    def match_to_bio(
        self,
        match: Dict,
        word_tokenizer: Callable,
        lang: str,
        other_marker: str = 'O'
    ) -> Dict:

        words = match['token']
        n_words = len(words)

        tag = match['tag']
        entity = match['entity']

        if tag == other_marker:
            tags = [other_marker] * n_words
            entities = [other_marker] * n_words
            lemmas = [other_marker] * n_words
        else:
            tags = [f"B-{tag}"] + [f"I-{tag}"] * (n_words - 1)
            entities = [f"B-{entity}"] + [f"I-{entity}"] * (n_words - 1)
            lemmas = word_tokenizer(match['lemma'], lang=lang)

        return {
            'words': words,
            'tags': tags,
            'entities': entities,
            'lemmas': lemmas
        }

    def process_pair(
        self,
        f_annotated: Path,
        f_raw: Path,
        sent_tokenizer: Callable,
        word_tokenizer: Callable,
        lang: str
    ) -> List[Dict]:

        file_id, annotations = self.load_annotated(f_annotated,
                                                   word_tokenizer,
                                                   lang=lang)
        txt_id, lang, creation_date, url, title, text = self.load_raw(f_raw)

        for sentence in sent_tokenizer(text, lang=lang):
            s_match = {}
            tokens = word_tokenizer(sentence, lang=lang)
            for match in self.match_annotations_in_sentence(tokens,
                                                            annotations):
                bios = self.match_to_bio(match, word_tokenizer=word_tokenizer,
                                         lang=lang)

                # Use whatever came in if we still have an empty dict
                if not s_match:
                    s_match = bios
                # Append to the respective keys otherwise
                else:
                    for key in s_match:
                        s_match[key].extend(bios[key])

            s_match['id'] = txt_id
            yield s_match

    def list_file_pairs(
        self,
        topic_lang_prefix: Path,
        annotation_prefix: str = 'annotated',
        raw_prefix: str = 'raw'
    ) -> Tuple[str, str]:

        for a_path in topic_lang_prefix.glob('*' + self.output_file_suffix):
            r_str = str(a_path.with_suffix('.txt'))
            r_path = Path(r_str.replace(f"/{annotation_prefix}/",
                                        f"/{raw_prefix}/"))
            yield a_path, r_path
