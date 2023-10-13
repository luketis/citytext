import numpy as np
from scipy.spatial.distance import cosine
from sentence_transformers import SentenceTransformer, util



class TextEncoder():
    def __init__(self):
        pass

    def try_wrap(self, func):
        def wrap(arg):
            try:
                return func(arg)
            except Exception as e:
                print('Occured run-time error:', e,
                      'with argument: ', arg)

        return wrap


    def encode(self, word):
        if isinstance(word, str):
            return self.encode_func(word)

        elif isinstance(word, list):
            return np.array([self.encode_func(sentence) for sentence in word])

        return word.transform(self.encode_func)

    def cosine_dist(self, word_a, word_b):
        return 1 - cosine(word_a, word_b)


class TransformerEncoder:
    def __init__(self, model_name='xlm-r-100langs-bert-base-nli-stsb-mean-tokens', device='cuda'):
        self.embedder = SentenceTransformer(model_name, device=device)
        print('max sentence lenght:', self.embedder.max_seq_length)
    
    def encode(self, text):
        return self.embedder.encode(text, convert_to_tensor=True)

    def cos_dist(self, vec1, vec2):
        return util.pytorch_cos_sim(vec1, vec2)[0]
    
    def many_cos_dist(self, vecs):
        return util.pytorch_cos_sim(vecs, vecs)
    
    def cos_dist_cartesian(self, vecs1, vecs2):
        return util.pytorch_cos_sim(vecs1, vecs2)