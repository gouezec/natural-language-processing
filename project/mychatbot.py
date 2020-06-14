import datasets
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import random
from utils import unpickle_file

class MyChatBot(object):
    def __init__(self, paths):
        self.embeddings, self.embeddings_dim = self._load_embeddings(paths['DIALOGUE_EMBEDDINGS'])
        self.question_vectors = unpickle_file(paths['QUESTION_VECTORS'])
        self.dialogues = datasets.readCornellData(paths['DIALOGUE_FOLDER'], max_len=100)

    def get_response(self, question):
        ranked_candidates = self._rank_candidates(question)
        i = ranked_candidates[random.choice(range(3))]
        answer = self.dialogues[i][2]
        return answer 
        
    def _load_embeddings(self, path):
        embeddings_dim = 100
        embeddings = {}
        with open(path, 'r', encoding="ascii") as file:
            for line in file:
                arr = line.split('\t')
                word = arr[0]
                vec = np.array([w for w in arr[1:]], dtype=np.float32)
                embeddings[word] = vec
        return embeddings, embeddings_dim
   
    def _question_to_vec(self, question):
        if question == '':
            return np.zeros(dim)
        e = np.array([self.embeddings[w] for w in question.split() if w in self.embeddings])
        if e.shape[0] == 0:
            return np.zeros(dim)    
        return np.mean(e, axis=0)

    def _rank_candidates(self, question):
        prepared_question = datasets.extractText(question)
        question_vector = np.array([self._question_to_vec(prepared_question)])
        cosine_matrix = cosine_similarity(question_vector, self.question_vectors)
        cosine_matrix = cosine_matrix.tolist()[0]
        l = [(idx, cosine) for idx, cosine in enumerate(cosine_matrix)]
        l = sorted(l, key=lambda t:t[1], reverse=True)
        return [idx for idx, _ in l]
