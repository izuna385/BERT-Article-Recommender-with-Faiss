import faiss
import numpy as np
import pdb

class ArticleTitleIndexerWithFaiss:
    def __init__(self, config, mention_idx2emb, dsr, kbemb_dim=768):
        self.config = config
        self.kbemb_dim = kbemb_dim
        self.article_num = len(mention_idx2emb)
        self.mention_idx2emb = mention_idx2emb
        self.dsr = dsr
        self.search_method_for_faiss = self.config.search_method_for_faiss
        self._indexed_faiss_loader()
        self.KBmatrix, self.kb_idx2mention_idx = self._KBmatrixloader()
        self._indexed_faiss_KBemb_adder(KBmatrix=self.KBmatrix)

    def _KBmatrixloader(self):
        KBemb = np.random.randn(self.article_num, self.kbemb_dim).astype('float32')
        kb_idx2mention_idx = {}
        for idx, (mention_idx, emb) in enumerate(self.mention_idx2emb.items()):
            KBemb[idx] = emb
            kb_idx2mention_idx.update({idx: mention_idx})

        return KBemb, kb_idx2mention_idx

    def _indexed_faiss_loader(self):
        if self.search_method_for_faiss == 'indexflatl2':  # L2
            self.indexed_faiss = faiss.IndexFlatL2(self.kbemb_dim)
        elif self.search_method_for_faiss == 'indexflatip':  #
            self.indexed_faiss = faiss.IndexFlatIP(self.kbemb_dim)
        elif self.search_method_for_faiss == 'cossim':  # innerdot * Beforehand-Normalization must be done.
            self.indexed_faiss = faiss.IndexFlatIP(self.kbemb_dim)

    def _indexed_faiss_KBemb_adder(self, KBmatrix):
        if self.search_method_for_faiss == 'cossim':
            KBemb_normalized_for_cossimonly = np.random.randn(self.article_num, self.kbemb_dim).astype('float32')
            for idx, emb in enumerate(KBmatrix):
                if np.linalg.norm(emb, ord=2, axis=0) != 0:
                    KBemb_normalized_for_cossimonly[idx] = emb / np.linalg.norm(emb, ord=2, axis=0)
            self.indexed_faiss.add(KBemb_normalized_for_cossimonly)
        else:
            self.indexed_faiss.add(KBmatrix)

    def _indexed_faiss_returner(self):
        return self.indexed_faiss

    def search_with_emb(self, emb):
        _, faiss_search_candidate_result_kb_idxs = self.indexed_faiss.search(
            np.array([emb]).astype('float32'),
            self.config.how_many_top_hits_preserved)
        top_titles = []
        for kb_idx in faiss_search_candidate_result_kb_idxs:
            mention_idx = self.kb_idx2mention_idx[kb_idx]
            candidate_title = self.dsr.mention_id2data[mention_idx]['title']
            top_titles.append(candidate_title)

        pdb.set_trace()