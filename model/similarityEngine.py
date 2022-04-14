from sentence_transformers import SentenceTransformer, util
import numpy as np
# from transformers import AutoTokenizer, AutoModel
# import torch
# import torch.nn.functional as F


class SimilarityEngine(object):

    def __init__(self):
        self.MODEL = SentenceTransformer('sentence-transformers/all-distilroberta-v1')

    def get_embedding(self,sentences):
        embeddings = self.MODEL.encode(sentences, convert_to_tensor=True)  
        return embeddings
    
    def get_similarity(self,sentence_embedding, corpus_embeddings):
        cos_scores = util.pytorch_cos_sim(sentence_embedding, corpus_embeddings)[0]
        return cos_scores

    def compute_weighted_similarity(self,usecase_sim_scores,industry_sim_scores,vertical_sim_scores):
        cos_scores = ( industry_sim_scores*0.10 + 0.55 * usecase_sim_scores + 0.35*vertical_sim_scores ) / 3
        
        print(' \n ******** \n Scores : Before Weighted Avg. Usecase, Industry, Vertical\n' , usecase_sim_scores,industry_sim_scores,vertical_sim_scores)
        print('After Weighted Avg.' , cos_scores)
        return cos_scores

    def get_top_reco(self,top_k,cos_scores,reference_corpus):
        top_results = np.argpartition(-cos_scores, range(top_k))[0:top_k]
        #print("Sentence:", sentence, "\n")
        print("Top", top_k, "most similar sentences in corpus:")
        for idx in top_results[0:top_k]:
            print(reference_corpus[idx], "(Score: %.4f)" % (cos_scores[idx]))


    def get_top_k(self,top_k,demo_id_referece,reference_corpus,cos_scores):
        top_res = []
        top_results = np.argpartition(-cos_scores, range(top_k))[0:top_k]
        #print("Sentence:", sentence, "\n")
        print("Top", top_k, "most similar sentences in corpus:")
        for idx in top_results[0:top_k]:
            res = {}
            res['demo_uuid'] = demo_id_referece[idx]
            res['referece'] = reference_corpus[idx]
            res['score'] = format(float(cos_scores[idx]),'.4f')

            top_res.append((res))
            print(reference_corpus[idx],demo_id_referece[idx], "(Score: %.4f)" % (cos_scores[idx]))

        return top_res


    # def get_top_recomendations(self,top_k,user_input_response,graph_response):
    #     # if you have any common values you wanna replace the value for in the list. Use the Null Replacer
    #     null_mapper = {'Unknown':''}
    #     null_replacer = null_mapper.get

    #     ## User Inputs
    #     usecase = user_input_response.get("usecase")
    #     industry = user_input_response.get("industry")
    #     vertical = user_input_response.get("vertical")
    #     client_name = user_input_response.get("client_name")
    #     client_details = extract_profile(client_name)

    #     # Graph Results
    #     demo_id_reference = [graph_response[each].get('usecase').get('demo_uuid') for each in range(0,len(graph_response))]
    #     usecase_reference = [graph_response[each].get('usecase').get('usecase') for each in range(0,len(graph_response))]
    #     industry_reference = [graph_response[each].get('demo').get('industry') for each in range(0,len(graph_response))]
    #     vertical_reference = [graph_response[each].get('usecase').get('vertical') for each in range(0,len(graph_response))]
    #     client_reference = [graph_response[each].get('demo').get('client_name') for each in range(0,len(graph_response))]
    #     client_reference = [null_replacer(n, n) for n in client_reference]
    #     client_details_reference = [ extract_profile(n) for n in client_reference]

    #     usecase_sim_scores = self.get_similarity(self.get_embedding(usecase),self.get_embedding(usecase_reference))
    #     industry_sim_scores = self.get_similarity(self.get_embedding(industry),self.get_embedding(industry_reference))
    #     vertical_sim_scores  = self.get_similarity(self.get_embedding(vertical),self.get_embedding(vertical_reference))
    #     client_sim_scores = self.get_similarity(self.get_embedding(client_details),self.get_embedding(client_details_reference))

    #     sim_scores = self.compute_weighted_similarity(usecase_sim_scores,industry_sim_scores,vertical_sim_scores)
    #     result = self.get_top_k(top_k,demo_id_reference,usecase_reference,sim_scores)

    #     return result
