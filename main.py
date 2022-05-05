from lib2to3.pytree import Base
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional
from model import similarityEngine
from neo4j import GraphDatabase, Session
import config

# print(config.uri,config.password,config.user)
class user_input(BaseModel):
    client_name: str
    client_persona: str
    industry: str
    storefront_asset: str
    vertical: str
    usecase: str




def connection():
    driver = GraphDatabase.driver(uri=config.uri,auth=(config.user,config.password))
    # print(driver)
    return driver

def get_top_recomendations(top_k,user_input_response,graph_response,se):
    # if you have any common values you wanna replace the value for in the list. Use the Null Replacer
    null_mapper = {'Unknown':''}
    null_replacer = null_mapper.get

    ## User Inputs
    usecase = user_input_response.usecase
    industry = user_input_response.industry
    vertical = user_input_response.vertical
    client_name = user_input_response.client_name
    # client_details = extract_profile(client_name)

   
    # Graph Results
    demo_id_reference = [graph_response[each].get('usecase').get('demo_uuid') for each in range(0,len(graph_response))]
    usecase_reference = [graph_response[each].get('usecase').get('usecase') for each in range(0,len(graph_response))]
    industry_reference = [graph_response[each].get('demo').get('industry') for each in range(0,len(graph_response))]
    vertical_reference = [graph_response[each].get('usecase').get('vertical') for each in range(0,len(graph_response))]
    client_reference = [graph_response[each].get('demo').get('client_name') for each in range(0,len(graph_response))]
    client_reference = [null_replacer(n, n) for n in client_reference]
    # client_details_reference = [ extract_profile(n) for n in client_reference]
    # u_emb = se.get_embedding(usecase)
    # print(u_emb)
    
    usecase_sim_scores = se.get_similarity(se.get_embedding(usecase),se.get_embedding(usecase_reference))
    industry_sim_scores = se.get_similarity(se.get_embedding(industry),se.get_embedding(industry_reference))
    vertical_sim_scores  = se.get_similarity(se.get_embedding(vertical),se.get_embedding(vertical_reference))
    # client_sim_scores = se.get_similarity(se.get_embedding(client_details),se.get_embedding(client_details_reference))

    sim_scores = se.compute_weighted_similarity(usecase_sim_scores,industry_sim_scores,vertical_sim_scores)
    result = se.get_top_k(top_k,demo_id_reference,usecase_reference,sim_scores)

    print('\n\n Final Recomendatations : \n')
    return result


app = FastAPI()

@app.get("/")
def read_root():
    return {(config.uri,config.password,config.user)}


@app.get('/get_all_usecases')
def get_usecases():
    conn = connection()
    session =  conn.session()

    #Query 
    query = ''' 
    match (demo:Demo) -[:has_usecase]->(usecase:Usecase) return demo,usecase
    ''' 
    q_result = session.run(query).data()
    usecases = [q_result[each].get('usecase').get('usecase') for each in range(0,len(q_result))]
    return usecases


@app.post('/get_recos')
def get_recommendations(user_input_response : user_input):
    print(user_input_response.usecase)
    se = similarityEngine.SimilarityEngine()
    top_k = 3
    conn = connection()
    session =  conn.session()

    #Query 
    query = ''' 
    match (demo:Demo) -[:has_usecase]->(usecase:Usecase) return demo,usecase
    ''' 
    graph_response = session.run(query).data()
    
    top_recos = get_top_recomendations(top_k,user_input_response,graph_response,se)
    return top_recos




    



    
    