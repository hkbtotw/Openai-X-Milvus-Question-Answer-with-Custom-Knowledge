### conda env vicuna
### ref: https://github.com/towhee-io/examples/blob/main/nlp/question_answering/1_build_question_answering_engine.ipynb

## milvus
from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection, utility

import pandas as pd
import openai
from pypdf import PdfReader
import re 

from langchain.text_splitter import CharacterTextSplitter,  RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.docstore.document import Document
from langchain.prompts import PromptTemplate
from langchain.indexes.vectorstore import VectorstoreIndexCreator

from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI

## embedding
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.embeddings import LlamaCppEmbeddings
from openai.embeddings_utils import get_embedding

## gpt4all llm
from langchain.llms import LlamaCpp

### compute tiktoken
import tiktoken
import  numpy as np

### read PDF
from langchain.document_loaders import UnstructuredPDFLoader
from langchain.document_loaders import PyPDFLoader # for loading the pdf

## environment
import os
import time

def vector_similarity(x, y):
    """
    Returns the similarity between two vectors.
    
    Because OpenAI Embeddings are normalized to length 1, the cosine similarity is the same as the dot product.
    """
    # print(len(x),' --- ',len(y))
    return np.dot(np.array(x), np.array(y))

def get_embedding(text, model):
    result = openai.Embedding.create(
      model=model,
      input=text
    )
    return result["data"][0]["embedding"]

def compute_doc_embeddings(df, model):
    """
    Create an embedding for each row in the dataframe using the OpenAI Embeddings API.
    
    Return a dictionary that maps between each embedding vector and the index of the row that it corresponds to.
    """
    return {
        idx: get_embedding(r.content, model) for idx, r in df.iterrows()
    }

def compute_doc_embeddings_delay(df, model):
    """
    Create an embedding for each row in the dataframe using the OpenAI Embeddings API.
    
    Return a dictionary that maps between each embedding vector and the index of the row that it corresponds to.
    """
    output={}
    count=0
    for idx, r in df.iterrows():
        count=count+1
        print(' embedding : ',count,' from ',len(df))
        output.update({idx: get_embedding(r.content, model) })
        time.sleep(2)
    return output

def Construct_Context(dfData,MAX_SECTION_LEN):

    dfData.dropna(subset=['content'],inplace=True)
    most_relevant_document_sections = list(dfData['order'])

    chosen_sections = []
    chosen_sections_len = 0
    chosen_sections_indexes = []

    df=dfData.copy()
    for section_index in most_relevant_document_sections:
        # Add contexts until we run out of space.        
        document_section = df.loc[section_index]
        
        chosen_sections_len += document_section.tokens + separator_len
        # print(' section : ',document_section,' :: ',chosen_sections_len)
        if chosen_sections_len > MAX_SECTION_LEN:
            break
            
        chosen_sections.append(SEPARATOR + document_section.content.replace("\n", " "))
        # print(' chosen section : ',chosen_sections)
        chosen_sections_indexes.append(str(section_index))
            
    # Useful diagnostic information
    # print(f"Selected {len(chosen_sections)} document sections:")
    # print("\n".join(chosen_sections_indexes))
    # print(' ===== ',chosen_sections)
    return chosen_sections

def answer_with_gpt( query, prompt,COMPLETIONS_API_PARAMS):
    messages = [
        {"role" : "system", "content":"You are a chatbot, only answer the question by using the provided context. If your are unable to answer the question using the provided context, say 'I don't know'"}
    ]   

    context= ""
    for article in prompt:
        context = context + article 

    context = context + '\n\n --- \n\n + ' + query

    messages.append({"role" : "user", "content":context})

    print('  *******************************************************  ')
    print('  *******************************************************  ')
    print('  Submitted message : ',messages)
    print('  *******************************************************  ')
    print('  *******************************************************  ')

    ### conpletion API
    ### ref: https://www.debugpoint.com/openai-chatgpt-api-python/
    response = openai.ChatCompletion.create(
        model=COMPLETIONS_MODEL,
        messages=messages,
        temperature=COMPLETIONS_API_PARAMS['temperature'],
        max_tokens=COMPLETIONS_API_PARAMS['max_tokens'],
        )
    return '\n' + response['choices'][0]['message']['content']


def answer_with_completion_gpt( query, prompt):
    messages = [
        {"role" : "system", "content":"You are a chatbot, only answer the question by using the provided context. If your are unable to answer the question using the provided context, say 'I don't know'"}
    ]   

    context= ""
    for article in prompt:
        context = context + article 

    context = context + '\n\n --- \n\n + ' + query

    messages.append({"role" : "user", "content":context})

    print('  *******************************************************  ')
    print('  *******************************************************  ')
    print('  Submitted message : ',messages)
    print('  *******************************************************  ')
    print('  *******************************************************  ')

    ### conpletion API
    ### ref: https://www.debugpoint.com/openai-chatgpt-api-python/
    response = openai.Completion.create(
        engine='text-davinci-003',
        prompt=messages,
        temperature=0,
        max_token=4000,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
        stop=None    
        )
    return '\n' + response['choices'][0]['text'].strip()

#####################
def parse_pdf(file) :
    pdf = PdfReader(file)
    output = []
    for page in pdf.pages:
        text = page.extract_text()
        # Merge hyphenated words
        text = re.sub(r"(\w+)-\n(\w+)", r"\1\2", text)
        # Fix newlines in the middle of sentences
        text = re.sub(r"(?<!\n\s)\n(?!\s\n)", " ", text.strip())
        # Remove multiple newlines
        text = re.sub(r"\n\s*\n", "\n\n", text)
        output.append(text)
    return output

def parse_text(file) :
    print('---- load text data -----')
    ## read datafile and prepare for processing
    with open(base_path+'\\data\\'+data_file_name, encoding="utf8", errors='ignore') as f:
        text = f.read()
    
    # Merge hyphenated words
    text = re.sub(r"(\w+)-\n(\w+)", r"\1\2", text)
    # Fix newlines in the middle of sentences
    text = re.sub(r"(?<!\n\s)\n(?!\s\n)", " ", text.strip())
    # Remove multiple newlines
    text = re.sub(r"\n\s*\n", "\n\n", text)
        
    return text

## pymilvus
def create_milvus_collection(collection_name, dim):
    if utility.has_collection(collection_name):
        utility.drop_collection(collection_name)
    
    fields = [
    FieldSchema(name='id', dtype=DataType.VARCHAR, descrition='ids', max_length=500, is_primary=True, auto_id=False),    
    FieldSchema(name='embedding', dtype=DataType.FLOAT_VECTOR, descrition='embedding vectors', dim=dim) 
    ]
    schema = CollectionSchema(fields=fields, description='question and answer')
    collection = Collection(name=collection_name, schema=schema)

    # create IVF_FLAT index for collection.
    ## ref : https://milvus.io/docs/index.md  , Set params nlist
    index_params = {
        'metric_type':'L2',
        'index_type':"IVF_FLAT",
        'params':{"nlist":2048}    # max 65536
    }
    collection.create_index(field_name="embedding", index_params=index_params)
    return collection

## rearrange data to vector format
def Vectorize_DataFrame(dfIn, columnList, start_id):
    dfIn=dfIn[columnList].copy()
    outputList=[]
    dfIn['id']=dfIn['id'].astype(int)
    dfIn['id']=dfIn['id']+start_id
    dfIn['id']=dfIn['id'].astype(str)
    for column in columnList:
        # print(dfIn[column].values,' ------ ',type(dfIn[column].values)) 
        outputList.append(dfIn[column].values.tolist())
    return outputList

#######################################################################################################
### milvus
### connect to pymilvus
#### set up Milvus according to the documentation on its website , standalone : https://milvus.io/docs/install_standalone-operator.md
#### input host ip , ip of machine with Milvus installed
connections.connect("default", host="10.8.222.129", port="19530")   

## openai
openai.api_key='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'    #### input your oopenai API key here

EMBEDDING_MODEL = "text-embedding-ada-002"
COMPLETIONS_MODEL = "gpt-3.5-turbo"
MAX_SECTION_LEN = 2000
SEPARATOR = "\n* "
ENCODING = "gpt2"  # encoding for text-davinci-003

#######################################
### Separator in context
encoding = tiktoken.get_encoding(ENCODING)
separator_len = len(encoding.encode(SEPARATOR))
print(f"Context separator contains {separator_len} tokens")

# max_varchar_length=65535  # milvus varchar limittation
dim=1536  ### openai embeeding vector length / limittation
#####################################################################################################

#####################################################
### Specify Path and Testing data file
base_path='D:\\DataWarehouse\\GPT\\Milvus\\'

###### Local Knowledge to be embedded
# data_file_name='test1.txt'
data_file_name='test1_th.txt'
# pdf_name= "Full-เอกสารประกอบการสอน-วิชาการเมืองการปกคร.pdf"
# pdf_name = "article_20230217170704.pdf"
pdf_name = "BOTMAG6-64.pdf"

### reference table
reference_table_file='reference_file.parquet'

### Specify query
# query = "Who is the current prime minister of Thailand? and What does Prayut relate with this person"
# query = "ระบุ ชื่อนายกรัฐมนตรีของประเทศไทย หลังจากการเกิดรัฐประหาร แต่ละครั้ง และ ปีที่เกิดรัฐประหารขึ้น"   ### answer in thai is ok now, need to make sure the local knowledge is readable in utf-8
# query = "รัฐประหารในไทยเกิดเมื่อไรบ้างและเป็นอย่างไร"
# query = "พลเอกประยุทธ์ จัทร์โอชา เป็นนายกรัฐมนตรีคนที่เท่าไร ของประเทศไทย"
# query = "ต้องทำอย่างไรจึงป้องกันการเกิดรัฐประหารในไทยให้ไม่เกิดขึ้นอีก"
# query = "รัฐประหารในไทยเกิดเมื่อไรบ้างและ ควรทำอย่างไรเพื่อให้ไม่เกิดขึ้นอีก"
# query = "อธิบายบทบาทของนายกรัฐมนตรี กับ การเมืองการปกครองของไทย"
# query = "ใครคือนายกรัฐมนตรีของประเทศไทยในปัจจุบัน"
# query = " Provide qualitative estimation of Thailand's economy in 2023. "
query = "ใครคือพลเอกเปรม ติณสูลานนท์"
#####################################################
### Parameters 
### Set parameters to operate the code

if_new_data=0    ## 1 : Read data from file and save to parquet and processing / 0 : Read saved parquet file for further processing
## select new_data in txt or pdf format
if_pdf=1
if_text=0

if_insert_newdata=0               ## 1 : Insert new data to new Milvus table / 0 : Not insert any data 
if_insert_to_existingTable=0      ## 1 : Insert new data to existing Milvus table / 0 : Not insert any data 

if_query=1                      ## 1 : Search for content in local data most similar to the query string and submit it as context with query to openai / 0 : not doing anything

### Beware ? This is deleting the table
if_delete_collection=0
######################################################
if(if_delete_collection==1):
    print(' ---- Drop collection ----')
    utility.drop_collection('question_answer')
    print(' **** Dropping Done ****')
    exit()

if(if_new_data==1):
    text_data=''
    if(if_text==1):
        temp_prefix='txt'
        print('---- load text data -----')
        text_data= parse_text(base_path+'\\data\\'+data_file_name)
        print(' ------ Splitter ------')
        # text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""], chunk_overlap=0)
        texts = text_splitter.split_text(text_data)
        print(len(texts),' ---- text splitted ---- ',type(texts))
        
    elif(if_pdf==1):
        print(' ****** load pdf *****')
        temp_prefix='pdf'
        pages = parse_pdf(base_path+'\\data\\'+pdf_name)
        print(len(pages), ' ---- ',pages[0],' :: ',type(pages[0]),' :: ',type(pages))

        # loader = PyPDFLoader(base_path+'\\data\\'+pdf_name)
        # pages = loader.load_and_split()
        # # print(len(pages),' ---',pages[0].page_content,' :: ',type(pages[0].page_content),' :: ',type(pages))

        texts=[]
        print(' ------ Splitter by Page ------')
        ### textsplitter , recommended Recursive /  ref :https://python.langchain.com/en/latest/modules/indexes/text_splitters/examples/recursive_text_splitter.html
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        # text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""], chunk_overlap=0)
        for page in pages:      
            # print(page,' *////////////////////  ',text_splitter.split_text(page))                  
            texts.extend(text_splitter.split_text(page))
        print(len(texts),' ---- text splitted ---- ',type(texts))
    

    else:
        print(' ******************************** ')
        print(' ----      E r R o R         ---- ')
        print(' ******************************** ')
        exit()

    enc = tiktoken.encoding_for_model("gpt-4")
    tokens_per_section = []
    df=pd.DataFrame()
    for text in texts:
        # print(' ****************************************************** ')
        tokens = enc.encode(text)
        # print(' ==== text ===> ',text,' :: ',len(tokens))
        df=df.append({'index':text,'content':text,'tokens':len(tokens)},ignore_index=True)

    df.set_index(['index'],inplace=True)    

    ### original embeeding , not delay version (for demo)
    # document_embeddings = compute_doc_embeddings(df,EMBEDDING_MODEL)
    #####################################################
    ## test
    # df=df.head(30)
    ####################################################
    ### Operation version : due to openai limit for 60 embedding calls per minute, forcing delayed call to avoid this limit
    document_embeddings = compute_doc_embeddings_delay(df,EMBEDDING_MODEL)

    print(len(document_embeddings),' ---- embedded ----', document_embeddings,' :: ',type(document_embeddings))
    outputDf=pd.DataFrame()
    outputList=list(document_embeddings)
    for outputId in outputList:
        tokens = enc.encode(outputId)        
        outputDf=outputDf.append({'content':outputId,'vector':document_embeddings[outputId],'length':len(document_embeddings[outputId]),'tokens':len(tokens)},ignore_index=True)
    outputDf.to_parquet(os.getcwd()+'\\temp\\'+temp_prefix+'_embedding_ver2.parquet',index=False)

    try:
        print(' -- updating Existing reference table ----')
        reference_table_df=pd.read_parquet(os.getcwd()+'\\temp\\'+reference_table_file)
        reference_table_df=reference_table_df.append(outputDf).reset_index(drop=True)
        reference_table_df.to_parquet(os.getcwd()+'\\temp\\'+reference_table_file,index=False)
    except:
        print(' -- Creating new reference table --')
        outputDf.to_parquet(os.getcwd()+'\\temp\\'+reference_table_file,index=False)

    print(' ----- read new data Done : exit ----')
    exit()

else:
    if(if_text==1):
        temp_prefix='txt'
    elif(if_pdf==1):
        temp_prefix='pdf'
    else:
        print(' --- E r R o r --- ')
        exit()
    dfData=pd.read_parquet(os.getcwd()+'\\temp\\'+temp_prefix+'_embedding_ver2.parquet')
    dfData['id']=dfData.index ### id  will be used to link search result with content in original data
    ###################
    ## test
    # dfData=dfData.head(3)
    ###################    
    ### cast all to specified types as defined in schema otherwise Milvus might assume type according to the content
    dfData['vector'] = dfData['vector'].apply(lambda x: list(x))
    dfData['id']=dfData['id'].astype(str)
    print(len(dfData),' --- read in : saved embedding data ---- ',dfData.head(3),' :: ',dfData.columns)

    reference_table_df=pd.read_parquet(os.getcwd()+'\\temp\\'+reference_table_file)
    print(len(reference_table_df),' ---  reference table ----',reference_table_df.head(3),' :: ',reference_table_df.columns)
    reference_table_df.to_excel(os.getcwd()+'\\'+'check_reference_Table.xlsx',index=False)

if(if_insert_newdata==1):   
    start_id=0 
    columnList=['id','vector']
    entities=Vectorize_DataFrame(dfData, columnList, start_id)
    print(' entities : ',entities)
    collection = create_milvus_collection('question_answer', dim)
    insert_result = collection.insert(entities)
    # # After final entity is inserted, it is best to call flush to have no growing segments left in memory
    collection.flush() 
    print('Total number of inserted data is {}.'.format(collection.num_entities))

if(if_insert_to_existingTable==1):
    collection = Collection("question_answer")      # Get an existing collection according to the name. eg. "question answer" is a name of collection (table)
    collection.load(replica_number=1)    
    print('Total number of entitiy is {}.'.format(collection.num_entities))    
    columnList=['id','vector']
    start_id=int(collection.num_entities)  ### 
    entities=Vectorize_DataFrame(dfData, columnList, start_id)    
    # print(' entities : ',entities)  
    insert_result = collection.insert(entities)
    # # After final entity is inserted, it is best to call flush to have no growing segments left in memory
    collection.flush() 
    print('Total number of inserted data is {}.'.format(collection.num_entities))

if(if_query==1):
    print(' ----- querying ------')
    reference_table_df=pd.read_parquet(os.getcwd()+'\\temp\\'+reference_table_file)    
    print(len(reference_table_df),' ---  reference table ----',reference_table_df.head(3),' :: ',reference_table_df.columns)
    dfData=reference_table_df.copy()
    dfData['id']=dfData.index ### 
    dfData['id']=dfData['id'].astype(str)

    collection = Collection("question_answer")      # Get an existing collection according to the name. eg. "question answer" is a name of collection (table)
    collection.load(replica_number=1)    
    print('Total number of entitiy is {}.'.format(collection.num_entities))

    #### Get embedding of the query
    query_embedding = get_embedding(query,EMBEDDING_MODEL)    
    print(' query :: ',query,' : ',type(query_embedding),' -- ',len(query_embedding))

    ##  serach param , ref: https://milvus.io/docs/v2.1.x/index.md
    search_params = {
        "metric_type": "L2",
        "params": {"nprobe": 2048},   # 2048
    }
    result = collection.search([query_embedding], "embedding", search_params, limit= 16384 , output_fields=["id"])    ## limit= 16384
    collection.release()    
    print(' : ',result[0].ids,' :: ',result[0].distances)
    resultDf=pd.DataFrame(zip(result[0].ids, result[0].distances),columns=['id','distance'])
    resultDf=resultDf.merge(dfData,on='id',how='left')
    resultDf['order']=resultDf.index
    # print(' Output : ',resultDf.head(3))
    resultDf.to_excel(os.getcwd()+'\\'+'check_Result.xlsx',index=False)

    #### Obtain context
    chosen_sections=Construct_Context(resultDf,MAX_SECTION_LEN)
    
    COMPLETIONS_API_PARAMS = {
        # We use temperature of 0.0 because it gives the most predictable, factual answer.
        "temperature": 0.0,
        "max_tokens": 2000,    #2000
        "model": COMPLETIONS_MODEL,
    }

    #### Ask GPT with question + context constructed from local knowledge
    ### chat completion/ ref: https://platform.openai.com/docs/guides/chat
    response = answer_with_gpt(query,chosen_sections, COMPLETIONS_API_PARAMS)
    # response = answer_with_completion_gpt(query,chosen_sections)  ## still error

    print(' ------------------------------------------------ ')
    print(' ------------------------------------------------ ')
    print(' Question : ',query)
    print(' ANSWER : ',response)
    print(' ------------------------------------------------ ')
    print(' ------------------------------------------------ ')

    print(' *********** DONE ************* ')



 