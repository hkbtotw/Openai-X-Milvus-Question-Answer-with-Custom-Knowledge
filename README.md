# Openai-X-Milvus-Question-Answer-with-Custom-Knowledge
Openai X Milvus - Question Answer with Custom Knowledge - Chatbot
 <br>
 <br>
## Directory Structure: <br>
data\: local knowledge file <br>
temp\: temporary file used in processing <br>
OpenaiXMilvus_question_and_answer_local_knowledge_ver2 : Main code <br>
test_pymilvus_ver1 :Code used to learn how to interact with Milvus <br> <br>

## How:  <br>
1.Set up Milvus standalone on local machine according to instruction on its website.  <br>
     (htps://milvus.io/docs/install_standalone-docker.md)  <br>
    Recommend: Use docker by installing docker desktop and call docker compose downloaded from this site  <br>
2.Clone repo and set up directory structure as indicated  <br>
3.Set up environment according to the requirement.txt  <br>
4.Try running the maincode to see how the program work  <br>
 <br> <br>
# note: <br>
1.Need to tune the code to obtain best performance, should you want to use it in production <br>
2.Add read PDF document function 2023-04-21  <br>
3.from Experiment: Found that using different text splitter results in different prediction results  <br>
  Best result obtains from using Recursivecharactertextsplitter for text reader and using Charactertextsplitter for pdf reader  <br>
