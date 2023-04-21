#### conda env vicuna

from pymilvus import (
    connections,
    utility,
    FieldSchema,
    CollectionSchema,
    DataType,
    Collection,
)

connections.connect("default", host="xxx.xxx.xxx.xxx", port="19530")

## https://github.com/milvus-io/milvus/issues/19090
fields = [
    FieldSchema(name="pk", dtype=DataType.INT64, is_primary=True, auto_id=False),
    FieldSchema(name="random", dtype=DataType.VARCHAR, max_length=65535 ),     # DataType.DOUBLE    DataType.VARCHAR
    FieldSchema(name="embeddings", dtype=DataType.FLOAT_VECTOR, dim=8)
]
schema = CollectionSchema(fields, "hello_milvus is the simplest demo")
hello_milvus = Collection("hello_milvus", schema)

import random

max_len=10
entities = [
    [i for i in range(max_len)],  # field pk
    [str('A_')+str(float(random.randrange(-20, -10))) for _ in range(max_len)],  # field random in varchar
    # [float(random.randrange(-20, -10)) for _ in range(max_len)],  # field random
    [[random.random() for _ in range(8)] for _ in range(max_len)],  # field embeddings
]

print(' entities : ',entities)


insert_result = hello_milvus.insert(entities)
# # After final entity is inserted, it is best to call flush to have no growing segments left in memory
hello_milvus.flush()  

### Description of parameters in index creation
### ref: https://milvus.io/docs/build_index.md
### ref: https://milvus.io/docs/index.md
index = {
    "index_type": "IVF_FLAT",
    "metric_type": "L2",
    "params": {"nlist": 128},
}
hello_milvus.create_index("embeddings", index)


hello_milvus.load()
print(' shape : ',len(entities), ' ---> ',len(entities[0]))
vectors_to_search = entities[-1][-1:]
text_=entities[1][-1:]
print( '  query : ',vectors_to_search)
print( '  text : ',text_)

search_params = {
    "metric_type": "L2",
    "params": {"nprobe": 10},
}
result = hello_milvus.search(vectors_to_search, "embeddings", search_params, limit=3, output_fields=["random"])
print(' 1 ===> ',result)

# result = hello_milvus.query(expr="random > -14",limit=3, output_fields=["random", "embeddings"])
# print(' 2 ===> ',result)

# result = hello_milvus.search(vectors_to_search, "embeddings", search_params, limit=3, expr="random > -12", output_fields=["random"])
# print(' 3 ===> ',result)


### Delete content in collection
# expr = """pk in [0,3000]"""
# hello_milvus.delete(expr)

#### Drop database
# utility.drop_collection("hello_milvus")


