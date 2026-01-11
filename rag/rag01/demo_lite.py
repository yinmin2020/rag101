# 文件名：demo_lite.py
import os
from dotenv import load_dotenv
from langchain_milvus import Milvus
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader

# 加载环境变量
load_dotenv()

# 1. 加载文档
loader = TextLoader("test.md", encoding="utf-8")
documents = loader.load()

# 2. 文档切块
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,  # 每块500字符
    chunk_overlap=50  # 重叠50字符避免语义截断
)
docs = text_splitter.split_documents(documents)
print(f"文档切分为{len(docs)} 块")

# 3. 初始化OpenAI Embedding（text-embedding-ada-002，1536维）
embeddings = OpenAIEmbeddings(
    model="text-embedding-ada-002",
    openai_api_key=os.getenv("OPENAI_API_KEY"),
    openai_api_base=os.getenv("OPENAI_BASE_URL")
)

# 测试Embedding维度
test_vector = embeddings.embed_query("测试")
print(f"Embedding维度: {len(test_vector)}")  # 应该输出1536

# 4. 创建Milvus Lite向量存储（数据存在本地）
vector_store = Milvus.from_documents(
    documents=docs,
    embedding=embeddings,
    connection_args={"uri": "./milvus_demo.db"}# Lite模式直接指定文件路径
)
print("数据入库成功")

# 5. 相似度搜索
query = "什么是RAG？"
results = vector_store.similarity_search_with_score(query, k=3)

print(f"\n查询：{query}")
for i, (doc, score) in enumerate(results, 1):
    print(f"\n结果 {i}（相似度: {score:.4f}）:")
    print(doc.page_content)
