# 文件名：demo_standalone.py
import os
from dotenv import load_dotenv
from langchain_milvus import Milvus
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from pymilvus import connections, utility

# 加载环境变量
load_dotenv()

# 1. 连接Milvus Standalone
try:
    connections.connect(
        alias="default",
        host="localhost",  # Docker部署在本机
        port="19530",  # 检查端口映射是否正确
        timeout=10# 连接超时10秒
    )
    print(f"Milvus连接成功，版本: {utility.get_server_version()}")
except Exception as e:
    print(f"Milvus连接失败: {e}")
    exit(1)

# 2. 加载文档
loader = TextLoader("test.md", encoding="utf-8")
documents = loader.load()
print(f"成功加载文档: {len(documents)} 个")

# 3. 文档切块
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,  # 每块500字符
    chunk_overlap=50  # 重叠50字符避免语义截断
)
docs = text_splitter.split_documents(documents)
print(f"文档切分为{len(docs)} 块")

# 4. 初始化OpenAI Embedding（text-embedding-ada-002，1536维）
embeddings = OpenAIEmbeddings(
    model="text-embedding-ada-002",
    openai_api_key=os.getenv("OPENAI_API_KEY"),
    openai_api_base=os.getenv("OPENAI_BASE_URL")
)

# 测试Embedding维度
test_vector = embeddings.embed_query("测试")
print(f"Embedding维度: {len(test_vector)}")  # 应该输出1536

# 5. 创建Milvus Standalone向量存储
vector_store = Milvus.from_documents(
    documents=docs,
    embedding=embeddings,
    connection_args={"host": "localhost", "port": "19530"},
    collection_name="demo_collection"
)
print("数据入库成功")

# 6. 相似度搜索
query = "什么是RAG？"
results = vector_store.similarity_search_with_score(query, k=3)

print(f"\n查询：{query}")
for i, (doc, score) in enumerate(results, 1):
    print(f"\n结果 {i}（相似度: {score:.4f}）:")
    print(doc.page_content)