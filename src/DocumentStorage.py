# 导入必要模块
from langchain.agents import ReActAgent, ZeroShotAgent, AgentExecutor

from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings


# 定义文档存储模块
class DocumentStorage:
    def __init__(self, directory_path, index_path="faiss_index"):
        """
        初始化文档存储模块，加载文档并创建FAISS向量存储。
        :param directory_path: 企业文档的存储路径
        :param index_path: FAISS向量存储路径
        """
        self.directory_path = directory_path
        self.index_path = index_path
        self.embeddings = OpenAIEmbeddings()
        self.vector_store = self.create_vector_store()

    def create_vector_store(self):
        """加载文档并创建FAISS向量存储"""
        loader = TextLoader(self.directory_path)
        documents = loader.load()
        vector_store = FAISS.from_documents(documents, self.embeddings)
        vector_store.save_local(self.index_path)
        return vector_store

    def search_documents(self, query, top_k=3):
        """
        在向量存储中搜索与查询最相似的文档。
        :param query: 查询内容
        :param top_k: 返回的相似文档数量
        :return: 相似文档内容列表
        """
        query_embedding = self.embeddings.embed(query)
        similar_docs = self.vector_store.similarity_search_by_vector(query_embedding, k=top_k)

        # 输出文档内容
        print("\n相似文档内容:")
        for i, doc in enumerate(similar_docs):
            print(f"文档{i + 1}:\n{doc.page_content[:200]}...\n")

        return [doc.page_content for doc in similar_docs]