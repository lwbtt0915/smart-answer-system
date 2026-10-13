

# 导入必要模块
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
import os

from DataLoaderModule import DataLoaderModule



# 嵌入生成与存储模块
class EmbeddingStorageModule:
    def __init__(self, data_loader, faiss_index_path="faiss_index"):
        """
        初始化嵌入生成与存储模块，设置数据加载模块和FAISS索引存储路径。
        :param data_loader: 已加载和清洗的DataLoaderModule对象
        :param faiss_index_path: FAISS向量存储的路径
        """
        self.data_loader = data_loader  # 加载后的数据
        self.embeddings = OpenAIEmbeddings()  # 嵌入模型
        self.faiss_index_path = faiss_index_path  # FAISS向量存储路径
        self.vector_store = None  # 初始化向量存储对象

    def generate_embeddings(self):
        """生成嵌入向量"""
        documents = self.data_loader.documents
        print(f"正在为{len(documents)}个文档生成嵌入向量...")
        document_embeddings = [self.embeddings.embed(doc.page_content) for doc in documents]
        return document_embeddings

    def create_faiss_vector_store(self):
        """创建FAISS向量存储并保存到本地"""
        # 使用生成的嵌入向量创建FAISS向量存储
        print("开始创建FAISS向量存储...")
        self.vector_store = FAISS.from_documents(self.data_loader.documents, self.embeddings)

        # 持久化向量存储到磁盘
        self.vector_store.save_local(self.faiss_index_path)
        print(f"FAISS向量存储已创建并保存至 {self.faiss_index_path}")

    def load_faiss_vector_store(self):
        """加载已保存的FAISS向量存储"""
        if os.path.exists(self.faiss_index_path):
            print(f"加载本地FAISS向量存储：{self.faiss_index_path}")
            self.vector_store = FAISS.load_local(self.faiss_index_path, self.embeddings)
        else:
            print(f"未检测到FAISS向量存储文件：{self.faiss_index_path}，请先创建。")

    def search_similar_documents(self, query, top_k=3):
        """
        根据用户查询执行相似性搜索，并返回最相似的文档内容。
        :param query: 用户查询的文本
        :param top_k: 返回的相似文档数
        :return: 最相似的文档内容列表
        """
        if not self.vector_store:
            print("FAISS向量存储尚未加载，请先创建或加载存储。")
            return []

        # 将查询文本嵌入为向量并执行相似性搜索
        query_embedding = self.embeddings.embed(query)
        similar_docs = self.vector_store.similarity_search_by_vector(query_embedding, k=top_k)

        # 打印并返回搜索结果
        print(f"\n查询 '{query}' 的相似文档内容：")
        for i, doc in enumerate(similar_docs):
            print(f"文档{i + 1}:\n{doc.page_content[:200]}...\n")  # 显示前200字符
        return [doc.page_content for doc in similar_docs]


# 模块运行测试
if __name__ == "__main__":
    # 初始化数据加载模块
    data_loader = DataLoaderModule(directory_path="../data")
    data_loader.load_all_documents(json_path="../data/sample.json", markdown_path="../data/sample.md")

    # 显示数据加载模块的文档概要
    data_loader.display_summary()

    # 初始化嵌入生成与存储模块
    embedding_storage = EmbeddingStorageModule(data_loader)

    # 生成嵌入并创建FAISS向量存储
    embedding_storage.generate_embeddings()
    embedding_storage.create_faiss_vector_store()

    # 加载并执行相似性搜索
    embedding_storage.load_faiss_vector_store()
    query = "文艺复兴的主要贡献"
    embedding_storage.search_similar_documents(query=query, top_k=3)