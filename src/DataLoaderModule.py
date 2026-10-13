# 导入必要模块
from langchain_community.document_loaders import DirectoryLoader, JSONLoader, TextLoader
from langchain_text_splitters import CharacterTextSplitter



# 定义数据加载和预处理模块
class  DataLoaderModule:
    def __init__(self, directory_path, chunk_size=500, chunk_overlap=50):
        """
        初始化数据加载模块，设置文件目录路径、分块大小和重叠。
        :param directory_path: 要加载的文件目录路径
        :param chunk_size: 文档分割的每个块大小
        :param chunk_overlap: 文档块之间的重叠大小
        """
        self.directory_path = directory_path
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.documents = []
        self.text_splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    def load_text_documents(self):
        """加载文本文件"""
        loader = DirectoryLoader(path=self.directory_path, glob="**/*.txt", show_progress=True)
        raw_documents = loader.load()
        self.documents.extend(self.text_splitter.split_documents(raw_documents))
        print(f"加载的文本文件数：{len(raw_documents)}")

    def load_json_documents(self, json_path):
        """加载JSON文件（新版 JSONLoader 需显式指定 jq_schema）"""
        loader = JSONLoader(file_path=json_path, jq_schema=".[]")
        json_documents = loader.load()
        split_docs = self.text_splitter.split_documents(json_documents)
        self.documents.extend(split_docs)
        print(f"加载的JSON文件条目数：{len(json_documents)}")

    def load_markdown_documents(self, markdown_path):
        """加载Markdown文件（新版 langchain-community 已移除 MarkdownLoader，改用 TextLoader）"""
        loader = TextLoader(file_path=markdown_path)
        markdown_documents = loader.load()
        split_docs = self.text_splitter.split_documents(markdown_documents)
        self.documents.extend(split_docs)
        print(f"加载的Markdown文件数：{len(markdown_documents)}")

    def validate_and_clean_data(self):
        """数据验证和清洗：去除空内容，删除重复条目"""
        # 去除空内容
        self.documents = [doc for doc in self.documents if doc.page_content.strip()]

        # 去重处理
        unique_docs = {doc.page_content: doc for doc in self.documents}
        self.documents = list(unique_docs.values())

        print(f"清洗后剩余文档数：{len(self.documents)}")

    def load_all_documents(self, json_path, markdown_path):
        """加载所有文档：文本、JSON、Markdown"""
        self.load_text_documents()
        self.load_json_documents(json_path)
        self.load_markdown_documents(markdown_path)

        # 数据清洗
        self.validate_and_clean_data()

    def display_summary(self):
        """显示数据加载和处理的结果统计"""
        print(f"总文档数：{len(self.documents)}")
        for i, doc in enumerate(self.documents[:5]):  # 显示前5个文档内容概览
            print(f"\n文档{i + 1}内容预览:\n{doc.page_content[:200]}...")  # 只显示前200字符


# 模块运行测试
if __name__ == "__main__":
    # 初始化DataLoaderModule，指定文件路径
    data_loader = DataLoaderModule(directory_path="../data")

    # 加载所有文件，并执行数据清洗
    data_loader.load_all_documents(json_path="../data/sample.json", markdown_path="../data/sample.md")

    # 输出加载与处理结果的概览
    data_loader.display_summary()