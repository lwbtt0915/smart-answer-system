# 单元测试1：数据加载模块测试
from unittest import TestCase
from src import DataLoaderModule,EmbeddingStorageModule,PromptEngineeringModule,CustomCallbackHandler,EnterpriseAgentSystem

class TestDataLoaderModule(TestCase):
    def setUp(self):
        # 初始化数据加载模块
        self.data_loader = DataLoaderModule.DataLoaderModule(directory_path="./test_data")
        # 加载和预处理文档
        self.data_loader.load_all_documents(json_path="./test_data/sample.json", markdown_path="./test_data/sample.md")

    def test_document_count(self):
        """测试数据加载后文档数量"""
        self.assertTrue(len(self.data_loader.documents) > 0, "数据加载失败，文档数应大于0")
        print(f"文档总数: {len(self.data_loader.documents)}")

    def test_document_cleaning(self):
        """测试文档清洗"""
        initial_count = len(self.data_loader.documents)
        self.data_loader.validate_and_clean_data()
        cleaned_count = len(self.data_loader.documents)
        self.assertTrue(cleaned_count <= initial_count, "清洗后文档数量应不大于初始数量")
        print(f"清洗后文档数: {cleaned_count}")


# 单元测试2：嵌入生成与存储模块测试
from unittest import TestCase


class TestEmbeddingStorageModule(TestCase):
    def setUp(self):
        # 初始化嵌入生成与存储模块
        self.data_loader = DataLoaderModule.DataLoaderModule(directory_path="./test_data")
        self.data_loader.load_all_documents(json_path="./test_data/sample.json", markdown_path="./test_data/sample.md")
        self.embedding_storage = EmbeddingStorageModule.EmbeddingStorageModule(data_loader=self.data_loader)

    def test_embedding_generation(self):
        """测试嵌入生成"""
        embeddings = self.embedding_storage.generate_embeddings()
        self.assertEqual(len(embeddings), len(self.data_loader.documents), "生成嵌入数量应等于文档数量")
        print(f"生成的嵌入向量数量: {len(embeddings)}")

    def test_faiss_storage_creation(self):
        """测试FAISS向量存储创建"""
        self.embedding_storage.create_faiss_vector_store()
        self.assertTrue(os.path.exists(self.embedding_storage.faiss_index_path), "FAISS索引文件未创建")
        print("FAISS索引文件已创建")


# 单元测试3：提示词工程模块测试
from unittest import TestCase


class TestPromptEngineeringModule(TestCase):
    def setUp(self):
        # 初始化提示词工程模块
        self.prompt_engineering = PromptEngineeringModule.PromptEngineeringModule()

    def test_prompt_template_creation(self):
        """测试Prompt模板创建"""
        query = "公司财务审核流程是什么？"
        prompt = self.prompt_engineering.create_prompt_template(query)
        self.assertIn("公司财务审核流程是什么", prompt, "Prompt模板生成错误")
        print("Prompt模板生成成功:", prompt)

    def test_response_parsing(self):
        """测试响应解析"""
        mock_response = '{"question": "公司财务审核流程是什么？", "answer": "财务审核流程包括预算编制、审批和执行。", "details": "详细的审核流程涉及财务部门与项目团队的多轮审批。"}'
        parsed_answer = self.prompt_engineering.parse_response(mock_response)
        self.assertEqual(parsed_answer.question, "公司财务审核流程是什么？", "问题解析错误")
        self.assertEqual(parsed_answer.answer, "财务审核流程包括预算编制、审批和执行。", "简洁回答解析错误")
        print("响应解析成功:", parsed_answer)


# 单元测试4：任务链设计模块测试
from unittest import TestCase


class TestDebuggableTaskChain(TestCase):
    def setUp(self):
        # 初始化任务链模块和回调处理程序
        self.callback_handler = CustomCallbackHandler.CustomCallbackHandler()
        self.task_chain = CustomCallbackHandler.DebuggableTaskChain(callback_handler=self.callback_handler)

    def test_chain_execution(self):
        """测试任务链执行"""
        query = "解释年度预算编制流程"
        result = self.task_chain.execute_chain(query)
        self.assertIsNotNone(result, "任务链执行失败")
        print("任务链执行成功，结果:", result)


# 单元测试5：Agent系统模块测试
from unittest import TestCase


class TestEnterpriseAgentSystem(TestCase):
    def setUp(self):
        # 初始化Agent系统
        self.document_directory = "./test_data/enterprise_docs"
        self.enterprise_agent = EnterpriseAgentSystem.EnterpriseAgentSystem(document_directory=self.document_directory)

    def test_react_agent_response(self):
        """测试ReAct Agent响应"""
        query = "年度增长策略有哪些内容？"
        response = self.enterprise_agent.execute_query(query=query, use_react=True)
        self.assertIsNotNone(response, "ReAct Agent执行失败")
        print("ReAct Agent响应成功，结果:", response)

    def test_zero_shot_agent_response(self):
        """测试Zero-shot Agent响应"""
        query = "公司的远程办公政策是什么？"
        response = self.enterprise_agent.execute_query(query=query, use_react=False)
        self.assertIsNotNone(response, "Zero-shot Agent执行失败")
        print("Zero-shot Agent响应成功，结果:", response)


# 单元测试6：回调机制与监控模块测试
from unittest import TestCase


class TestCustomCallbackHandler(TestCase):
    def setUp(self):
        # 初始化回调处理程序和任务链
        self.callback_handler = CustomCallbackHandler.CustomCallbackHandler()
        self.debuggable_task_chain = CustomCallbackHandler.DebuggableTaskChain(callback_handler=self.callback_handler)

    def test_callback_logging(self):
        """测试回调处理程序的日志记录"""
        query = "公司的年度增长计划是什么？"
        self.debuggable_task_chain.execute_chain(query)

        # 验证日志文件是否存在且内容是否正确
        log_file = self.callback_handler.log_file
        self.assertTrue(os.path.exists(log_file), "回调日志文件未创建")
        with open(log_file, "r") as f:
            log_content = f.read()
            self.assertIn("启动任务链", log_content, "日志记录内容错误")
        print("回调日志记录测试成功，日志内容:\n", log_content)


# 导入必要模块和类
import os
from unittest import TestCase


# 集成测试类
class TestEnterpriseQuestionAnsweringSystem(TestCase):
    def setUp(self):
        # 设置数据加载和向量存储路径
        self.document_directory = "./test_data/enterprise_docs"

        # 初始化各个模块
        print("初始化数据加载模块...")
        self.data_loader = DataLoaderModule.DataLoaderModule(directory_path=self.document_directory)
        self.data_loader.load_all_documents(json_path="./test_data/sample.json", markdown_path="./test_data/sample.md")

        print("初始化嵌入生成与存储模块...")
        self.embedding_storage = EmbeddingStorageModule.EmbeddingStorageModule(data_loader=self.data_loader)
        self.embedding_storage.create_faiss_vector_store()

        print("初始化提示词工程模块...")
        self.prompt_engineering = PromptEngineeringModule.PromptEngineeringModule()

        print("初始化任务链模块和回调处理程序...")
        self.callback_handler = CustomCallbackHandler.CustomCallbackHandler()
        self.task_chain = CustomCallbackHandler.DebuggableTaskChain(callback_handler=self.callback_handler)

        print("初始化企业级Agent系统...")
        self.enterprise_agent = EnterpriseAgentSystem.EnterpriseAgentSystem(document_directory=self.document_directory)

    def test_complete_qa_flow(self):
        """
        集成测试完整的问答流程，从文档加载、嵌入生成到任务链执行和Agent选择。
        """
        # 模拟用户查询
        query = "公司的年度增长策略是什么？"

        # 执行数据加载模块
        print("执行数据加载模块...")
        initial_doc_count = len(self.data_loader.documents)
        self.assertTrue(initial_doc_count > 0, "数据加载失败，文档数应大于0")
        print(f"数据加载模块成功，文档总数: {initial_doc_count}")

        # 执行嵌入生成与存储模块
        print("执行嵌入生成与存储模块...")
        embeddings = self.embedding_storage.generate_embeddings()
        self.assertEqual(len(embeddings), initial_doc_count, "生成的嵌入数量应与文档数量一致")
        print("嵌入生成与存储模块成功")

        # 执行提示词工程模块
        print("执行提示词工程模块...")
        prompt = self.prompt_engineering.create_prompt_template(query)
        self.assertIn("公司的年度增长策略", prompt, "Prompt模板生成错误")
        print("提示词工程模块成功")

        # 执行任务链模块
        print("执行任务链模块...")
        task_chain_result = self.task_chain.execute_chain(query)
        self.assertIsNotNone(task_chain_result, "任务链执行失败")
        print("任务链模块成功，结果:", task_chain_result)

        # 执行Agent系统模块
        print("执行Agent系统模块...")
        agent_response = self.enterprise_agent.execute_query(query=query, use_react=True)
        self.assertIsNotNone(agent_response, "Agent系统执行失败")
        print("Agent系统模块成功，Agent响应:", agent_response)

        # 验证回调日志记录
        print("验证回调日志记录...")
        log_file = self.callback_handler.log_file
        self.assertTrue(os.path.exists(log_file), "回调日志文件未创建")
        with open(log_file, "r") as f:
            log_content = f.read()
            self.assertIn("启动任务链", log_content, "日志记录内容错误")
        print("回调日志记录测试成功，日志内容:\n", log_content)


# 运行集成测试
if __name__ == "__main__":
    import unittest

    unittest.main()
