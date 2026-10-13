import time
from statistics import mean

from langchain_core.prompts import PromptTemplate


from src import EnterpriseQASystem
class EnterpriseQATester:
    def __init__(self, document_directory: str):
        """初始化测试环境，包括基础和优化配置"""
        self.qa_system_basic = EnterpriseQASystem.EnterpriseQASystem(document_directory=document_directory)
        self.qa_system_optimized = EnterpriseQASystem.EnterpriseQASystem(document_directory=document_directory)

    def basic_query(self, query: str):
        """
        基础配置：普通向量搜索，无自动修复解析器和输出校验
        """
        # 使用普通向量搜索
        relevant_docs = self.qa_system_basic.vector_store.similarity_search(query, k=3)
        combined_docs = "\n".join(doc.page_content for doc in relevant_docs)

        prompt_template = PromptTemplate(
            template=(
                "请回答以下问题，并提供简洁的答案和详细解释：\n"
                "问题: {question}\n"
                "相关内容:\n{context}\n"
            ),
            input_variables=["question", "context"]
        )
        prompt = prompt_template.format(question=query, context=combined_docs)

        # 调用LLM生成响应
        raw_response = self.qa_system_basic.llm(prompt)
        return raw_response

    def optimized_query(self, query: str):
        """
        优化配置：MMR搜索 + 自动修复解析器 + 输出校验
        """
        # 使用MMR搜索并生成结构化输出
        structured_answer = self.qa_system_optimized.generate_answer(query)
        if not self.qa_system_optimized.validate_output(structured_answer):
            return None
        return structured_answer

    def perform_test(self, query: str, iterations: int = 10):
        """
        执行测试，计算基础配置和优化配置的平均响应时间和成功解析率。
        :param query: 测试查询内容
        :param iterations: 每种配置的测试次数
        """
        basic_times = []
        optimized_times = []
        optimized_success_count = 0

        # 基础配置测试
        print("\n基础配置测试...")
        for i in range(iterations):
            start_time = time.time()
            response = self.basic_query(query)
            end_time = time.time()
            basic_times.append(end_time - start_time)
            print(f"基础配置第{i + 1}次查询响应时间: {end_time - start_time:.4f} 秒")

        # 优化配置测试
        print("\n优化配置测试...")
        for i in range(iterations):
            start_time = time.time()
            response = self.optimized_query(query)
            end_time = time.time()
            optimized_times.append(end_time - start_time)
            print(f"优化配置第{i + 1}次查询响应时间: {end_time - start_time:.4f} 秒")

            # 检查解析器成功输出的结果
            if response:
                optimized_success_count += 1

        # 结果统计
        print("\n测试结果统计:")
        print(f"基础配置平均响应时间: {mean(basic_times):.4f} 秒")
        print(f"优化配置平均响应时间: {mean(optimized_times):.4f} 秒")
        print(f"优化配置成功解析率: {optimized_success_count / iterations * 100:.2f}%")


# 运行测试
if __name__ == "__main__":
    document_directory = "./data/enterprise_docs"
    tester = EnterpriseQATester(document_directory=document_directory)
    query = "请简要说明公司的年度增长策略"

    # 执行测试，定量分析优化效果
    tester.perform_test(query=query, iterations=10)

