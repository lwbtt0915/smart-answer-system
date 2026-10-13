# 导入必要模块
try:
    from src import DocumentStorage, ZeroShotAgentSystem, ReActAgentSystem
except ImportError:
    import DocumentStorage
    import ZeroShotAgentSystem
    import ReActAgentSystem


# 集成Agent执行系统
class EnterpriseAgentSystem:
    def __init__(self, document_directory):
        """
        初始化企业级Agent系统，结合ReAct Agent和Zero-shot Agent。
        :param document_directory: 企业文档存储路径
        """
        self.document_storage = DocumentStorage.DocumentStorage(directory_path=document_directory)
        self.react_agent = ReActAgentSystem.ReActAgentSystem(document_storage=self.document_storage)
        self.zero_shot_agent = ZeroShotAgentSystem.ZeroShotAgentSystem()

    def execute_query(self, query, use_react=True):
        """
        执行企业问答查询，根据查询内容选择适当的Agent。
        :param query: 用户的查询内容
        :param use_react: 是否使用ReAct Agent，默认为True
        :return: 最终回答
        """
        # 使用文档存储库查找相关文档
        related_docs = self.document_storage.search_documents(query=query)

        # 根据是否使用ReAct选择Agent
        if use_react:
            print("\n使用ReAct Agent进行回答：")
            response = self.react_agent.generate_response(query)
        else:
            print("\n使用Zero-shot Agent进行回答：")
            response = self.zero_shot_agent.generate_response(query)

        print("\n最终回答:", response)
        return response


# 模块运行测试
if __name__ == "__main__":
    # 初始化企业级Agent系统
    document_directory = "./data/enterprise_docs"
    enterprise_agent = EnterpriseAgentSystem(document_directory=document_directory)

    # 测试企业问答用例1：使用ReAct Agent
    query1 = "请说明公司的年度战略计划"
    print("\n查询1结果：")
    enterprise_agent.execute_query(query=query1, use_react=True)

    # 测试企业问答用例2：使用Zero-shot Agent
    query2 = "公司的远程办公政策是什么？"
    print("\n查询2结果：")
    enterprise_agent.execute_query(query=query2, use_react=False)