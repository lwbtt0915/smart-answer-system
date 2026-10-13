
from flask import Flask, jsonify, request
import os

from src import EmbeddingStorageModule
from src import DataLoaderModule
from src import PromptEngineeringModule
from src import CustomCallbackHandler
from src import EnterpriseAgentSystem
# 初始化 Flask 应用
app = Flask(__name__)

# 模块初始化
print("初始化数据加载模块...")
data_loader = DataLoaderModule.DataLoaderModule(directory_path="./data")
data_loader.load_all_documents(json_path="./data/sample.json", markdown_path="./data/sample.md")

print("初始化嵌入生成与存储模块...")
embedding_storage = EmbeddingStorageModule.EmbeddingStorageModule(data_loader=data_loader)
embedding_storage.create_faiss_vector_store()

print("初始化提示词工程模块...")
prompt_engineering = PromptEngineeringModule.PromptEngineeringModule()

print("初始化任务链模块和回调处理程序...")
callback_handler = CustomCallbackHandler.CustomCallbackHandler()
task_chain = CustomCallbackHandler.DebuggableTaskChain(callback_handler=callback_handler)

print("初始化企业级Agent系统...")
enterprise_agent = EnterpriseAgentSystem.EnterpriseAgentSystem(document_directory="./data")

# 定义API接口

@app.route("/load_data", methods=["POST"])
def load_data():
    """API接口: 数据加载"""
    directory = request.json.get("directory", "./data")
    json_path = request.json.get("json_path", "./data/sample.json")
    markdown_path = request.json.get("markdown_path", "./data/sample.md")

    # 加载文档
    data_loader.directory_path = directory
    data_loader.load_all_documents(json_path=json_path, markdown_path=markdown_path)
    return jsonify({"status": "success", "message": "数据加载完成", "document_count": len(data_loader.documents)})

@app.route("/generate_embeddings", methods=["POST"])
def generate_embeddings():
    """API接口: 嵌入生成与存储"""
    embeddings = embedding_storage.generate_embeddings()
    embedding_storage.create_faiss_vector_store()
    return jsonify({"status": "success", "message": "嵌入生成并存储完成", "embedding_count": len(embeddings)})

@app.route("/query_chain", methods=["POST"])
def query_chain():
    """API接口: 任务链执行"""
    query = request.json.get("query", "")
    if not query:
        return jsonify({"status": "error", "message": "查询不能为空"}), 400

    result = task_chain.execute_chain(query)
    if result is None:
        return jsonify({"status": "error", "message": "任务链执行失败"}), 500
    return jsonify({"status": "success", "result": result})

@app.route("/agent_query", methods=["POST"])
def agent_query():
    """API接口: 企业级Agent问答"""
    query = request.json.get("query", "")
    use_react = request.json.get("use_react", True)

    if not query:
        return jsonify({"status": "error", "message": "查询不能为空"}), 400

    response = enterprise_agent.execute_query(query=query, use_react=use_react)
    if response is None:
        return jsonify({"status": "error", "message": "Agent执行失败"}), 500
    return jsonify({"status": "success", "response": response})

@app.route("/generate_prompt", methods=["POST"])
def generate_prompt():
    """API接口: 提示词生成"""
    query = request.json.get("query", "")
    if not query:
        return jsonify({"status": "error", "message": "查询不能为空"}), 400

    prompt = prompt_engineering.create_prompt_template(query)
    return jsonify({"status": "success", "prompt": prompt})

@app.route("/get_logs", methods=["GET"])
def get_logs():
    """API接口: 获取任务链执行日志"""
    log_file = callback_handler.log_file
    if not os.path.exists(log_file):
        return jsonify({"status": "error", "message": "日志文件不存在"}), 404

    with open(log_file, "r") as f:
        logs = f.readlines()
    return jsonify({"status": "success", "logs": logs})

# 启动Flask应用
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)











