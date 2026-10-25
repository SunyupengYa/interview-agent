# interview-agent — Java 面试知识库问答 Agent

Java 后端视角的 AI Agent 完整项目：LangChain4j + LangGraph4j + DeepSeek + 进程内中文 Embedding。

## 技术栈

| 层         | 选型                                  | 说明                                                     |
| --------- | ----------------------------------- | ------------------------------------------------------ |
| 框架        | Spring Boot 3.5.16 / Java 21        |                                                        |
| 模型接入      | LangChain4j 1.22.0（OpenAI 兼容客户端）    | DeepSeek `deepseek-chat`，baseUrl `api.deepseek.com/v1` |
| Embedding | bge-small-zh-v15（ONNX 进程内推理）        | 零外部依赖、零成本、离线；中文优化                                      |
| 向量库       | **双后端可切换**：InMemory（开发）/ **Milvus 2.4 standalone**（生产主流） | 面向 `EmbeddingStore` 接口编程，业务零改动                                  |
| Agent 编排  | LangGraph4j 1.9.6（core）             | 状态机：意图路由 → 检索 → 生成 → 校验（开发中）                           |
| 语料解析      | jsoup 1.23.2                        | 三种手册模板兼容                                               |

## 运行

```bash
export DEEPSEEK_API_KEY="sk-xxx"        # platform.deepseek.com
export VECTOR_STORE_TYPE=milvus         # memory（默认）| milvus
mvn spring-boot:run -Dspring-boot.run.jvmArguments="-Dserver.port=18080"
```

Milvus 启动（首次需要拉镜像，官方源被墙走 daocloud 镜像源）：

```bash
docker run -d --name milvus-standalone \
  -e ETCD_USE_EMBED=true -e ETCD_DATA_DIR=/var/lib/milvus/etcd \
  -e ETCD_CONFIG_PATH=/milvus/configs/embedEtcd.yaml \
  -e COMMON_STORAGETYPE=local \
  -v ./data/milvus:/var/lib/milvus \
  -v ./data/embedEtcd.yaml:/milvus/configs/embedEtcd.yaml \
  -p 19530:19530 -p 9091:9091 \
  milvusdb/milvus:v2.4.17 milvus run standalone
```

注意：`embedEtcd.yaml` 必须挂载进去（镜像里没有这个文件，缺了会 panic），内容见 `data/embedEtcd.yaml`。v2.6 的单容器 embedded etcd 配置已变更，v2.4.17 是官方脚本验证过的稳定选择。

首次启动自动扫描 `corpus/*.html` → 解析 → 向量化入库（~1000 chunk，约 10s）→ 持久化到 `data/embeddings.store`；之后再启动直接加载索引（秒级）。

## 接口

- `GET /api/search?q=主从复制&topK=5` — 纯检索调试（不需要 LLM key）
- `POST /api/chat {"message":"..."}` — RAG 问答，返回答案 + 引用来源

## 语料

`corpus/` 下 9 份面试手册（约 325 题）：JVM 冲刺 80 题、MySQL、Redis、消息队列、分布式与高可用、Spring Cloud、框架、HTTP、K8s。

## 项目进度

- [x] 骨架：Spring Boot + DeepSeek 连通（流式后补）
- [x] 语料解析：三模板兼容，1001 chunk，字段/模块归属正确
- [x] 检索：语义检索 topK + minScore 0.40
- [x] RAG 问答链路：引用标注 + 幻觉抑制（实测会正确拒答库外问题）
- [x] LangGraph4j 状态机：意图路由 classify→(retrieve→generate | chat)，多轮会话记忆
- [x] Web UI：聊天页面 + 引用来源展示 + 意图标识
- [x] 评测：hit@3 = 90%（10 用例，见 eval/）

## 评测

```bash
python3 eval/eval.py   # 服务启动后运行，输出 hit@k
```

当前 hit@3 = 90%。唯一 MISS（分布式锁）是评测标注歧义：sprint 手册与 redis 手册均有专题，top1 命中 sprint 并非检索错误。

## 简历项目描述（参考话术）

> **面试知识库问答 Agent**（个人项目 · Java 21 / Spring Boot 3 / LangChain4j / LangGraph4j / DeepSeek）
> - 基于 LangGraph4j 构建意图路由状态机（意图分类 → 向量检索 → 增强生成 / 直答），实现多轮会话记忆与上下文指代消解；
> - 自研语料解析器兼容 3 种 HTML 模板变体，将 9 份技术手册（约 325 题）结构化为 1000+ 检索块，进程内 ONNX 中文 Embedding（bge-small-zh）9 秒完成全量向量化并持久化；
> - 设计幻觉抑制机制：检索结果与问题无关时明确拒答并给查阅方向，回答强制标注引用来源；检索质量可量化评测（hit@3 = 90%）；
> - 解决 LangGraph4j 状态深拷贝序列化限制：图状态与富对象解耦，节点间仅传递可序列化数据。

### 面试可能被追问的点（提前准备）

1. **为什么图状态只能放纯类型？** → LangGraph4j 节点间对 state 深拷贝（Java 序列化），record 不实现 Serializable 会炸；解耦后图无状态、记忆外置，也方便换 Redis。
2. **Embedding 为什么不用 DeepSeek？** → DeepSeek API 不提供 embedding 接口；bge-small-zh ONNX 进程内推理零成本零延迟，中文效果好。
3. **怎么解决幻觉？** → 三层：检索 minScore 过滤 / prompt 强约束只答 context / 模型侧拒答兜底，实测 HashMap 扩容（库外题）正确拒答。
4. **检索怎么调优？** → 分块策略（题目级 3 类块 + 前缀上下文锚点 + 句子边界二级切分适配 512 token 窗口）、minScore 0.40、hit@k 评测驱动。

