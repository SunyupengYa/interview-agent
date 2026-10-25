package com.syp.interviewagent;

import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;

/**
 * 应用入口：Java 后端面试辅导 Agent。
 *
 * 启动时序（由 Spring 管理）：
 *  1. 装配 ChatModel（DeepSeek）与 EmbeddingModel（bge，进程内 ONNX）
 *  2. 按配置装配向量库后端（memory / milvus，见 {@link com.syp.interviewagent.config.VectorStoreConfig}）
 *  3. KnowledgeBaseService 作为 ApplicationRunner 在容器就绪后构建/加载知识库索引
 *  4. 对外暴露 /api/chat（Agent 问答）与 /api/search（检索调试）
 */
@SpringBootApplication
public class InterviewAgentApplication {

    public static void main(String[] args) {
        SpringApplication.run(InterviewAgentApplication.class, args);
    }
}
