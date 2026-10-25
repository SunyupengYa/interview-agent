package com.syp.interviewagent.config;

import dev.langchain4j.model.embedding.EmbeddingModel;
import dev.langchain4j.model.embedding.onnx.bgesmallzhv15.BgeSmallZhV15EmbeddingModel;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

/**
 * Embedding 模型配置：bge-small-zh-v15。
 *
 * 选型理由：
 *  - 中文优化：语料全是中文面试手册，bge 中文检索效果显著好于多语言通用模型；
 *  - 进程内 ONNX 推理：模型文件随 jar 分发，零外部依赖（不需要任何 API key、不出网），
 *    千级语料全量向量化仅数秒；
 *  - 为什么不用 DeepSeek embedding：DeepSeek API 不提供 embedding 接口，这也是
 *    「一个厂商解决不了全部链路」的真实案例。
 *
 * 不变式：输出向量 512 维，必须与 VectorStoreConfig 里集合的 dimension 一致，
 * 否则 Milvus 建集合/插入时直接报维度不匹配。
 */
@Configuration
public class EmbeddingConfig {

    /**
     * 单例 Embedding 模型，底层 ONNX Runtime 会话线程安全，可被并发调用。
     *
     * @return 512 维中文 embedding 模型（bge-small-zh-v15）
     */
    @Bean
    public EmbeddingModel embeddingModel() {
        return new BgeSmallZhV15EmbeddingModel();
    }
}
