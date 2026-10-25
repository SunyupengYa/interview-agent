package com.syp.interviewagent.config;

import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.store.embedding.EmbeddingStore;
import dev.langchain4j.store.embedding.milvus.MilvusEmbeddingStore;
import io.milvus.param.IndexType;
import io.milvus.param.MetricType;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.boot.autoconfigure.condition.ConditionalOnProperty;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

/**
 * 向量库装配：通过 vector-store.type 切换 memory / milvus 两种后端。
 *
 * memory 后端是 KnowledgeBaseService 内部 new 的（不需要这里的 Bean），
 * 本类只负责 milvus 后端的装配；memory 后端零外部依赖，用于开发/CI 环境。
 *
 * 三个关键参数的选型依据：
 *  - dimension=512：必须与 bge-small-zh-v15 的输出维度一致（见 EmbeddingConfig）；
 *  - IndexType.FLAT：千级向量暴力检索（精确）足够快（毫秒级），百万级再换 HNSW/IVF；
 *  - MetricType.COSINE：bge 系列模型的训练目标就是余弦相似度，score 越接近 1 越相关。
 *
 * 注意：MilvusEmbeddingStore 在构造时就会自动创建集合（含 schema），
 * 所以业务代码里清库必须用 removeAll()（保 schema 清数据），不能 drop 集合。
 */
@Configuration
public class VectorStoreConfig {

    /**
     * 仅当 vector-store.type=milvus 时装配此 Bean（条件装配，memory 环境下不创建连接）。
     *
     * @param host       Milvus 地址（本地 standalone 为 localhost）
     * @param port       Milvus gRPC 端口，默认 19530
     * @param collection 集合名，默认 interview_kb
     * @return 512 维、FLAT、COSINE 的 Milvus 向量库客户端
     */
    @Bean
    @ConditionalOnProperty(name = "vector-store.type", havingValue = "milvus")
    public EmbeddingStore<TextSegment> milvusStore(
            @Value("${vector-store.milvus.host}") String host,
            @Value("${vector-store.milvus.port}") int port,
            @Value("${vector-store.milvus.collection}") String collection) {
        return MilvusEmbeddingStore.builder()
                .host(host)
                .port(port)
                .collectionName(collection)
                .dimension(512)
                .indexType(IndexType.FLAT)
                .metricType(MetricType.COSINE)
                .build();
    }
}
