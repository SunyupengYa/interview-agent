package com.syp.interviewagent.rag;

import dev.langchain4j.data.document.Metadata;
import dev.langchain4j.data.embedding.Embedding;
import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.model.embedding.EmbeddingModel;
import dev.langchain4j.store.embedding.EmbeddingSearchRequest;
import dev.langchain4j.store.embedding.EmbeddingSearchResult;
import dev.langchain4j.store.embedding.EmbeddingStore;
import dev.langchain4j.store.embedding.inmemory.InMemoryEmbeddingStore;
import dev.langchain4j.store.embedding.milvus.MilvusEmbeddingStore;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.ObjectProvider;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.boot.ApplicationArguments;
import org.springframework.boot.ApplicationRunner;
import org.springframework.stereotype.Service;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;

/**
 * 知识库构建与检索服务：RAG 链路的检索层核心。
 *
 * 双后端设计，由 vector-store.type 切换：
 *  - memory：进程内 InMemoryEmbeddingStore，索引持久化到本地文件，重启秒级加载（开发/CI 用）
 *  - milvus：外部 Milvus standalone（生产主流），每次启动全量重灌（本项目 1000 块仅 ~11 秒）
 *
 * 为什么检索层代码两种后端完全一致：都面向 EmbeddingStore 接口编程，
 * 换向量库（Qdrant/ES/pgvector）只是换一个 Bean，业务零改动 —— 这是分层的价值。
 *
 * 启动时序：实现 ApplicationRunner，容器就绪后自动建库/加载，建库完成前
 * search() 不会被调用（HTTP 端口尚未开始正常服务流量前的窗口可忽略）。
 */
@Service
public class KnowledgeBaseService implements ApplicationRunner {

    private static final Logger log = LoggerFactory.getLogger(KnowledgeBaseService.class);

    private final CorpusParser parser;
    private final EmbeddingModel embeddingModel;

    /**
     * milvus 后端的 Bean 通过 ObjectProvider 惰性获取：memory 环境下容器里
     * 根本没有这个 Bean（条件装配），构造器直接注入会启动失败。
     */
    private final ObjectProvider<EmbeddingStore<TextSegment>> milvusProvider;

    /** 实际使用的向量库；run() 里赋值后只读，volatile 保证跨线程可见 */
    private volatile EmbeddingStore<TextSegment> store;

    /** 当前 chunk 总数，仅用于 /api/search 的 totalChunks 展示 */
    private volatile int chunkCount = 0;

    /** 语料目录（application.yml corpus.dir），启动时扫描其中全部 .html */
    @Value("${corpus.dir}")
    private String corpusDir;

    /** 内存索引持久化文件路径（仅 memory 后端使用） */
    @Value("${kb.store-path:./data/embeddings.store}")
    private String storePath;

    /** 向量库后端类型：memory | milvus */
    @Value("${vector-store.type:memory}")
    private String storeType;

    /** Milvus 集合名 */
    @Value("${vector-store.milvus.collection:interview_kb}")
    private String milvusCollection;

    public KnowledgeBaseService(CorpusParser parser,
                                EmbeddingModel embeddingModel,
                                ObjectProvider<EmbeddingStore<TextSegment>> milvusProvider) {
        this.parser = parser;
        this.embeddingModel = embeddingModel;
        this.milvusProvider = milvusProvider;
    }

    /**
     * 启动钩子：按后端类型初始化知识库。
     * memory：有持久化文件就秒级加载，否则全量重建并落盘；
     * milvus：每次启动全量重灌（简单可靠，千级数据重建成本可忽略）。
     */
    @Override
    public void run(ApplicationArguments args) throws Exception {
        if ("milvus".equals(storeType)) {
            store = milvusProvider.getObject();
            rebuild();
        } else {
            Path storeFile = Path.of(storePath);
            if (Files.exists(storeFile)) {
                loadStore(storeFile);
            } else {
                store = new InMemoryEmbeddingStore<>();
                rebuild();
            }
        }
        log.info("向量库后端: {}, 当前 {} chunks", storeType, chunkCount);
    }

    /**
     * 全量重建知识库：解析语料 → 分批向量化 → 入库 → （memory 后端）落盘。
     *
     * synchronized：防止并发重建导致 store 数据错乱；当前只有启动钩子调用，
     * 预留给后续「POST /api/kb/rebuild」热更新场景。
     *
     * 分批 embedAll（64 条/批）：单次请求过大易超时/占内存，批间无依赖可顺序执行。
     *
     * @throws IOException 语料文件读取失败
     */
    public synchronized void rebuild() throws IOException {
        List<CorpusParser.Chunk> chunks = parseCorpus();
        if (chunks.isEmpty()) {
            log.warn("corpus 目录（{}）没有解析到任何内容，知识库为空", corpusDir);
            return;
        }

        long start = System.currentTimeMillis();
        // 清空数据但保留集合 schema（集合由 MilvusEmbeddingStore 构造时自动创建，
        // drop 掉后插入会报 100 not found）。百万级数据要真正重建 schema 时再考虑 drop+重连。
        store.removeAll();

        List<TextSegment> segments = toSegments(chunks);
        int batchSize = 64;
        for (int i = 0; i < segments.size(); i += batchSize) {
            List<TextSegment> batch = segments.subList(i, Math.min(i + batchSize, segments.size()));
            List<Embedding> embeddings = embeddingModel.embedAll(batch).content();
            store.addAll(embeddings, batch);
        }
        chunkCount = chunks.size();

        // memory 后端：序列化整个索引到磁盘 JSON，重启时秒级加载，避免每次启动重算 embedding
        if (store instanceof InMemoryEmbeddingStore<TextSegment> inMemory) {
            Path storeFile = Path.of(storePath);
            Files.createDirectories(storeFile.getParent());
            inMemory.serializeToFile(storeFile);
            log.info("内存索引已持久化到 {}", storeFile.toAbsolutePath());
        }

        log.info("知识库构建完成：{} 个 chunk，耗时 {} ms，后端 {}",
                chunks.size(), System.currentTimeMillis() - start, storeType);
    }

    /** 当前知识库 chunk 总数（展示用，非强一致读） */
    public int size() {
        return chunkCount;
    }

    /**
     * 从磁盘加载 memory 后端的持久化索引（跳过向量化的快速启动路径）。
     * 文件存在性由调用方保证；文件损坏会抛异常导致启动失败（fail-fast，比静默空库好）。
     */
    private void loadStore(Path storeFile) {
        long start = System.currentTimeMillis();
        InMemoryEmbeddingStore<TextSegment> loaded = InMemoryEmbeddingStore.fromFile(storeFile);
        store = loaded;
        chunkCount = loaded.size();
        log.info("知识库索引已加载：{} 条向量，来源 {}，耗时 {} ms",
                chunkCount, storeFile, System.currentTimeMillis() - start);
    }

    /**
     * chunk → TextSegment：把溯源元数据写进每条向量的 metadata，
     * 检索命中后可直接还原「来自哪份手册哪道题」，无需二次查表。
     */
    private List<TextSegment> toSegments(List<CorpusParser.Chunk> chunks) {
        List<TextSegment> segments = new ArrayList<>(chunks.size());
        for (CorpusParser.Chunk c : chunks) {
            Metadata meta = new Metadata()
                    .put("source", c.sourceFile())
                    .put("section", c.section())
                    .put("question_no", c.questionNo())
                    .put("question", c.question())
                    .put("level", c.level())
                    .put("type", c.type());
            segments.add(TextSegment.from(c.text(), meta));
        }
        return segments;
    }

    /**
     * 扫描语料目录下全部 .html 并解析为 chunk 列表。
     * 目录不存在或为空时返回空列表（由 rebuild 的空检查兜底告警）。
     */
    private List<CorpusParser.Chunk> parseCorpus() throws IOException {
        List<CorpusParser.Chunk> all = new ArrayList<>();
        File dir = new File(corpusDir);
        File[] files = dir.listFiles((d, name) -> name.toLowerCase().endsWith(".html"));
        if (files == null) {
            return all;
        }
        for (File f : files) {
            all.addAll(parser.parse(f));
        }
        return all;
    }

    /**
     * 单条检索命中结果。
     *
     * @param score      相关度 0~1（COSINE），越接近 1 越相关；低于 minScore 的不会出现在结果里
     * @param text       chunk 正文（已含【题目】前缀锚点）
     * @param source     来源手册文件名，如 redis-handbook.html
     * @param section    所属模块标题（h2），可能为空串（语料未标注模块时）
     * @param questionNo 题号，如 "Q12"
     * @param question   题干
     * @param level      难度标签，可能为空串
     * @param type       chunk 类型：answer（答案正文）/ follow（追问）/ pitfall（踩坑）
     */
    public record SearchHit(double score, String text, String source, String section,
                            String questionNo, String question, String level, String type) {}

    /**
     * 语义检索核心入口（Controller 与 Agent 节点共用）。
     * 流程：查询文本 → 向量化 → 向量库 top-K 检索 → 还原元数据。
     * minScore=0.40 是实测调出来的过滤线：低于 0.4 的命中基本是"沾边但不相关"，
     * 放进 prompt 只会诱导模型硬答；宁缺毋滥，配合生成层的拒答规则形成双保险。
     *
     * @param query      查询文本（自然语言）
     * @param maxResults 返回条数上限（Agent 主链路用 5）
     * @return 按 score 降序的命中列表，可能为空
     */
    public List<SearchHit> search(String query, int maxResults) {
        EmbeddingSearchResult<TextSegment> result = store.search(new EmbeddingSearchRequest(
                embeddingModel.embed(query).content(),
                maxResults,
                0.40,   // minScore：低于此相关度的直接丢弃
                null));
        return result.matches().stream()
                .map(m -> {
                    TextSegment seg = m.embedded();
                    Metadata md = seg.metadata();
                    return new SearchHit(
                            m.score(),
                            seg.text(),
                            md.getString("source"),
                            md.getString("section"),
                            md.getString("question_no"),
                            md.getString("question"),
                            md.getString("level"),
                            md.getString("type"));
                })
                .toList();
    }
}
