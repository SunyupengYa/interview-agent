package com.syp.interviewagent.api;

import com.syp.interviewagent.rag.KnowledgeBaseService;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;

import java.util.List;

/**
 * 检索调试接口：不经过 LLM，直接验证向量检索质量。
 *
 * 这是调 RAG 的第一入口 —— 检索不行，生成层怎么调都没用。
 * 排查回答质量问题时先打本接口：看 topK 命中的是哪题、score 多少、有没有被 minScore 过滤。
 */
@RestController
@RequestMapping("/api")
public class SearchController {

    private final KnowledgeBaseService kb;

    public SearchController(KnowledgeBaseService kb) {
        this.kb = kb;
    }

    /**
     * 检索响应体。
     *
     * @param totalChunks 知识库当前 chunk 总数（校验建库是否完整）
     * @param hits        命中列表，按相关度降序
     */
    public record SearchResponse(int totalChunks, List<KnowledgeBaseService.SearchHit> hits) {}

    /**
     * 语义检索知识库。
     *
     * HTTP: GET /api/search?q=xxx&topK=5，无需鉴权。
     * 纯向量计算（本地 ONNX + 向量库），无 LLM 调用，耗时通常在百毫秒级。
     * 相关度低于 minScore（0.40）的结果会被直接丢弃，hits 可能为空列表。
     *
     * @param q     查询文本（自然语言问题或关键词）
     * @param topK  返回条数上限，默认 5
     * @return chunk 总数 + 命中列表（含 score/来源手册/题号/类型等元数据）
     */
    @GetMapping("/search")
    public SearchResponse search(@RequestParam String q,
                                 @RequestParam(defaultValue = "5") int topK) {
        return new SearchResponse(kb.size(), kb.search(q, topK));
    }
}
