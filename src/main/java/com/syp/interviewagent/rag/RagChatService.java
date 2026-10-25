package com.syp.interviewagent.rag;

import dev.langchain4j.model.chat.ChatModel;
import org.springframework.stereotype.Service;

import java.util.List;
import java.util.stream.Collectors;

/**
 * 朴素 RAG 问答服务（无 Agent 图的对照链路）：检索 → 组装增强 prompt → LLM 生成 → 带引用返回。
 *
 * 与 InterviewAgent 的关系：本项目主链路走 LangGraph4j 状态机（意图路由 + 多轮记忆），
 * 本类保留「一次检索 + 一次生成」的最简 RAG 实现，用途：
 *  1. 对照演示「直接 RAG」与「Agent 编排」的差异（面试可讲的设计演进）；
 *  2. 图故障时的降级路径候选。
 *
 * 设计原则：
 *  1. 只答知识库里有的内容，检索不到就说检索不到（抑制幻觉）
 *  2. 回答必须标注引用来源（题目 + 出处手册），可溯源
 *  3. 检索层和生成层解耦：/api/search 可独立调试检索质量
 */
@Service
public class RagChatService {

    /**
     * system prompt 与 InterviewAgent.GENERATE_SYSTEM 几乎一致（少多轮记忆那条）。
     * 注意两处 prompt 是有意分开维护的：朴素链路没有 <history> 标签，规则集不该引用它。
     */
    private static final String SYSTEM_PROMPT = """
            你是「肥仔」，一个 Java 后端面试辅导 Agent。严格按以下规则回答：

            1. 只依据 <context> 标签内提供的知识库内容回答，禁止编造 context 之外的知识点。
            2. 回答结构：先给一句话核心结论，再展开要点；如果 context 里有追问/踩坑条目，挑最相关的附上。
            3. 引用标注：在用到的要点后面标注来源，格式 [来源: 手册文件名 · 题号]。
            4. 如果 context 为空或与问题无关，直接回答："知识库里没有覆盖这个问题"，并给出你认为应该查阅的方向。不要硬答。
            5. 语言：简体中文，风格务实，说人话，不说空话套话。
            """;

    private final KnowledgeBaseService kb;
    private final ChatModel chatModel;

    public RagChatService(KnowledgeBaseService kb, ChatModel chatModel) {
        this.kb = kb;
        this.chatModel = chatModel;
    }

    /**
     * RAG 回答结果。
     *
     * @param answer  生成文本（含行内 [来源: ...] 标注）
     * @param sources 检索命中的来源列表（与 answer 的行内引用对应，供前端渲染可点引用）
     */
    public record RagAnswer(String answer, List<Source> sources) {}

    /**
     * 单条引用来源。
     *
     * @param questionNo 题号（如 Q12）
     * @param question   题干
     * @param source     来源手册文件名
     * @param type       块类型：answer / follow / pitfall
     * @param score      相关度 0~1
     */
    public record Source(String questionNo, String question, String source, String type, double score) {}

    /**
     * 执行一次朴素 RAG 问答。
     * 与 Agent 链路的区别：无意图分类（一律检索）、无多轮记忆（每问独立）。
     *
     * @param question 用户问题
     * @return 答案 + 引用来源；检索无命中时 context 为空串，模型按规则 4 拒答
     */
    public RagAnswer ask(String question) {
        // ① 检索 top5（与 Agent 的 retrieve 节点同参数，保证两条链路可比）
        List<KnowledgeBaseService.SearchHit> hits = kb.search(question, 5);
        String context = hits.stream()
                .map(h -> "<doc source=\"%s\" question=\"%s\" type=\"%s\">\n%s\n</doc>"
                        .formatted(h.source(), (h.questionNo() + " " + h.question()).trim(), h.type(), h.text()))
                .collect(Collectors.joining("\n\n"));

        // ② 组装增强 prompt：<question> + <context>，与 Agent 生成节点的格式保持一致
        String userPrompt = """
                <question>%s</question>

                <context>
                %s
                </context>
                """.formatted(question, context);

        // ③ 生成
        String answer = chatModel.chat(SYSTEM_PROMPT + "\n\n" + userPrompt);

        // ④ 引用来源随答案一起返回（无论模型是否实际引用，都把候选来源给前端展示）
        List<Source> sources = hits.stream()
                .map(h -> new Source(h.questionNo(), h.question(), h.source(), h.type(), h.score()))
                .toList();
        return new RagAnswer(answer, sources);
    }
}
