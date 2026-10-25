package com.syp.interviewagent.api;

import com.syp.interviewagent.agent.AgentService;
import com.syp.interviewagent.rag.KnowledgeBaseService;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

import java.util.List;

/**
 * Agent 问答接口：LangGraph4j 状态机驱动（意图路由 → 检索 → 生成 / 直答）。
 *
 * 与 RagChatService 的区别：本接口走完整的 Agent 图（含意图分类与多轮记忆），
 * RagChatService 是不带图的朴素 RAG 链路（保留用于对照/演示两层方案的差异）。
 */
@RestController
@RequestMapping("/api")
public class ChatController {

    private final AgentService agentService;

    public ChatController(AgentService agentService) {
        this.agentService = agentService;
    }

    /**
     * 问答请求体。
     *
     * @param sessionId 会话 ID；同一 ID 的多次调用共享多轮记忆。传 null/空白则用 "default"
     * @param message   用户本轮输入，不可为空白（空白会导致 LLM 收到空问题）
     */
    public record ChatRequest(String sessionId, String message) {}

    /**
     * 问答响应体。
     *
     * @param answer  模型生成的答案文本（已含 [来源: ...] 行内引用标注）
     * @param intent  意图分类结果：knowledge（知识问答）/ chat（寒暄），用于前端展示路由标签
     * @param sources 检索命中的来源列表；chat 路径恒为空列表
     */
    public record ChatResponse(String answer, String intent, List<KnowledgeBaseService.SearchHit> sources) {}

    /**
     * 提交一个问题，走完整 Agent 状态机并返回答案。
     *
     * HTTP: POST /api/chat，无需鉴权（本地/演示项目）。
     * 同步阻塞调用：内部含 1-2 次 LLM 往返（分类 + 生成），正常耗时数秒，客户端应设置足够超时。
     * 副作用：更新该 sessionId 的会话记忆（见 AgentService.remember）。
     *
     * @param request 问题与可选会话 ID
     * @return 答案 + 意图 + 引用来源
     */
    @PostMapping("/chat")
    public ChatResponse chat(@RequestBody ChatRequest request) {
        String sessionId = request.sessionId() == null || request.sessionId().isBlank()
                ? "default" : request.sessionId();
        AgentService.AgentResult result = agentService.chat(sessionId, request.message());
        return new ChatResponse(result.answer(), result.intent(), result.sources());
    }
}
