package com.syp.interviewagent.agent;

import com.syp.interviewagent.rag.KnowledgeBaseService;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.stereotype.Service;

import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Deque;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

/**
 * 会话级 Agent 服务：管理多轮对话记忆，驱动状态机执行。
 *
 * 分层职责：本类是「图执行的外壳」——图（InterviewAgent）无状态、可并发复用；
 * 会话记忆由本层持有并注入图输入。这样设计的收益：
 *  - 记忆实现可独立演进（当前进程内 Map → 后续 Redis / LangGraph checkpoint），图代码零改动；
 *  - 同一图可同时服务多个会话，记忆天然按 sessionId 隔离。
 *
 * 并发模型：sessionHistory 用 ConcurrentHashMap 保证不同会话互不阻塞；
 * 同一会话的并发请求共用一个 Deque，写入以 synchronized(deque) 串行化。
 * 记忆只存进程内存，应用重启即丢失（演示项目可接受）。
 */
@Service
public class AgentService {

    private static final Logger log = LoggerFactory.getLogger(AgentService.class);

    /** 每会话保留的最大记忆条数：6 条 = 最近 3 轮 Q/A。太小丢上下文，太大撑爆 prompt 且引入无关干扰 */
    private static final int MAX_HISTORY = 6;

    private final InterviewAgent agent;

    /** sessionId → 会话记忆队列（按时间序，队头最旧）；computeIfAbsent 保证惰性创建 */
    private final Map<String, Deque<String>> sessionHistory = new ConcurrentHashMap<>();

    public AgentService(InterviewAgent agent) {
        this.agent = agent;
    }

    /**
     * 一次问答的完整结果。
     *
     * @param answer  最终答案文本
     * @param intent  意图路由结果（knowledge / chat）
     * @param route   实际走过的节点序列，预留用于前端展示执行轨迹（当前恒为空列表）
     * @param sources 检索引用来源；chat 路径为空
     */
    public record AgentResult(String answer, String intent, List<String> route,
                              List<KnowledgeBaseService.SearchHit> sources) {}

    /**
     * 处理一轮对话：取记忆 → 跑状态机 → 回写记忆。
     *
     * @param sessionId 会话 ID，首次出现时自动创建记忆队列
     * @param question  本轮用户问题
     * @return 答案 + 意图 + 引用来源
     */
    public AgentResult chat(String sessionId, String question) {
        // 拷贝一份快照传给图，避免图执行期间外部写队列造成不一致
        List<String> history = new ArrayList<>(sessionHistory
                .computeIfAbsent(sessionId, k -> new ArrayDeque<>()));

        long start = System.currentTimeMillis();
        Map<String, Object> result = agent.runOnce(question, history);
        long cost = System.currentTimeMillis() - start;

        String intent = String.valueOf(result.getOrDefault(GraphState.INTENT, "knowledge"));
        String answer = String.valueOf(result.getOrDefault(GraphState.ANSWER, ""));

        // 图 state 里只能放 Map（序列化约束），在这里还原回强类型 SearchHit
        @SuppressWarnings("unchecked")
        List<Map<String, Object>> sourceMaps =
                (List<Map<String, Object>>) result.getOrDefault(GraphState.SOURCES, List.of());
        List<KnowledgeBaseService.SearchHit> sources = sourceMaps.stream()
                .map(m -> new KnowledgeBaseService.SearchHit(
                        ((Number) m.get("score")).doubleValue(),
                        String.valueOf(m.get("text")),
                        String.valueOf(m.get("source")),
                        String.valueOf(m.get("section")),
                        String.valueOf(m.get("questionNo")),
                        String.valueOf(m.get("question")),
                        String.valueOf(m.get("level")),
                        String.valueOf(m.get("type"))))
                .toList();

        // 记忆写入在本轮图执行完成后进行：本轮问题/答案不进本轮的 prompt，下一轮才生效
        remember(sessionId, "用户: " + question, "肥仔: " + truncate(answer));

        log.info("会话 {} 意图={} 耗时={}ms 来源数={}", sessionId, intent, cost, sources.size());
        return new AgentResult(answer, intent, List.of(), sources);
    }

    /**
     * 向指定会话追加记忆并裁剪到 MAX_HISTORY 以内。
     * 队列可能被并发请求触碰，用 synchronized(deque) 保证 addLast/removeFirst 原子。
     *
     * @param sessionId 会话 ID；队列不存在时静默跳过（会话已被清理的场景）
     * @param lines     追加的记忆行（每轮两条：用户 + 助手）
     */
    private void remember(String sessionId, String... lines) {
        Deque<String> deque = sessionHistory.get(sessionId);
        if (deque == null) {
            return;
        }
        synchronized (deque) {
            for (String line : lines) {
                deque.addLast(line);
            }
            // 滑动窗口：从队头淘汰最旧记忆
            while (deque.size() > MAX_HISTORY) {
                deque.removeFirst();
            }
        }
    }

    /** 记忆里的答案截断到 200 字：记忆只需提供话题锚点，全文进 prompt 徒占 token */
    private String truncate(String s) {
        return s.length() <= 200 ? s : s.substring(0, 200) + "…";
    }
}
